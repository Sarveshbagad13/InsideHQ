"""Unit and Integration Tests for Model Fallback & Redundancy (Feature 14).

Tests:
1. Exponential backoff retry around Gemini and fallback on repeated failure:
   - Gemini fails transiently, succeeds on retry.
   - Gemini fails repeatedly, returns clearly labeled raw chunk text and citations.
   - Streaming mode fallback when Gemini is unreachable.
2. Cross-encoder reranker failure fallback:
   - Falls back to Reciprocal Rank Fusion (RRF) candidate order and logs trigger.
3. Embedding model failure fallback:
   - Falls back to PostgreSQL keyword-only full-text search and logs trigger.
4. Startup local models health check:
   - Verifies health endpoint reports model loading status.
"""

from __future__ import annotations

import json
import logging
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_current_user, get_db
from app.core.security import create_access_token
from app.main import app
from app.models.user import User
from app.services.embeddings import EmbeddingService
from app.services.generation import (
    FALLBACK_LABEL,
    GenerationService,
    build_fallback_answer_and_citations,
    extract_all_chunk_citations,
)
from app.services.reranking import RerankerService
from app.services.retrieval import RetrievalResult, hybrid_search, retrieve_and_rerank


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def test_user():
    """Create test employee user."""
    return User(
        id=uuid.uuid4(),
        name="Fallback Tester",
        email="fallback_tester@insidehq.com",
        role="employee",
        department="Engineering",
        is_active=True,
    )


@pytest.fixture
def auth_headers(test_user):
    """Generate valid Bearer authorization header."""
    token = create_access_token(
        subject=test_user.id,
        role=test_user.role,
        department=test_user.department,
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_candidates():
    """Create sample retrieval candidates with distinct RRF scores."""
    c1 = RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        chunk_text="Parental leave provides 16 weeks of paid leave for primary caregivers.",
        chunk_metadata={"filename": "parental_leave.pdf", "section_heading": "Primary Leave"},
        document_title="Employee Handbook",
        rrf_score=0.045,
        vector_rank=1,
        keyword_rank=2,
    )
    c2 = RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        chunk_text="Secondary caregivers are granted 8 weeks of paid leave.",
        chunk_metadata={"filename": "parental_leave.pdf", "section_heading": "Secondary Leave"},
        document_title="Employee Handbook",
        rrf_score=0.038,
        vector_rank=2,
        keyword_rank=3,
    )
    c3 = RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        chunk_text="Unpaid parental leave can be extended up to an additional 12 weeks.",
        chunk_metadata={"filename": "parental_leave.pdf", "section_heading": "Extended Leave"},
        document_title="Employee Handbook",
        rrf_score=0.025,
        vector_rank=4,
        keyword_rank=4,
    )
    return [c1, c2, c3]


class MockStreamChunk:
    def __init__(self, text: str):
        self.text = text


# ---------------------------------------------------------------------------
# 1. Gemini Retry with Backoff & Fallback on Repeated Failure
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gemini_retry_succeeds_on_transient_failure(sample_candidates, caplog):
    """Verify Gemini calls retry on transient failure and succeed if subsequent attempt passes."""
    c1 = sample_candidates[0]
    call_attempts = 0

    def flaky_generate(*args, **kwargs):
        nonlocal call_attempts
        call_attempts += 1
        if call_attempts == 1:
            raise ConnectionError("Transient network timeout")
        return [MockStreamChunk(f"Answer grounded in [{c1.chunk_id}].")]

    service = GenerationService()
    service._client = MagicMock()
    service._client.models.generate_content_stream.side_effect = flaky_generate

    with caplog.at_level(logging.WARNING):
        answer, citations = await service.agenerate_answer(
            question="What is parental leave?",
            chunks=sample_candidates,
        )

    # Should have retried and succeeded on attempt 2
    assert call_attempts == 2
    assert f"[{c1.chunk_id}]" in answer
    assert len(citations) == 1
    assert "Retrying in" in caplog.text


@pytest.mark.asyncio
async def test_gemini_repeated_failure_falls_back_to_raw_chunks_and_citations(sample_candidates, caplog):
    """Verify that when Gemini repeatedly fails across all retries, service returns labeled fallback text + citations."""
    c1 = sample_candidates[0]
    c2 = sample_candidates[1]

    service = GenerationService()
    service._client = MagicMock()
    service._client.models.generate_content_stream.side_effect = RuntimeError("Gemini 503 Service Unavailable")

    with caplog.at_level(logging.WARNING):
        answer, citations = await service.agenerate_answer(
            question="What is the leave policy?",
            chunks=[c1, c2],
        )

    # Must contain clear fallback banner
    assert FALLBACK_LABEL in answer
    assert "The generative AI assistant is temporarily unavailable" in answer
    assert "Parental leave provides 16 weeks" in answer
    assert f"[{c1.chunk_id}]" in answer
    assert f"[{c2.chunk_id}]" in answer

    # Must provide structured citations for all retrieved chunks
    assert len(citations) == 2
    assert citations[0].chunk_id == str(c1.chunk_id)
    assert citations[1].chunk_id == str(c2.chunk_id)

    # Verify fallback was logged
    assert "Fallback Triggered" in caplog.text


@pytest.mark.asyncio
async def test_gemini_streaming_repeated_failure_fallback(sample_candidates, caplog):
    """Verify streaming generation yields fallback text and citations when Gemini is unreachable."""
    service = GenerationService()
    service._client = MagicMock()
    service._client.models.generate_content_stream.side_effect = RuntimeError("API connection failure")

    with caplog.at_level(logging.WARNING):
        events = []
        async for event in service.astream_answer(
            question="What is the leave policy?",
            chunks=sample_candidates,
        ):
            events.append(event)

    token_events = [e for e in events if e["type"] == "token"]
    citation_events = [e for e in events if e["type"] == "citations"]
    done_events = [e for e in events if e["type"] == "done"]

    assert len(token_events) == 1
    assert FALLBACK_LABEL in token_events[0]["content"]
    assert len(citation_events) == 1
    assert len(citation_events[0]["citations"]) == 3
    assert len(done_events) == 1
    assert FALLBACK_LABEL in done_events[0]["full_answer"]
    assert "Fallback Triggered in stream" in caplog.text


# ---------------------------------------------------------------------------
# 2. Reranker Failure -> Fallback to RRF Order
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_reranker_failure_falls_back_to_rrf_order(test_user, sample_candidates, caplog):
    """Verify when cross-encoder reranker raises an exception, retrieve_and_rerank falls back to RRF ordering."""
    mock_db = AsyncMock()

    # Mock hybrid_search returning candidates sorted by RRF score
    with patch("app.services.retrieval.hybrid_search", new_callable=AsyncMock) as mock_hybrid:
        mock_hybrid.return_value = sample_candidates

        # Mock reranker service that raises an exception
        mock_reranker = MagicMock(spec=RerankerService)
        mock_reranker.arerank.side_effect = RuntimeError("CUDA Out of Memory in Reranker")

        with caplog.at_level(logging.WARNING):
            results = await retrieve_and_rerank(
                db=mock_db,
                query="What is the parental leave policy?",
                user=test_user,
                top_k_candidates=20,
                top_n_rerank=2,
                reranker_service=mock_reranker,
            )

        # Must return top 2 in original RRF order
        assert len(results) == 2
        assert results[0].chunk_id == sample_candidates[0].chunk_id
        assert results[1].chunk_id == sample_candidates[1].chunk_id
        assert results[0].rerank_rank == 1
        assert results[1].rerank_rank == 2

        # Verify fallback logging
        assert "Fallback Triggered: Cross-encoder reranker failed" in caplog.text


# ---------------------------------------------------------------------------
# 3. Embedding Failure -> Fallback to Keyword-Only Search
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_embedding_failure_falls_back_to_keyword_search(test_user, sample_candidates, caplog):
    """Verify when embedding model fails, hybrid_search falls back to PostgreSQL keyword-only search."""
    mock_db = AsyncMock()

    # Mock embedding service that raises an exception on aembed_query
    mock_embed_svc = MagicMock(spec=EmbeddingService)
    mock_embed_svc.aembed_query.side_effect = RuntimeError("Embedding model weights corrupt / unreachable")

    # Mock search_keyword_candidates returning keyword results
    keyword_candidate = sample_candidates[0]
    with patch("app.services.retrieval.search_keyword_candidates", new_callable=AsyncMock) as mock_kw_search:
        mock_kw_search.return_value = [keyword_candidate]

        with caplog.at_level(logging.WARNING):
            results = await hybrid_search(
                db=mock_db,
                query="parental leave policy",
                user=test_user,
                top_k_candidates=20,
                final_top_k=5,
                embedding_service=mock_embed_svc,
            )

        assert len(results) == 1
        assert results[0].chunk_id == keyword_candidate.chunk_id
        assert results[0].keyword_rank == 1
        assert results[0].rrf_score > 0.0

        # Verify fallback logging
        assert "Fallback Triggered: Embedding service failed" in caplog.text


# ---------------------------------------------------------------------------
# 4. Startup & Health-Check Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_health_check_reports_models_status():
    """Verify /health and /api/v1/health endpoints report model loading statuses."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response_root = await client.get("/health")
        assert response_root.status_code == 200
        data_root = response_root.json()
        assert "status" in data_root
        assert "models" in data_root
        assert "embeddings" in data_root["models"]
        assert "reranker" in data_root["models"]

        response_api = await client.get("/api/v1/health")
        assert response_api.status_code == 200
        data_api = response_api.json()
        assert data_api["status"] in ("healthy", "degraded")
        assert "embeddings" in data_api["models"]
        assert "reranker" in data_api["models"]
