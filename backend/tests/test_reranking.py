"""Unit and Integration Tests for Cross-Encoder Reranking (Feature 10).

Covers:
    - RerankerService singleton instantiation and properties.
    - Synchronous and asynchronous reranking via local cross-encoder.
    - Top 5 candidate truncation and rank assignment.
    - Inference latency logging.
    - Core requirement test: Top result changes appropriately for a query where initial
      keyword/vector rank and cross-encoder semantic relevance disagree.
    - Empty candidate list and empty query edge cases.
    - End-to-end retrieve_and_rerank pipeline.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import numpy as np
import pytest

from app.core.config import settings
from app.models.document import Document, DocumentACL, DocumentChunk
from app.models.user import User
from app.services.embeddings import EmbeddingService
from app.services.reranking import (
    DEFAULT_RERANK_TOP_N,
    RerankerService,
    get_reranker_service,
    rerank_candidates,
)
from app.services.retrieval import (
    RetrievalResult,
    hybrid_search,
    retrieve_and_rerank,
)


# ---------------------------------------------------------------------------
# Mock CrossEncoder for Fast Deterministic Unit Tests
# ---------------------------------------------------------------------------
class DummyCrossEncoderModel:
    """Deterministic Mock CrossEncoder computing relevance scores based on semantic match."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        self.model_name = model_name

    def predict(self, pairs: List[List[str]]) -> List[float]:
        scores = []
        for query, text in pairs:
            q_lower = query.lower()
            t_lower = text.lower()

            score = 0.0
            # Meaning match: deeper context matching
            if "primary caregiver" in q_lower and "primary caregivers" in t_lower:
                score += 8.5
            if "welcoming a new child" in q_lower or "birth or adoption" in t_lower:
                score += 4.0
            if "duration of paid time off" in q_lower and "16 weeks of fully paid leave" in t_lower:
                score += 9.0

            # Superficial keyword match: low semantic score
            if "keyword spam" in t_lower or "unrelated mention" in t_lower:
                score -= 5.0

            # Baseline noise for deterministic variance
            score += float(len(t_lower) % 10) * 0.01
            scores.append(score)
        return scores


@pytest.fixture
def mock_reranker_service():
    """Create a RerankerService instance with a mocked CrossEncoder."""
    service = RerankerService(model_name=settings.RERANKER_MODEL_NAME)
    service._model = DummyCrossEncoderModel(model_name=settings.RERANKER_MODEL_NAME)
    return service


# ---------------------------------------------------------------------------
# Service Unit Tests
# ---------------------------------------------------------------------------
def test_singleton_reranker_service():
    """Verify get_reranker_service returns singleton instance with correct model name."""
    s1 = get_reranker_service()
    s2 = get_reranker_service()
    assert s1 is s2
    assert s1.model_name == "BAAI/bge-reranker-large"


def test_rerank_candidates_sorts_and_truncates_to_top_5(mock_reranker_service):
    """Verify reranking scores candidates, assigns 1-based ranks, and returns top 5."""
    query = "What is the duration of paid time off for primary caregivers welcoming a new child?"

    candidates = [
        RetrievalResult(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            version_id=uuid.uuid4(),
            chunk_text="Irrelevant text 1 about server maintenance and network cables.",
            rrf_score=0.03,
        ),
        RetrievalResult(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            version_id=uuid.uuid4(),
            chunk_text="Parental leave policy provides 16 weeks of fully paid leave for primary caregivers after birth or adoption.",
            rrf_score=0.01,  # Lower initial rank
        ),
        RetrievalResult(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            version_id=uuid.uuid4(),
            chunk_text="Travel expense policy reimbursement within 30 days.",
            rrf_score=0.02,
        ),
        RetrievalResult(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            version_id=uuid.uuid4(),
            chunk_text="Office parking permits and visitor garage access.",
            rrf_score=0.025,
        ),
        RetrievalResult(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            version_id=uuid.uuid4(),
            chunk_text="Kitchen cleanliness and refrigerator cleaning schedule.",
            rrf_score=0.028,
        ),
        RetrievalResult(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            version_id=uuid.uuid4(),
            chunk_text="Security badges and visitor sign-in logs.",
            rrf_score=0.027,
        ),
        RetrievalResult(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            version_id=uuid.uuid4(),
            chunk_text="Holiday calendar for upcoming calendar year.",
            rrf_score=0.026,
        ),
    ]

    reranked = mock_reranker_service.rerank(query=query, candidates=candidates, top_n=5)

    # Must return exactly top 5 chunks
    assert len(reranked) == 5

    # Top result should be the parental leave chunk
    assert "16 weeks of fully paid leave" in reranked[0].chunk_text
    assert reranked[0].rerank_rank == 1
    assert reranked[0].rerank_score is not None

    # Check rank ordering
    for i, c in enumerate(reranked, start=1):
        assert c.rerank_rank == i
        assert c.rerank_score is not None
        if i > 1:
            assert c.rerank_score <= reranked[i - 2].rerank_score


def test_rerank_reorders_when_keyword_and_semantic_rank_disagree(mock_reranker_service):
    """CORE REQUIREMENT: Top result changes appropriately when keyword search and semantic relevance disagree.

    Scenario:
    - Candidate A has repeated keyword match ('leave leave leave') but is completely off-topic (unrelated mention).
      In initial hybrid search, Candidate A was at rank 1 due to high keyword frequency.
    - Candidate B is the actual policy chunk answering the exact question ('16 weeks paid leave for primary caregivers').
      In initial hybrid search, Candidate B was at rank 3.
    - After cross-encoder reranking, Candidate B must be elevated to Rank 1.
    """
    query = "What is the duration of paid time off for primary caregivers welcoming a new child?"

    # Initial hybrid search results where keyword spammer is Rank 1
    chunk_a_spam = RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        chunk_text="Keyword spam: leave leave leave time off child paid primary caregiver unrelated mention without any actual policy details.",
        rrf_score=0.050,  # Rank 1 initially
        keyword_rank=1,
        vector_rank=5,
    )
    chunk_b_target = RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        chunk_text="Parental leave policy provides 16 weeks of fully paid leave for primary caregivers following the birth or adoption of a child.",
        rrf_score=0.025,  # Rank 2 initially
        keyword_rank=4,
        vector_rank=1,
    )
    chunk_c_other = RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        chunk_text="Annual vacation policy: employees accrue 15 days of PTO per calendar year.",
        rrf_score=0.020,
        keyword_rank=3,
        vector_rank=4,
    )

    initial_candidates = [chunk_a_spam, chunk_b_target, chunk_c_other]

    # Pre-condition: Candidate A is currently at rank 1 in the list
    assert initial_candidates[0].chunk_id == chunk_a_spam.chunk_id

    # Execute cross-encoder reranking
    reranked = mock_reranker_service.rerank(query=query, candidates=initial_candidates, top_n=5)

    # Post-condition: Candidate B is promoted to Rank 1 due to cross-encoder semantic score
    assert len(reranked) == 3
    assert reranked[0].chunk_id == chunk_b_target.chunk_id
    assert reranked[0].rerank_rank == 1
    assert "16 weeks of fully paid leave" in reranked[0].chunk_text
    assert reranked[0].rerank_score > reranked[1].rerank_score

    # Candidate A is penalized/demoted
    assert reranked[0].chunk_id != chunk_a_spam.chunk_id


def test_rerank_empty_and_edge_cases(mock_reranker_service):
    """Verify reranking handles empty lists, whitespace queries, and small lists gracefully."""
    # 1. Empty candidates list
    assert mock_reranker_service.rerank("some query", []) == []

    # 2. Empty query string returns original candidates up to top_n
    candidates = [
        RetrievalResult(chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), version_id=uuid.uuid4(), chunk_text=f"Doc {i}")
        for i in range(3)
    ]
    res_empty_q = mock_reranker_service.rerank("   ", candidates, top_n=2)
    assert len(res_empty_q) == 2

    # 3. Fewer than top_n candidates
    res_small = mock_reranker_service.rerank("valid query", candidates, top_n=5)
    assert len(res_small) == 3


def test_rerank_logs_inference_time(mock_reranker_service, caplog):
    """Verify reranker logs inference execution time for observability."""
    query = "Test query for latency logging"
    candidates = [
        RetrievalResult(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            version_id=uuid.uuid4(),
            chunk_text="Test chunk content for latency measurement.",
        )
    ]

    with caplog.at_level(logging.INFO):
        mock_reranker_service.rerank(query=query, candidates=candidates, top_n=5)

    # Verify log output mentions latency in ms
    assert any("Reranked 1 candidates in" in record.message for record in caplog.records)
    assert any("ms (model=" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_arerank_async(mock_reranker_service):
    """Verify asynchronous arerank function produces expected results."""
    query = "What is the duration of paid time off for primary caregivers?"
    candidates = [
        RetrievalResult(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            version_id=uuid.uuid4(),
            chunk_text="Parental leave policy provides 16 weeks of fully paid leave for primary caregivers.",
        ),
        RetrievalResult(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            version_id=uuid.uuid4(),
            chunk_text="Office supply reimbursement rules.",
        ),
    ]

    results = await rerank_candidates(query=query, candidates=candidates, top_n=5, reranker_service=mock_reranker_service)
    assert len(results) == 2
    assert results[0].rerank_rank == 1
    assert "16 weeks of fully paid leave" in results[0].chunk_text


@pytest.mark.asyncio
async def test_retrieve_and_rerank_pipeline():
    """Verify retrieve_and_rerank executes hybrid retrieval followed by cross-encoder reranking."""
    user = User(
        id=uuid.uuid4(),
        name="Alice Employee",
        email="alice@insidehq.com",
        role="employee",
        department="Engineering",
    )
    doc_id = uuid.uuid4()
    mock_db = AsyncMock()

    candidate = RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        version_id=uuid.uuid4(),
        chunk_text="Engineering on-call rotation guidelines: primary on-call shifts are 7 days.",
        rrf_score=0.03,
    )

    mock_reranker = RerankerService(model_name="BAAI/bge-reranker-large")
    mock_reranker._model = DummyCrossEncoderModel()

    with patch("app.services.retrieval.hybrid_search", new_callable=AsyncMock) as mock_hybrid:
        mock_hybrid.return_value = [candidate]

        final_results = await retrieve_and_rerank(
            db=mock_db,
            query="How long are engineering on-call shifts?",
            user=user,
            top_k_candidates=20,
            top_n_rerank=5,
            reranker_service=mock_reranker,
        )

        assert mock_hybrid.called
        assert len(final_results) == 1
        assert final_results[0].rerank_rank == 1
        assert final_results[0].rerank_score is not None
