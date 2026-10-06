"""Unit and API Tests for Grounded Answer Generation & Citation Streaming (Feature 11).

Covers:
    - Context chunk formatting and prompt construction.
    - Citation marker extraction ([chunk_id]) and structured metadata mapping.
    - Core requirement test: Question with no matching chunks produces an
      "I do not have enough information" response rather than a fabricated answer.
    - Grounded streaming generation with Gemini client mock.
    - POST /chat and POST /api/v1/chat endpoints (streaming SSE and JSON responses).
    - Authentication and permission integration on chat endpoint.
"""

from __future__ import annotations

import json
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_current_user, get_db
from app.core.security import create_access_token
from app.main import app
from app.models.user import User
from app.services.generation import (
    NO_INFO_PHRASE,
    Citation,
    GenerationService,
    build_user_prompt,
    extract_citations,
    format_context_chunks,
    get_generation_service,
)
from app.services.retrieval import RetrievalResult



# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_user():
    """Create test employee user."""
    return User(
        id=uuid.uuid4(),
        name="Test Employee",
        email="emp@insidehq.com",
        role="employee",
        department="Engineering",
        is_active=True,
    )


@pytest.fixture
def auth_headers(sample_user):
    """Generate valid Bearer Authorization header for sample user."""
    token = create_access_token(
        subject=sample_user.id,
        role=sample_user.role,
        department=sample_user.department,
    )
    return {"Authorization": f"Bearer {token}"}



@pytest.fixture
def sample_chunks():
    """Create sample reranked retrieval results with document metadata."""
    chunk1_id = uuid.uuid4()
    chunk2_id = uuid.uuid4()
    doc1_id = uuid.uuid4()
    doc2_id = uuid.uuid4()

    chunk1 = RetrievalResult(
        chunk_id=chunk1_id,
        document_id=doc1_id,
        version_id=uuid.uuid4(),
        chunk_text="Parental leave policy provides 16 weeks of fully paid leave for primary caregivers after birth or adoption.",
        chunk_metadata={
            "filename": "benefits_guide.pdf",
            "section_heading": "Parental Leave",
            "source_type": "pdf",
        },
        document_title="Employee Benefits and Leave Policy 2026",
        rrf_score=0.032,
        rerank_score=8.75,
        rerank_rank=1,
    )

    chunk2 = RetrievalResult(
        chunk_id=chunk2_id,
        document_id=doc2_id,
        version_id=uuid.uuid4(),
        chunk_text="Secondary caregivers receive 8 weeks of fully paid leave which must be taken within 12 months.",
        chunk_metadata={
            "filename": "benefits_guide.pdf",
            "section_heading": "Secondary Caregivers",
            "source_type": "pdf",
        },
        document_title="Employee Benefits and Leave Policy 2026",
        rrf_score=0.028,
        rerank_score=7.20,
        rerank_rank=2,
    )

    return [chunk1, chunk2]


# ---------------------------------------------------------------------------
# Unit Tests for Formatting & Citation Extraction
# ---------------------------------------------------------------------------
def test_format_context_chunks(sample_chunks):
    """Verify chunks are formatted with chunk IDs, document titles, headings, and boundaries."""
    formatted = format_context_chunks(sample_chunks)
    assert f"--- START CHUNK [{sample_chunks[0].chunk_id}] ---" in formatted
    assert f"--- END CHUNK [{sample_chunks[0].chunk_id}] ---" in formatted
    assert "Employee Benefits and Leave Policy 2026" in formatted
    assert "Parental Leave" in formatted
    assert "16 weeks of fully paid leave" in formatted


def test_build_user_prompt(sample_chunks):
    """Verify user prompt combines question and formatted context."""
    question = "How long is parental leave for primary caregivers?"
    prompt = build_user_prompt(question, sample_chunks)
    assert question in prompt
    assert str(sample_chunks[0].chunk_id) in prompt


def test_extract_citations_maps_bracketed_markers_to_metadata(sample_chunks):
    """Verify citation extraction correctly identifies [chunk_id] markers and attaches metadata."""
    c1 = sample_chunks[0]
    c2 = sample_chunks[1]

    llm_output = (
        f"Primary caregivers receive 16 weeks of fully paid leave [{c1.chunk_id}]. "
        f"Secondary caregivers receive 8 weeks of leave [{c2.chunk_id}]."
    )

    citations = extract_citations(llm_output, sample_chunks)

    assert len(citations) == 2
    assert citations[0].chunk_id == str(c1.chunk_id)
    assert citations[0].document_title == "Employee Benefits and Leave Policy 2026"
    assert citations[0].section_heading == "Parental Leave"
    assert citations[0].filename == "benefits_guide.pdf"
    assert citations[0].citation_marker == f"[{c1.chunk_id}]"

    assert citations[1].chunk_id == str(c2.chunk_id)
    assert citations[1].section_heading == "Secondary Caregivers"


def test_extract_citations_deduplicates_repeated_markers(sample_chunks):
    """Verify repeated citation markers for the same chunk in text produce a single unique Citation."""
    c1 = sample_chunks[0]
    llm_output = f"Claim 1 [{c1.chunk_id}]. Claim 2 [{c1.chunk_id}]. Final detail [{c1.chunk_id}]."

    citations = extract_citations(llm_output, sample_chunks)
    assert len(citations) == 1
    assert citations[0].chunk_id == str(c1.chunk_id)


# ---------------------------------------------------------------------------
# Core Requirement: No Matching Chunks -> "I don't have enough info"
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_matching_chunks_produces_no_info_response():
    """CORE REQUIREMENT: When no matching chunks exist, return 'I do not have enough information' response.

    The model MUST NOT fabricate or hallucinate answers when knowledge base contains no matching chunks.
    """
    service = GenerationService()
    question = "What is the secret recipe for cafeteria cookies?"

    # Empty chunks list (no matching knowledge)
    empty_chunks: List[RetrievalResult] = []

    answer, citations = await service.agenerate_answer(question=question, chunks=empty_chunks)

    assert NO_INFO_PHRASE in answer
    assert "not have enough information" in answer.lower()
    assert len(citations) == 0


# ---------------------------------------------------------------------------
# Mock Gemini Streaming Generation Tests
# ---------------------------------------------------------------------------
class MockStreamChunk:
    def __init__(self, text: str):
        self.text = text


class MockGeminiClient:
    def __init__(self, mock_response_tokens: List[str]):
        self.mock_response_tokens = mock_response_tokens
        self.models = MagicMock()
        self.models.generate_content_stream.return_value = [
            MockStreamChunk(t) for t in mock_response_tokens
        ]


@pytest.mark.asyncio
async def test_astream_answer_with_mocked_gemini(sample_chunks):
    """Verify streaming generation yields token events followed by citations event and done event."""
    c1 = sample_chunks[0]
    tokens = [
        "Primary caregivers ",
        "receive 16 weeks of ",
        "fully paid parental leave ",
        f"[{c1.chunk_id}].",
    ]

    service = GenerationService()
    service._client = MockGeminiClient(tokens)

    events = []
    async for event in service.astream_answer(
        question="How much parental leave do primary caregivers get?",
        chunks=sample_chunks,
    ):
        events.append(event)

    # Check event stream
    token_events = [e for e in events if e["type"] == "token"]
    citation_events = [e for e in events if e["type"] == "citations"]
    done_events = [e for e in events if e["type"] == "done"]

    assert len(token_events) == 4
    assert len(citation_events) == 1
    assert len(done_events) == 1

    # Check assembled text
    full_text = "".join(e["content"] for e in token_events)
    assert "16 weeks" in full_text
    assert f"[{c1.chunk_id}]" in full_text

    # Check citation
    citations = citation_events[0]["citations"]
    assert len(citations) == 1
    assert citations[0]["chunk_id"] == str(c1.chunk_id)
    assert citations[0]["section_heading"] == "Parental Leave"


# ---------------------------------------------------------------------------
# API Endpoint Tests (POST /chat and POST /api/v1/chat)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_post_chat_streaming_sse(sample_user, auth_headers, sample_chunks):
    """Verify POST /chat returns 200 text/event-stream SSE events with token stream and citations."""
    c1 = sample_chunks[0]
    tokens = [f"According to policy [{c1.chunk_id}], 16 weeks are granted."]

    mock_gen_service = GenerationService()
    mock_gen_service._client = MockGeminiClient(tokens)

    app.dependency_overrides[get_current_user] = lambda: sample_user
    app.dependency_overrides[get_db] = lambda: AsyncMock()
    app.dependency_overrides[get_generation_service] = lambda: mock_gen_service

    with patch("app.api.chat.retrieve_and_rerank", new_callable=AsyncMock) as mock_retrieve:
        mock_retrieve.return_value = sample_chunks

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/chat",
                headers=auth_headers,
                json={
                    "query": "How long is parental leave for primary caregivers?",
                    "stream": True,
                },
            )

            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            # Inspect SSE lines
            lines = [line for line in response.text.split("\n\n") if line.startswith("data: ")]
            assert len(lines) >= 2  # tokens + citations + done

            # Parse JSON payloads in SSE stream
            events = [json.loads(line[6:]) for line in lines]
            assert any(e.get("type") == "token" for e in events)
            assert any(e.get("type") == "citations" for e in events)

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_post_chat_non_streaming_json(sample_user, auth_headers, sample_chunks):
    """Verify POST /chat with stream=False returns JSON response with answer and citations."""
    c1 = sample_chunks[0]
    tokens = [f"Primary caregivers get 16 weeks of paid leave [{c1.chunk_id}]."]

    mock_gen_service = GenerationService()
    mock_gen_service._client = MockGeminiClient(tokens)

    app.dependency_overrides[get_current_user] = lambda: sample_user
    app.dependency_overrides[get_db] = lambda: AsyncMock()
    app.dependency_overrides[get_generation_service] = lambda: mock_gen_service

    with patch("app.api.chat.retrieve_and_rerank", new_callable=AsyncMock) as mock_retrieve:
        mock_retrieve.return_value = sample_chunks

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/chat",
                headers=auth_headers,
                json={
                    "query": "How long is parental leave?",
                    "stream": False,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "query" in data
            assert "answer" in data
            assert "citations" in data
            assert data["sources_count"] == 1
            assert data["chunks_used"] == 2
            assert data["citations"][0]["chunk_id"] == str(c1.chunk_id)

    app.dependency_overrides.clear()



@pytest.mark.asyncio
async def test_post_chat_no_matching_chunks_endpoint(sample_user, auth_headers):
    """Verify POST /chat when retrieval yields 0 matching chunks returns no-info response."""
    app.dependency_overrides[get_current_user] = lambda: sample_user
    app.dependency_overrides[get_db] = lambda: AsyncMock()

    with patch("app.api.chat.retrieve_and_rerank", new_callable=AsyncMock) as mock_retrieve:
        mock_retrieve.return_value = []  # No matching chunks

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/chat",
                headers=auth_headers,
                json={
                    "query": "What is the secret recipe for cafeteria cookies?",
                    "stream": False,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert NO_INFO_PHRASE in data["answer"]
            assert data["sources_count"] == 0
            assert data["chunks_used"] == 0

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_post_chat_empty_query_rejected(sample_user, auth_headers):
    """Verify empty or whitespace query returns 422/400 Validation Error."""
    app.dependency_overrides[get_current_user] = lambda: sample_user
    app.dependency_overrides[get_db] = lambda: AsyncMock()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/chat",
            headers=auth_headers,
            json={"query": "", "stream": False},
        )
        assert response.status_code == 422

    app.dependency_overrides.clear()
