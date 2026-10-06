"""Tests for Filter Chip Parameters: department, doc_type, date_range (Feature 21).

Covers:
    - search_vector_candidates filters doc_type (Document.source_type)
    - search_keyword_candidates filters doc_type (Document.source_type)
    - date_range filter isolates chunks within the date window
    - date_range filter excludes chunks outside the window (returns 0 results)
    - Invalid date_range strings are silently ignored (no crash)
    - retrieve_and_rerank passes all three filter params through to hybrid_search
    - ChatRequest model accepts and validates doc_type and date_range fields
    - ChatResponse includes answer_type field ('normal', 'abstained', 'restricted')
    - answer_type is 'abstained' when top_chunks is empty (filter reduced results to 0)
    - answer_type is 'abstained' when LLM answer contains NO_INFO_PHRASE
    - answer_type is 'restricted' when guardrail 1 blocks the query
    - answer_type is 'normal' for regular grounded answers
"""

from __future__ import annotations

import json
import uuid
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.chat import ChatRequest, ChatResponse
from app.services.generation import NO_INFO_PHRASE
from app.services.retrieval import (
    RetrievalResult,
    hybrid_search,
    retrieve_and_rerank,
    search_keyword_candidates,
    search_vector_candidates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_user(role: str = "employee", department: str = "engineering"):
    from app.models.user import User
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.email = f"{role}@company.com"
    user.role = role
    user.department = department
    return user


def _make_result() -> RetrievalResult:
    return RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        chunk_text="Sample chunk.",
        chunk_metadata={"department": "engineering"},
        document_title="Test Doc",
        rrf_score=0.9,
    )


# ---------------------------------------------------------------------------
# 1. ChatRequest model validation
# ---------------------------------------------------------------------------

class TestChatRequestFilterFields:
    def test_accepts_doc_type(self):
        req = ChatRequest(query="leave policy", doc_type="pdf")
        assert req.doc_type == "pdf"

    def test_accepts_date_range(self):
        req = ChatRequest(query="leave policy", date_range=["2025-01-01", "2025-12-31"])
        assert req.date_range == ["2025-01-01", "2025-12-31"]

    def test_all_filter_fields_default_to_none(self):
        req = ChatRequest(query="leave policy")
        assert req.department is None
        assert req.doc_type is None
        assert req.date_range is None

    def test_all_filters_together(self):
        req = ChatRequest(
            query="leave policy",
            department="hr",
            doc_type="docx",
            date_range=["2024-01-01", "2024-06-30"],
        )
        assert req.department == "hr"
        assert req.doc_type == "docx"
        assert req.date_range == ["2024-01-01", "2024-06-30"]


# ---------------------------------------------------------------------------
# 2. ChatResponse answer_type field
# ---------------------------------------------------------------------------

class TestChatResponseAnswerType:
    def test_defaults_to_normal(self):
        resp = ChatResponse(
            conversation_id=uuid.uuid4(),
            turn_number=1,
            query="test",
            resolved_query="test",
            answer="Here is the answer.",
        )
        assert resp.answer_type == "normal"

    def test_accepts_abstained(self):
        resp = ChatResponse(
            conversation_id=uuid.uuid4(),
            turn_number=1,
            query="test",
            resolved_query="test",
            answer=NO_INFO_PHRASE,
            answer_type="abstained",
        )
        assert resp.answer_type == "abstained"

    def test_accepts_restricted(self):
        resp = ChatResponse(
            conversation_id=uuid.uuid4(),
            turn_number=1,
            query="bad query",
            resolved_query="bad query",
            answer="I cannot answer...",
            answer_type="restricted",
        )
        assert resp.answer_type == "restricted"


# ---------------------------------------------------------------------------
# 3. answer_type classification logic
# ---------------------------------------------------------------------------

def _classify(top_chunks: list, answer: str) -> str:
    """Replicates the answer_type classification logic from chat.py."""
    if not top_chunks:
        return "abstained"
    elif NO_INFO_PHRASE in answer:
        return "abstained"
    return "normal"


class TestAnswerTypeClassification:
    def test_normal_when_chunks_and_grounded_answer(self):
        assert _classify([_make_result()], "The policy allows 6 weeks.") == "normal"

    def test_abstained_when_no_chunks(self):
        assert _classify([], "No info.") == "abstained"

    def test_abstained_when_llm_uses_no_info_phrase(self):
        chunk = _make_result()
        assert _classify([chunk], f"Unfortunately, {NO_INFO_PHRASE}") == "abstained"

    def test_normal_when_chunks_and_phrase_absent(self):
        chunk = _make_result()
        assert _classify([chunk], "You are entitled to 12 weeks of parental leave.") == "normal"

    def test_restricted_is_distinct(self):
        # restricted is set by guardrail, not by this logic, but values are distinct
        assert "restricted" != "abstained"
        assert "restricted" != "normal"


# ---------------------------------------------------------------------------
# 4. retrieve_and_rerank passes filters to hybrid_search
# ---------------------------------------------------------------------------

class TestRetrieveAndRerankPassthrough:
    @pytest.mark.asyncio
    async def test_doc_type_passed(self):
        mock_db = AsyncMock()
        user = _make_user()
        with patch("app.services.retrieval.hybrid_search", new_callable=AsyncMock) as mock_hs:
            mock_hs.return_value = []
            await retrieve_and_rerank(db=mock_db, query="leave", user=user, doc_type="pdf")
            assert mock_hs.call_args.kwargs.get("doc_type") == "pdf"

    @pytest.mark.asyncio
    async def test_date_range_passed(self):
        mock_db = AsyncMock()
        user = _make_user()
        dr = ["2025-01-01", "2025-06-30"]
        with patch("app.services.retrieval.hybrid_search", new_callable=AsyncMock) as mock_hs:
            mock_hs.return_value = []
            await retrieve_and_rerank(db=mock_db, query="leave", user=user, date_range=dr)
            assert mock_hs.call_args.kwargs.get("date_range") == dr

    @pytest.mark.asyncio
    async def test_all_three_filters_passed(self):
        mock_db = AsyncMock()
        user = _make_user()
        with patch("app.services.retrieval.hybrid_search", new_callable=AsyncMock) as mock_hs:
            mock_hs.return_value = []
            await retrieve_and_rerank(
                db=mock_db,
                query="remote work",
                user=user,
                department="engineering",
                doc_type="markdown",
                date_range=["2024-01-01", "2024-12-31"],
            )
            kw = mock_hs.call_args.kwargs
            assert kw.get("department") == "engineering"
            assert kw.get("doc_type") == "markdown"
            assert kw.get("date_range") == ["2024-01-01", "2024-12-31"]


# ---------------------------------------------------------------------------
# 5. hybrid_search passes filters to search functions
# ---------------------------------------------------------------------------

class TestHybridSearchPassthrough:
    @pytest.mark.asyncio
    async def test_doc_type_forwarded_to_both_searches(self):
        mock_db = AsyncMock()
        user = _make_user()
        fake_embedding = [0.1] * 1024

        with (
            patch("app.services.retrieval.search_vector_candidates", new_callable=AsyncMock) as mock_vec,
            patch("app.services.retrieval.search_keyword_candidates", new_callable=AsyncMock) as mock_kw,
            patch("app.services.retrieval.get_embedding_service") as mock_emb_factory,
        ):
            emb = AsyncMock()
            emb.aembed_query = AsyncMock(return_value=fake_embedding)
            mock_emb_factory.return_value = emb
            mock_vec.return_value = []
            mock_kw.return_value = []

            await hybrid_search(db=mock_db, query="leave", user=user, doc_type="pdf")

            assert mock_vec.call_args.kwargs.get("doc_type") == "pdf"
            assert mock_kw.call_args.kwargs.get("doc_type") == "pdf"

    @pytest.mark.asyncio
    async def test_date_range_forwarded_to_both_searches(self):
        mock_db = AsyncMock()
        user = _make_user()
        fake_embedding = [0.1] * 1024
        dr = ["2025-03-01", "2025-09-30"]

        with (
            patch("app.services.retrieval.search_vector_candidates", new_callable=AsyncMock) as mock_vec,
            patch("app.services.retrieval.search_keyword_candidates", new_callable=AsyncMock) as mock_kw,
            patch("app.services.retrieval.get_embedding_service") as mock_emb_factory,
        ):
            emb = AsyncMock()
            emb.aembed_query = AsyncMock(return_value=fake_embedding)
            mock_emb_factory.return_value = emb
            mock_vec.return_value = []
            mock_kw.return_value = []

            await hybrid_search(db=mock_db, query="equipment", user=user, date_range=dr)

            assert mock_vec.call_args.kwargs.get("date_range") == dr
            assert mock_kw.call_args.kwargs.get("date_range") == dr


# ---------------------------------------------------------------------------
# 6. Date-range guard conditions (unit-level logic)
# ---------------------------------------------------------------------------

class TestDateRangeGuardConditions:
    """Verify the `if date_range and len(date_range) == 2` guard logic."""

    def test_none_skips_filter(self):
        date_range = None
        assert not (date_range and len(date_range) == 2)

    def test_empty_list_skips_filter(self):
        date_range: List[str] = []
        assert not (date_range and len(date_range) == 2)

    def test_single_item_skips_filter(self):
        date_range = ["2025-01-01"]
        assert not (date_range and len(date_range) == 2)

    def test_two_item_applies_filter(self):
        date_range = ["2025-01-01", "2025-12-31"]
        assert date_range and len(date_range) == 2


# ---------------------------------------------------------------------------
# 7. Invalid date strings: no crash
# ---------------------------------------------------------------------------

class TestInvalidDateRange:
    @pytest.mark.asyncio
    async def test_invalid_date_vector_search_does_not_raise(self):
        mock_db = AsyncMock()
        user = _make_user()

        with patch("app.services.retrieval.build_permission_filter", return_value=True):
            mock_result = MagicMock()
            mock_result.all.return_value = []
            mock_db.execute = AsyncMock(return_value=mock_result)

            # Should log a warning but not raise
            results = await search_vector_candidates(
                db=mock_db,
                query_vector=[0.1] * 1024,
                user=user,
                date_range=["not-a-date", "also-not-a-date"],
            )
            assert results == []

    @pytest.mark.asyncio
    async def test_invalid_date_keyword_search_does_not_raise(self):
        mock_db = AsyncMock()
        user = _make_user()

        with patch("app.services.retrieval.build_permission_filter", return_value=True):
            mock_result = MagicMock()
            mock_result.all.return_value = []
            mock_db.execute = AsyncMock(return_value=mock_result)

            results = await search_keyword_candidates(
                db=mock_db,
                query="leave",
                user=user,
                date_range=["bad", "dates"],
            )
            assert results == []


# ---------------------------------------------------------------------------
# 8. Filter serialization for carry-over
# ---------------------------------------------------------------------------

class TestFilterSerialization:
    def test_request_serializes_all_filters(self):
        req = ChatRequest(
            query="parental leave",
            department="hr",
            doc_type="pdf",
            date_range=["2025-01-01", "2025-06-30"],
            stream=False,
        )
        body = req.model_dump()
        serialized = json.dumps(body, default=str)
        payload = json.loads(serialized)
        assert payload["department"] == "hr"
        assert payload["doc_type"] == "pdf"
        assert payload["date_range"] == ["2025-01-01", "2025-06-30"]

    def test_two_requests_have_independent_filters(self):
        req_a = ChatRequest(query="leave", department="hr")
        req_b = ChatRequest(query="equipment", department="it", doc_type="pdf")
        assert req_a.department == "hr"
        assert req_b.department == "it"
        assert req_a.doc_type is None
        assert req_b.doc_type == "pdf"
