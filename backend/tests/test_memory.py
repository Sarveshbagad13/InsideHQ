"""Unit and Integration Tests for Conversation & Session Memory (Feature 13).

Tests:
1. Referential query detection heuristics (pronouns, anaphoric phrases, short questions).
2. Gemini-based query rewriting of referential follow-ups using conversation history.
3. Full two-turn conversation flow over POST /chat (non-streaming and streaming):
   - Turn 1: Initial standalone question.
   - Turn 2: Referential follow-up using same conversation_id.
   - Verifies query rewriting, retrieval using resolved query, and Postgres turn persistence.
"""

from typing import Optional
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
from app.models.conversation import ConversationTurn
from app.models.user import User
from app.services.generation import GenerationService, get_generation_service
from app.services.memory import (
    build_query_rewrite_prompt,
    get_conversation_history,
    is_referential_query,
    rewrite_query,
    save_conversation_turn,
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
        name="Memory Test User",
        email="memtest@insidehq.com",
        role="employee",
        department="Engineering",
        is_active=True,
    )


@pytest.fixture
def auth_headers(sample_user):
    """Generate valid Bearer authorization header."""
    token = create_access_token(
        subject=sample_user.id,
        role=sample_user.role,
        department=sample_user.department,
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def primary_leave_chunk():
    """Retrieved chunk for primary caregiver leave."""
    return RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        chunk_text="Primary caregivers receive 16 weeks of fully paid parental leave.",
        chunk_metadata={"filename": "leave_policy.pdf", "section_heading": "Primary Leave"},
        document_title="Company Leave Policy",
        rrf_score=0.04,
        rerank_score=9.1,
        rerank_rank=1,
    )


@pytest.fixture
def secondary_leave_chunk():
    """Retrieved chunk for secondary caregiver leave."""
    return RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        chunk_text="Secondary caregivers receive 8 weeks of fully paid parental leave.",
        chunk_metadata={"filename": "leave_policy.pdf", "section_heading": "Secondary Leave"},
        document_title="Company Leave Policy",
        rrf_score=0.038,
        rerank_score=8.8,
        rerank_rank=1,
    )


class MockStreamChunk:
    def __init__(self, text: str):
        self.text = text


class MockGeminiClientForMemory:
    """Mock Gemini client supporting both query rewriting and grounded answer generation."""

    def __init__(self, rewrite_output: str, answer_tokens: List[str]):
        self.rewrite_output = rewrite_output
        self.answer_tokens = answer_tokens
        self.models = MagicMock()
        self.models.generate_content_stream.side_effect = self._generate_content_stream

    def _generate_content_stream(self, contents, **kwargs):
        # If the contents contain query rewrite prompt, return the rewritten query
        contents_str = str(contents)
        if "Rewritten Standalone Search Query:" in contents_str or "Conversation History" in contents_str:
            return [MockStreamChunk(self.rewrite_output)]
        # Otherwise return the answer tokens
        return [MockStreamChunk(t) for t in self.answer_tokens]


# ---------------------------------------------------------------------------
# Unit Tests: Referential Query Detection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query,expected_referential",
    [
        ("How long is it for secondary caregivers?", True),
        ("What about them?", True),
        ("Can they take that within 12 months?", True),
        ("What about secondary caregivers?", True),
        ("And for contractors?", True),
        ("How much is that?", True),
        ("Why is that?", True),
        ("Tell me more about this policy.", True),
        ("How long?", True),
        ("What is the company parental leave policy for primary caregivers?", False),
        ("How do I request a replacement laptop from IT?", False),
        ("Where can I find the 2026 holiday calendar?", False),
    ],
)
def test_is_referential_query(query: str, expected_referential: bool):
    """Verify referential detection heuristic catches pronouns, anaphoric phrases, and short follow-ups."""
    assert is_referential_query(query) is expected_referential


def test_build_query_rewrite_prompt():
    """Verify rewrite prompt formats previous turns and follow-up question properly."""
    conv_id = uuid.uuid4()
    turn1 = ConversationTurn(
        id=uuid.uuid4(),
        conversation_id=conv_id,
        turn_number=1,
        question="What is the parental leave policy for primary caregivers?",
        resolved_query="What is the parental leave policy for primary caregivers?",
        answer="Primary caregivers receive 16 weeks of fully paid leave.",
        retrieved_chunk_ids=[],
    )

    prompt = build_query_rewrite_prompt("How long is it for secondary caregivers?", [turn1])

    assert "Turn 1:" in prompt
    assert "primary caregivers" in prompt
    assert "How long is it for secondary caregivers?" in prompt
    assert "Rewritten Standalone Search Query:" in prompt


@pytest.mark.asyncio
async def test_rewrite_query_uses_gemini():
    """Verify rewrite_query invokes Gemini and returns the resolved standalone query."""
    conv_id = uuid.uuid4()
    turn1 = ConversationTurn(
        id=uuid.uuid4(),
        conversation_id=conv_id,
        turn_number=1,
        question="What is the primary caregiver leave?",
        resolved_query="What is the primary caregiver leave?",
        answer="Primary caregivers get 16 weeks.",
        retrieved_chunk_ids=[],
    )

    rewritten_target = "What is the secondary caregiver parental leave policy?"
    mock_gen = GenerationService()
    mock_gen._client = MockGeminiClientForMemory(
        rewrite_output=f'"{rewritten_target}"',
        answer_tokens=["dummy"],
    )

    result = await rewrite_query(
        current_query="What about secondary caregivers?",
        history=[turn1],
        generation_service=mock_gen,
    )

    assert result == rewritten_target


@pytest.mark.asyncio
async def test_rewrite_query_skips_non_referential():
    """Verify non-referential standalone queries are returned directly without calling LLM."""
    conv_id = uuid.uuid4()
    turn1 = ConversationTurn(
        id=uuid.uuid4(),
        conversation_id=conv_id,
        turn_number=1,
        question="What is parental leave?",
        resolved_query="What is parental leave?",
        answer="16 weeks.",
        retrieved_chunk_ids=[],
    )

    mock_gen = GenerationService()
    mock_gen._client = MagicMock()

    standalone_q = "How do I configure AWS credentials for local development?"
    result = await rewrite_query(
        current_query=standalone_q,
        history=[turn1],
        generation_service=mock_gen,
    )

    assert result == standalone_q
    mock_gen._client.models.generate_content_stream.assert_not_called()


# ---------------------------------------------------------------------------
# Integration Tests: Two-Turn Conversation Flow
# ---------------------------------------------------------------------------
class InMemoryTurnDatabase:
    """In-memory mock for session memory turns across the two-turn conversation."""

    def __init__(self):
        self.turns: List[ConversationTurn] = []

    async def get_history(self, db, conversation_id: uuid.UUID, limit: int = 3) -> List[ConversationTurn]:
        filtered = [t for t in self.turns if t.conversation_id == conversation_id]
        filtered.sort(key=lambda t: t.turn_number)
        return filtered[-limit:]

    async def save_turn(
        self,
        db,
        conversation_id: uuid.UUID,
        turn_number: int,
        question: str,
        resolved_query: str,
        answer: str,
        retrieved_chunk_ids: List[str],
        user_id: Optional[uuid.UUID] = None,
        **kwargs,
    ) -> ConversationTurn:
        turn = ConversationTurn(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            turn_number=turn_number,
            question=question,
            resolved_query=resolved_query,
            answer=answer,
            retrieved_chunk_ids=retrieved_chunk_ids,
        )
        self.turns.append(turn)
        return turn



@pytest.mark.asyncio
async def test_two_turn_conversation_flow(
    sample_user,
    auth_headers,
    primary_leave_chunk,
    secondary_leave_chunk,
):
    """CORE REQUIREMENT: Test a complete two-turn conversation.

    Turn 1: User asks initial question -> gets answer with citation, turn 1 recorded.
    Turn 2: User asks referential follow-up with same conversation_id -> query is rewritten
            using Turn 1 context, retrieval uses rewritten query, turn 2 recorded.
    """
    conversation_id = uuid.uuid4()
    in_memory_db = InMemoryTurnDatabase()

    app.dependency_overrides[get_current_user] = lambda: sample_user
    app.dependency_overrides[get_db] = lambda: AsyncMock()

    # --- TURN 1 ---
    turn1_answer_tokens = [f"Primary caregivers receive 16 weeks of fully paid leave [{primary_leave_chunk.chunk_id}]."]
    mock_gen_turn1 = GenerationService()
    mock_gen_turn1._client = MockGeminiClientForMemory(
        rewrite_output="",
        answer_tokens=turn1_answer_tokens,
    )
    app.dependency_overrides[get_generation_service] = lambda: mock_gen_turn1

    with patch("app.api.chat.get_conversation_history", side_effect=in_memory_db.get_history), \
         patch("app.api.chat.save_conversation_turn", side_effect=in_memory_db.save_turn), \
         patch("app.api.chat.retrieve_and_rerank", new_callable=AsyncMock) as mock_retrieve_t1:

        mock_retrieve_t1.return_value = [primary_leave_chunk]

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response1 = await client.post(
                "/chat",
                headers=auth_headers,
                json={
                    "conversation_id": str(conversation_id),
                    "query": "What is the parental leave policy for primary caregivers?",
                    "stream": False,
                },
            )

            assert response1.status_code == 200
            data1 = response1.json()
            assert data1["conversation_id"] == str(conversation_id)
            assert data1["turn_number"] == 1
            assert data1["query"] == "What is the parental leave policy for primary caregivers?"
            assert data1["resolved_query"] == "What is the parental leave policy for primary caregivers?"
            assert "16 weeks" in data1["answer"]
            assert data1["citations"][0]["chunk_id"] == str(primary_leave_chunk.chunk_id)

            # Check retrieve_and_rerank was called with turn 1 query
            mock_retrieve_t1.assert_called_once()
            assert mock_retrieve_t1.call_args.kwargs["query"] == "What is the parental leave policy for primary caregivers?"

    # Verify Turn 1 is stored in session memory
    assert len(in_memory_db.turns) == 1
    assert in_memory_db.turns[0].turn_number == 1
    assert in_memory_db.turns[0].retrieved_chunk_ids == [str(primary_leave_chunk.chunk_id)]

    # --- TURN 2: Referential Follow-up ---
    rewritten_query_t2 = "What is the company parental leave policy for secondary caregivers?"
    turn2_answer_tokens = [f"Secondary caregivers receive 8 weeks of paid leave [{secondary_leave_chunk.chunk_id}]."]

    mock_gen_turn2 = GenerationService()
    mock_gen_turn2._client = MockGeminiClientForMemory(
        rewrite_output=rewritten_query_t2,
        answer_tokens=turn2_answer_tokens,
    )
    app.dependency_overrides[get_generation_service] = lambda: mock_gen_turn2

    with patch("app.api.chat.get_conversation_history", side_effect=in_memory_db.get_history), \
         patch("app.api.chat.save_conversation_turn", side_effect=in_memory_db.save_turn), \
         patch("app.api.chat.retrieve_and_rerank", new_callable=AsyncMock) as mock_retrieve_t2:

        mock_retrieve_t2.return_value = [secondary_leave_chunk]

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response2 = await client.post(
                "/chat",
                headers=auth_headers,
                json={
                    "conversation_id": str(conversation_id),
                    "query": "How long is it for secondary caregivers?",
                    "stream": False,
                },
            )

            assert response2.status_code == 200
            data2 = response2.json()
            assert data2["conversation_id"] == str(conversation_id)
            assert data2["turn_number"] == 2
            assert data2["query"] == "How long is it for secondary caregivers?"
            # Query rewriting must have resolved 'it' to parental leave policy
            assert data2["resolved_query"] == rewritten_query_t2
            assert "8 weeks" in data2["answer"]
            assert data2["citations"][0]["chunk_id"] == str(secondary_leave_chunk.chunk_id)

            # CRITICAL: Retrieval for Turn 2 must use the REWRITTEN query!
            mock_retrieve_t2.assert_called_once()
            assert mock_retrieve_t2.call_args.kwargs["query"] == rewritten_query_t2

    # Verify both turns exist in session memory
    assert len(in_memory_db.turns) == 2
    assert in_memory_db.turns[1].turn_number == 2
    assert in_memory_db.turns[1].question == "How long is it for secondary caregivers?"
    assert in_memory_db.turns[1].resolved_query == rewritten_query_t2
    assert in_memory_db.turns[1].retrieved_chunk_ids == [str(secondary_leave_chunk.chunk_id)]

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_two_turn_streaming_conversation(
    sample_user,
    auth_headers,
    primary_leave_chunk,
    secondary_leave_chunk,
):
    """Verify streaming response for multi-turn conversation includes turn metadata and rewritten query in done event."""
    conversation_id = uuid.uuid4()
    in_memory_db = InMemoryTurnDatabase()

    # Pre-populate turn 1
    await in_memory_db.save_turn(
        db=None,
        conversation_id=conversation_id,
        turn_number=1,
        question="What is the AWS policy?",
        resolved_query="What is the AWS policy?",
        answer="Engineers have sandbox access.",
        retrieved_chunk_ids=[str(primary_leave_chunk.chunk_id)],
    )

    rewritten_query = "What is the AWS sandbox policy for production access?"
    mock_gen = GenerationService()
    mock_gen._client = MockGeminiClientForMemory(
        rewrite_output=rewritten_query,
        answer_tokens=[f"Production access requires approvals [{secondary_leave_chunk.chunk_id}]."],
    )

    app.dependency_overrides[get_current_user] = lambda: sample_user
    app.dependency_overrides[get_db] = lambda: AsyncMock()
    app.dependency_overrides[get_generation_service] = lambda: mock_gen

    with patch("app.api.chat.get_conversation_history", side_effect=in_memory_db.get_history), \
         patch("app.api.chat.save_conversation_turn", side_effect=in_memory_db.save_turn), \
         patch("app.api.chat.retrieve_and_rerank", new_callable=AsyncMock) as mock_retrieve:

        mock_retrieve.return_value = [secondary_leave_chunk]

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/chat",
                headers=auth_headers,
                json={
                    "conversation_id": str(conversation_id),
                    "query": "And for production?",
                    "stream": True,
                },
            )

            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            lines = [line for line in response.text.split("\n\n") if line.startswith("data: ")]
            events = [json.loads(line[6:]) for line in lines]

            done_events = [e for e in events if e.get("type") == "done"]
            assert len(done_events) == 1
            assert done_events[0]["conversation_id"] == str(conversation_id)
            assert done_events[0]["turn_number"] == 2
            assert done_events[0]["resolved_query"] == rewritten_query

    app.dependency_overrides.clear()
