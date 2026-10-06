"""Pytest test suite for Feature 17: User Feedback Loop.

Tests:
1. Linkage of feedback (conversation_id, turn_number, user_id, rating, flagged_chunk_id, reason).
2. Submitting feedback for user's own conversations.
3. Enforcement that users CANNOT rate others' conversations (returns HTTP 403).
4. Admin / Curator endpoint listing documents with repeated negative feedback.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import uuid

from fastapi import HTTPException, status
from fastapi.testclient import TestClient
import pytest

from app.core.security import create_access_token
from app.main import app
from app.models.conversation import ConversationTurn
from app.models.document import Document, DocumentChunk
from app.models.feedback import Feedback
from app.models.user import User
from app.services.feedback import list_documents_with_negative_feedback, submit_user_feedback


@pytest.fixture
def user_a() -> User:
    """Fixture creating user A (standard employee)."""
    user = User(
        id=uuid.uuid4(),
        name="Employee User A",
        email="usera@company.com",
        role="employee",
        department="Engineering",
        is_active=True,
    )
    return user


@pytest.fixture
def user_b() -> User:
    """Fixture creating user B (different employee)."""
    user = User(
        id=uuid.uuid4(),
        name="Employee User B",
        email="userb@company.com",
        role="employee",
        department="Finance",
        is_active=True,
    )
    return user


@pytest.fixture
def curator_user() -> User:
    """Fixture creating a curator user."""
    user = User(
        id=uuid.uuid4(),
        name="Curator Admin",
        email="curator@company.com",
        role="curator",
        department="Human Resources",
        is_active=True,
    )
    return user


@pytest.fixture
def token_user_a(user_a: User) -> str:
    return create_access_token(subject=user_a.id, role=user_a.role, department=user_a.department)


@pytest.fixture
def token_user_b(user_b: User) -> str:
    return create_access_token(subject=user_b.id, role=user_b.role, department=user_b.department)


@pytest.fixture
def token_curator(curator_user: User) -> str:
    return create_access_token(
        subject=curator_user.id, role=curator_user.role, department=curator_user.department
    )


@pytest.mark.asyncio
async def test_submit_feedback_own_conversation_success(user_a: User):
    """Test submitting feedback for own conversation turn succeeds."""
    mock_db = AsyncMock()

    # Mock no existing turns/feedback from other users
    mock_db.execute.return_value.scalars.return_value.all.return_value = []
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_db.refresh = AsyncMock()

    conv_id = uuid.uuid4()
    chunk_id = uuid.uuid4()

    feedback = await submit_user_feedback(
        db=mock_db,
        conversation_id=conv_id,
        turn_number=1,
        user_id=user_a.id,
        rating=-1,
        flagged_chunk_id=chunk_id,
        reason="Outdated policy info",
    )

    assert feedback.conversation_id == conv_id
    assert feedback.turn_number == 1
    assert feedback.user_id == user_a.id
    assert feedback.rating == -1
    assert feedback.flagged_chunk_id == chunk_id
    assert feedback.reason == "Outdated policy info"


@pytest.mark.asyncio
async def test_cannot_rate_others_conversations(user_a: User, user_b: User):
    """Test user B cannot submit feedback for a conversation turn owned by user A."""
    mock_db = AsyncMock()
    conv_id = uuid.uuid4()

    # Existing conversation turn owned by user A
    turn_user_a = ConversationTurn(
        id=uuid.uuid4(),
        conversation_id=conv_id,
        turn_number=1,
        question="What is the remote work policy?",
        resolved_query="What is the remote work policy?",
        answer="Employees may work remotely 2 days per week.",
        retrieved_chunk_ids=[],
        user_id=user_a.id,
    )

    mock_db.execute.return_value.scalars.return_value.all.return_value = [turn_user_a]

    with pytest.raises(HTTPException) as exc_info:
        await submit_user_feedback(
            db=mock_db,
            conversation_id=conv_id,
            turn_number=1,
            user_id=user_b.id,
            rating=-1,
            reason="Incorrect details",
        )

    assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
    assert "Cannot submit feedback for a conversation owned by another user" in exc_info.value.detail


@pytest.mark.asyncio
async def test_list_documents_with_negative_feedback_linkage():
    """Test resolving negative feedback to documents via flagged_chunk_id and turn retrieved_chunk_ids."""
    mock_db = AsyncMock()

    doc_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    conv_id = uuid.uuid4()
    user_id = uuid.uuid4()

    doc = Document(id=doc_id, title="Parental Leave Policy 2026", source_type="pdf", status="approved")
    chunk = DocumentChunk(id=chunk_id, document_id=doc_id, chunk_text="Leave details")

    feedback = Feedback(
        id=uuid.uuid4(),
        conversation_id=conv_id,
        turn_number=1,
        user_id=user_id,
        rating=-1,
        flagged_chunk_id=chunk_id,
        reason="Incorrect maternity duration",
    )

    turn = ConversationTurn(
        id=uuid.uuid4(),
        conversation_id=conv_id,
        turn_number=1,
        question="Leave details?",
        resolved_query="Leave details?",
        answer="Details...",
        retrieved_chunk_ids=[str(chunk_id)],
        user_id=user_id,
    )

    # Setup mock query responses for execute()
    async def mock_execute(stmt):
        mock_res = MagicMock()
        stmt_str = str(stmt)
        if "feedback" in stmt_str.lower():
            mock_res.scalars.return_value.all.return_value = [feedback]
        elif "conversation_turns" in stmt_str.lower():
            mock_res.scalar_one_or_none.return_value = turn
        elif "document_chunks" in stmt_str.lower():
            mock_res.scalars.return_value.all.return_value = [chunk]
        elif "documents" in stmt_str.lower():
            mock_res.scalars.return_value.all.return_value = [doc]
        else:
            mock_res.scalars.return_value.all.return_value = []
        return mock_res

    mock_db.execute.side_effect = mock_execute

    results = await list_documents_with_negative_feedback(db=mock_db, min_negative_count=1)

    assert len(results) == 1
    assert results[0]["document_id"] == str(doc_id)
    assert results[0]["title"] == "Parental Leave Policy 2026"
    assert results[0]["negative_feedback_count"] == 1
    assert str(chunk_id) in results[0]["flagged_chunks"]
    assert "Incorrect maternity duration" in results[0]["reasons"]


def test_api_submit_feedback_and_curator_negative_docs(
    user_a: User, user_b: User, curator_user: User, token_user_a: str, token_user_b: str, token_curator: str
):
    """Test HTTP API endpoint integration for submitting feedback and curator negative feedback listing."""
    client = TestClient(app)

    active_user = user_a

    async def mock_get_user():
        return active_user

    mock_session = AsyncMock()

    async def mock_get_db():
        yield mock_session

    from app.api.deps import get_current_user, get_db
    app.dependency_overrides[get_current_user] = mock_get_user
    app.dependency_overrides[get_db] = mock_get_db

    try:
        conv_id = str(uuid.uuid4())

        # 1. User A submits feedback for own conversation turn
        mock_session.execute.return_value.scalars.return_value.all.return_value = []
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        headers_a = {"Authorization": f"Bearer {token_user_a}"}
        payload = {
            "conversation_id": conv_id,
            "turn_number": 1,
            "rating": -1,
            "reason": "Missing medical benefit details",
        }

        active_user = user_a
        res = client.post("/api/v1/feedback", json=payload, headers=headers_a)
        assert res.status_code == status.HTTP_201_CREATED, res.text
        data = res.json()
        assert data["conversation_id"] == conv_id
        assert data["turn_number"] == 1
        assert data["rating"] == -1

        # 2. User B attempts to rate User A's conversation turn -> expect 403
        turn_a = ConversationTurn(
            id=uuid.uuid4(),
            conversation_id=uuid.UUID(conv_id),
            turn_number=1,
            question="Q",
            resolved_query="Q",
            answer="A",
            retrieved_chunk_ids=[],
            user_id=user_a.id,
        )
        mock_session.execute.return_value.scalars.return_value.all.return_value = [turn_a]

        active_user = user_b
        headers_b = {"Authorization": f"Bearer {token_user_b}"}
        res_forbidden = client.post("/api/v1/feedback", json=payload, headers=headers_b)
        assert res_forbidden.status_code == status.HTTP_403_FORBIDDEN
        assert "another user" in res_forbidden.json()["detail"].lower()

        # 3. Regular employee (User A) attempts to access curator negative feedback endpoint -> expect 403
        active_user = user_a
        res_emp = client.get("/api/v1/feedback/negative-documents", headers=headers_a)
        assert res_emp.status_code == status.HTTP_403_FORBIDDEN

        # 4. Curator accesses negative feedback endpoint -> expect 200 OK
        active_user = curator_user
        mock_session.execute.return_value.scalars.return_value.all.return_value = []
        res_curator = client.get(
            "/api/v1/feedback/negative-documents",
            headers={"Authorization": f"Bearer {token_curator}"},
        )
        assert res_curator.status_code == status.HTTP_200_OK
        assert isinstance(res_curator.json(), list)

    finally:
        app.dependency_overrides.clear()
