"""Pytest test suite for Feature 19: Rate Limiting & Abuse Protection.

Tests:
1. Per-user sliding-window rate limiter (blocked at the limit, allowed after reset).
2. Stricter sliding-window limits for expensive operations (streaming / department filtered).
3. Daily token-cost budget per user returning HTTP 429 when exceeded.
4. HTTP API integration on POST /chat endpoint.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import uuid

from fastapi import HTTPException, status
from fastapi.testclient import TestClient
import pytest

from app.core.security import create_access_token
from app.main import app
from app.models.user import User
from app.services.rate_limiter import RateLimiter, rate_limiter


@pytest.fixture
def test_user() -> User:
    """Fixture creating a test user."""
    return User(
        id=uuid.uuid4(),
        name="Rate Limit Test User",
        email="ratelimit@company.com",
        role="employee",
        department="Engineering",
        is_active=True,
    )


@pytest.fixture
def user_token(test_user: User) -> str:
    return create_access_token(
        subject=test_user.id, role=test_user.role, department=test_user.department
    )


def test_sliding_window_rate_limiter_blocked_at_limit(test_user: User):
    """Test user requests are allowed up to the limit and blocked with 429 at limit."""
    limiter = RateLimiter(default_window_seconds=60.0, default_request_limit=3)
    user_id = test_user.id

    # 1. First 3 requests allowed
    limiter.check_rate_limit(user_id=user_id, is_expensive=False, now=100.0)
    limiter.check_rate_limit(user_id=user_id, is_expensive=False, now=101.0)
    limiter.check_rate_limit(user_id=user_id, is_expensive=False, now=102.0)

    # 2. 4th request within same window is blocked with HTTP 429
    with pytest.raises(HTTPException) as exc_info:
        limiter.check_rate_limit(user_id=user_id, is_expensive=False, now=103.0)

    assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert "Rate limit exceeded" in exc_info.value.detail
    assert "Retry-After" in exc_info.value.headers


def test_sliding_window_rate_limiter_allowed_after_reset(test_user: User):
    """Test user blocked at limit is allowed again after resetting rate limiter or time progression."""
    limiter = RateLimiter(default_window_seconds=60.0, default_request_limit=2)
    user_id = test_user.id

    # Fill capacity at t=100
    limiter.check_rate_limit(user_id=user_id, now=100.0)
    limiter.check_rate_limit(user_id=user_id, now=101.0)

    # Blocked at t=102
    with pytest.raises(HTTPException):
        limiter.check_rate_limit(user_id=user_id, now=102.0)

    # Option A: Reset user rate limits explicitly
    limiter.reset_user(user_id)
    limiter.check_rate_limit(user_id=user_id, now=103.0)  # Now allowed!

    # Fill capacity again at t=103 & t=104
    limiter.check_rate_limit(user_id=user_id, now=104.0)

    # Blocked at t=105
    with pytest.raises(HTTPException):
        limiter.check_rate_limit(user_id=user_id, now=105.0)

    # Option B: Allowed naturally after sliding window expires (at t=165, > 103 + 60)
    limiter.check_rate_limit(user_id=user_id, now=165.0)  # Allowed!


def test_expensive_operation_stricter_limit(test_user: User):
    """Test expensive operations (e.g. streaming) use a stricter limit."""
    limiter = RateLimiter(
        default_window_seconds=60.0,
        default_request_limit=5,
        expensive_request_limit=2,
    )
    user_id = test_user.id

    # Expensive request 1 & 2 allowed
    limiter.check_rate_limit(user_id=user_id, is_expensive=True, now=10.0)
    limiter.check_rate_limit(user_id=user_id, is_expensive=True, now=11.0)

    # Expensive request 3 blocked
    with pytest.raises(HTTPException) as exc_info:
        limiter.check_rate_limit(user_id=user_id, is_expensive=True, now=12.0)

    assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert "Maximum 2 requests" in exc_info.value.detail


def test_daily_token_budget_blocked_and_reset(test_user: User):
    """Test daily token budget enforcement and reset."""
    limiter = RateLimiter(daily_token_budget=1000)
    user_id = test_user.id

    # Record 800 tokens -> within budget
    limiter.check_daily_token_budget(user_id, estimated_tokens=100)
    limiter.record_token_usage(user_id, 800)

    # Requesting another 300 tokens -> exceeds 1000 budget!
    with pytest.raises(HTTPException) as exc_info:
        limiter.check_daily_token_budget(user_id, estimated_tokens=300)

    assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert "Daily token budget exceeded" in exc_info.value.detail

    # Reset user tokens -> allowed again
    limiter.reset_user(user_id)
    limiter.check_daily_token_budget(user_id, estimated_tokens=300)  # Allowed!


def test_chat_endpoint_rate_limiting_integration(test_user: User, user_token: str):
    """Test /chat endpoint returns 429 when rate limit is reached."""
    client = TestClient(app)

    async def mock_get_user():
        return test_user

    mock_db = AsyncMock()

    async def mock_get_db():
        yield mock_db

    from app.api.deps import get_current_user, get_db
    from app.services.generation import get_generation_service

    mock_gen_service = MagicMock()

    async def mock_astream_answer(*args, **kwargs):
        yield {"type": "done", "full_answer": "Policy details..."}

    mock_gen_service.astream_answer = mock_astream_answer

    app.dependency_overrides[get_current_user] = mock_get_user
    app.dependency_overrides[get_db] = mock_get_db
    app.dependency_overrides[get_generation_service] = lambda: mock_gen_service

    try:
        # Save original global rate limiter settings & reset
        rate_limiter.reset_all()
        orig_expensive = rate_limiter.expensive_request_limit
        rate_limiter.expensive_request_limit = 2

        headers = {"Authorization": f"Bearer {user_token}"}
        payload = {"query": "What is company parental leave policy?", "stream": True}

        # Mock retrieval service return empty chunks
        from unittest.mock import patch
        with patch("app.api.chat.retrieve_and_rerank", new=AsyncMock(return_value=[])):
            with patch("app.api.chat.save_conversation_turn", new=AsyncMock()):
                # Request 1 & 2 -> Success (200)
                res1 = client.post("/chat", json=payload, headers=headers)
                assert res1.status_code == status.HTTP_200_OK, res1.text

                res2 = client.post("/chat", json=payload, headers=headers)
                assert res2.status_code == status.HTTP_200_OK, res2.text

                # Request 3 -> Rate limit exceeded (429)
                res3 = client.post("/chat", json=payload, headers=headers)
                assert res3.status_code == status.HTTP_429_TOO_MANY_REQUESTS
                assert res3.headers.get("Retry-After") is not None

                # Reset limits for user -> Request 4 succeeds (200)
                rate_limiter.reset_user(test_user.id)
                res4 = client.post("/chat", json=payload, headers=headers)
                assert res4.status_code == status.HTTP_200_OK, res4.text

    finally:
        rate_limiter.expensive_request_limit = orig_expensive
        rate_limiter.reset_all()
        app.dependency_overrides.clear()
