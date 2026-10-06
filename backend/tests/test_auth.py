"""JWT authentication tests.

Covers:
    - POST /api/v1/auth/login: valid credentials return a signed JWT
    - POST /api/v1/auth/login: wrong password is rejected (401)
    - POST /api/v1/auth/login: unknown email is rejected (401)
    - GET /api/v1/auth/me: valid token resolves to the correct user
    - GET /api/v1/auth/me: missing / malformed token is rejected (401/403)
    - GET /api/v1/auth/me: expired token is rejected (401)
    - JWT payload contains the expected role and department claims
    - Disabled (is_active=False) user is rejected at login (403)
"""

import uuid
from datetime import timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from jose import jwt

from app.core.config import settings
from app.core.security import create_access_token, hash_password
from app.db.session import SyncSessionLocal
from app.main import app
from app.models.user import User


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE = "/api/v1"

async def _client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _get_seeded_user(email: str) -> User | None:
    session = SyncSessionLocal()
    try:
        return session.query(User).filter_by(email=email).first()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Login tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_login_valid_credentials_returns_token():
    """Valid email+password must return HTTP 200 and a valid JWT."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"{BASE}/auth/login",
            json={"email": "alice.admin@insidehq.com", "password": "admin123"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["role"] == "admin"
    assert data["department"] == "Executive"


@pytest.mark.asyncio
async def test_login_wrong_password_is_rejected():
    """Wrong password must return HTTP 401."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"{BASE}/auth/login",
            json={"email": "alice.admin@insidehq.com", "password": "wrongpass"},
        )
    assert resp.status_code == 401
    assert "Incorrect" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_login_unknown_email_is_rejected():
    """Unknown email must return HTTP 401 (no user enumeration)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"{BASE}/auth/login",
            json={"email": "nobody@insidehq.com", "password": "whatever"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_login_each_role_produces_correct_claims():
    """Each seeded role must appear verbatim in the JWT payload."""
    role_cases = [
        ("bob.hr@insidehq.com", "hr123", "hr", "Human Resources"),
        ("charlie.eng@insidehq.com", "eng123", "engineering", "Engineering"),
        ("diana.fin@insidehq.com", "finance123", "finance", "Finance"),
        ("edward.emp@insidehq.com", "employee123", "employee", "Operations"),
    ]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for email, password, expected_role, expected_dept in role_cases:
            resp = await client.post(
                f"{BASE}/auth/login",
                json={"email": email, "password": password},
            )
            assert resp.status_code == 200, f"Login failed for {email}"
            data = resp.json()
            # Decode without verification to inspect claims
            payload = jwt.decode(
                data["access_token"],
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
            )
            assert payload["role"] == expected_role, f"Wrong role for {email}"
            assert payload["dept"] == expected_dept, f"Wrong dept for {email}"


# ---------------------------------------------------------------------------
# Protected endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_me_with_valid_token():
    """GET /auth/me with a valid token returns the correct user profile."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        login = await client.post(
            f"{BASE}/auth/login",
            json={"email": "charlie.eng@insidehq.com", "password": "eng123"},
        )
        token = login.json()["access_token"]

        resp = await client.get(
            f"{BASE}/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 200
    me = resp.json()
    assert me["email"] == "charlie.eng@insidehq.com"
    assert me["role"] == "engineering"
    assert me["department"] == "Engineering"
    assert me["is_active"] is True


@pytest.mark.asyncio
async def test_get_me_without_token_returns_401():
    """GET /auth/me without any Authorization header must return 401."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(f"{BASE}/auth/me")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_get_me_with_malformed_token_returns_401():
    """GET /auth/me with a garbled token must return 401."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            f"{BASE}/auth/me",
            headers={"Authorization": "Bearer this.is.garbage"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_get_me_with_expired_token_returns_401():
    """GET /auth/me with an already-expired token must return 401."""
    # Craft a token that expired 1 hour ago
    expired_token = create_access_token(
        subject=uuid.uuid4(),
        role="employee",
        department="Operations",
        expires_delta=timedelta(hours=-1),
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            f"{BASE}/auth/me",
            headers={"Authorization": f"Bearer {expired_token}"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_get_me_with_wrong_secret_returns_401():
    """JWT signed with a different secret key must be rejected."""
    bad_token = jwt.encode(
        {"sub": str(uuid.uuid4()), "role": "admin", "dept": "IT"},
        "totallywrongsecret",
        algorithm="HS256",
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            f"{BASE}/auth/me",
            headers={"Authorization": f"Bearer {bad_token}"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_disabled_user_cannot_login():
    """A user with is_active=False must be rejected at login with 403."""
    session = SyncSessionLocal()
    try:
        # Temporarily disable Edward
        user = session.query(User).filter_by(email="edward.emp@insidehq.com").first()
        assert user is not None
        user.is_active = False
        session.commit()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"{BASE}/auth/login",
                json={"email": "edward.emp@insidehq.com", "password": "employee123"},
            )
        assert resp.status_code == 403
        assert "disabled" in resp.json()["detail"].lower()
    finally:
        # Always re-enable
        user.is_active = True
        session.commit()
        session.close()


@pytest.mark.asyncio
async def test_token_sub_matches_user_id():
    """The 'sub' JWT claim must match the user's actual UUID in the DB."""
    user = _get_seeded_user("bob.hr@insidehq.com")
    assert user is not None

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"{BASE}/auth/login",
            json={"email": "bob.hr@insidehq.com", "password": "hr123"},
        )
    payload = jwt.decode(
        resp.json()["access_token"],
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],
    )
    assert payload["sub"] == str(user.id)
