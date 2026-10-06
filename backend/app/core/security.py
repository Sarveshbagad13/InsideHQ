"""Security utilities: password hashing (passlib/bcrypt) and JWT (python-jose)."""

from datetime import datetime, timedelta, timezone
from typing import Any
import uuid

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    """Return the bcrypt hash of *plain*."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True when *plain* matches *hashed*."""
    return _pwd_context.verify(plain, hashed)


# ---------------------------------------------------------------------------
# JWT utilities
# ---------------------------------------------------------------------------

_ALGORITHM = settings.JWT_ALGORITHM
_SECRET = settings.JWT_SECRET_KEY


def create_access_token(
    subject: str | uuid.UUID,
    role: str,
    department: str,
    expires_delta: timedelta | None = None,
) -> str:
    """Create and return a signed JWT access token.

    Payload claims:
        sub  – user id (str)
        role – user role string
        dept – user department string
        exp  – expiry timestamp
        iat  – issued-at timestamp
        jti  – unique token id (prevents replay in future blacklist feature)
    """
    now = datetime.now(timezone.utc)
    expire = now + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload: dict[str, Any] = {
        "sub": str(subject),
        "role": role,
        "dept": department,
        "exp": expire,
        "iat": now,
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, _SECRET, algorithm=_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT.  Raises JWTError on any failure."""
    return jwt.decode(token, _SECRET, algorithms=[_ALGORITHM])
