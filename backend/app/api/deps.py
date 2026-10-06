"""FastAPI dependency: resolves and validates the JWT bearer token on protected routes."""

from collections.abc import Callable
import uuid
from typing import Annotated, List, Sequence

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_access_token
from app.db.session import get_db
from app.models.user import User

_bearer = HTTPBearer(auto_error=True)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Validate the JWT bearer token and return the corresponding active User.

    Raises HTTP 401 for any token problem and HTTP 403 when the account is
    disabled.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(credentials.credentials)
        user_id_str: str | None = payload.get("sub")
        if user_id_str is None:
            raise credentials_exception
        user_id = uuid.UUID(user_id_str)
    except (JWTError, ValueError):
        raise credentials_exception

    result = await db.execute(select(User).where(User.id == user_id))
    user: User | None = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    return user


# Typed shorthand for use in route signatures
CurrentUser = Annotated[User, Depends(get_current_user)]

_bearer_optional = HTTPBearer(auto_error=False)


async def get_current_user_optional(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_optional)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User | None:
    """Resolve user from bearer token if present; return None if unauthenticated."""
    if not credentials:
        return None
    try:
        payload = decode_access_token(credentials.credentials)
        user_id_str: str | None = payload.get("sub")
        if user_id_str is None:
            return None
        user_id = uuid.UUID(user_id_str)
        result = await db.execute(select(User).where(User.id == user_id))
        user: User | None = result.scalar_one_or_none()
        if user and not user.is_active:
            return None
        return user
    except Exception:
        return None


OptionalCurrentUser = Annotated[User | None, Depends(get_current_user_optional)]


def require_roles(allowed_roles: Sequence[str]) -> Callable:
    """Dependency factory restricting route access to users with specific roles."""
    normalized_allowed = {r.strip().lower() for r in allowed_roles}

    async def role_checker(current_user: CurrentUser) -> User:
        user_role = (current_user.role or "").strip().lower()
        if user_role not in normalized_allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User role '{current_user.role}' is not authorized. Allowed roles: {', '.join(allowed_roles)}",
            )
        return current_user

    return role_checker


# Role-restricted dependency shortcuts
RequireAdmin = Annotated[User, Depends(require_roles(["admin"]))]
RequireCuratorOrAdmin = Annotated[User, Depends(require_roles(["admin", "curator"]))]
