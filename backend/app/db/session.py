"""Database session configuration and Base declarative model for InsideHQ."""
from collections.abc import AsyncGenerator
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings

# Async engine & session for FastAPI async operations
async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

def _get_sync_engine():
    try:
        return create_engine(
            settings.SYNC_DATABASE_URL,
            echo=False,
            future=True,
            pool_pre_ping=True,
        )
    except Exception:
        return None

sync_engine = None
try:
    sync_engine = _get_sync_engine()
except Exception:
    pass

if sync_engine is not None:
    SyncSessionLocal = sessionmaker(
        bind=sync_engine,
        autocommit=False,
        autoflush=False,
    )
else:
    def SyncSessionLocal():
        raise RuntimeError("Synchronous database engine is not available (psycopg2/sync driver issue). Use AsyncSessionLocal instead.")




class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for providing an async database session to FastAPI endpoints."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
