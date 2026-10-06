"""Pytest configuration and global fixtures for InsideHQ backend tests."""

import pytest_asyncio
from app.db.session import async_engine


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_connections():
    """Dispose the async engine connection pool after each test.

    This prevents asyncpg connections created in one test's event loop
    from being accessed in another test's event loop, avoiding
    'Event loop is closed' or 'NoneType object has no attribute send'.
    """
    yield
    await async_engine.dispose()
