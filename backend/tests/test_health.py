import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.core.config import settings


@pytest.mark.asyncio
async def test_root_health_endpoint():
    """Verify that GET /health returns 200 OK with expected fields."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "project" in data
        assert "version" in data
        assert "environment" in data


@pytest.mark.asyncio
async def test_api_v1_health_endpoint():
    """Verify that GET /api/v1/health returns 200 OK."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_root_index_endpoint():
    """Verify root index endpoint."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "docs_url" in data
        assert data["health_url"] == "/health"


def test_settings_loaded():
    """Verify application configuration loads correctly."""
    assert settings.PROJECT_NAME == "InsideHQ Knowledge Assistant"
    assert settings.PORT == 8000
    assert "http://localhost:5173" in settings.CORS_ORIGINS
