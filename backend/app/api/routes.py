from fastapi import APIRouter
from app.core.config import settings
from app.api.chat import router as chat_router
from app.api.curation import router as curation_router
from app.api.documents import router as documents_router
from app.api.feedback import router as feedback_router
from app.api.v1.auth import router as auth_router

from app.services.embeddings import get_embedding_service
from app.services.reranking import get_reranker_service

api_router = APIRouter()


@api_router.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint to verify backend operational status and local models health."""
    embed_loaded = get_embedding_service().is_loaded()
    rerank_loaded = get_reranker_service().is_loaded()
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "models": {
            "embeddings": "loaded" if embed_loaded else "ready",
            "reranker": "loaded" if rerank_loaded else "ready",
        },
    }


# Mount sub-routers
api_router.include_router(auth_router)
api_router.include_router(documents_router)
api_router.include_router(curation_router)
api_router.include_router(chat_router)
api_router.include_router(feedback_router)

