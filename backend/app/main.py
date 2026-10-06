from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import api_router
from app.core.config import settings
from app.services.embeddings import get_embedding_service
from app.services.reranking import get_reranker_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager: pre-warm and verify local embedding & reranking models at startup."""
    logger.info("InsideHQ starting up: running startup health-check on local models...")
    try:
        embed_service = get_embedding_service()
        embed_service.load_model()
        logger.info("Embedding service startup health-check passed (model: %s).", embed_service.model_name)
    except Exception as e:
        logger.warning("Embedding service startup health-check notice: %s", e)

    try:
        rerank_service = get_reranker_service()
        rerank_service.load_model()
        logger.info("Reranker service startup health-check passed (model: %s).", rerank_service.model_name)
    except Exception as e:
        logger.warning("Reranker service startup health-check notice: %s", e)

    yield
    logger.info("InsideHQ shutting down...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Internal AI assistant that lets employees ask questions about company knowledge with permission awareness.",
    debug=settings.DEBUG,
    lifespan=lifespan,
)

# CORS middleware for React Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
async def root_health_check():
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


from app.api.chat import router as chat_router

# Include versioned API router under prefix
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(chat_router)



@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with basic service info."""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "docs_url": "/docs",
        "health_url": "/health",
        "api_v1": settings.API_V1_STR,
    }
