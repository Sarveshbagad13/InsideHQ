"""Local embedding generation service using SentenceTransformers.

Loads the BAAI/bge-large-en-v1.5 model (dimension 1024) once locally and provides
batch embedding generation for chunks and queries without external API dependencies.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Singleton service for generating vector embeddings locally via SentenceTransformers."""

    _instance: Optional[EmbeddingService] = None

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.dimension = settings.EMBEDDING_DIMENSION
        self._model = None

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None) -> EmbeddingService:
        """Get or create singleton instance of EmbeddingService."""
        if cls._instance is None:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    @property
    def model(self):
        """Lazy load the sentence-transformers model."""
        if self._model is None:
            self.load_model()
        return self._model

    def load_model(self) -> None:
        """Load the SentenceTransformer model into memory with graceful fallback."""
        if self._model is None:
            logger.info("Loading local embedding model: %s ...", self.model_name)
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                logger.info(
                    "Embedding model '%s' loaded successfully (dimension=%d)",
                    self.model_name,
                    self.dimension,
                )
            except Exception as e:
                logger.warning(
                    "SentenceTransformers/PyTorch embedding model unavailable (%s). Using keyword retrieval.",
                    e,
                )
                self._model = "fallback"

    def is_loaded(self) -> bool:
        """Check if the embedding model is already loaded in memory."""
        return self._model is not None and self._model != "fallback"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate normalized vector embeddings for a batch of text chunks synchronously."""
        if not texts or self.model == "fallback":
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        return [list(vec) for vec in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """Generate normalized vector embedding for a search query."""
        if not query or not query.strip() or self.model == "fallback":
            return []

        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        return list(embedding)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously generate normalized vector embeddings for a batch of text chunks.

        Runs the CPU/GPU-bound encoding in a separate thread to prevent blocking
        the async event loop.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors of length 1024.
        """
        if not texts:
            return []
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, query: str) -> List[float]:
        """Asynchronously generate normalized vector embedding for a search query.

        Runs encoding in a separate thread.

        Args:
            query: The search query string.

        Returns:
            Embedding vector as a list of floats of length 1024.
        """
        return await asyncio.to_thread(self.embed_query, query)


def get_embedding_service() -> EmbeddingService:
    """Dependency helper to get the singleton EmbeddingService instance."""
    return EmbeddingService.get_instance()
