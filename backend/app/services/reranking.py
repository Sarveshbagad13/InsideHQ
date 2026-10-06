"""Cross-Encoder Reranking Service (Feature 10).

Uses a local sentence-transformers CrossEncoder (BAAI/bge-reranker-large) loaded once
at startup to compute joint relevance scores between the query and permission-filtered candidate chunks,
returning the top 5 chunks in refined order and logging inference latency for observability.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional

import numpy as np

from app.core.config import settings
from app.services.retrieval import RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_RERANK_TOP_N = 5


class RerankerService:
    """Singleton service for local cross-encoder reranking using SentenceTransformers CrossEncoder."""

    _instance: Optional[RerankerService] = None

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self._model = None

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None) -> RerankerService:
        """Get or create singleton instance of RerankerService."""
        if cls._instance is None:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    @property
    def model(self):
        """Lazy load the CrossEncoder model."""
        if self._model is None:
            self.load_model()
        return self._model

    def load_model(self) -> None:
        """Load the CrossEncoder model into memory with graceful fallback."""
        if self._model is None:
            logger.info("Loading local cross-encoder reranker model: %s ...", self.model_name)
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
                logger.info("Reranker model '%s' loaded successfully.", self.model_name)
            except Exception as e:
                logger.warning("CrossEncoder reranker model unavailable (%s). Using RRF candidate ordering.", e)
                self._model = "fallback"

    def is_loaded(self) -> bool:
        """Check if the reranker model is already loaded in memory."""
        return self._model is not None and self._model != "fallback"

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_n: int = DEFAULT_RERANK_TOP_N,
    ) -> List[RetrievalResult]:
        """Score and re-rank candidate chunks against the query using the cross-encoder."""
        if not candidates:
            return []

        if self.model == "fallback":
            # Assign rank based on RRF candidate order
            fallback_chunks = candidates[:top_n]
            for rank, candidate in enumerate(fallback_chunks, start=1):
                candidate.rerank_rank = rank
            return fallback_chunks

        clean_query = query.strip()
        if not clean_query:
            return candidates[:top_n]

        start_time = time.perf_counter()

        # Construct pairs: [query, chunk_text]
        pairs = [[clean_query, candidate.chunk_text] for candidate in candidates]

        # Compute cross-encoder relevance scores
        raw_scores = self.model.predict(pairs)

        if isinstance(raw_scores, np.ndarray):
            scores = raw_scores.tolist()
        elif isinstance(raw_scores, (list, tuple)):
            scores = [float(s) for s in raw_scores]
        else:
            scores = [float(raw_scores)]

        # Attach rerank score to candidates
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)

        # Sort candidates primarily by rerank_score DESC, secondary by RRF score
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (
                c.rerank_score if c.rerank_score is not None else -999.0,
                c.rrf_score,
            ),
            reverse=True,
        )

        # Assign 1-based rerank_rank
        for rank, candidate in enumerate(sorted_candidates, start=1):
            candidate.rerank_rank = rank

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Reranked %d candidates in %.2fms (model=%s, top_score=%.4f)",
            len(candidates),
            elapsed_ms,
            self.model_name,
            sorted_candidates[0].rerank_score if sorted_candidates else 0.0,
        )

        return sorted_candidates[:top_n]

    async def arerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_n: int = DEFAULT_RERANK_TOP_N,
    ) -> List[RetrievalResult]:
        """Asynchronously rerank candidates in a background thread to prevent blocking event loop.

        Args:
            query: The user search query string.
            candidates: List of candidate RetrievalResult items from hybrid retrieval.
            top_n: Number of top reranked chunks to return (default 5).

        Returns:
            Top-N candidate chunks sorted by rerank_score descending.
        """
        if not candidates:
            return []
        return await asyncio.to_thread(self.rerank, query, candidates, top_n)


def get_reranker_service() -> RerankerService:
    """Dependency helper to get the singleton RerankerService instance."""
    return RerankerService.get_instance()


async def rerank_candidates(
    query: str,
    candidates: List[RetrievalResult],
    top_n: int = DEFAULT_RERANK_TOP_N,
    reranker_service: Optional[RerankerService] = None,
) -> List[RetrievalResult]:
    """Helper function to rerank candidate chunks using the default or provided reranker service.

    Args:
        query: The user search query string.
        candidates: List of candidate RetrievalResult items.
        top_n: Number of top reranked chunks to return (default 5).
        reranker_service: Optional custom or mocked RerankerService instance.

    Returns:
        Top-N candidate chunks ordered by cross-encoder relevance score.
    """
    if reranker_service is None:
        reranker_service = get_reranker_service()
    return await reranker_service.arerank(query=query, candidates=candidates, top_n=top_n)
