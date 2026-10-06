"""Unit tests for the Local Embedding Service (Feature 6).

Covers:
    - EmbeddingService singleton instantiation and properties.
    - Synchronous batch embedding generation (returns List[List[float]]).
    - Dimension matches settings.EMBEDDING_DIMENSION (1024).
    - Normalized vector output (L2 norm ≈ 1.0).
    - Query embedding generation (returns List[float] of length 1024).
    - Async embedding generation (aembed_documents and aembed_query).
    - Edge cases: empty texts list, whitespace-only query.
"""

from __future__ import annotations

import math
from typing import List
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from app.core.config import settings
from app.services.embeddings import EmbeddingService, get_embedding_service


class DummyMockModel:
    """Mock SentenceTransformer model for fast deterministic unit tests."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def encode(
        self,
        sentences: str | List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        if isinstance(sentences, str):
            vec = np.ones(self.dim, dtype=np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                vec = vec / (norm if norm > 0 else 1.0)
            return vec

        arr = []
        for idx, s in enumerate(sentences):
            # Create unique deterministic pseudo-vector for each text
            v = np.full(self.dim, fill_value=(idx + 1) * 0.1, dtype=np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(v)
                v = v / (norm if norm > 0 else 1.0)
            arr.append(v)
        return np.array(arr)


@pytest.fixture
def mock_embedding_service():
    """Create an EmbeddingService with a mocked lightweight model."""
    service = EmbeddingService(model_name=settings.EMBEDDING_MODEL_NAME)
    service._model = DummyMockModel(dim=settings.EMBEDDING_DIMENSION)
    return service


def test_singleton_embedding_service():
    """Verify get_embedding_service returns singleton instance."""
    s1 = get_embedding_service()
    s2 = get_embedding_service()
    assert s1 is s2
    assert s1.dimension == 1024
    assert s1.model_name == "BAAI/bge-large-en-v1.5"


def test_embed_documents_batch(mock_embedding_service):
    """Verify embed_documents returns vectors of correct length and shape."""
    texts = [
        "InsideHQ internal knowledge base",
        "Permission-aware role based retrieval system",
        "FastAPI PostgreSQL pgvector architecture",
    ]
    vectors = mock_embedding_service.embed_documents(texts)

    assert len(vectors) == 3
    for vec in vectors:
        assert len(vec) == 1024
        assert isinstance(vec, list)
        assert isinstance(vec[0], float)
        # Check normalization (L2 norm should be approx 1.0)
        norm = math.sqrt(sum(x * x for x in vec))
        assert pytest.approx(norm, rel=1e-3) == 1.0


def test_embed_empty_documents_list(mock_embedding_service):
    """Verify empty document list returns empty list."""
    assert mock_embedding_service.embed_documents([]) == []


def test_embed_query(mock_embedding_service):
    """Verify embed_query returns a 1024-dimensional normalized vector."""
    query = "How do I request vacation time?"
    vec = mock_embedding_service.embed_query(query)

    assert isinstance(vec, list)
    assert len(vec) == 1024
    norm = math.sqrt(sum(x * x for x in vec))
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_embed_empty_query(mock_embedding_service):
    """Verify whitespace or empty query returns zero vector."""
    vec = mock_embedding_service.embed_query("   ")
    assert len(vec) == 1024
    assert all(x == 0.0 for x in vec)


@pytest.mark.asyncio
async def test_aembed_documents_async(mock_embedding_service):
    """Verify asynchronous aembed_documents produces expected embeddings."""
    texts = ["Async block 1", "Async block 2"]
    vectors = await mock_embedding_service.aembed_documents(texts)

    assert len(vectors) == 2
    for vec in vectors:
        assert len(vec) == 1024


@pytest.mark.asyncio
async def test_aembed_query_async(mock_embedding_service):
    """Verify asynchronous aembed_query produces expected vector."""
    query = "What is the engineering on-call policy?"
    vec = await mock_embedding_service.aembed_query(query)

    assert len(vec) == 1024
    assert isinstance(vec, list)
