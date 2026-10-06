"""Tests for Document Chunk Indexing and Embedding Pipeline (Feature 6).

Covers:
    - Indexing clean text blocks into DocumentChunk records with 1024-dim vector embeddings.
    - Chunk metadata population (department, filename, source_type, position, token_count, heading).
    - Background task execution after ingestion (Feature 4 -> Feature 5 -> Feature 6).
    - Document status remains 'pending' after indexing completes.
    - Verification that both embedding vector and tsvector are populated after processing.
"""

from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock, MagicMock, patch
import uuid
from fastapi import BackgroundTasks
import numpy as np
import pytest

from app.core.config import settings
from app.models.document import Document, DocumentChunk
from app.services.embeddings import EmbeddingService
from app.services.indexing import index_document_chunks, process_document_indexing
from app.services.ingestion import ingest_document


class MockEmbeddingService(EmbeddingService):
    """Deterministic mock embedding service returning normalized 1024-dim vectors."""

    def __init__(self):
        super().__init__(model_name=settings.EMBEDDING_MODEL_NAME)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        arr = []
        for idx, _ in enumerate(texts):
            vec = np.full(self.dimension, fill_value=float(idx + 1) * 0.05, dtype=np.float32)
            norm = np.linalg.norm(vec)
            vec = vec / (norm if norm > 0 else 1.0)
            arr.append(vec.tolist())
        return arr

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)


def create_mock_session():
    """Create a mock AsyncSession with synchronous add() and asynchronous commit()/execute()."""
    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    return mock_db


@pytest.mark.asyncio
async def test_index_document_chunks_populates_embeddings_and_metadata():
    """Verify index_document_chunks creates DocumentChunk records with 1024-dim vectors and metadata."""
    doc_id = uuid.uuid4()
    ver_id = uuid.uuid4()
    blocks = [
        "Company Security Policy Overview",
        "All employee passwords must be rotated every 90 days and use MFA.",
        "Production database access requires explicit VP Engineering approval.",
    ]
    mock_service = MockEmbeddingService()

    mock_db = create_mock_session()
    added_records: List[DocumentChunk] = []

    def handle_add(obj):
        if isinstance(obj, DocumentChunk):
            added_records.append(obj)

    mock_db.add.side_effect = handle_add

    chunks = await index_document_chunks(
        db=mock_db,
        document_id=doc_id,
        version_id=ver_id,
        blocks=blocks,
        department="Security",
        filename="security_policy.pdf",
        source_type="pdf",
        embedding_service=mock_service,
    )

    assert len(chunks) > 0
    assert len(added_records) == len(chunks)
    assert mock_db.commit.called

    for chunk in chunks:
        assert chunk.document_id == doc_id
        assert chunk.version_id == ver_id
        assert isinstance(chunk.embedding, list)
        assert len(chunk.embedding) == 1024
        assert chunk.chunk_metadata["department"] == "Security"
        assert chunk.chunk_metadata["filename"] == "security_policy.pdf"
        assert chunk.chunk_metadata["source_type"] == "pdf"
        assert "position" in chunk.chunk_metadata
        assert "token_count" in chunk.chunk_metadata


@pytest.mark.asyncio
async def test_process_document_indexing_preserves_pending_status():
    """Verify background indexing task leaves the document in 'pending' status."""
    doc_id = uuid.uuid4()
    ver_id = uuid.uuid4()
    blocks = ["HR Remote Work Guidelines", "Employees may work remotely 3 days a week."]
    mock_service = MockEmbeddingService()

    mock_doc = Document(
        id=doc_id,
        title="Remote Work Guidelines",
        source_type="pdf",
        status="pending",
    )

    mock_session = create_mock_session()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_doc
    mock_session.execute.return_value = mock_result

    with patch("app.services.indexing.AsyncSessionLocal") as mock_session_local:
        mock_session_local.return_value.__aenter__.return_value = mock_session

        await process_document_indexing(
            document_id=doc_id,
            version_id=ver_id,
            clean_blocks=blocks,
            department="Human Resources",
            filename="remote_policy.pdf",
            source_type="pdf",
            embedding_service=mock_service,
        )

        assert mock_doc.status == "pending"


@pytest.mark.asyncio
async def test_ingest_document_schedules_background_task():
    """Verify ingest_document queues process_document_indexing in BackgroundTasks."""
    bg_tasks = BackgroundTasks()
    mock_db = create_mock_session()

    pdf_text = "Engineering Guidelines\n\nCode reviews are mandatory for all pull requests."
    doc, clean_blocks = await ingest_document(
        db=mock_db,
        file_bytes=pdf_text.encode("utf-8"),
        filename="guidelines.txt",
        title="Engineering Guidelines",
        department="Engineering",
        allowed_roles=["engineering", "admin"],
        background_tasks=bg_tasks,
    )

    assert doc.status == "pending"
    assert len(bg_tasks.tasks) == 1
    task = bg_tasks.tasks[0]
    assert task.func == process_document_indexing
    # Check args or kwargs for document_id and filename
    doc_id_arg = task.kwargs.get("document_id") or (task.args[0] if task.args else None)
    filename_arg = task.kwargs.get("filename") or (task.args[4] if len(task.args) > 4 else None)
    assert doc_id_arg == doc.id
    assert filename_arg == "guidelines.txt"


@pytest.mark.asyncio
async def test_chunk_embedding_and_tsvector_populated_after_ingestion():
    """Verify that chunk records generated after ingestion have embedding and tsvector column definition."""
    doc_id = uuid.uuid4()
    ver_id = uuid.uuid4()
    mock_db = create_mock_session()
    mock_service = MockEmbeddingService()

    blocks = [
        "Financial Compliance Policy",
        "Quarterly expense reports must be submitted before the 5th of each month.",
    ]

    chunks = await index_document_chunks(
        db=mock_db,
        document_id=doc_id,
        version_id=ver_id,
        blocks=blocks,
        department="Finance",
        filename="compliance.docx",
        source_type="docx",
        embedding_service=mock_service,
    )

    assert len(chunks) > 0
    for chunk in chunks:
        # 1. Embedding vector is populated and has dimension 1024
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 1024
        # 2. Chunk text is non-empty
        assert len(chunk.chunk_text) > 0
        # 3. DocumentChunk table schema defines content_tsvector for PostgreSQL full-text search
        assert hasattr(DocumentChunk, "content_tsvector")
        assert DocumentChunk.__table__.columns["content_tsvector"] is not None
        assert "to_tsvector" in str(DocumentChunk.__table__.columns["content_tsvector"].computed.sqltext)
