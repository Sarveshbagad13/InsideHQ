"""Document indexing service: chunks text, generates embeddings, and persists document chunks."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence
import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.models.document import Document, DocumentChunk
from app.services.chunking import TextChunk, chunk_text_blocks
from app.services.embeddings import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


async def index_document_chunks(
    db: AsyncSession,
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    blocks: Sequence[str],
    department: str = "",
    filename: str = "",
    source_type: str = "",
    extra_metadata: Optional[Dict[str, Any]] = None,
    embedding_service: Optional[EmbeddingService] = None,
    chunk_size: int = 500,
    overlap: int = 75,
) -> List[DocumentChunk]:
    """Chunk clean text blocks, generate embeddings in batch, and persist chunks to the database.

    Args:
        db: Async database session.
        document_id: UUID of the parent Document record.
        version_id: UUID of the DocumentVersion record.
        blocks: Clean text blocks parsed from the source file.
        department: Department name for metadata tagging.
        filename: Source filename for metadata tagging.
        source_type: Document format/type (pdf, docx, markdown, etc.).
        extra_metadata: Optional additional metadata dict.
        embedding_service: Optional custom or mocked EmbeddingService instance.
        chunk_size: Token limit per chunk (default 500).
        overlap: Overlap tokens between consecutive chunks (default 75).

    Returns:
        List of created DocumentChunk ORM instances with embeddings populated.
    """
    if embedding_service is None:
        embedding_service = get_embedding_service()

    # 1. Chunk text blocks into token-bounded TextChunks with headings and positions
    text_chunks: List[TextChunk] = chunk_text_blocks(
        blocks=blocks,
        document_id=document_id,
        version_id=version_id,
        source_filename=filename,
        source_type=source_type,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not text_chunks:
        logger.warning(
            "No chunks produced for document_id=%s version_id=%s",
            document_id,
            version_id,
        )
        return []

    # 2. Extract chunk texts and compute embeddings in batch
    texts = [c.text for c in text_chunks]
    logger.info(
        "Generating embeddings for %d chunks of doc %s (model: %s) ...",
        len(texts),
        document_id,
        embedding_service.model_name,
    )
    embeddings = await embedding_service.aembed_documents(texts)

    # 3. Clean up any existing placeholder/unembedded chunks for this version
    await db.execute(
        delete(DocumentChunk).where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.version_id == version_id,
        )
    )

    # 4. Create and persist DocumentChunk records
    db_chunks: List[DocumentChunk] = []
    base_meta = extra_metadata.copy() if extra_metadata else {}

    for text_chunk, vector in zip(text_chunks, embeddings):
        chunk_meta = {
            **base_meta,
            "department": department,
            "filename": filename,
            "source_type": source_type,
            "position": text_chunk.position,
            "token_count": text_chunk.token_count,
            "section_heading": text_chunk.section_heading,
        }

        chunk_record = DocumentChunk(
            id=uuid.uuid4(),
            document_id=document_id,
            version_id=version_id,
            chunk_text=text_chunk.text,
            embedding=vector,
            chunk_metadata=chunk_meta,
            # content_tsvector is a PostgreSQL Computed column: to_tsvector('english', chunk_text)
        )
        db.add(chunk_record)
        db_chunks.append(chunk_record)

    await db.commit()
    logger.info(
        "Successfully indexed %d chunks for document %s with embeddings.",
        len(db_chunks),
        document_id,
    )
    return db_chunks


async def process_document_indexing(
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    clean_blocks: List[str],
    department: str,
    filename: str,
    source_type: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
    embedding_service: Optional[EmbeddingService] = None,
) -> None:
    """Background task handler: chunks and embeds a document after ingestion.

    This function opens an independent database session, executes the indexing pipeline,
    and ensures the document remains in 'pending' status (awaiting Feature 7 approval).

    Args:
        document_id: UUID of the document.
        version_id: UUID of the version.
        clean_blocks: List of clean parsed text blocks.
        department: Department name.
        filename: Original file name.
        source_type: Source format (e.g. pdf, docx, markdown).
        extra_metadata: Optional additional metadata.
        embedding_service: Optional embedding service instance.
    """
    logger.info(
        "Starting background indexing for document_id=%s (filename=%s)",
        document_id,
        filename,
    )
    try:
        async with AsyncSessionLocal() as session:
            # Confirm document exists and verify it's in pending status
            stmt = select(Document).where(Document.id == document_id)
            result = await session.execute(stmt)
            doc = result.scalar_one_or_none()
            if not doc:
                logger.error("Document %s not found for background indexing.", document_id)
                return

            await index_document_chunks(
                db=session,
                document_id=document_id,
                version_id=version_id,
                blocks=clean_blocks,
                department=department,
                filename=filename,
                source_type=source_type,
                extra_metadata=extra_metadata,
                embedding_service=embedding_service,
            )

            # Ensure document status remains pending until approved in Feature 7
            if doc.status != "pending":
                doc.status = "pending"
                await session.commit()

            logger.info(
                "Background indexing complete for document %s (status remains 'pending').",
                document_id,
            )
    except Exception as exc:
        logger.exception(
            "Error during background indexing for document %s: %s",
            document_id,
            exc,
        )
