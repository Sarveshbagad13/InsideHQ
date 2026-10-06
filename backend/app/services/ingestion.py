"""Document parsing and ingestion service using the Unstructured library."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid

from fastapi import BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentACL, DocumentChunk, DocumentVersion
from app.services.indexing import index_document_chunks, process_document_indexing


SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
}


def parse_file_to_blocks(file_bytes: bytes, filename: str) -> Tuple[str, List[str]]:
    """Parse raw file bytes into clean text blocks using Unstructured.

    Args:
        file_bytes: The raw binary content of the file.
        filename: Original file name used to determine source type.

    Returns:
        A tuple of (source_type, list_of_clean_text_blocks).
    """
    ext = Path(filename).suffix.lower()
    source_type = SUPPORTED_EXTENSIONS.get(ext, ext.lstrip(".") or "unknown")

    elements = []
    try:
        if source_type == "pdf":
            from unstructured.partition.pdf import partition_pdf
            elements = partition_pdf(file=io.BytesIO(file_bytes), strategy="fast")
        elif source_type == "docx":
            from unstructured.partition.docx import partition_docx
            elements = partition_docx(file=io.BytesIO(file_bytes))
        elif source_type == "markdown":
            from unstructured.partition.md import partition_md
            text_content = file_bytes.decode("utf-8", errors="replace")
            elements = partition_md(text=text_content)
        elif source_type == "text":
            from unstructured.partition.text import partition_text
            text_content = file_bytes.decode("utf-8", errors="replace")
            elements = partition_text(text=text_content)
        else:
            # Fallback to auto partition
            from unstructured.partition.auto import partition
            elements = partition(file=io.BytesIO(file_bytes))
    except Exception:
        # Fallback if partitioner raises on raw content or import failure
        try:
            from unstructured.partition.auto import partition
            elements = partition(file=io.BytesIO(file_bytes))
        except Exception:
            text_fallback = file_bytes.decode("utf-8", errors="replace")
            return source_type, [p.strip() for p in text_fallback.split("\n\n") if p.strip()]

    clean_blocks: List[str] = []
    for el in elements:
        text = str(el).strip()
        if text and len(text) > 0:
            clean_blocks.append(text)

    # If unstructured returned no blocks, fall back to plain text decoding
    if not clean_blocks and len(file_bytes) > 0:
        try:
            raw_text = file_bytes.decode("utf-8", errors="replace")
            clean_blocks = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
        except Exception:
            clean_blocks = []

    return source_type, clean_blocks


async def ingest_document(
    db: AsyncSession,
    file_bytes: bytes,
    filename: str,
    title: str,
    department: str,
    allowed_roles: List[str],
    current_user_id: Optional[uuid.UUID] = None,
    background_tasks: Optional[BackgroundTasks] = None,
    run_indexing_sync: bool = False,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Document, List[str]]:
    """Parse an uploaded document, persist metadata and ACLs, and trigger chunking/embedding.

    The document is stored with status="pending" (awaiting approval in Feature 7).
    Chunking and embedding run automatically as a background task (or synchronously
    if run_indexing_sync=True).

    Args:
        db: Async database session.
        file_bytes: File contents in bytes.
        filename: Name of the uploaded file.
        title: User-provided or extracted title for the document.
        department: Owning department name.
        allowed_roles: List of roles allowed to read the document.
        current_user_id: Optional UUID of the user who uploaded the document.
        background_tasks: Optional FastAPI BackgroundTasks instance to schedule embedding generation.
        run_indexing_sync: If True, executes chunking and embedding synchronously.
        extra_metadata: Optional additional metadata dictionary.

    Returns:
        Tuple of (persisted_document, list_of_clean_text_blocks).
    """
    # Run parsing in thread pool to prevent blocking the async event loop
    source_type, clean_blocks = await asyncio.to_thread(
        parse_file_to_blocks, file_bytes, filename
    )

    doc_id = uuid.uuid4()
    version_id = uuid.uuid4()

    # 1. Store document metadata with status='pending'
    doc = Document(
        id=doc_id,
        title=title,
        source_type=source_type,
        status="pending",
    )
    db.add(doc)

    # 2. Store initial version (v1, is_current=True)
    version = DocumentVersion(
        id=version_id,
        document_id=doc_id,
        version_number=1,
        is_current=True,
    )
    db.add(version)

    # 3. Store ACL entries for allowed roles
    normalized_roles = {r.strip().lower() for r in allowed_roles if r.strip()}
    for role in normalized_roles:
        acl = DocumentACL(
            id=uuid.uuid4(),
            document_id=doc_id,
            allowed_role=role,
            permission_type="read",
        )
        db.add(acl)

    # If owner user ID is provided, grant admin permission on the document
    if current_user_id:
        user_acl = DocumentACL(
            id=uuid.uuid4(),
            document_id=doc_id,
            allowed_user=current_user_id,
            permission_type="admin",
        )
        db.add(user_acl)

    await db.commit()
    await db.refresh(doc)

    # 4. Trigger chunking and embedding generation
    if run_indexing_sync:
        await index_document_chunks(
            db=db,
            document_id=doc_id,
            version_id=version_id,
            blocks=clean_blocks,
            department=department,
            filename=filename,
            source_type=source_type,
            extra_metadata=extra_metadata,
        )
    elif background_tasks is not None:
        background_tasks.add_task(
            process_document_indexing,
            document_id=doc_id,
            version_id=version_id,
            clean_blocks=clean_blocks,
            department=department,
            filename=filename,
            source_type=source_type,
            extra_metadata=extra_metadata,
        )
    else:
        # Fallback: store placeholder un-embedded chunks or schedule task
        for idx, block in enumerate(clean_blocks):
            chunk = DocumentChunk(
                id=uuid.uuid4(),
                document_id=doc_id,
                version_id=version_id,
                chunk_text=block,
                chunk_metadata={
                    "department": department,
                    "filename": filename,
                    "block_index": idx,
                    "source_type": source_type,
                },
                embedding=None,
            )
            db.add(chunk)
        await db.commit()

    return doc, clean_blocks
