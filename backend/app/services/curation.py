"""Content approval and curation workflow service (Feature 7).

Provides functions for reviewing pending documents, approving them for retrieval,
rejecting documents with reasons, and deprecating outdated documents.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import List, Optional, Tuple
import uuid

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.document import Document, DocumentACL, DocumentChunk, DocumentVersion
from app.models.user import User

logger = logging.getLogger(__name__)


def is_document_searchable(doc: Document | str) -> bool:
    """Check if a document or document status string is eligible for retrieval.

    Only documents with status 'approved' are searchable by employees.
    Documents in 'pending', 'rejected', or 'deprecated' statuses are excluded.

    Args:
        doc: Document ORM instance or status string.

    Returns:
        True if the document is approved and searchable, False otherwise.
    """
    doc_status = doc.status if isinstance(doc, Document) else str(doc)
    return doc_status.strip().lower() == "approved"


async def get_document_with_relations(
    db: AsyncSession, document_id: uuid.UUID
) -> Optional[Document]:
    """Retrieve document by ID with ACLs, versions, and chunks loaded."""
    stmt = (
        select(Document)
        .where(Document.id == document_id)
        .options(
            selectinload(Document.acls),
            selectinload(Document.versions),
            selectinload(Document.chunks),
            selectinload(Document.approver),
        )
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def list_pending_documents(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
) -> List[Document]:
    """List all documents currently awaiting approval (status='pending').

    Args:
        db: Async database session.
        skip: Pagination offset.
        limit: Maximum number of records to return.

    Returns:
        List of Document instances in 'pending' status with relations loaded.
    """
    stmt = (
        select(Document)
        .where(Document.status == "pending")
        .options(
            selectinload(Document.acls),
            selectinload(Document.chunks),
            selectinload(Document.versions),
        )
        .order_by(Document.created_at.asc())
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def list_documents_by_status(
    db: AsyncSession,
    status_filter: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
) -> List[Document]:
    """List documents optionally filtered by lifecycle status.

    Args:
        db: Async database session.
        status_filter: Optional filter ('pending', 'approved', 'rejected', 'deprecated').
        skip: Pagination offset.
        limit: Maximum number of records to return.

    Returns:
        List of Document instances matching status filter.
    """
    stmt = select(Document).options(
        selectinload(Document.acls),
        selectinload(Document.chunks),
        selectinload(Document.versions),
        selectinload(Document.approver),
    )
    if status_filter:
        stmt = stmt.where(Document.status == status_filter.strip().lower())

    stmt = stmt.order_by(Document.created_at.desc()).offset(skip) .limit(limit)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def approve_document(
    db: AsyncSession,
    document_id: uuid.UUID,
    approver: User,
) -> Document:
    """Approve a pending document, making it eligible for retrieval search.

    Records the approving user's UUID and approval timestamp.

    Args:
        db: Async database session.
        document_id: UUID of document to approve.
        approver: User instance performing the approval (must be admin or curator).

    Returns:
        The updated Document instance with status='approved'.

    Raises:
        HTTPException 404: If document is not found.
    """
    doc = await get_document_with_relations(db, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    doc.status = "approved"
    doc.approved_by = approver.id
    doc.approved_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(doc)
    logger.info(
        "Document %s ('%s') approved by %s (role: %s)",
        doc.id,
        doc.title,
        approver.email,
        approver.role,
    )
    return doc


async def reject_document(
    db: AsyncSession,
    document_id: uuid.UUID,
    reviewer: User,
    reason: str,
) -> Document:
    """Reject a pending document with a mandatory reason.

    The document is flagged as 'rejected' and excluded from retrieval.

    Args:
        db: Async database session.
        document_id: UUID of document to reject.
        reviewer: User instance performing the rejection.
        reason: Required explanation for rejection.

    Returns:
        The updated Document instance with status='rejected'.

    Raises:
        HTTPException 400: If rejection reason is empty.
        HTTPException 404: If document is not found.
    """
    if not reason or not reason.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Rejection reason is required and cannot be empty.",
        )

    doc = await get_document_with_relations(db, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    doc.status = "rejected"
    # Store rejection reason in chunk_metadata of first chunk or document state if applicable
    doc.approved_by = reviewer.id
    doc.approved_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(doc)
    logger.info(
        "Document %s ('%s') rejected by %s (role: %s). Reason: %s",
        doc.id,
        doc.title,
        reviewer.email,
        reviewer.role,
        reason,
    )
    return doc


async def deprecate_document(
    db: AsyncSession,
    document_id: uuid.UUID,
    reviewer: User,
    reason: Optional[str] = None,
) -> Document:
    """Deprecate a previously approved document.

    The document is retained for audit trails but excluded from all subsequent
    retrieval queries.

    Args:
        db: Async database session.
        document_id: UUID of document to deprecate.
        reviewer: User instance performing deprecation.
        reason: Optional explanation for deprecation.

    Returns:
        The updated Document instance with status='deprecated'.

    Raises:
        HTTPException 404: If document is not found.
    """
    doc = await get_document_with_relations(db, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    doc.status = "deprecated"
    await db.commit()
    await db.refresh(doc)
    logger.info(
        "Document %s ('%s') deprecated by %s (role: %s).",
        doc.id,
        doc.title,
        reviewer.email,
        reviewer.role,
    )
    return doc
