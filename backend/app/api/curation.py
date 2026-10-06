"""Content curation and approval workflow API endpoints (Feature 7).

Endpoints for reviewing pending documents, approving them for retrieval,
rejecting with required reasons, and deprecating outdated documents.
Accessible only to users with 'admin' or 'curator' roles.
"""

from __future__ import annotations

from typing import List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import RequireCuratorOrAdmin
from app.db.session import get_db
from app.models.document import Document
from app.models.user import User
from app.services.curation import (
    approve_document,
    deprecate_document,
    list_documents_by_status,
    list_pending_documents,
    reject_document,
)

router = APIRouter(prefix="/curation", tags=["Curation & Approval"])


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------


class RejectDocumentRequest(BaseModel):
    reason: str = Field(
        ...,
        min_length=1,
        description="Mandatory explanation detailing why the document was rejected",
    )


class DeprecateDocumentRequest(BaseModel):
    reason: Optional[str] = Field(
        default=None,
        description="Optional reason explaining why the document is being deprecated",
    )


class CurationDocumentItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: str
    title: str
    source_type: str
    status: str
    department: Optional[str] = None
    allowed_roles: List[str]
    chunks_count: int
    created_at: str
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None


class CurationActionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: str
    title: str
    status: str
    message: str
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None


def _format_doc_item(doc: Document) -> CurationDocumentItem:
    """Format a Document ORM model into CurationDocumentItem."""
    allowed_roles = [acl.allowed_role for acl in doc.acls if acl.allowed_role]
    department = None
    if doc.chunks and len(doc.chunks) > 0:
        first_chunk_meta = doc.chunks[0].chunk_metadata or {}
        department = first_chunk_meta.get("department")

    return CurationDocumentItem(
        document_id=str(doc.id),
        title=doc.title,
        source_type=doc.source_type,
        status=doc.status,
        department=department,
        allowed_roles=allowed_roles,
        chunks_count=len(doc.chunks),
        created_at=doc.created_at.isoformat() if doc.created_at else "",
        approved_by=str(doc.approved_by) if doc.approved_by else None,
        approved_at=doc.approved_at.isoformat() if doc.approved_at else None,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/pending",
    response_model=List[CurationDocumentItem],
    summary="List all documents awaiting approval",
)
async def get_pending_documents(
    current_user: RequireCuratorOrAdmin,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> List[CurationDocumentItem]:
    """Retrieve list of all pending documents requiring review and approval.

    Restricted to users with 'admin' or 'curator' roles.
    """
    docs = await list_pending_documents(db=db, skip=skip, limit=limit)
    return [_format_doc_item(doc) for doc in docs]


@router.get(
    "/documents",
    response_model=List[CurationDocumentItem],
    summary="List documents filtered by status",
)
async def get_curation_documents(
    current_user: RequireCuratorOrAdmin,
    status_filter: Optional[str] = Query(
        None,
        alias="status",
        description="Filter by lifecycle status ('pending', 'approved', 'rejected', 'deprecated')",
    ),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> List[CurationDocumentItem]:
    """List documents with optional status filtering for curators and admins."""
    docs = await list_documents_by_status(
        db=db, status_filter=status_filter, skip=skip, limit=limit
    )
    return [_format_doc_item(doc) for doc in docs]


@router.post(
    "/{document_id}/approve",
    response_model=CurationActionResponse,
    summary="Approve a document for retrieval search",
)
async def approve_pending_document(
    document_id: uuid.UUID,
    current_user: RequireCuratorOrAdmin,
    db: AsyncSession = Depends(get_db),
) -> CurationActionResponse:
    """Approve a document to make it searchable in the employee knowledge assistant.

    Records the approving user ID and timestamp, and changes document status to 'approved'.
    Restricted to users with 'admin' or 'curator' roles.
    """
    doc = await approve_document(db=db, document_id=document_id, approver=current_user)
    return CurationActionResponse(
        document_id=str(doc.id),
        title=doc.title,
        status=doc.status,
        message="Document approved successfully and is now searchable.",
        approved_by=str(doc.approved_by) if doc.approved_by else None,
        approved_at=doc.approved_at.isoformat() if doc.approved_at else None,
    )


@router.post(
    "/{document_id}/reject",
    response_model=CurationActionResponse,
    summary="Reject a document with reason",
)
async def reject_pending_document(
    document_id: uuid.UUID,
    payload: RejectDocumentRequest,
    current_user: RequireCuratorOrAdmin,
    db: AsyncSession = Depends(get_db),
) -> CurationActionResponse:
    """Reject a pending document with a mandatory reason.

    Document status is changed to 'rejected' and excluded from retrieval.
    Restricted to users with 'admin' or 'curator' roles.
    """
    if not payload.reason.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Rejection reason cannot be empty",
        )

    doc = await reject_document(
        db=db,
        document_id=document_id,
        reviewer=current_user,
        reason=payload.reason.strip(),
    )
    return CurationActionResponse(
        document_id=str(doc.id),
        title=doc.title,
        status=doc.status,
        message=f"Document rejected: {payload.reason.strip()}",
        approved_by=str(doc.approved_by) if doc.approved_by else None,
        approved_at=doc.approved_at.isoformat() if doc.approved_at else None,
    )


@router.post(
    "/{document_id}/deprecate",
    response_model=CurationActionResponse,
    summary="Deprecate an outdated document",
)
async def deprecate_active_document(
    document_id: uuid.UUID,
    current_user: RequireCuratorOrAdmin,
    payload: Optional[DeprecateDocumentRequest] = None,
    db: AsyncSession = Depends(get_db),
) -> CurationActionResponse:
    """Deprecate a document, excluding it from future retrieval while preserving audit trails.

    Restricted to users with 'admin' or 'curator' roles.
    """
    reason = payload.reason if payload else None
    doc = await deprecate_document(
        db=db,
        document_id=document_id,
        reviewer=current_user,
        reason=reason,
    )
    return CurationActionResponse(
        document_id=str(doc.id),
        title=doc.title,
        status=doc.status,
        message="Document has been deprecated and excluded from new search queries.",
        approved_by=str(doc.approved_by) if doc.approved_by else None,
        approved_at=doc.approved_at.isoformat() if doc.approved_at else None,
    )


@router.get(
    "/negative-feedback",
    summary="List documents with repeated negative feedback",
)
async def get_curation_negative_feedback(
    current_user: RequireCuratorOrAdmin,
    min_negative_count: int = Query(1, ge=1),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve documents with repeated negative user feedback for curation oversight.

    Restricted to Curator or Admin users.
    """
    from app.services.feedback import list_documents_with_negative_feedback

    return await list_documents_with_negative_feedback(
        db=db,
        min_negative_count=min_negative_count,
        skip=skip,
        limit=limit,
    )
