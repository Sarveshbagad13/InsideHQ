"""Documents API: endpoints for document upload, ingestion, and inspection."""

import json
from typing import List, Optional
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import OptionalCurrentUser
from app.db.session import get_db
from app.models.document import Document
from app.services.ingestion import ingest_document

router = APIRouter(prefix="/documents", tags=["Documents"])


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------


class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: str
    title: str
    source_type: str
    status: str
    department: str
    allowed_roles: List[str]
    chunks_count: int
    created_at: str


class DocumentDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: str
    title: str
    source_type: str
    status: str
    created_at: str
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    allowed_roles: List[str]
    chunks_count: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document (PDF, DOCX, Markdown)",
)
@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to parse and ingest"),
    title: str = Form(..., description="Document title"),
    department: str = Form(..., description="Owning department name"),
    allowed_roles: str = Form(
        ...,
        description="Comma-separated roles (e.g. 'hr,admin') or JSON array of roles",
    ),
    db: AsyncSession = Depends(get_db),
    current_user: OptionalCurrentUser = None,
) -> DocumentUploadResponse:
    """Ingest a company document (PDF, Word, Markdown) into the knowledge base.

    The file is parsed into clean text blocks using the Unstructured library,
    stored with status 'pending' in the documents table, and tagged with
    appropriate ACLs in document_acl. Chunking and embedding generation
    run automatically as a background task.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a valid filename",
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    # Parse allowed_roles from string / JSON
    roles: List[str] = []
    trimmed = allowed_roles.strip()
    if trimmed.startswith("["):
        try:
            parsed = json.loads(trimmed)
            if isinstance(parsed, list):
                roles = [str(r).strip() for r in parsed if str(r).strip()]
            else:
                roles = [str(parsed).strip()]
        except Exception:
            roles = [r.strip() for r in trimmed.split(",") if r.strip()]
    else:
        roles = [r.strip() for r in trimmed.split(",") if r.strip()]

    if not roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one allowed role must be specified in allowed_roles",
        )

    user_id = current_user.id if current_user else None

    # Ingest document into DB with status='pending' and trigger background chunking/embedding
    doc, clean_blocks = await ingest_document(
        db=db,
        file_bytes=file_bytes,
        filename=file.filename,
        title=title,
        department=department,
        allowed_roles=roles,
        current_user_id=user_id,
        background_tasks=background_tasks,
    )

    return DocumentUploadResponse(
        document_id=str(doc.id),
        title=doc.title,
        source_type=doc.source_type,
        status=doc.status,
        department=department,
        allowed_roles=roles,
        chunks_count=len(clean_blocks),
        created_at=doc.created_at.isoformat() if doc.created_at else "",
    )


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document metadata and ingestion status",
)
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> DocumentDetailResponse:
    """Retrieve document status, ACLs, and metadata."""
    stmt = (
        select(Document)
        .where(Document.id == document_id)
        .options(
            selectinload(Document.acls),
            selectinload(Document.chunks),
        )
    )
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    allowed_roles = [acl.allowed_role for acl in doc.acls if acl.allowed_role]

    return DocumentDetailResponse(
        document_id=str(doc.id),
        title=doc.title,
        source_type=doc.source_type,
        status=doc.status,
        created_at=doc.created_at.isoformat() if doc.created_at else "",
        approved_by=str(doc.approved_by) if doc.approved_by else None,
        approved_at=doc.approved_at.isoformat() if doc.approved_at else None,
        allowed_roles=allowed_roles,
        chunks_count=len(doc.chunks),
    )
