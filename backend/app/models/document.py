import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.db.session import Base

if TYPE_CHECKING:
    from app.models.user import User


class Document(Base):
    """Document metadata model with approval lifecycle and source tracking."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    source_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # e.g., 'pdf', 'docx', 'markdown', 'html'
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
    )  # 'pending', 'approved', 'deprecated', 'rejected'
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    approved_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    approved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'approved', 'deprecated', 'rejected')",
            name="chk_document_status_valid",
        ),
    )

    # Relationships
    approver: Mapped[Optional["User"]] = relationship(
        "User", back_populates="approved_documents", foreign_keys=[approved_by]
    )
    versions: Mapped[List["DocumentVersion"]] = relationship(
        "DocumentVersion",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="desc(DocumentVersion.version_number)",
    )
    acls: Mapped[List["DocumentACL"]] = relationship(
        "DocumentACL", back_populates="document", cascade="all, delete-orphan"
    )
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Document {self.title} [{self.status}]>"


class DocumentVersion(Base):
    """Document versions tracking revisions and active version flag."""

    __tablename__ = "document_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version_number: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1
    )
    is_current: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "document_id", "version_number", name="uq_document_version_number"
        ),
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document", back_populates="versions"
    )
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="version", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<DocumentVersion doc={self.document_id} v={self.version_number} current={self.is_current}>"


class DocumentACL(Base):
    """Access Control List (ACL) granting role or user permissions to documents."""

    __tablename__ = "document_acl"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    allowed_role: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, index=True
    )
    allowed_user: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    permission_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default="read"
    )  # e.g., 'read', 'write', 'admin'

    __table_args__ = (
        # Ensure either allowed_role or allowed_user is specified
        CheckConstraint(
            "(allowed_role IS NOT NULL) OR (allowed_user IS NOT NULL)",
            name="chk_document_acl_role_or_user_not_null",
        ),
        # Fast permission lookup indexes
        Index(
            "ix_document_acl_role_lookup",
            "allowed_role",
            "document_id",
            "permission_type",
        ),
        Index(
            "ix_document_acl_user_lookup",
            "allowed_user",
            "document_id",
            "permission_type",
        ),
        Index(
            "ix_document_acl_doc_perm",
            "document_id",
            "permission_type",
        ),
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="acls")
    user: Mapped[Optional["User"]] = relationship("User", back_populates="acls")

    def __repr__(self) -> str:
        target = f"role={self.allowed_role}" if self.allowed_role else f"user={self.allowed_user}"
        return f"<DocumentACL doc={self.document_id} {target} perm={self.permission_type}>"


class DocumentChunk(Base):
    """Document chunks holding text, pgvector embeddings, metadata, and tsvector for hybrid search."""

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(settings.EMBEDDING_DIMENSION), nullable=True
    )
    chunk_metadata: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    content_tsvector: Mapped[Optional[Any]] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', chunk_text)", persisted=True),
        nullable=True,
    )

    __table_args__ = (
        # HNSW index on embedding for vector similarity search
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # GIN index on content_tsvector for full-text search
        Index(
            "ix_document_chunks_content_tsvector_gin",
            "content_tsvector",
            postgresql_using="gin",
        ),
        # Composite index for document & version filtering
        Index(
            "ix_document_chunks_doc_version",
            "document_id",
            "version_id",
        ),
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document", back_populates="chunks"
    )
    version: Mapped["DocumentVersion"] = relationship(
        "DocumentVersion", back_populates="chunks"
    )

    def __repr__(self) -> str:
        return f"<DocumentChunk id={self.id} doc={self.document_id} v={self.version_id}>"
