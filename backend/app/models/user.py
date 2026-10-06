import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List
from sqlalchemy import DateTime, String, func, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base

if TYPE_CHECKING:
    from app.models.document import Document, DocumentACL


class User(Base):
    """User model representing internal enterprise employees."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    role: Mapped[str] = mapped_column(
        String(50), index=True, nullable=False
    )  # e.g., 'admin', 'hr', 'engineering', 'finance', 'employee'
    department: Mapped[str] = mapped_column(
        String(100), index=True, nullable=False
    )  # e.g., 'Engineering', 'Human Resources', 'Finance', 'Legal'
    hashed_password: Mapped[str] = mapped_column(
        String(255), nullable=False, default=""
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    approved_documents: Mapped[List["Document"]] = relationship(
        "Document", back_populates="approver", foreign_keys="Document.approved_by"
    )
    acls: Mapped[List["DocumentACL"]] = relationship(
        "DocumentACL", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User {self.email} ({self.role} - {self.department})>"
