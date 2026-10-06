"""SQLAlchemy model for User Feedback Loop (Feature 17).

Stores user feedback (ratings, flagged chunks, reasons) per conversation turn
and tracks feedback linkage to documents for curation oversight.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional
import uuid

from sqlalchemy import DateTime, ForeignKey, Integer, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base

if TYPE_CHECKING:
    from app.models.document import DocumentChunk
    from app.models.user import User


class Feedback(Base):
    """Postgres table storing user ratings and feedback on RAG response turns."""

    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), index=True, nullable=False
    )
    turn_number: Mapped[int] = mapped_column(
        Integer, nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    rating: Mapped[int] = mapped_column(
        Integer, nullable=False
    )  # -1 (negative / thumbs down), +1 (positive / thumbs up), or 1-5 scale
    flagged_chunk_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_chunks.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    reason: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship("User")
    flagged_chunk: Mapped[Optional["DocumentChunk"]] = relationship("DocumentChunk")

    def __repr__(self) -> str:
        return (
            f"<Feedback conv={self.conversation_id} turn={self.turn_number} "
            f"user={self.user_id} rating={self.rating}>"
        )
