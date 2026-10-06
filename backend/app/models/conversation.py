"""SQLAlchemy model for Conversation & Session Memory (Feature 13).

Stores multi-turn dialogue history per conversation_id, capturing original questions,
rewritten standalone queries, answers, and retrieved chunk IDs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional
import uuid

from sqlalchemy import DateTime, Integer, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class ConversationTurn(Base):
    """Postgres table model for session memory tracking conversation turns."""

    __tablename__ = "conversation_turns"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), index=True, nullable=False
    )
    turn_number: Mapped[int] = mapped_column(
        Integer, nullable=False
    )
    question: Mapped[str] = mapped_column(
        Text, nullable=False
    )
    resolved_query: Mapped[str] = mapped_column(
        Text, nullable=False
    )
    answer: Mapped[str] = mapped_column(
        Text, nullable=False
    )
    retrieved_chunk_ids: Mapped[List[str]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), index=True, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("conversation_id", "turn_number", name="uq_conversation_turn_number"),
    )
