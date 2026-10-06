"""User Feedback Loop API Endpoints (Feature 17).

Endpoints for submitting ratings/feedback on conversation turns and viewing
documents with repeated negative feedback.
"""

from __future__ import annotations

from typing import List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, RequireCuratorOrAdmin, get_db
from app.services.feedback import (
    list_documents_with_negative_feedback,
    submit_user_feedback,
)

router = APIRouter(prefix="/feedback", tags=["User Feedback Loop"])


class SubmitFeedbackRequest(BaseModel):
    """Payload to submit user feedback on a conversation turn."""

    conversation_id: uuid.UUID = Field(..., description="UUID of the conversation session")
    turn_number: int = Field(..., ge=1, description="1-based turn index")
    rating: int = Field(
        ..., description="Rating score (e.g. -1 for negative / thumbs down, 1 for positive / thumbs up)"
    )
    flagged_chunk_id: Optional[uuid.UUID] = Field(
        default=None, description="Optional UUID of specific document chunk flagged"
    )
    reason: Optional[str] = Field(
        default=None, description="Optional explanation or reason for feedback"
    )


class FeedbackResponse(BaseModel):
    """Response model for submitted feedback."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    conversation_id: uuid.UUID
    turn_number: int
    user_id: uuid.UUID
    rating: int
    flagged_chunk_id: Optional[uuid.UUID] = None
    reason: Optional[str] = None
    created_at: str


class NegativeFeedbackDocumentItem(BaseModel):
    """Summary of a document with repeated negative feedback."""

    document_id: str
    title: str
    source_type: str
    status: str
    negative_feedback_count: int
    flagged_chunks: List[str]
    reasons: List[str]


@router.post(
    "",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit feedback for own conversation turn",
)
async def submit_feedback(
    payload: SubmitFeedbackRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> FeedbackResponse:
    """Submit rating and optional flagged chunk / reason for a conversation turn.

    Restricted to user's own conversations.
    """
    feedback = await submit_user_feedback(
        db=db,
        conversation_id=payload.conversation_id,
        turn_number=payload.turn_number,
        user_id=current_user.id,
        rating=payload.rating,
        flagged_chunk_id=payload.flagged_chunk_id,
        reason=payload.reason,
    )
    return FeedbackResponse(
        id=feedback.id,
        conversation_id=feedback.conversation_id,
        turn_number=feedback.turn_number,
        user_id=feedback.user_id,
        rating=feedback.rating,
        flagged_chunk_id=feedback.flagged_chunk_id,
        reason=feedback.reason,
        created_at=feedback.created_at.isoformat() if feedback.created_at else "",
    )


@router.get(
    "/negative-documents",
    response_model=List[NegativeFeedbackDocumentItem],
    summary="List documents with repeated negative feedback (Curator/Admin)",
)
async def get_negative_feedback_documents(
    current_user: RequireCuratorOrAdmin,
    min_negative_count: int = Query(1, ge=1, description="Minimum negative feedback count"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> List[NegativeFeedbackDocumentItem]:
    """List documents with repeated negative feedback.

    Restricted to Curator or Admin users.
    """
    return await list_documents_with_negative_feedback(
        db=db,
        min_negative_count=min_negative_count,
        skip=skip,
        limit=limit,
    )
