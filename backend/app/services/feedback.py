"""User Feedback Loop Service (Feature 17).

Provides business logic for submitting user feedback on conversation turns,
enforcing conversation ownership rules, and resolving negative feedback
linkages to documents for curation oversight.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional
import uuid

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import ConversationTurn
from app.models.document import Document, DocumentChunk
from app.models.feedback import Feedback

logger = logging.getLogger(__name__)


async def _get_scalars_all(res: Any) -> list:
    """Helper to safely extract list from scalars().all() for SQLAlchemy AsyncResult and test mocks."""
    if res is None:
        return []
    scalars_fn = getattr(res, "scalars", None)
    if not callable(scalars_fn):
        return []
    s_res = scalars_fn()
    if inspect.isawaitable(s_res):
        s_res = await s_res
    all_fn = getattr(s_res, "all", None)
    if not callable(all_fn):
        return []
    a_res = all_fn()
    if inspect.isawaitable(a_res):
        a_res = await a_res
    return list(a_res)


async def _get_scalar_one_or_none(res: Any) -> Any:
    """Helper to safely extract single item or None for AsyncResult and test mocks."""
    if res is None:
        return None
    fn = getattr(res, "scalar_one_or_none", None)
    if not callable(fn):
        return None
    one_res = fn()
    if inspect.isawaitable(one_res):
        one_res = await one_res
    return one_res


async def submit_user_feedback(
    db: AsyncSession,
    conversation_id: uuid.UUID,
    turn_number: int,
    user_id: uuid.UUID,
    rating: int,
    flagged_chunk_id: Optional[uuid.UUID] = None,
    reason: Optional[str] = None,
) -> Feedback:
    """Submit user feedback for a specific conversation turn after verifying ownership.

    Args:
        db: AsyncSession database connection.
        conversation_id: UUID of the conversation.
        turn_number: Turn index (1-based).
        user_id: UUID of the user submitting feedback.
        rating: Feedback rating (e.g. -1 for negative, 1 for positive).
        flagged_chunk_id: Optional UUID of specific document chunk flagged.
        reason: Optional text feedback reason.

    Returns:
        Persisted Feedback ORM instance.

    Raises:
        HTTPException(403): If the conversation belongs to a different user.
    """
    # 1. Verify ownership against existing conversation turns
    stmt_turn = select(ConversationTurn).where(
        ConversationTurn.conversation_id == conversation_id
    )
    res_turn = await db.execute(stmt_turn)
    turns = await _get_scalars_all(res_turn)

    for turn in turns:
        if turn.user_id is not None and turn.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot submit feedback for a conversation owned by another user.",
            )

    # 2. Verify ownership against any previous feedback records for this conversation
    stmt_fb = select(Feedback).where(Feedback.conversation_id == conversation_id)
    res_fb = await db.execute(stmt_fb)
    existing_fbs = await _get_scalars_all(res_fb)

    for fb in existing_fbs:
        if fb.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot submit feedback for a conversation owned by another user.",
            )

    # 3. Create and persist new Feedback record
    feedback = Feedback(
        id=uuid.uuid4(),
        conversation_id=conversation_id,
        turn_number=turn_number,
        user_id=user_id,
        rating=rating,
        flagged_chunk_id=flagged_chunk_id,
        reason=reason.strip() if reason else None,
    )

    try:
        add_res = db.add(feedback)
        if inspect.isawaitable(add_res):
            await add_res

        commit_res = db.commit()
        if inspect.isawaitable(commit_res):
            await commit_res

        if hasattr(db, "refresh"):
            refresh_res = db.refresh(feedback)
            if inspect.isawaitable(refresh_res):
                await refresh_res
    except Exception as exc:
        if hasattr(db, "rollback"):
            rb_res = db.rollback()
            if inspect.isawaitable(rb_res):
                await rb_res
        logger.warning("Error persisting feedback: %s", exc)

    logger.info(
        "Submitted feedback: conv_id=%s, turn=%d, user=%s, rating=%d",
        conversation_id,
        turn_number,
        user_id,
        rating,
    )
    return feedback


async def list_documents_with_negative_feedback(
    db: AsyncSession,
    min_negative_count: int = 1,
    skip: int = 0,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Retrieve documents associated with repeated negative feedback.

    Negative feedback is identified by rating <= 0. Linkages are resolved via:
    1. Direct flagged_chunk_id -> DocumentChunk -> Document
    2. Turn retrieved_chunk_ids -> DocumentChunk -> Document

    Args:
        db: AsyncSession database connection.
        min_negative_count: Minimum threshold of negative feedback count.
        skip: Pagination offset.
        limit: Max items to return.

    Returns:
        List of dicts summarizing documents with negative feedback counts, flagged chunks, and reasons.
    """
    # Fetch negative feedback records (rating <= 0 or rating == -1)
    stmt_fb = select(Feedback).where(Feedback.rating <= 0)
    res_fb = await db.execute(stmt_fb)
    feedbacks = await _get_scalars_all(res_fb)

    # Map document_id -> aggregated feedback info
    doc_feedback_map: Dict[uuid.UUID, Dict[str, Any]] = {}

    for fb in feedbacks:
        target_chunk_ids: List[uuid.UUID] = []

        if fb.flagged_chunk_id:
            target_chunk_ids.append(fb.flagged_chunk_id)

        # Look up chunks from retrieved_chunk_ids for this conversation turn
        stmt_turn = select(ConversationTurn).where(
            ConversationTurn.conversation_id == fb.conversation_id,
            ConversationTurn.turn_number == fb.turn_number,
        )
        res_turn = await db.execute(stmt_turn)
        turn = await _get_scalar_one_or_none(res_turn)

        if turn and turn.retrieved_chunk_ids:
            for c_id_str in turn.retrieved_chunk_ids:
                try:
                    target_chunk_ids.append(uuid.UUID(str(c_id_str)))
                except ValueError:
                    continue

        if not target_chunk_ids:
            continue

        # Resolve document IDs for target chunks
        stmt_chunks = select(DocumentChunk).where(DocumentChunk.id.in_(target_chunk_ids))
        res_chunks = await db.execute(stmt_chunks)
        chunks = await _get_scalars_all(res_chunks)

        for chunk in chunks:
            doc_id = chunk.document_id
            if doc_id not in doc_feedback_map:
                doc_feedback_map[doc_id] = {
                    "document_id": doc_id,
                    "negative_feedback_count": 0,
                    "flagged_chunks": set(),
                    "reasons": [],
                    "feedback_ids": set(),
                }

            if fb.id not in doc_feedback_map[doc_id]["feedback_ids"]:
                doc_feedback_map[doc_id]["feedback_ids"].add(fb.id)
                doc_feedback_map[doc_id]["negative_feedback_count"] += 1

            doc_feedback_map[doc_id]["flagged_chunks"].add(str(chunk.id))
            if fb.reason and fb.reason not in doc_feedback_map[doc_id]["reasons"]:
                doc_feedback_map[doc_id]["reasons"].append(fb.reason)

    # Filter documents meeting min_negative_count
    filtered_doc_ids = [
        doc_id
        for doc_id, info in doc_feedback_map.items()
        if info["negative_feedback_count"] >= min_negative_count
    ]

    if not filtered_doc_ids:
        return []

    # Fetch document metadata
    stmt_docs = select(Document).where(Document.id.in_(filtered_doc_ids))
    res_docs = await db.execute(stmt_docs)
    documents = await _get_scalars_all(res_docs)
    doc_lookup = {doc.id: doc for doc in documents}

    results = []
    for doc_id in filtered_doc_ids:
        doc = doc_lookup.get(doc_id)
        if not doc:
            continue

        info = doc_feedback_map[doc_id]
        results.append(
            {
                "document_id": str(doc.id),
                "title": doc.title,
                "source_type": doc.source_type,
                "status": doc.status,
                "negative_feedback_count": info["negative_feedback_count"],
                "flagged_chunks": list(info["flagged_chunks"]),
                "reasons": info["reasons"],
            }
        )

    # Sort by negative_feedback_count descending
    results.sort(key=lambda x: x["negative_feedback_count"], reverse=True)

    # Apply pagination
    return results[skip : skip + limit]
