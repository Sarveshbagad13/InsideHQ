"""Chat and Answer Generation API Endpoint.

Provides POST /chat (and POST /api/v1/chat) for asking questions against the employee
knowledge base with permission-aware retrieval, cross-encoder reranking, grounded
Groq answer generation, and structured citation streaming.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, get_current_user, get_db
from app.db.session import AsyncSessionLocal, get_db
from app.models.user import User
from app.services.generation import (
    Citation,
    GenerationService,
    get_generation_service,
)
from app.services.guardrails import REFUSAL_MESSAGE, validate_query_safety
from app.services.memory import (
    get_conversation_history,
    is_referential_query,
    rewrite_query,
    save_conversation_turn,
)
from app.services.rate_limiter import rate_limiter
from app.services.retrieval import retrieve_and_rerank
from app.services.tracing import end_trace, start_trace

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


class ChatRequest(BaseModel):
    """Chat question request payload."""

    query: str = Field(
        ...,
        min_length=1,
        description="Question to ask the knowledge assistant",
        json_schema_extra={"example": "What is the company parental leave policy?"},
    )
    conversation_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Optional conversation ID for multi-turn session memory",
    )
    stream: bool = Field(default=True, description="Whether to stream response tokens via Server-Sent Events")


class ChatResponse(BaseModel):
    """Non-streaming chat response model with answer, citations, and conversation metadata."""

    conversation_id: uuid.UUID
    turn_number: int
    query: str
    resolved_query: str
    answer: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    sources_count: int = 0
    chunks_used: int = 0
    # Feature 21: allows client to distinguish normal / abstained / restricted answers
    answer_type: str = Field(default="normal", description="'normal' | 'abstained' | 'restricted'")


@router.post(
    "/chat",
    response_model=None,
    summary="Ask a question to the RAG Knowledge Assistant",
    description="Performs permission-aware retrieval, cross-encoder reranking, and streams grounded Groq answer with citations.",
)
async def chat_endpoint(
    request: ChatRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    generation_service: GenerationService = Depends(get_generation_service),
):
    """Execute grounded RAG workflow for the authenticated employee with session memory."""
    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty or whitespace.",
        )

    # Feature 19: Rate Limiting & Daily Token Budget Protection
    # Streaming / expensive operations use stricter sliding window limits
    rate_limiter.check_rate_limit(user_id=current_user.id, is_expensive=request.stream)
    rate_limiter.check_daily_token_budget(user_id=current_user.id, estimated_tokens=len(query) // 4 + 50)

    conversation_id = request.conversation_id or uuid.uuid4()

    # Start Langfuse / Distributed Trace for this /chat request
    start_trace(
        user=current_user,
        query=query,
        conversation_id=conversation_id,
        metadata={
            "stream": request.stream,
        },
    )

    # Guardrail 1: Block abusive/out-of-scope queries BEFORE retrieval
    is_safe, refusal_reason = validate_query_safety(query)
    if not is_safe:
        logger.warning(
            "Guardrail 1 Triggered in chat endpoint: user=%s, reason='%s', query='%s'",
            current_user.email,
            refusal_reason,
            query,
        )
        end_trace(
            output_answer=REFUSAL_MESSAGE,
            metadata={"status": "refused_by_guardrail_1", "refusal_reason": refusal_reason},
        )
        if request.stream:
            async def refusal_event_generator():
                yield f"data: {json.dumps({'type': 'token', 'content': REFUSAL_MESSAGE})}\n\n"
                yield f"data: {json.dumps({'type': 'citations', 'citations': []})}\n\n"
                # answer_type='restricted' signals guardrail 1 refusal (blocked query)
                yield f"data: {json.dumps({'type': 'done', 'full_answer': REFUSAL_MESSAGE, 'conversation_id': str(conversation_id), 'turn_number': 1, 'resolved_query': query, 'answer_type': 'restricted'})}\n\n"

            return StreamingResponse(
                refusal_event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return ChatResponse(
                conversation_id=conversation_id,
                turn_number=1,
                query=query,
                resolved_query=query,
                answer=REFUSAL_MESSAGE,
                citations=[],
                sources_count=0,
                chunks_used=0,
                answer_type="restricted",
            )

    # 1. Retrieve recent turns from PostgreSQL session memory (last 2-3 turns)
    history = await get_conversation_history(db=db, conversation_id=conversation_id, limit=3)
    turn_number = (history[-1].turn_number + 1) if history else 1

    # 2. Referential query resolution: Rewrite if follow-up question references previous context
    if history and is_referential_query(query):
        resolved_query = await rewrite_query(
            current_query=query,
            history=history,
            generation_service=generation_service,
        )
        logger.info(
            "Multi-turn query rewritten: conv_id=%s, turn=%d, original='%s' -> resolved='%s'",
            conversation_id,
            turn_number,
            query,
            resolved_query,
        )
    else:
        resolved_query = query

    logger.info(
        "Chat request processing: user=%s (role=%s, dept=%s), conv_id=%s, turn=%d, query='%s', resolved='%s', stream=%s",
        current_user.email,
        current_user.role,
        current_user.department,
        conversation_id,
        turn_number,
        query,
        resolved_query,
        request.stream,
    )

    # 3. Permission-aware retrieval & cross-encoder reranking using resolved query
    top_chunks = await retrieve_and_rerank(
        db=db,
        query=resolved_query,
        user=current_user,
        top_k_candidates=20,
        top_n_rerank=5,
    )

    logger.info(
        "Retrieved %d reranked chunks for resolved query '%s' (user: %s)",
        len(top_chunks),
        resolved_query,
        current_user.email,
    )

    retrieved_chunk_ids = [str(c.chunk_id) for c in top_chunks]

    # 4. Handle Streaming vs Non-streaming response and persist turn
    if request.stream:
        async def event_generator():
            full_answer_text = ""
            async for event in generation_service.astream_answer(question=resolved_query, chunks=top_chunks):
                if event["type"] == "done":
                    full_answer_text = event.get("full_answer", "")
                    event["conversation_id"] = str(conversation_id)
                    event["turn_number"] = turn_number
                    event["resolved_query"] = resolved_query
                    # Feature 21: classify answer type for frontend badges
                    from app.services.generation import NO_INFO_PHRASE
                    if not top_chunks:
                        event["answer_type"] = "abstained"  # no matching chunks after filters
                    elif NO_INFO_PHRASE in full_answer_text:
                        event["answer_type"] = "abstained"  # LLM explicitly abstained
                    else:
                        event["answer_type"] = "normal"

                payload = json.dumps(event)
                yield f"data: {payload}\n\n"

            # Finalize trace for streaming response
            end_trace(
                output_answer=full_answer_text,
                metadata={
                    "resolved_query": resolved_query,
                    "turn_number": turn_number,
                    "retrieved_chunks_count": len(retrieved_chunk_ids),
                },
            )

            # Persist session memory turn asynchronously and record token usage
            try:
                rate_limiter.record_token_usage(
                    current_user.id, (len(query) + len(full_answer_text)) // 4 + 50
                )
                async with AsyncSessionLocal() as session:
                    await save_conversation_turn(
                        db=session,
                        conversation_id=conversation_id,
                        turn_number=turn_number,
                        question=query,
                        resolved_query=resolved_query,
                        answer=full_answer_text,
                        retrieved_chunk_ids=retrieved_chunk_ids,
                        user_id=current_user.id,
                    )
            except Exception as e:
                logger.warning("Could not persist streaming turn in session memory: %s", e)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        answer, citations = await generation_service.agenerate_answer(
            question=resolved_query,
            chunks=top_chunks,
        )

        # Feature 21: classify answer type for frontend badges
        from app.services.generation import NO_INFO_PHRASE
        if not top_chunks:
            answer_type = "abstained"  # no chunks returned after applying filters
        elif NO_INFO_PHRASE in answer:
            answer_type = "abstained"  # LLM explicitly abstained due to lack of context
        else:
            answer_type = "normal"

        # Finalize trace for non-streaming response
        end_trace(
            output_answer=answer,
            metadata={
                "resolved_query": resolved_query,
                "turn_number": turn_number,
                "citations_count": len(citations),
                "chunks_used": len(top_chunks),
                "answer_type": answer_type,
            },
        )

        # Persist session memory turn and record token usage
        try:
            rate_limiter.record_token_usage(
                current_user.id, (len(query) + len(answer)) // 4 + 50
            )
            await save_conversation_turn(
                db=db,
                conversation_id=conversation_id,
                turn_number=turn_number,
                question=query,
                resolved_query=resolved_query,
                answer=answer,
                retrieved_chunk_ids=retrieved_chunk_ids,
                user_id=current_user.id,
            )
        except Exception as e:
            logger.warning("Could not persist non-streaming turn in session memory: %s", e)

        return ChatResponse(
            conversation_id=conversation_id,
            turn_number=turn_number,
            query=query,
            resolved_query=resolved_query,
            answer=answer,
            citations=[c.to_dict() for c in citations],
            sources_count=len(citations),
            chunks_used=len(top_chunks),
            answer_type=answer_type,
        )
