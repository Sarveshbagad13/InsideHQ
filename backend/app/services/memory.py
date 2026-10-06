"""Conversation & Session Memory Service (Feature 13).

Provides multi-turn dialogue management:
- Loads previous conversation turns from Postgres for a given conversation_id.
- Detects whether a follow-up user query is referential (contains pronouns/anaphoric references).
- Uses Google Gemini to rewrite referential follow-ups into standalone, self-contained search queries
  using context from the last 2-3 turns.
- Persists completed conversation turns with questions, resolved queries, answers, and chunk IDs.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Sequence
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import ConversationTurn
from app.services.generation import GenerationService

logger = logging.getLogger(__name__)

# Patterns indicative of referential or contextual follow-up questions
REFERENTIAL_PATTERNS = [
    r"(?i)\b(?:it|its|they|them|their|theirs|this|that|these|those|he|him|his|she|her|hers)\b",
    r"(?i)\b(?:what|how)\s+about\b",
    r"(?i)\b(?:and\s+for|same\s+for|what\s+if|tell\s+me\s+more|give\s+more\s+details|why\s+is\s+that)\b",
    r"(?i)\b(?:the\s+former|the\s+latter|which\s+one|who\s+is\s+responsible|can\s+they|can\s+it|does\s+it|is\s+it)\b",
    r"(?i)\b(?:how\s+much(?:\s+is\s+that)?|how\s+long|where\s+can\s+i\s+find\s+it)\b",
    r"(?i)\b(?:and\s+the|and\s+what)\b",
]

REWRITE_SYSTEM_PROMPT = """You are an expert search query resolver.
Your task is to reformulate an employee's follow-up question into a single, fully self-contained, standalone search query using the provided recent conversation history.

CRITICAL RULES:
1. Resolve all pronouns (it, they, this, that, etc.) and implicit references by substituting the actual subjects/entities from previous turns.
2. Preserve the specific intent and technical/business terms from the follow-up question.
3. Output ONLY the rewritten standalone query string. Do NOT provide answers, explanations, greetings, or punctuation quotes.
4. If the question is already fully self-contained and does not depend on prior turns, return the question unchanged.
"""


def is_referential_query(query: str) -> bool:
    """Determine if a follow-up query contains pronouns or referential cues.

    Args:
        query: Current user query string.

    Returns:
        True if the query appears referential; False otherwise.
    """
    clean_query = query.strip()
    if not clean_query:
        return False

    # Ignore standalone uppercase 'IT' (Information Technology) to avoid false positives on workplace queries
    text_for_check = re.sub(r"\bIT\b", "dept_it", clean_query)

    # Check for pronoun or referential phrase patterns
    for pattern in REFERENTIAL_PATTERNS:
        if re.search(pattern, text_for_check):
            return True

    # Very short follow-up questions (<= 3 words, e.g. "And for engineering?", "How much?")
    words = clean_query.split()
    if len(words) <= 3 and clean_query.endswith("?"):
        if words[0].lower() in ("and", "what", "how", "why", "where", "who", "which", "can", "does", "is"):
            return True

    return False


def build_query_rewrite_prompt(current_query: str, history: Sequence[ConversationTurn]) -> str:
    """Format recent turns and the current query into a rewriting prompt for Gemini.

    Args:
        current_query: Latest user follow-up question.
        history: Sequence of previous ConversationTurn records (last 2-3 turns).

    Returns:
        Formatted prompt string for the LLM.
    """
    history_blocks = []
    for turn in history:
        history_blocks.append(
            f"Turn {turn.turn_number}:\n"
            f"User: {turn.question.strip()}\n"
            f"Assistant: {turn.answer.strip()[:300]}"
        )

    history_str = "\n\n".join(history_blocks)

    return (
        f"Conversation History (Last {len(history)} turns):\n"
        f"{history_str}\n\n"
        f"Latest Follow-up Question:\n"
        f"{current_query.strip()}\n\n"
        f"Rewritten Standalone Search Query:"
    )


async def rewrite_query(
    current_query: str,
    history: Sequence[ConversationTurn],
    generation_service: GenerationService,
) -> str:
    """Rewrite a referential follow-up query into a standalone query using Groq.

    Args:
        current_query: Latest user query.
        history: Recent conversation history turns (up to 3).
        generation_service: Initialized GenerationService instance.

    Returns:
        Standalone search query string.
    """
    if not history or not is_referential_query(current_query):
        return current_query.strip()

    prompt = build_query_rewrite_prompt(current_query, history)

    try:
        import asyncio
        from app.services.generation import extract_chunk_text

        client = generation_service.client
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            create_res = client.chat.completions.create(
                model=generation_service.model_name,
                messages=[
                    {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                stream=True,
            )
        elif hasattr(client, "models") and hasattr(client.models, "generate_content_stream"):
            create_res = client.models.generate_content_stream(
                model=generation_service.model_name,
                contents=prompt,
            )
        else:
            create_res = client.generate_content_stream(prompt)

        response_stream = await create_res if asyncio.iscoroutine(create_res) else create_res

        tokens = []
        if hasattr(response_stream, "__aiter__"):
            async for chunk in response_stream:
                token_text = extract_chunk_text(chunk)
                if token_text:
                    tokens.append(token_text)
        else:
            for chunk in response_stream:
                token_text = extract_chunk_text(chunk)
                if token_text:
                    tokens.append(token_text)

        rewritten = "".join(tokens).strip()

        # Clean any wrapping quotes
        if (rewritten.startswith('"') and rewritten.endswith('"')) or (
            rewritten.startswith("'") and rewritten.endswith("'")
        ):
            rewritten = rewritten[1:-1].strip()

        if rewritten:
            logger.info(
                "Query rewritten by Groq: original='%s' -> resolved='%s'",
                current_query,
                rewritten,
            )
            return rewritten

    except Exception as exc:
        logger.warning(
            "Failed to rewrite query via Groq: %s. Falling back to original query.",
            exc,
        )

    return current_query.strip()



async def get_conversation_history(
    db: AsyncSession,
    conversation_id: uuid.UUID,
    limit: int = 3,
) -> List[ConversationTurn]:
    """Retrieve the most recent conversation turns for a conversation_id in chronological order.

    Args:
        db: AsyncSession database connection.
        conversation_id: Unique conversation UUID.
        limit: Number of recent turns to fetch (default: 3).

    Returns:
        List of ConversationTurn objects ordered chronologically.
    """
    try:
        stmt = (
            select(ConversationTurn)
            .where(ConversationTurn.conversation_id == conversation_id)
            .order_by(ConversationTurn.turn_number.desc())
            .limit(limit)
        )
        result = await db.execute(stmt)
        if hasattr(result, "scalars"):
            scalar_res = result.scalars()
            if hasattr(scalar_res, "all"):
                turns = list(scalar_res.all())
            else:
                turns = []
        else:
            turns = []
        # Reverse to return chronological order: [turn 1, turn 2, turn 3]
        turns.reverse()
        return turns
    except Exception as exc:
        logger.warning("Could not retrieve conversation history for %s: %s", conversation_id, exc)
        return []


import inspect

async def save_conversation_turn(
    db: AsyncSession,
    conversation_id: uuid.UUID,
    turn_number: int,
    question: str,
    resolved_query: str,
    answer: str,
    retrieved_chunk_ids: List[str],
    user_id: Optional[uuid.UUID] = None,
) -> ConversationTurn:
    """Persist a conversation turn record in PostgreSQL session memory.

    Args:
        db: AsyncSession database connection.
        conversation_id: Unique conversation identifier.
        turn_number: Turn index (1, 2, ...).
        question: Original user question.
        resolved_query: Standalone query used for retrieval.
        answer: Generated assistant answer.
        retrieved_chunk_ids: List of chunk ID strings used in generation.
        user_id: Optional user UUID who owns this conversation turn.

    Returns:
        Persisted ConversationTurn instance.
    """
    turn = ConversationTurn(
        id=uuid.uuid4(),
        conversation_id=conversation_id,
        turn_number=turn_number,
        question=question,
        resolved_query=resolved_query,
        answer=answer,
        retrieved_chunk_ids=retrieved_chunk_ids,
        user_id=user_id,
    )
    try:
        add_res = db.add(turn)
        if inspect.isawaitable(add_res):
            await add_res

        commit_res = db.commit()
        if inspect.isawaitable(commit_res):
            await commit_res

        if hasattr(db, "refresh"):
            refresh_res = db.refresh(turn)
            if inspect.isawaitable(refresh_res):
                await refresh_res
    except Exception as exc:
        if hasattr(db, "rollback"):
            rb_res = db.rollback()
            if inspect.isawaitable(rb_res):
                await rb_res
        logger.warning("Could not persist conversation turn: %s", exc)

    logger.info(
        "Saved conversation turn: conv_id=%s, turn=%d, q='%s', resolved='%s'",
        conversation_id,
        turn_number,
        question,
        resolved_query,
    )
    return turn
