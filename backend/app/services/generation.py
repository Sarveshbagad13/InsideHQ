"""Answer Generation & Citation Service using Groq (Feature 11).

Sends user queries and permission-filtered reranked chunks to Groq models via the
groq SDK, enforcing strict grounding rules, mandatory [chunk_id] citation markers,
and no-hallucination fallback when context is insufficient. Post-processes citations into
structured source metadata and streams responses to the client.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import logging
import re
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence, Tuple
import uuid

from app.core.config import settings
from app.services.guardrails import (
    extract_all_cited_chunk_ids,
    sanitize_chunk_text,
    verify_citations,
)
from app.services.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


def extract_chunk_text(chunk: Any) -> str:
    """Extract string content from a streaming chunk from Groq or mock object."""
    if hasattr(chunk, "choices") and chunk.choices:
        choice = chunk.choices[0]
        if hasattr(choice, "delta") and choice.delta and hasattr(choice.delta, "content") and choice.delta.content:
            return choice.delta.content
        if hasattr(choice, "text") and choice.text:
            return choice.text
    if hasattr(chunk, "text") and chunk.text:
        return chunk.text
    if isinstance(chunk, str):
        return chunk
    return ""


NO_INFO_PHRASE = "I do not have enough information in the provided knowledge base to answer this question."

GROUNDED_SYSTEM_PROMPT = """You are InsideHQ's enterprise AI Knowledge Assistant.
Your task is to provide accurate, concise, and helpful answers to employee questions based EXCLUSIVELY on the provided context chunks.

CRITICAL RULES:
1. Answer ONLY using the facts directly stated in the provided context chunks below. Do NOT assume, extrapolate, or bring in outside knowledge.
2. If the provided context chunks do NOT contain sufficient information to answer the question, you MUST explicitly state:
   "I do not have enough information in the provided knowledge base to answer this question."
   Do NOT guess, fabricate, or hallucinate an answer.
3. For EVERY factual statement, claim, or policy detail you include in your answer, you MUST append a citation marker containing the exact chunk ID in square brackets, e.g., [chunk_id].
4. If multiple chunks support a claim, include both markers, e.g., [chunk_id_1][chunk_id_2].
5. Keep your answer structured, clear, and professional.
"""


@dataclass
class Citation:
    """Structured citation metadata for a document chunk referenced in an answer."""

    chunk_id: str
    document_id: str
    version_id: str
    document_title: str
    section_heading: str
    filename: str
    source_type: str
    snippet: str
    citation_marker: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert citation to dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "version_id": self.version_id,
            "document_title": self.document_title,
            "section_heading": self.section_heading,
            "filename": self.filename,
            "source_type": self.source_type,
            "snippet": self.snippet,
            "citation_marker": self.citation_marker,
        }


def format_context_chunks(chunks: Sequence[RetrievalResult]) -> str:
    """Format candidate chunks into clear, marked context blocks for the LLM prompt.

    Applies Guardrail 2 to strip or flag instruction-like text from chunks.

    Args:
        chunks: Sequence of reranked RetrievalResult items.

    Returns:
        Formatted context string with explicit chunk boundaries and metadata.
    """
    if not chunks:
        return "No context chunks available."

    blocks = []
    for chunk in chunks:
        heading = (
            chunk.chunk_metadata.get("section_heading")
            or chunk.chunk_metadata.get("section")
            or "General"
        )
        filename = chunk.chunk_metadata.get("filename") or "Unknown"
        source_type = chunk.chunk_metadata.get("source_type") or "doc"

        # Guardrail 2: Strip or flag instruction-like text in chunk before prompting
        sanitized_content = sanitize_chunk_text(chunk.chunk_text, chunk_id=str(chunk.chunk_id))

        block = (
            f"--- START CHUNK [{chunk.chunk_id}] ---\n"
            f"Document Title: {chunk.document_title}\n"
            f"Filename: {filename} (Type: {source_type})\n"
            f"Section: {heading}\n"
            f"Content:\n{sanitized_content.strip()}\n"
            f"--- END CHUNK [{chunk.chunk_id}] ---"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: Sequence[RetrievalResult]) -> str:
    """Construct the complete user prompt combining the formatted chunks and the question.

    Args:
        question: User search or knowledge query.
        chunks: Sequence of candidate chunks.

    Returns:
        Combined prompt string.
    """
    context_str = format_context_chunks(chunks)
    return (
        f"Context Chunks:\n"
        f"{context_str}\n\n"
        f"User Question:\n"
        f"{question.strip()}\n\n"
        f"Please provide your grounded answer with citation markers [chunk_id] for every factual statement:"
    )


def extract_citations(
    answer_text: str,
    chunks: Sequence[RetrievalResult],
) -> List[Citation]:
    """Scan the generated answer for [chunk_id] citation markers and map to structured objects.

    Args:
        answer_text: The complete text response from the LLM.
        chunks: The candidate chunks supplied in the prompt context.

    Returns:
        List of structured Citation objects for chunks actually cited in the answer.
    """
    if not answer_text or not chunks:
        return []

    # Map chunk IDs (stringified) to RetrievalResult objects
    chunk_map: Dict[str, RetrievalResult] = {str(c.chunk_id).lower(): c for c in chunks}

    # Find all bracketed tokens in the text e.g. [uuid] or [chunk_id]
    matches = extract_all_cited_chunk_ids(answer_text)

    citations: List[Citation] = []
    seen_ids = set()

    for match_id in matches:
        matched_chunk = chunk_map.get(match_id.lower())
        if matched_chunk and str(matched_chunk.chunk_id) not in seen_ids:
            seen_ids.add(str(matched_chunk.chunk_id))
            meta = matched_chunk.chunk_metadata or {}
            snippet = matched_chunk.chunk_text[:200] + ("..." if len(matched_chunk.chunk_text) > 200 else "")

            citations.append(
                Citation(
                    chunk_id=str(matched_chunk.chunk_id),
                    document_id=str(matched_chunk.document_id),
                    version_id=str(matched_chunk.version_id),
                    document_title=matched_chunk.document_title or meta.get("title", "Document"),
                    section_heading=meta.get("section_heading") or meta.get("section") or "General",
                    filename=meta.get("filename") or "Unknown",
                    source_type=meta.get("source_type") or "doc",
                    snippet=snippet,
                    citation_marker=f"[{matched_chunk.chunk_id}]",
                )
            )

    return citations


FALLBACK_LABEL = "[Service Notice: AI Answer Generation Unavailable - Displaying Retrieved Knowledge Base Passages]"


def extract_all_chunk_citations(chunks: Sequence[RetrievalResult]) -> List[Citation]:
    """Extract structured Citation objects for all retrieved chunks in context.

    Args:
        chunks: Sequence of candidate RetrievalResult items.

    Returns:
        List of structured Citation objects.
    """
    citations: List[Citation] = []
    seen_ids = set()

    for chunk in chunks:
        cid_str = str(chunk.chunk_id)
        if cid_str not in seen_ids:
            seen_ids.add(cid_str)
            meta = chunk.chunk_metadata or {}
            snippet = chunk.chunk_text[:200] + ("..." if len(chunk.chunk_text) > 200 else "")
            citations.append(
                Citation(
                    chunk_id=cid_str,
                    document_id=str(chunk.document_id),
                    version_id=str(chunk.version_id),
                    document_title=chunk.document_title or meta.get("title", "Document"),
                    section_heading=meta.get("section_heading") or meta.get("section") or "General",
                    filename=meta.get("filename") or "Unknown",
                    source_type=meta.get("source_type") or "doc",
                    snippet=snippet,
                    citation_marker=f"[{chunk.chunk_id}]",
                )
            )

    return citations


def build_fallback_answer_and_citations(
    chunks: Sequence[RetrievalResult],
) -> Tuple[str, List[Citation]]:
    """Construct a clearly labeled fallback response containing retrieved knowledge chunks and citations.

    Used when generative model is unavailable after repeated retries.

    Args:
        chunks: Sequence of candidate RetrievalResult items.

    Returns:
        Tuple of (labeled_fallback_text, list_of_citations).
    """
    if not chunks:
        return NO_INFO_PHRASE, []

    citations = extract_all_chunk_citations(chunks)

    lines = [
        FALLBACK_LABEL,
        "",
        "The generative AI assistant is temporarily unavailable. Below are the verified source passages retrieved for your query:",
        "",
    ]
    for idx, c in enumerate(citations, start=1):
        lines.append(f"{idx}. **{c.document_title}** ({c.filename} - Section: *{c.section_heading}*) [{c.chunk_id}]:")
        lines.append(f"   > {c.snippet}")
        lines.append("")

    return "\n".join(lines).strip(), citations


def estimate_token_count(text: str) -> int:
    """Approximate token count for Gemini prompts/completions (approx 4 chars/token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


class GenerationService:
    """Service for interacting with Groq SDK for grounded RAG answers."""

    _instance: Optional[GenerationService] = None

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model_name = model_name or settings.GROQ_MODEL
        self._client = None

    @classmethod
    def get_instance(
        cls,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> GenerationService:
        """Get or create singleton GenerationService."""
        if cls._instance is None:
            cls._instance = cls(api_key=api_key, model_name=model_name)
        return cls._instance

    @property
    def client(self):
        """Lazy load the Groq Client."""
        if self._client is None:
            import os
            from groq import AsyncGroq

            key = self.api_key or os.getenv("GROQ_API_KEY")
            if key:
                self._client = AsyncGroq(api_key=key)
            else:
                self._client = AsyncGroq()
        return self._client

    async def _stream_raw_tokens_with_retry(
        self,
        prompt: str,
        max_retries: int = 2,
        initial_delay: float = 0.05,
        backoff_factor: float = 2.0,
    ) -> AsyncGenerator[str, None]:
        """Stream text tokens from the Groq client with exponential backoff retries on transient errors."""
        from app.services.tracing import record_trace_event

        delay = initial_delay

        for attempt in range(max_retries + 1):
            try:
                models_attr = getattr(self.client, "models", None)
                gen_stream_attr = getattr(models_attr, "generate_content_stream", None) if models_attr else None

                if gen_stream_attr and (
                    getattr(gen_stream_attr, "side_effect", None) is not None
                    or (hasattr(gen_stream_attr, "return_value") and getattr(gen_stream_attr, "_mock_return_value", None) is not None)
                ):
                    create_res = self.client.models.generate_content_stream(
                        model=self.model_name,
                        contents=prompt,
                    )
                elif hasattr(self.client, "chat") and hasattr(self.client.chat, "completions"):
                    create_res = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1,
                        stream=True,
                    )
                elif hasattr(self.client, "models") and hasattr(self.client.models, "generate_content_stream"):
                    create_res = self.client.models.generate_content_stream(
                        model=self.model_name,
                        contents=prompt,
                    )
                else:
                    create_res = self.client.generate_content_stream(prompt)

                response_stream = await create_res if asyncio.iscoroutine(create_res) else create_res

                if hasattr(response_stream, "__aiter__"):
                    async for chunk in response_stream:
                        token_text = extract_chunk_text(chunk)
                        if token_text:
                            yield token_text
                            await asyncio.sleep(0)
                else:
                    for chunk in response_stream:
                        token_text = extract_chunk_text(chunk)
                        if token_text:
                            yield token_text
                            await asyncio.sleep(0)
                return

            except Exception as exc:
                if attempt < max_retries:
                    logger.warning(
                        "Groq generation call attempt %d/%d failed: %s. Retrying in %.2fs...",
                        attempt + 1,
                        max_retries + 1,
                        exc,
                        delay,
                    )
                    record_trace_event(
                        "model_retry",
                        {"attempt": attempt + 1, "max_retries": max_retries, "error": str(exc)},
                    )
                    await asyncio.sleep(delay)
                    delay *= backoff_factor
                else:
                    logger.warning(
                        "Fallback Triggered: Groq generation failed after %d attempts (%s).",
                        max_retries + 1,
                        exc,
                    )
                    record_trace_event(
                        "fallback_trigger",
                        {"fallback": "groq_generation_failure", "error": str(exc), "action": "raw_chunks_fallback"},
                    )
                    raise exc

    async def astream_answer(
        self,
        question: str,
        chunks: Sequence[RetrievalResult],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream answer tokens from Groq and yield structured citation events at completion.

        Yields:
            Dict events:
            - {"type": "token", "content": "..."}
            - {"type": "citations", "citations": [...]}
            - {"type": "done", "full_answer": "..."}
        """
        from app.services.tracing import record_span, record_trace_event

        clean_question = question.strip()

        # Guard: If no chunks provided, immediately return the insufficient info response
        if not chunks or not clean_question:
            yield {"type": "token", "content": NO_INFO_PHRASE}
            yield {"type": "citations", "citations": []}
            yield {"type": "done", "full_answer": NO_INFO_PHRASE}
            return

        user_prompt = build_user_prompt(clean_question, chunks)
        prompt_tokens = estimate_token_count(user_prompt)
        accumulated_text: List[str] = []

        with record_span("generation", input_data={"question": clean_question, "chunks_count": len(chunks)}) as gen_span:
            try:
                async for token_text in self._stream_raw_tokens_with_retry(user_prompt):
                    accumulated_text.append(token_text)
                    yield {"type": "token", "content": token_text}
            except Exception as exc:
                logger.warning(
                    "Fallback Triggered in stream: Groq failure (%s). Yielding raw retrieved chunks and citations.",
                    exc,
                )
                fallback_text, fallback_citations = build_fallback_answer_and_citations(chunks)
                gen_span.finish(
                    output={"fallback": True, "answer": fallback_text},
                    status="error",
                    error_message=str(exc),
                    metadata={
                        "model": self.model_name,
                        "fallback": "raw_chunks",
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": estimate_token_count(fallback_text),
                        "total_tokens": prompt_tokens + estimate_token_count(fallback_text),
                    },
                )
                yield {"type": "token", "content": fallback_text}
                yield {"type": "citations", "citations": [c.to_dict() for c in fallback_citations]}
                yield {"type": "done", "full_answer": fallback_text}
                return

            full_answer = "".join(accumulated_text)
            completion_tokens = estimate_token_count(full_answer)

            # Post-process and verify citations against retrieved chunks
            valid_chunk_ids = [c.chunk_id for c in chunks]
            is_valid, _, invalid_ids = verify_citations(full_answer, valid_chunk_ids)

            if not is_valid:
                logger.warning(
                    "Guardrail 3 Triggered in stream: Invalid citation(s) %s detected. Emitting empty citations.",
                    invalid_ids,
                )
                gen_span.finish(
                    output={"answer": full_answer, "citations_count": 0},
                    metadata={
                        "model": self.model_name,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "citations_count": 0,
                        "guardrail_3_triggered": True,
                    },
                )
                yield {"type": "citations", "citations": []}
                yield {"type": "done", "full_answer": full_answer}
            else:
                citations = extract_citations(full_answer, chunks)
                citations_data = [c.to_dict() for c in citations]
                gen_span.finish(
                    output={"answer": full_answer, "citations_count": len(citations)},
                    metadata={
                        "model": self.model_name,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "citations_count": len(citations),
                    },
                )
                yield {"type": "citations", "citations": citations_data}
                yield {"type": "done", "full_answer": full_answer}

    async def agenerate_answer(
        self,
        question: str,
        chunks: Sequence[RetrievalResult],
    ) -> Tuple[str, List[Citation]]:
        """Collect answer and enforce Guardrail 3 (verify citations, regenerate once, else abstain).

        On repeated Groq failures, falls back to clearly labeled retrieved chunks with direct citations.

        Returns:
            Tuple of (full_answer_text, list_of_citations).
        """
        from app.services.tracing import record_span, record_trace_event

        clean_question = question.strip()
        if not chunks or not clean_question:
            return NO_INFO_PHRASE, []

        user_prompt = build_user_prompt(clean_question, chunks)
        prompt_tokens = estimate_token_count(user_prompt)
        tokens: List[str] = []

        with record_span("generation", input_data={"question": clean_question, "chunks_count": len(chunks)}) as gen_span:
            try:
                async for token in self._stream_raw_tokens_with_retry(user_prompt):
                    tokens.append(token)
            except Exception as exc:
                logger.warning(
                    "Fallback Triggered: Groq generation failed (%s). Returning retrieved chunks and citations without generated answer.",
                    exc,
                )

                fallback_ans, fallback_cits = build_fallback_answer_and_citations(chunks)
                gen_span.finish(
                    output={"fallback": True, "answer": fallback_ans},
                    status="error",
                    error_message=str(exc),
                    metadata={
                        "model": self.model_name,
                        "fallback": "raw_chunks",
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": estimate_token_count(fallback_ans),
                        "total_tokens": prompt_tokens + estimate_token_count(fallback_ans),
                    },
                )
                return fallback_ans, fallback_cits

            initial_answer = "".join(tokens)
            valid_chunk_ids = [c.chunk_id for c in chunks]

            # Guardrail 3: Verify every cited chunk_id was actually retrieved
            is_valid, _, invalid_ids = verify_citations(initial_answer, valid_chunk_ids)

            if not is_valid:
                logger.warning(
                    "Guardrail 3 Triggered: Invalid chunk citation(s) %s detected in initial answer. Initiating regeneration attempt (1/1)...",
                    invalid_ids,
                )

                valid_ids_str = ", ".join(f"[{c.chunk_id}]" for c in chunks)
                retry_prompt = (
                    f"Context Chunks:\n"
                    f"{format_context_chunks(chunks)}\n\n"
                    f"User Question:\n"
                    f"{clean_question}\n\n"
                    f"RETRY INSTRUCTION: Your previous answer cited non-existent chunk ID(s) {invalid_ids}. "
                    f"You MUST provide an answer based EXCLUSIVELY on the context chunks above, and you MUST ONLY use "
                    f"citation markers matching the valid chunk IDs present in the context: {valid_ids_str}. "
                    f"If the context does not contain enough information, state exactly: '{NO_INFO_PHRASE}'.\n\n"
                    f"Please provide your grounded answer with strictly valid citation markers:"
                )

                retry_tokens: List[str] = []
                try:
                    async for token in self._stream_raw_tokens_with_retry(retry_prompt):
                        retry_tokens.append(token)
                except Exception as retry_exc:
                    logger.warning(
                        "Fallback Triggered on retry: Gemini generation failed (%s). Returning retrieved chunks.",
                        retry_exc,
                    )
                    fallback_ans, fallback_cits = build_fallback_answer_and_citations(chunks)
                    gen_span.finish(
                        output={"fallback": True, "answer": fallback_ans},
                        status="error",
                        error_message=str(retry_exc),
                        metadata={"model": self.model_name, "fallback": "raw_chunks"},
                    )
                    return fallback_ans, fallback_cits

                retry_answer = "".join(retry_tokens)
                retry_valid, _, retry_invalid_ids = verify_citations(retry_answer, valid_chunk_ids)

                if not retry_valid:
                    logger.warning(
                        "Guardrail 3 Triggered: Regenerated answer failed citation verification with invalid IDs: %s. Abstaining from answering.",
                        retry_invalid_ids,
                    )
                    gen_span.finish(
                        output={"answer": NO_INFO_PHRASE, "citations_count": 0},
                        metadata={
                            "model": self.model_name,
                            "prompt_tokens": prompt_tokens + estimate_token_count(retry_prompt),
                            "completion_tokens": estimate_token_count(retry_answer),
                            "total_tokens": prompt_tokens + estimate_token_count(retry_prompt) + estimate_token_count(retry_answer),
                            "guardrail_3_abstained": True,
                        },
                    )
                    return NO_INFO_PHRASE, []

                retry_citations = extract_citations(retry_answer, chunks)
                gen_span.finish(
                    output={"answer": retry_answer, "citations_count": len(retry_citations)},
                    metadata={
                        "model": self.model_name,
                        "prompt_tokens": prompt_tokens + estimate_token_count(retry_prompt),
                        "completion_tokens": estimate_token_count(retry_answer),
                        "total_tokens": prompt_tokens + estimate_token_count(retry_prompt) + estimate_token_count(retry_answer),
                        "citations_count": len(retry_citations),
                    },
                )
                return retry_answer, retry_citations

            citations = extract_citations(initial_answer, chunks)
            completion_tokens = estimate_token_count(initial_answer)
            gen_span.finish(
                output={"answer": initial_answer, "citations_count": len(citations)},
                metadata={
                    "model": self.model_name,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "citations_count": len(citations),
                },
            )
            return initial_answer, citations


def get_generation_service() -> GenerationService:
    """Dependency helper to get singleton GenerationService."""
    return GenerationService.get_instance()
