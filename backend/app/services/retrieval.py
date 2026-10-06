"""Hybrid Retrieval Service with Permission-Aware Filtering (Features 8 & 9).

Combines vector similarity search (pgvector cosine distance) and keyword search
(PostgreSQL tsvector / full-text search) over approved document chunks, strictly filtered
by the current user's role, department, and user ID via SQL document_acl joins before
any chunk reaches memory or the LLM, merging results using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional
import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentChunk
from app.models.user import User
from app.services.embeddings import EmbeddingService, get_embedding_service
from app.services.permissions import build_permission_filter

# Sentinel for parse errors in date_range filter
_INVALID_DATE = None

logger = logging.getLogger(__name__)

# Default constant k for Reciprocal Rank Fusion (standard value 60)
DEFAULT_RRF_K = 60


@dataclass
class RetrievalResult:
    """Represents a single retrieved document chunk with similarity scores and ranking metadata."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    version_id: uuid.UUID
    chunk_text: str
    chunk_metadata: Dict[str, Any] = field(default_factory=dict)
    document_title: str = ""
    rrf_score: float = 0.0
    vector_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    vector_distance: Optional[float] = None
    keyword_score: Optional[float] = None
    rerank_score: Optional[float] = None
    rerank_rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert retrieval result to dictionary representation."""
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "version_id": str(self.version_id),
            "chunk_text": self.chunk_text,
            "chunk_metadata": self.chunk_metadata,
            "document_title": self.document_title,
            "rrf_score": self.rrf_score,
            "vector_rank": self.vector_rank,
            "keyword_rank": self.keyword_rank,
            "vector_distance": self.vector_distance,
            "keyword_score": self.keyword_score,
            "rerank_score": self.rerank_score,
            "rerank_rank": self.rerank_rank,
        }



def compute_rrf_score(
    vector_rank: Optional[int],
    keyword_rank: Optional[int],
    rrf_k: int = DEFAULT_RRF_K,
) -> float:
    """Calculate the Reciprocal Rank Fusion (RRF) score for a single chunk.

    RRF_score = sum(1 / (k + rank_i)) for each retrieval method where the chunk appears.

    Args:
        vector_rank: 1-based rank in vector search results (or None if not in vector top-k).
        keyword_rank: 1-based rank in keyword search results (or None if not in keyword top-k).
        rrf_k: Constant smoothing parameter (default 60).

    Returns:
        The combined RRF score.
    """
    score = 0.0
    if vector_rank is not None and vector_rank > 0:
        score += 1.0 / (rrf_k + vector_rank)
    if keyword_rank is not None and keyword_rank > 0:
        score += 1.0 / (rrf_k + keyword_rank)
    return score


def reciprocal_rank_fusion(
    vector_results: List[RetrievalResult],
    keyword_results: List[RetrievalResult],
    rrf_k: int = DEFAULT_RRF_K,
    top_k: Optional[int] = None,
) -> List[RetrievalResult]:
    """Merge and re-rank vector search and keyword search result sets using RRF.

    Args:
        vector_results: Ranked list of chunks from vector cosine similarity search.
        keyword_results: Ranked list of chunks from full-text search.
        rrf_k: Constant smoothing parameter (default 60).
        top_k: Optional cutoff for maximum results to return.

    Returns:
        Combined list of RetrievalResult items sorted by rrf_score in descending order.
    """
    merged: Dict[uuid.UUID, RetrievalResult] = {}

    # Process vector results
    for rank, item in enumerate(vector_results, start=1):
        item.vector_rank = rank
        merged[item.chunk_id] = item

    # Process keyword results
    for rank, item in enumerate(keyword_results, start=1):
        item.keyword_rank = rank
        if item.chunk_id in merged:
            existing = merged[item.chunk_id]
            existing.keyword_rank = rank
            existing.keyword_score = item.keyword_score
        else:
            merged[item.chunk_id] = item

    # Calculate final RRF scores
    for item in merged.values():
        item.rrf_score = compute_rrf_score(
            vector_rank=item.vector_rank,
            keyword_rank=item.keyword_rank,
            rrf_k=rrf_k,
        )

    # Sort primarily by rrf_score DESC, with secondary tie-breakers
    ranked_results = sorted(
        merged.values(),
        key=lambda r: (
            r.rrf_score,
            -(r.vector_distance if r.vector_distance is not None else 999.0),
            r.keyword_score if r.keyword_score is not None else 0.0,
        ),
        reverse=True,
    )

    if top_k is not None and top_k > 0:
        return ranked_results[:top_k]
    return ranked_results


async def search_vector_candidates(
    db: AsyncSession,
    query_vector: List[float],
    user: User,
    limit: int = 20,
) -> List[RetrievalResult]:
    """Execute vector similarity search (cosine distance) against permitted approved document chunks.

    Enforces permission-aware filtering directly in the SQL query using document_acl.

    Args:
        db: Async database session.
        query_vector: 1024-dimensional normalized embedding vector.
        user: The current authenticated User.
        limit: Max candidate chunks to retrieve.

    Returns:
        Ranked list of RetrievalResult instances (1-based vector_rank populated).
    """
    if not query_vector:
        return []

    # Cosine distance operator <=> in pgvector
    dist_expr = DocumentChunk.embedding.cosine_distance(query_vector).label("distance")

    # Construct permission filter for the requesting user
    perm_filter = build_permission_filter(user=user, document_id_col=DocumentChunk.document_id)

    stmt = (
        select(DocumentChunk, Document.title, dist_expr)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            Document.status == "approved",
            DocumentChunk.embedding.isnot(None),
            perm_filter,
        )
    )

    stmt = stmt.order_by(dist_expr.asc()).limit(limit)

    result = await db.execute(stmt)
    rows = result.all()

    candidates: List[RetrievalResult] = []
    for rank, (chunk, doc_title, distance) in enumerate(rows, start=1):
        candidates.append(
            RetrievalResult(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                version_id=chunk.version_id,
                chunk_text=chunk.chunk_text,
                chunk_metadata=chunk.chunk_metadata or {},
                document_title=doc_title or "",
                vector_rank=rank,
                vector_distance=float(distance) if distance is not None else None,
            )
        )

    return candidates


async def search_keyword_candidates(
    db: AsyncSession,
    query: str,
    user: User,
    limit: int = 20,
) -> List[RetrievalResult]:
    """Execute PostgreSQL full-text search against content_tsvector of permitted approved chunks.

    Enforces permission-aware filtering directly in the SQL query using document_acl.
    Uses multi-stage fallback: websearch_to_tsquery -> plainto_tsquery -> OR keyword tsquery -> ILIKE term search.

    Args:
        db: Async database session.
        query: Query string.
        user: The current authenticated User.
        limit: Max candidate chunks to retrieve.

    Returns:
        Ranked list of RetrievalResult instances (1-based keyword_rank populated).
    """
    import re
    from sqlalchemy import or_, literal

    clean_query = query.strip()
    if not clean_query:
        return []

    # Construct permission filter for the requesting user
    perm_filter = build_permission_filter(user=user, document_id_col=DocumentChunk.document_id)

    rows = []

    # Stage 1: websearch_to_tsquery (exact phrase/multi-term match)
    try:
        tsquery = func.websearch_to_tsquery("english", clean_query)
        rank_expr = func.ts_rank_cd(DocumentChunk.content_tsvector, tsquery).label("rank_score")
        stmt = (
            select(DocumentChunk, Document.title, rank_expr)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                Document.status == "approved",
                DocumentChunk.content_tsvector.op("@@")(tsquery),
                perm_filter,
            )
            .order_by(rank_expr.desc())
            .limit(limit)
        )
        res = await db.execute(stmt)
        rows = res.all()
    except Exception as exc:
        logger.debug("Stage 1 websearch_to_tsquery exception: %s", exc)

    # Stage 2: plainto_tsquery (AND match without syntax constraints)
    if not rows:
        try:
            tsquery_plain = func.plainto_tsquery("english", clean_query)
            rank_plain = func.ts_rank_cd(DocumentChunk.content_tsvector, tsquery_plain).label("rank_score")
            stmt = (
                select(DocumentChunk, Document.title, rank_plain)
                .join(Document, Document.id == DocumentChunk.document_id)
                .where(
                    Document.status == "approved",
                    DocumentChunk.content_tsvector.op("@@")(tsquery_plain),
                    perm_filter,
                )
                .order_by(rank_plain.desc())
                .limit(limit)
            )
            res = await db.execute(stmt)
            rows = res.all()
        except Exception as exc:
            logger.debug("Stage 2 plainto_tsquery exception: %s", exc)

    # Stage 3: OR keyword tsquery (flexible OR matching across meaningful query terms)
    if not rows:
        stopwords = {
            'how', 'many', 'does', 'a', 'an', 'the', 'is', 'are', 'what', 'where',
            'who', 'when', 'which', 'to', 'for', 'in', 'on', 'of', 'it', 'do', 'can',
            'should', 'with', 'be', 'from', 'receive', 'get', 'give', 'tell', 'find'
        }
        words = [re.sub(r'[^a-zA-Z0-9]', '', w.lower()) for w in clean_query.split()]
        keywords = [w for w in words if w and w not in stopwords]
        if keywords:
            or_expr = ' | '.join(keywords)
            try:
                tsquery_or = func.to_tsquery('english', or_expr)
                rank_or = func.ts_rank_cd(DocumentChunk.content_tsvector, tsquery_or).label("rank_score")
                stmt = (
                    select(DocumentChunk, Document.title, rank_or)
                    .join(Document, Document.id == DocumentChunk.document_id)
                    .where(
                        Document.status == "approved",
                        DocumentChunk.content_tsvector.op("@@")(tsquery_or),
                        perm_filter,
                    )
                    .order_by(rank_or.desc())
                    .limit(limit)
                )
                res = await db.execute(stmt)
                rows = res.all()
            except Exception as exc:
                logger.debug("Stage 3 OR tsquery exception: %s", exc)

    # Stage 4: ILIKE term fallback
    if not rows:
        words = [re.sub(r'[^a-zA-Z0-9]', '', w) for w in clean_query.split()]
        keywords = [w for w in words if len(w) > 3 and w.lower() not in stopwords]
        if keywords:
            ilike_clauses = [DocumentChunk.chunk_text.ilike(f"%{kw}%") for kw in keywords]
            stmt = (
                select(DocumentChunk, Document.title, literal(0.1).label("rank_score"))
                .join(Document, Document.id == DocumentChunk.document_id)
                .where(
                    Document.status == "approved",
                    or_(*ilike_clauses),
                    perm_filter,
                )
                .limit(limit)
            )
            res = await db.execute(stmt)
            rows = res.all()

    candidates: List[RetrievalResult] = []
    for rank, (chunk, doc_title, rank_score) in enumerate(rows, start=1):
        candidates.append(
            RetrievalResult(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                version_id=chunk.version_id,
                chunk_text=chunk.chunk_text,
                chunk_metadata=chunk.chunk_metadata or {},
                document_title=doc_title or "",
                keyword_rank=rank,
                keyword_score=float(rank_score) if rank_score is not None else None,
            )
        )

    return candidates


async def hybrid_search(
    db: AsyncSession,
    query: str,
    user: User,
    top_k_candidates: int = 20,
    final_top_k: int = 20,
    rrf_k: int = DEFAULT_RRF_K,
    embedding_service: Optional[EmbeddingService] = None,
) -> List[RetrievalResult]:
    """Perform permission-aware hybrid retrieval combining vector search and keyword FTS.

    Safety-Critical Workflow:
        1. Embed the query string locally using EmbeddingService.
        2. Execute pgvector cosine similarity search with user ACL SQL filtering for top ~20 candidates.
        3. Execute PostgreSQL full-text search with user ACL SQL filtering for top ~20 candidates.
        4. Fuse both permitted ranked lists using Reciprocal Rank Fusion (RRF) into a single list.
        Restricted chunks are never retrieved from the database into memory.

    Args:
        db: Async database session.
        query: Search question / query string.
        user: The requesting authenticated User.
        top_k_candidates: Number of top candidates retrieved from each search modality (default 20).
        final_top_k: Number of fused results to return (default 20).
        rrf_k: RRF constant factor k (default 60).
        embedding_service: Optional embedding service instance (defaults to singleton).

    Returns:
        List of fused RetrievalResult objects ordered by RRF score descending.
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    from app.services.permissions import get_user_permission_tokens
    from app.services.tracing import record_span, record_trace_event

    if embedding_service is None:
        embedding_service = get_embedding_service()

    with record_span("retrieval", input_data={"query": clean_query}) as retr_span:
        with record_span("permission_filtering", input_data={"user_role": user.role, "user_department": user.department}) as perm_span:
            # 1. Embed query vector asynchronously (with fallback to keyword-only search on failure)
            query_vector = None
            try:
                query_vector = await embedding_service.aembed_query(clean_query)
            except Exception as exc:
                logger.warning(
                    "Fallback Triggered: Embedding service failed (%s). Falling back to PostgreSQL keyword-only full-text search.",
                    exc,
                )
                record_trace_event(
                    "fallback_trigger",
                    {"fallback": "embedding_failure", "error": str(exc), "action": "keyword_only_search"},
                )

            # If embedding generation failed, execute keyword-only search fallback
            if query_vector is None:
                keyword_results = await search_keyword_candidates(
                    db=db,
                    query=clean_query,
                    user=user,
                    limit=top_k_candidates,
                )
                for rank, chunk in enumerate(keyword_results, start=1):
                    chunk.keyword_rank = rank
                    chunk.rrf_score = compute_rrf_score(vector_rank=None, keyword_rank=rank, rrf_k=rrf_k)

                perm_span.finish(
                    output={"allowed_chunks_count": len(keyword_results)},
                    metadata={"permission_tokens": list(get_user_permission_tokens(user))},
                )
                retr_span.finish(
                    output={"retrieved_count": len(keyword_results)},
                    metadata={"mode": "keyword_only_fallback", "keyword_candidates_count": len(keyword_results)},
                )
                return keyword_results[:final_top_k]

            # 2 & 3. Run permission-filtered vector search and keyword search concurrently
            vector_task = search_vector_candidates(
                db=db,
                query_vector=query_vector,
                user=user,
                limit=top_k_candidates,
            )
            keyword_task = search_keyword_candidates(
                db=db,
                query=clean_query,
                user=user,
                limit=top_k_candidates,
            )

            try:
                vector_results, keyword_results = await asyncio.gather(vector_task, keyword_task)
            except Exception as exc:
                logger.warning(
                    "Fallback Triggered: Vector search failed during execution (%s). Falling back to keyword-only search.",
                    exc,
                )
                record_trace_event(
                    "fallback_trigger",
                    {"fallback": "vector_search_failure", "error": str(exc), "action": "keyword_only_search"},
                )
                keyword_results = await search_keyword_candidates(
                    db=db,
                    query=clean_query,
                    user=user,
                    limit=top_k_candidates,
                )
                for rank, chunk in enumerate(keyword_results, start=1):
                    chunk.keyword_rank = rank
                    chunk.rrf_score = compute_rrf_score(vector_rank=None, keyword_rank=rank, rrf_k=rrf_k)
                perm_span.finish(output={"allowed_chunks_count": len(keyword_results)})
                retr_span.finish(output={"retrieved_count": len(keyword_results)})
                return keyword_results[:final_top_k]

            logger.debug(
                "Permission-aware hybrid search (user=%s, role=%s) retrieved %d vector candidates and %d keyword candidates for query: '%s'",
                user.email,
                user.role,
                len(vector_results),
                len(keyword_results),
                clean_query,
            )

            # 4. Merge permitted candidates using Reciprocal Rank Fusion
            fused_results = reciprocal_rank_fusion(
                vector_results=vector_results,
                keyword_results=keyword_results,
                rrf_k=rrf_k,
                top_k=final_top_k,
            )

            perm_span.finish(
                output={"allowed_chunks_count": len(fused_results)},
                metadata={"permission_tokens": list(get_user_permission_tokens(user))},
            )
            retr_span.finish(
                output={"retrieved_count": len(fused_results)},
                metadata={
                    "vector_candidates_count": len(vector_results),
                    "keyword_candidates_count": len(keyword_results),
                    "fused_candidates_count": len(fused_results),
                },
            )

    return fused_results


async def retrieve_and_rerank(
    db: AsyncSession,
    query: str,
    user: User,
    top_k_candidates: int = 20,
    top_n_rerank: int = 5,
    rrf_k: int = DEFAULT_RRF_K,
    embedding_service: Optional[EmbeddingService] = None,
    reranker_service: Optional[Any] = None,
) -> List[RetrievalResult]:
    """Execute end-to-end permission-aware retrieval and cross-encoder reranking.

    Workflow:
        1. Executes permission-aware hybrid search (vector + keyword FTS with RRF fusion)
           to fetch top_k_candidates (default 20).
        2. Applies local cross-encoder model (BAAI/bge-reranker-large) to score each
           candidate against the query.
        3. Returns top_n_rerank (default 5) highest scoring chunks.
        4. On cross-encoder failure, gracefully falls back to candidate order from RRF.

    Args:
        db: Async database session.
        query: User search query.
        user: Authenticated requesting User.
        top_k_candidates: Number of hybrid search candidates to retrieve before reranking (default 20).
        top_n_rerank: Number of final reranked chunks to return (default 5).
        rrf_k: RRF constant factor k (default 60).
        embedding_service: Optional embedding service instance.
        reranker_service: Optional reranker service instance.

    Returns:
        Top 5 reranked RetrievalResult items.
    """
    from app.services.reranking import rerank_candidates
    from app.services.tracing import record_span, record_trace_event

    candidates = await hybrid_search(
        db=db,
        query=query,
        user=user,
        top_k_candidates=top_k_candidates,
        final_top_k=top_k_candidates,
        rrf_k=rrf_k,
        embedding_service=embedding_service,
    )

    if not candidates:
        return []

    with record_span("reranking", input_data={"candidates_count": len(candidates), "top_n": top_n_rerank}) as rerank_span:
        try:
            reranked = await rerank_candidates(
                query=query,
                candidates=candidates,
                top_n=top_n_rerank,
                reranker_service=reranker_service,
            )
            rerank_span.finish(
                output={"reranked_count": len(reranked)},
                metadata={
                    "rerank_scores": [
                        {"chunk_id": str(c.chunk_id), "score": c.rerank_score, "rank": c.rerank_rank}
                        for c in reranked
                    ]
                },
            )
            return reranked
        except Exception as exc:
            logger.warning(
                "Fallback Triggered: Cross-encoder reranker failed (%s). Falling back to Reciprocal Rank Fusion (RRF) candidate order.",
                exc,
            )
            record_trace_event(
                "fallback_trigger",
                {"fallback": "reranker_failure", "error": str(exc), "action": "rrf_order"},
            )
            fallback_chunks = candidates[:top_n_rerank]
            for rank, chunk in enumerate(fallback_chunks, start=1):
                chunk.rerank_rank = rank
            rerank_span.finish(
                output={"reranked_count": len(fallback_chunks)},
                metadata={"status": "fallback_to_rrf"},
            )
            return fallback_chunks

