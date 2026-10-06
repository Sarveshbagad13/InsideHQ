"""Business logic and external integration services package for InsideHQ."""

from app.services.chunking import TextChunk, chunk_document, chunk_text_blocks
from app.services.curation import (
    approve_document,
    deprecate_document,
    is_document_searchable,
    list_documents_by_status,
    list_pending_documents,
    reject_document,
)
from app.services.embeddings import EmbeddingService, get_embedding_service
from app.services.indexing import index_document_chunks, process_document_indexing
from app.services.ingestion import ingest_document, parse_file_to_blocks
from app.services.permissions import (
    build_permission_filter,
    get_user_permission_tokens,
    has_document_access,
)
from app.services.generation import (
    Citation,
    GenerationService,
    extract_citations,
    format_context_chunks,
    get_generation_service,
)
from app.services.reranking import (
    RerankerService,
    get_reranker_service,
    rerank_candidates,
)
from app.services.retrieval import (
    RetrievalResult,
    compute_rrf_score,
    hybrid_search,
    reciprocal_rank_fusion,
    retrieve_and_rerank,
    search_keyword_candidates,
    search_vector_candidates,
)

__all__ = [
    "EmbeddingService",
    "get_embedding_service",
    "RerankerService",
    "get_reranker_service",
    "rerank_candidates",
    "Citation",
    "GenerationService",
    "get_generation_service",
    "extract_citations",
    "format_context_chunks",
    "TextChunk",
    "chunk_document",
    "chunk_text_blocks",
    "index_document_chunks",
    "process_document_indexing",
    "ingest_document",
    "parse_file_to_blocks",
    "approve_document",
    "deprecate_document",
    "is_document_searchable",
    "list_documents_by_status",
    "list_pending_documents",
    "reject_document",
    "build_permission_filter",
    "get_user_permission_tokens",
    "has_document_access",
    "RetrievalResult",
    "compute_rrf_score",
    "reciprocal_rank_fusion",
    "search_vector_candidates",
    "search_keyword_candidates",
    "hybrid_search",
    "retrieve_and_rerank",
]




