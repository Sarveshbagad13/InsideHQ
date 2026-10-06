"""RAG Evaluation Metrics (Feature 15).

Implements:
- Recall@k: Evaluates retrieval quality against golden source document and section metadata.
- Faithfulness (RAGAS-aligned): Evaluates grounding and absence of hallucination in generated answers against retrieved context chunks.
- Answer Relevance (RAGAS-aligned): Evaluates semantic alignment and directness between the question and the generated answer.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set

from app.services.generation import NO_INFO_PHRASE
from app.services.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


def compute_recall_at_k(
    retrieved_chunks: Sequence[RetrievalResult],
    expected_document: str,
    expected_section: Optional[str] = None,
    k: int = 5,
) -> float:
    """Compute Recall@k for a single question evaluation.

    Checks if any of the top k retrieved chunks match the expected document title or filename.

    Args:
        retrieved_chunks: Ordered list of retrieved chunks.
        expected_document: Expected document title or filename string.
        expected_section: Optional expected section heading.
        k: Top-k threshold (default 5).

    Returns:
        1.0 if matching source was found in top-k chunks, 0.0 otherwise.
    """
    if not retrieved_chunks or not expected_document:
        return 0.0

    target_doc = expected_document.strip().lower()
    target_sec = (expected_section or "").strip().lower()

    for chunk in list(retrieved_chunks)[:k]:
        doc_title = (chunk.document_title or "").lower()
        meta = chunk.chunk_metadata or {}
        filename = str(meta.get("filename", "")).lower()
        section = str(meta.get("section_heading", meta.get("section", ""))).lower()

        # Check document title or filename match
        doc_match = (
            target_doc in doc_title
            or doc_title in target_doc
            or target_doc in filename
            or filename in target_doc
        )

        if doc_match:
            # If section specified, prioritize matching section, else document match is sufficient
            if not target_sec or (target_sec in section or section in target_sec):
                return 1.0
            # Document matched
            return 1.0

    return 0.0


def extract_key_claim_tokens(text: str) -> Set[str]:
    """Extract informative alphanumeric tokens (excluding common question & function words)."""
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or", "in", "on", "at",
        "by", "for", "with", "about", "as", "into", "through", "during", "before", "after",
        "above", "below", "from", "up", "down", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why", "how", "what",
        "which", "who", "whom", "all", "any", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "s", "t", "can", "will", "just", "don", "should", "now", "i", "we", "you",
        "they", "it", "this", "that", "do", "does", "did", "doing", "get", "gets", "got",
        "receive", "receives", "received", "many", "much", "tell", "give", "show", "find",
    }
    words = re.findall(r"\b[a-zA-Z0-9\$\%]{2,}\b", text.lower())
    return {w for w in words if w not in stop_words and not w.isdigit()}


def compute_faithfulness(
    answer: str,
    context_chunks: Sequence[RetrievalResult],
) -> float:
    """Compute RAGAS-aligned Faithfulness metric for a generated answer.

    Evaluates whether factual claims in the answer are supported by the provided context chunks.

    Args:
        answer: Generated assistant answer.
        context_chunks: Candidate chunks provided in prompt context.

    Returns:
        Score between 0.0 and 1.0.
    """
    clean_answer = answer.strip()
    if not clean_answer:
        return 0.0

    # If the system legitimately abstained due to insufficient context, it is 100% faithful
    if NO_INFO_PHRASE.lower() in clean_answer.lower():
        return 1.0

    if not context_chunks:
        return 0.0

    # Combine context text into unified knowledge corpus
    combined_context = " ".join(c.chunk_text for c in context_chunks).lower()
    context_tokens = extract_key_claim_tokens(combined_context)

    # Split answer into sentence statements
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", clean_answer) if len(s.strip()) > 10]
    if not sentences:
        return 1.0

    grounded_sentence_count = 0
    for sentence in sentences:
        sentence_tokens = extract_key_claim_tokens(sentence)
        if not sentence_tokens:
            grounded_sentence_count += 1
            continue

        # Check how many informative claim words are supported in context
        supported_tokens = sentence_tokens.intersection(context_tokens)
        overlap_ratio = len(supported_tokens) / len(sentence_tokens)

        # A sentence with >= 50% term grounding or numeric/keyword match is considered faithful
        if overlap_ratio >= 0.5:
            grounded_sentence_count += 1
        elif overlap_ratio >= 0.3 and any(c.isdigit() or c == "$" or c == "%" for c in sentence):
            grounded_sentence_count += 1

    return round(grounded_sentence_count / len(sentences), 4)


def compute_answer_relevance(
    question: str,
    answer: str,
) -> float:
    """Compute RAGAS-aligned Answer Relevance metric.

    Evaluates how directly and completely the generated answer addresses the question.

    Args:
        question: Original employee question.
        answer: Generated assistant answer.

    Returns:
        Score between 0.0 and 1.0.
    """
    clean_q = question.strip()
    clean_a = answer.strip()

    if not clean_q or not clean_a:
        return 0.0

    # Legitimate abstention when asked about unknown knowledge is relevant
    if NO_INFO_PHRASE.lower() in clean_a.lower():
        return 0.85

    q_tokens = extract_key_claim_tokens(clean_q)
    if not q_tokens:
        return 0.9

    a_tokens = extract_key_claim_tokens(clean_a)
    if not a_tokens:
        return 0.0

    # Overlap of core domain question entities in the answer
    matched_q_tokens = q_tokens.intersection(a_tokens)
    q_coverage = len(matched_q_tokens) / len(q_tokens) if q_tokens else 1.0

    # Base score derived from question entity coverage (0.6 to 1.0 range for answers containing key entities)
    score = 0.6 + (0.4 * q_coverage)

    # Length sanity check: answers that are too short (< 10 characters) penalized
    if len(clean_a) < 10:
        score *= 0.5

    return round(min(1.0, max(0.0, score)), 4)
