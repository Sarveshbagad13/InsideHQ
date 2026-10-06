"""Chunking service: splits document text into token-bounded overlapping chunks.

Design decisions:
- Target chunk size: 500 tokens (cl100k_base encoding used by GPT-4/text-embedding-3).
- Overlap: 75 tokens (~15% of 500), pulled from the *end* of the preceding chunk.
- Boundary preference: always try to break on double-newline (paragraph) boundaries
  first, then single-newline, then mid-sentence as a last resort.
- Section-heading tracking: a paragraph that is short (<= HEADING_MAX_TOKENS tokens)
  and does not end with sentence-terminal punctuation is treated as a heading and
  carried as metadata into all subsequent chunks until the next heading appears.
- Metadata per chunk: document_id, version_id, position (0-based), section_heading
  (or None), source filename, and source_type.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import tiktoken


# ---------------------------------------------------------------------------
# Module-level encoder — cl100k_base is compatible with OpenAI text-embedding-3-*
# and Google text-embedding models generally use similar tokenisation.
# ---------------------------------------------------------------------------
_ENCODER = tiktoken.get_encoding("cl100k_base")

# ---------------------------------------------------------------------------
# Constants (tunable at call sites via keyword arguments)
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 500   # target tokens per chunk
DEFAULT_OVERLAP = 75       # overlap tokens between consecutive chunks (~15%)
HEADING_MAX_TOKENS = 25    # a block shorter than this with no terminal punct = heading
_TERMINAL_PUNCT = re.compile(r"[.!?…]\s*$")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TextChunk:
    """A single chunk ready for embedding.

    Attributes:
        text:            The chunk text content.
        token_count:     Precise token count (via tiktoken).
        position:        0-based order within the source document.
        section_heading: The most recent heading detected before this chunk, if any.
        metadata:        Arbitrary key-value pairs (document_id, source, …).
    """
    text: str
    token_count: int
    position: int
    section_heading: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count_tokens(text: str) -> int:
    """Return the number of cl100k_base tokens in *text*."""
    return len(_ENCODER.encode(text))


def _is_heading(text: str) -> bool:
    """Return True if *text* looks like a section heading.

    Heuristic: short (≤ HEADING_MAX_TOKENS) and does not end with
    sentence-terminal punctuation (.  !  ?  …).
    """
    stripped = text.strip()
    if not stripped:
        return False
    if _count_tokens(stripped) > HEADING_MAX_TOKENS:
        return False
    return not bool(_TERMINAL_PUNCT.search(stripped))


def _split_into_paragraphs(text: str) -> List[str]:
    """Split *text* at paragraph boundaries (blank-line separated blocks).

    Falls back to single-newline splits, then whole text if no newlines exist.
    """
    # Try double-newline paragraph splits first
    paras = re.split(r"\n\s*\n", text)
    result: List[str] = []
    for para in paras:
        # Within a paragraph, further split on single newlines to allow
        # finer-grained boundary selection without losing structure.
        lines = para.split("\n")
        current: List[str] = []
        for line in lines:
            stripped = line.rstrip()
            if stripped:
                current.append(stripped)
            else:
                if current:
                    result.append(" ".join(current))
                    current = []
        if current:
            result.append(" ".join(current))
    return [p for p in result if p.strip()]


def _split_sentence_boundary(text: str, max_tokens: int) -> List[str]:
    """Split a single paragraph that is too long to fit in *max_tokens*.

    Tries to cut at sentence boundaries (. ! ? followed by whitespace/end).
    """
    sentence_re = re.compile(r"(?<=[.!?…])\s+")
    sentences = sentence_re.split(text)
    groups: List[str] = []
    current_tokens = 0
    current_parts: List[str] = []

    for sentence in sentences:
        t = _count_tokens(sentence)
        if current_tokens + t > max_tokens and current_parts:
            groups.append(" ".join(current_parts))
            current_parts = [sentence]
            current_tokens = t
        else:
            current_parts.append(sentence)
            current_tokens += t

    if current_parts:
        groups.append(" ".join(current_parts))

    return [g for g in groups if g.strip()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_text_blocks(
    blocks: Sequence[str],
    document_id: uuid.UUID | str,
    version_id: uuid.UUID | str | None = None,
    source_filename: str = "",
    source_type: str = "",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[TextChunk]:
    """Chunk a list of parsed text blocks into overlapping token-bounded chunks.

    Args:
        blocks:          Ordered list of clean text blocks from the parser.
        document_id:     UUID of the parent Document record.
        version_id:      UUID of the DocumentVersion record (optional).
        source_filename: Original filename, stored in metadata.
        source_type:     File type (pdf / docx / markdown / …), stored in metadata.
        chunk_size:      Maximum tokens per chunk (default 500).
        overlap:         Number of overlap tokens to carry from the preceding chunk
                         (default 75 — roughly 15% of 500).

    Returns:
        Ordered list of :class:`TextChunk` objects with metadata attached.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    # -----------------------------------------------------------------------
    # Step 1: Flatten blocks → paragraphs, tracking heading context
    # -----------------------------------------------------------------------
    paragraphs: List[str] = []
    paragraph_headings: List[Optional[str]] = []  # heading active when paragraph starts
    current_heading: Optional[str] = None

    for block in blocks:
        sub_paras = _split_into_paragraphs(block)
        for para in sub_paras:
            if _is_heading(para):
                current_heading = para.strip()
            paragraphs.append(para.strip())
            paragraph_headings.append(current_heading)

    if not paragraphs:
        return []

    # -----------------------------------------------------------------------
    # Step 2: Greedily pack paragraphs into chunks; split oversized paragraphs
    # -----------------------------------------------------------------------
    # Each pending item: (text, heading_context)
    atoms: List[tuple[str, Optional[str]]] = []
    for para, heading in zip(paragraphs, paragraph_headings):
        para_tokens = _count_tokens(para)
        if para_tokens <= chunk_size:
            atoms.append((para, heading))
        else:
            # Oversized paragraph: split at sentence boundaries
            sub_parts = _split_sentence_boundary(para, chunk_size)
            for part in sub_parts:
                atoms.append((part, heading))

    # -----------------------------------------------------------------------
    # Step 3: Pack atoms into chunks with overlap
    # -----------------------------------------------------------------------
    chunks: List[TextChunk] = []
    position = 0

    # Sliding window: we accumulate atoms until we would exceed chunk_size.
    # On flush, the last `overlap`-tokens worth of content is *prepended* to
    # the next chunk so consecutive chunks share context.
    current_atoms: List[tuple[str, Optional[str]]] = []
    current_tokens = 0

    def flush(atoms_to_flush: List[tuple[str, Optional[str]]]) -> Optional[str]:
        """Flush accumulated atoms into a TextChunk; return overlap suffix."""
        nonlocal position

        combined_text = "\n\n".join(text for text, _ in atoms_to_flush)
        # Determine dominant heading: first heading in the batch that isn't None
        dominant_heading: Optional[str] = next(
            (h for _, h in atoms_to_flush if h is not None), None
        )

        token_count = _count_tokens(combined_text)
        chunks.append(
            TextChunk(
                text=combined_text,
                token_count=token_count,
                position=position,
                section_heading=dominant_heading,
                metadata={
                    "document_id": str(document_id),
                    "version_id": str(version_id) if version_id else None,
                    "source_filename": source_filename,
                    "source_type": source_type,
                    "position": position,
                },
            )
        )
        position += 1

        # Build the overlap tail: grab tokens from the end of combined_text
        if overlap <= 0:
            return None
        encoded = _ENCODER.encode(combined_text)
        if len(encoded) > overlap:
            overlap_tokens = encoded[-overlap:]
        else:
            overlap_tokens = encoded
        return _ENCODER.decode(overlap_tokens)

    overlap_prefix: Optional[str] = None

    for atom_text, atom_heading in atoms:
        atom_tokens = _count_tokens(atom_text)
        proposed_tokens = current_tokens + (1 if current_atoms else 0) * 2 + atom_tokens
        # Account for "\n\n" separator (2 tokens) between atoms

        if current_atoms and proposed_tokens > chunk_size:
            # Flush current batch and start fresh
            overlap_text = flush(current_atoms)
            current_atoms = []
            current_tokens = 0
            # Prepend overlap from the previous chunk
            if overlap_text:
                overlap_prefix = overlap_text
                current_atoms = [(overlap_prefix, atom_heading)]
                current_tokens = _count_tokens(overlap_prefix)
                overlap_prefix = None

        current_atoms.append((atom_text, atom_heading))
        current_tokens += (_count_tokens("\n\n") if len(current_atoms) > 1 else 0) + atom_tokens

    # Flush any remaining atoms
    if current_atoms:
        flush(current_atoms)

    return chunks


def chunk_document(
    blocks: Sequence[str],
    document_id: uuid.UUID | str,
    version_id: uuid.UUID | str | None = None,
    source_filename: str = "",
    source_type: str = "",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[dict]:
    """Convenience wrapper: returns plain dicts instead of TextChunk objects.

    Useful for serialisation / database insertion.
    """
    raw_chunks = chunk_text_blocks(
        blocks=blocks,
        document_id=document_id,
        version_id=version_id,
        source_filename=source_filename,
        source_type=source_type,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return [
        {
            "text": c.text,
            "token_count": c.token_count,
            "position": c.position,
            "section_heading": c.section_heading,
            "metadata": c.metadata,
        }
        for c in raw_chunks
    ]
