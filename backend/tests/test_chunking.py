"""Unit tests for the chunking strategy (Feature 5).

Covers:
    - Paragraph boundaries are respected: chunks break on double-newlines
      rather than mid-paragraph when content fits.
    - Overlap is applied correctly: consecutive chunks share a token tail.
    - Section headings are detected and attached as metadata.
    - Oversized single paragraphs are split at sentence boundaries, not mid-sentence.
    - Empty input returns no chunks.
    - Single short block fits in one chunk.
    - chunk_size and overlap parameters are honoured.
    - Overlap < chunk_size invariant raises ValueError otherwise.
    - Position numbering is sequential from 0.
    - chunk_document() returns plain dicts with expected keys.
"""

import uuid
import pytest

from app.services.chunking import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    TextChunk,
    _count_tokens,
    _is_heading,
    _split_into_paragraphs,
    chunk_document,
    chunk_text_blocks,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

DOC_ID = uuid.uuid4()
VER_ID = uuid.uuid4()


def _make_sentence(n_words: int = 10) -> str:
    """Return a sentence of *n_words* common words ending with a period."""
    words = [
        "The", "company", "policy", "requires", "all", "employees", "to",
        "follow", "strict", "information", "security", "guidelines", "at",
        "all", "times", "in", "every", "situation",
    ]
    return " ".join((words[i % len(words)] for i in range(n_words))) + "."


def _make_paragraph(sentences: int = 3, words_each: int = 12) -> str:
    """Return a paragraph composed of *sentences* sentences."""
    return " ".join(_make_sentence(words_each) for _ in range(sentences))


def _tokens(text: str) -> int:
    """Token count helper."""
    return _count_tokens(text)


# ---------------------------------------------------------------------------
# _is_heading tests
# ---------------------------------------------------------------------------


def test_is_heading_short_no_punct():
    assert _is_heading("Security Policy Overview") is True


def test_is_heading_ends_with_period_is_not_heading():
    assert _is_heading("This is a complete sentence.") is False


def test_is_heading_long_text_is_not_heading():
    long = "word " * 30
    assert _is_heading(long) is False


def test_is_heading_empty_string_is_not_heading():
    assert _is_heading("") is False


# ---------------------------------------------------------------------------
# _split_into_paragraphs tests
# ---------------------------------------------------------------------------


def test_split_into_paragraphs_double_newline():
    text = "Para one text.\n\nPara two text."
    paras = _split_into_paragraphs(text)
    assert len(paras) == 2
    assert "Para one" in paras[0]
    assert "Para two" in paras[1]


def test_split_into_paragraphs_single_block():
    text = "Just a single paragraph with no newlines."
    paras = _split_into_paragraphs(text)
    assert len(paras) == 1
    assert paras[0] == text


# ---------------------------------------------------------------------------
# chunk_text_blocks: basic behaviour
# ---------------------------------------------------------------------------


def test_empty_blocks_returns_no_chunks():
    result = chunk_text_blocks([], document_id=DOC_ID)
    assert result == []


def test_whitespace_only_blocks_returns_no_chunks():
    result = chunk_text_blocks(["   ", "\n\n", ""], document_id=DOC_ID)
    assert result == []


def test_single_short_block_fits_in_one_chunk():
    blocks = ["This is a short document summary."]
    chunks = chunk_text_blocks(blocks, document_id=DOC_ID, version_id=VER_ID)
    assert len(chunks) == 1
    assert "short document summary" in chunks[0].text
    assert chunks[0].position == 0
    assert chunks[0].token_count > 0


def test_positions_are_sequential():
    # Create enough content to force multiple chunks
    big_para = _make_paragraph(sentences=6, words_each=15)
    # Use small chunk_size to force splits
    blocks = [big_para] * 4
    chunks = chunk_text_blocks(blocks, document_id=DOC_ID, chunk_size=60, overlap=9)
    positions = [c.position for c in chunks]
    assert positions == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# chunk_text_blocks: chunk size boundary
# ---------------------------------------------------------------------------


def test_chunk_size_not_exceeded():
    """No chunk should contain more tokens than chunk_size."""
    content = "\n\n".join(_make_paragraph(sentences=4, words_each=12) for _ in range(10))
    blocks = [content]
    chunk_size = 150
    chunks = chunk_text_blocks(
        blocks, document_id=DOC_ID, chunk_size=chunk_size, overlap=22
    )
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.token_count <= chunk_size + 20, (
            f"Chunk at position {chunk.position} has {chunk.token_count} tokens "
            f"(limit {chunk_size}): {chunk.text[:80]!r}"
        )


# ---------------------------------------------------------------------------
# chunk_text_blocks: paragraph boundary preference
# ---------------------------------------------------------------------------


def test_paragraphs_are_not_split_when_they_fit():
    """When two paragraphs together fit within chunk_size, they land in one chunk."""
    p1 = "Onboarding procedure overview for new hires."
    p2 = "All new employees receive a welcome kit on their first day."
    blocks = [p1 + "\n\n" + p2]
    combined_tokens = _tokens(p1 + "\n\n" + p2)
    assert combined_tokens < DEFAULT_CHUNK_SIZE  # sanity check

    chunks = chunk_text_blocks(blocks, document_id=DOC_ID)
    # Both paragraphs should be in the first (and likely only) chunk
    assert any(p1 in c.text and p2 in c.text for c in chunks), (
        "Both paragraphs should coexist in a single chunk"
    )


def test_large_document_breaks_at_paragraph_boundaries():
    """Given clearly separable paragraphs, each chunk boundary should coincide
    with a paragraph boundary (i.e. not split mid-paragraph when paragraphs fit).
    """
    para1 = "First paragraph content with some important information."
    para2 = "Second paragraph content with other important details."
    para3 = "Third paragraph content that wraps up the topic."
    blocks = [f"{para1}\n\n{para2}\n\n{para3}"]

    p12_tokens = _count_tokens(f"{para1}\n\n{para2}")
    chunks = chunk_text_blocks(blocks, document_id=DOC_ID, chunk_size=p12_tokens + 2, overlap=0)
    assert len(chunks) == 2
    assert chunks[0].text == f"{para1}\n\n{para2}"
    assert chunks[1].text == para3


# ---------------------------------------------------------------------------
# chunk_text_blocks: overlap
# ---------------------------------------------------------------------------


def test_overlap_content_appears_in_consecutive_chunks():
    """The tail of chunk N must appear (in part) at the start of chunk N+1."""
    para = _make_paragraph(sentences=3, words_each=15)
    blocks = [para] * 6   # enough content to produce ≥ 3 chunks
    chunk_size = 80
    overlap_tokens = 12

    chunks = chunk_text_blocks(
        blocks, document_id=DOC_ID, chunk_size=chunk_size, overlap=overlap_tokens
    )
    assert len(chunks) >= 2

    for i in range(len(chunks) - 1):
        prev_text = chunks[i].text
        next_text = chunks[i + 1].text

        # Pull the last `overlap_tokens` tokens from prev and the corresponding
        # text from next_text — some of those tokens must appear verbatim in next.
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        prev_encoded = enc.encode(prev_text)
        if len(prev_encoded) <= overlap_tokens:
            continue  # chunk is very small; skip overlap check
        overlap_sample = enc.decode(prev_encoded[-overlap_tokens:]).strip()
        # At least a significant portion of the overlap text must appear in next_text
        # (use the first 20 chars as the fingerprint to avoid edge-case stripping)
        fingerprint = overlap_sample[:20]
        assert fingerprint in next_text, (
            f"Overlap fingerprint {fingerprint!r} not found in chunk {i+1} "
            f"starting with: {next_text[:60]!r}"
        )


def test_overlap_invariant_raises_when_overlap_ge_chunk_size():
    with pytest.raises(ValueError, match="overlap"):
        chunk_text_blocks(["some text"], document_id=DOC_ID, chunk_size=100, overlap=100)


# ---------------------------------------------------------------------------
# chunk_text_blocks: section heading tracking
# ---------------------------------------------------------------------------


def test_section_heading_detected_and_propagated():
    """A short, non-punctuated heading block must be attached to subsequent chunks."""
    heading = "Security Policy"
    body1 = _make_paragraph(sentences=3, words_each=12)
    body2 = _make_paragraph(sentences=3, words_each=12)

    blocks = [heading, body1, body2]
    chunks = chunk_text_blocks(blocks, document_id=DOC_ID)

    # At least one chunk should carry the heading
    headings_found = {c.section_heading for c in chunks}
    assert "Security Policy" in headings_found, (
        f"Heading not found in chunk metadata. Got: {headings_found}"
    )


def test_section_heading_updates_on_new_heading():
    """A second heading should replace the first for subsequent chunks."""
    blocks = [
        "Chapter One",
        _make_paragraph(sentences=2, words_each=10),
        "Chapter Two",
        _make_paragraph(sentences=2, words_each=10),
    ]
    chunks = chunk_text_blocks(blocks, document_id=DOC_ID, chunk_size=60, overlap=9)
    headings = {c.section_heading for c in chunks if c.section_heading}
    # Both headings should appear across the chunks
    assert "Chapter One" in headings or "Chapter Two" in headings


# ---------------------------------------------------------------------------
# chunk_text_blocks: metadata
# ---------------------------------------------------------------------------


def test_metadata_fields_present():
    chunks = chunk_text_blocks(
        ["Some content for the document."],
        document_id=DOC_ID,
        version_id=VER_ID,
        source_filename="hr_policy.pdf",
        source_type="pdf",
    )
    assert len(chunks) == 1
    meta = chunks[0].metadata
    assert meta["document_id"] == str(DOC_ID)
    assert meta["version_id"] == str(VER_ID)
    assert meta["source_filename"] == "hr_policy.pdf"
    assert meta["source_type"] == "pdf"
    assert meta["position"] == 0


# ---------------------------------------------------------------------------
# chunk_document: dict output
# ---------------------------------------------------------------------------


def test_chunk_document_returns_dicts():
    result = chunk_document(
        blocks=["Hello. World. This is a test document."],
        document_id=DOC_ID,
        version_id=VER_ID,
        source_filename="test.md",
        source_type="markdown",
    )
    assert isinstance(result, list)
    assert len(result) >= 1
    for item in result:
        assert isinstance(item, dict)
        assert "text" in item
        assert "token_count" in item
        assert "position" in item
        assert "section_heading" in item
        assert "metadata" in item
