"""Guardrails for RAG Chat (Feature 12).

Provides defensive guardrails around /chat:
1. Input safety filter: Block abusive, toxic, or out-of-scope adversarial queries before retrieval.
2. Chunk content sanitizer: Strip or flag instruction-like text (e.g. 'ignore previous instructions')
   in retrieved document chunks before prompting to prevent indirect prompt injection.
3. Citation verifier: Verify every cited [chunk_id] was actually in the retrieved candidate set;
   regenerate once if hallucinated citations are detected, else abstain.
Logs structured warning events on each guardrail trigger.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Sequence, Set, Tuple
import uuid

from app.services.tracing import record_trace_event

logger = logging.getLogger(__name__)

# Refusal message for abusive/out-of-scope user queries
REFUSAL_MESSAGE = (
    "I cannot answer this request as it is abusive, unsafe, or out-of-scope "
    "for the employee knowledge assistant."
)

# Replacement marker for stripped instruction-like injections in chunks
FLAGGED_INSTRUCTION_MARKER = "[FLAGGED_INSTRUCTION_REMOVED]"

# ---------------------------------------------------------------------------
# Guardrail 1: Input Safety Patterns (Abusive & Out-of-Scope Queries)
# ---------------------------------------------------------------------------
ABUSIVE_AND_OUT_OF_SCOPE_PATTERNS = [
    # Jailbreaks / Prompt Injections
    (
        r"(?i)\b(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|above|system)\s+(?:instructions|rules|prompts?)\b",
        "Prompt Injection / Instruction Override",
    ),
    (
        r"(?i)\b(?:you\s+are\s+now|act\s+as)\s+(?:dan|developer\s+mode|an?\s+unrestricted|an?\s+evil)\b",
        "Jailbreak Attempt (DAN/Developer Mode)",
    ),
    (
        r"(?i)\b(?:bypass|disable)\s+(?:all\s+)?(?:security|content|safety)\s+(?:filters?|restrictions?|guardrails?)\b",
        "Security Filter Bypass Attempt",
    ),
    (
        r"(?i)\b(?:show|reveal|leak|print|dump)\s+(?:system\s+prompt|all\s+passwords|api\s+keys?|jwt\s+secret|env\s+vars?)\b",
        "System Secret / Prompt Exfiltration Attempt",
    ),
    # Cyberattacks / Malicious Activities
    (
        r"(?i)\b(?:how\s+to\s+)?(?:write|create|execute|distribute|build|develop)\s+(?:an?\s+)?(?:malware|ransomware|keylogger|trojan|ddos\s+attack|sql\s+injection|exploit(?:\s+code)?)\b",
        "Malicious Cyberattack Instruction",
    ),
    (
        r"(?i)\b(?:hack\s+into|penetrate)\s+(?:the\s+)?(?:database|company\s+server|internal\s+network|admin\s+account)\b",
        "Unauthorized Hacking Request",
    ),
    # Physical Harm / Weapons / Violence
    (
        r"(?i)\b(?:how\s+to\s+)?(?:make|build|synthesize|manufacture)\s+(?:a\s+bomb|an?\s+explosive|poison|chemical\s+weapon|biological\s+weapon)\b",
        "Dangerous Weapons / Physical Harm Instruction",
    ),
    # Severe Harassment & Hate Speech Slurs
    (
        r"(?i)\b(?:i\s+hate\s+all|kill\s+all|threaten\s+to\s+kill|terrorist\s+attack)\b",
        "Hate Speech / Violent Threat",
    ),
    (
        r"(?i)\b(?:fuck\s+you|piece\s+of\s+shit|asshole|idiot\s+bot|shut\s+up\s+bitch)\b",
        "Abusive / Toxic Language",
    ),
]


def validate_query_safety(query: str) -> Tuple[bool, Optional[str]]:
    """Check if a user query is safe and within the scope of the knowledge assistant.

    Args:
        query: Raw user query string.

    Returns:
        Tuple of (is_safe, rejection_reason). If is_safe is False, rejection_reason
        contains the triggered category.
    """
    clean_query = query.strip()
    if not clean_query:
        return True, None

    for pattern, reason in ABUSIVE_AND_OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, clean_query):
            logger.warning(
                "Guardrail 1 Triggered: Blocked query before retrieval. Reason='%s', Query='%s'",
                reason,
                clean_query,
            )
            record_trace_event(
                "guardrail_trigger",
                {"guardrail": 1, "type": "input_safety", "reason": reason, "query": clean_query},
            )
            return False, reason

    return True, None


# ---------------------------------------------------------------------------
# Guardrail 2: Chunk Instruction Stripping / Neutralization
# ---------------------------------------------------------------------------
INSTRUCTION_LIKE_PATTERNS = [
    r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior|above|existing|system)\s+instructions\b[^\n.]*",
    r"(?i)\bdisregard\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|rules|context|prompts?)\b[^\n.]*",
    r"(?i)\bforget\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|rules|context)\b[^\n.]*",
    r"(?i)\b(?:you\s+are\s+now|act\s+as)\s+(?:an?\s+unrestricted|a\s+hacker|dan|developer\s+mode)[^\n.]*",
    r"(?i)\b(?:system\s+(?:prompt|instruction|override)|admin\s+override|assistant\s+instruction)\s*:[^\n]*",
    r"(?i)\b(?:new\s+(?:instructions?|system\s+prompt))\s*:[^\n]*",
    r"(?i)\[system(?:\s+message)?\].*?\[/system(?:\s+message)?\]",
    r"(?i)<!--\s*system\s+prompt.*?-->",
    r"(?i)\bdo\s+not\s+follow\s+(?:previous|system)\s+instructions\b[^\n.]*",
    r"(?i)\boverride\s+all\s+guidelines\b[^\n.]*",
]


def sanitize_chunk_text(text: str, chunk_id: Optional[str] = None) -> str:
    """Detect and strip or flag instruction-like text from document chunk text before prompting.

    Args:
        text: Raw text of the retrieved document chunk.
        chunk_id: Optional chunk identifier for logging.

    Returns:
        Sanitized chunk text with instruction-like phrases replaced by FLAGGED_INSTRUCTION_MARKER.
    """
    if not text:
        return ""

    sanitized = text
    for pattern in INSTRUCTION_LIKE_PATTERNS:
        matches = re.findall(pattern, sanitized)
        if matches:
            for match in matches:
                logger.warning(
                    "Guardrail 2 Triggered: Instruction-like text detected and sanitized in chunk %s. Matched: %r",
                    chunk_id or "unknown",
                    match.strip(),
                )
                record_trace_event(
                    "guardrail_trigger",
                    {
                        "guardrail": 2,
                        "type": "instruction_stripping",
                        "chunk_id": str(chunk_id or "unknown"),
                        "matched_pattern": match.strip(),
                    },
                )
            sanitized = re.sub(pattern, FLAGGED_INSTRUCTION_MARKER, sanitized)

    return sanitized


# ---------------------------------------------------------------------------
# Guardrail 3: Citation Verification
# ---------------------------------------------------------------------------
def extract_all_cited_chunk_ids(answer_text: str) -> List[str]:
    """Extract all bracketed chunk IDs from answer text (e.g. [uuid]).

    Args:
        answer_text: LLM generated answer string.

    Returns:
        List of extracted chunk ID strings in order of appearance.
    """
    if not answer_text:
        return []
    # Match standard UUIDs or hex IDs in square brackets
    return re.findall(r"\[([a-fA-F0-9\-]{32,36})\]", answer_text)


def verify_citations(
    answer_text: str,
    retrieved_chunk_ids: Sequence[str | uuid.UUID],
) -> Tuple[bool, List[str], List[str]]:
    """Verify that every chunk_id cited in answer_text exists in retrieved_chunk_ids.

    Args:
        answer_text: Generated answer containing citation markers.
        retrieved_chunk_ids: Sequence of chunk IDs that were actually retrieved.

    Returns:
        Tuple of (is_valid, valid_cited_ids, invalid_cited_ids).
        is_valid is True only if there are no cited IDs that are absent from retrieved_chunk_ids.
    """
    cited_ids = extract_all_cited_chunk_ids(answer_text)
    if not cited_ids:
        # No citations made (e.g. "I do not have enough information")
        return True, [], []

    valid_id_set: Set[str] = {str(cid).lower() for cid in retrieved_chunk_ids}

    valid_cited: List[str] = []
    invalid_cited: List[str] = []

    for cid in cited_ids:
        if cid.lower() in valid_id_set:
            valid_cited.append(cid)
        else:
            invalid_cited.append(cid)

    is_valid = len(invalid_cited) == 0

    if not is_valid:
        logger.warning(
            "Guardrail 3 Triggered: Invalid/hallucinated chunk citation(s) detected: %s. Valid retrieved IDs: %s",
            invalid_cited,
            list(valid_id_set),
        )
        record_trace_event(
            "guardrail_trigger",
            {
                "guardrail": 3,
                "type": "citation_hallucination",
                "invalid_chunk_ids": invalid_cited,
                "valid_retrieved_ids": list(valid_id_set),
            },
        )

    return is_valid, valid_cited, invalid_cited
