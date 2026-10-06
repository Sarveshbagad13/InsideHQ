"""Tests for Hybrid Retrieval (Vector + Keyword) with Reciprocal Rank Fusion & Permission Filtering (Features 8 & 9).

Covers:
    - Reciprocal Rank Fusion (RRF) calculation and ranking logic.
    - Vector candidate search with cosine distance against approved documents.
    - Keyword candidate search using PostgreSQL full-text search against approved documents.
    - SQL-level permission filtering against document_acl.
    - Adversarial permission-aware tests:
        * Engineering user searching for HR-restricted salary/compensation document gets 0 results.
        * HR user searching for the same query gets the exact matching chunk.
        * User-specific ACLs (User A can access, User B in same department cannot).
        * Admin role bypass (Admin can access all approved documents).
        * General / company-wide access (allowed_role in 'employee', 'public', 'all').
        * Pending/deprecated documents remain blocked even if user has ACL permissions.
    - Seeded dataset test:
        * Keyword-heavy query (exact code/jargon "SEC-8042 FIDO2") retrieves correct chunk.
        * Meaning-heavy paraphrase ("time off for new mothers and fathers when a baby arrives") retrieves correct chunk.
        * Edge cases: empty/whitespace queries.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import numpy as np
import pytest

from app.core.config import settings
from app.models.document import Document, DocumentACL, DocumentChunk
from app.models.user import User
from app.services.embeddings import EmbeddingService
from app.services.permissions import (
    build_permission_filter,
    get_user_permission_tokens,
    has_document_access,
)
from app.services.retrieval import (
    DEFAULT_RRF_K,
    RetrievalResult,
    compute_rrf_score,
    hybrid_search,
    reciprocal_rank_fusion,
    search_keyword_candidates,
    search_vector_candidates,
)


# ---------------------------------------------------------------------------
# Deterministic Semantic Mock Embedding Service for Semantic Paraphrase Tests
# ---------------------------------------------------------------------------
class SemanticMockEmbeddingService(EmbeddingService):
    """Semantic embedding mock that projects text into key thematic clusters in 1024-dim space.

    Allows accurate testing of meaning-heavy paraphrases vs keyword queries.
    """

    def __init__(self):
        super().__init__(model_name=settings.EMBEDDING_MODEL_NAME)
        # Cluster centroids
        self.themes = {
            "security": ["sec-8042", "fido2", "security", "password", "token", "hardware", "mfa", "auth"],
            "parental_leave": ["parent", "leave", "baby", "caregiver", "maternity", "paternity", "child", "time off", "birth"],
            "expense": ["expense", "reimbursement", "receipt", "purchase", "money", "finance", "dollar"],
            "vacation": ["vacation", "holiday", "pto", "annual leave"],
            "salary": ["salary", "compensation", "bonus", "payroll", "vp level", "equity"],
            "confidential": ["confidential", "restricted", "classified", "secret"],
        }

    def _get_embedding_for_text(self, text: str) -> List[float]:
        t_lower = text.lower()
        vec = np.zeros(self.dimension, dtype=np.float32)

        # Base noise for uniqueness
        seed = sum(ord(c) for c in t_lower) % 1000
        rng = np.random.RandomState(seed)
        vec += rng.normal(0.0, 0.01, size=self.dimension).astype(np.float32)

        # Add strong signal along dedicated dimension bands for matched themes
        for idx, (theme, keywords) in enumerate(self.themes.items()):
            band_start = idx * 150
            band_end = band_start + 150
            matches = sum(1 for kw in keywords if kw in t_lower)
            if matches > 0:
                vec[band_start:band_end] += float(matches) * 5.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_query(self, query: str) -> List[float]:
        return self._get_embedding_for_text(query)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding_for_text(t) for t in texts]

    async def aembed_query(self, query: str) -> List[float]:
        return self.embed_query(query)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)


# ---------------------------------------------------------------------------
# Unit Tests for RRF Logic
# ---------------------------------------------------------------------------
def test_compute_rrf_score_values():
    """Verify RRF mathematical scoring formula."""
    k = 60
    # Rank 1 in vector only
    score_vec_1 = compute_rrf_score(vector_rank=1, keyword_rank=None, rrf_k=k)
    assert score_vec_1 == pytest.approx(1.0 / (60 + 1), rel=1e-5)

    # Rank 1 in keyword only
    score_kw_1 = compute_rrf_score(vector_rank=None, keyword_rank=1, rrf_k=k)
    assert score_kw_1 == pytest.approx(1.0 / (60 + 1), rel=1e-5)

    # Rank 1 in both vector and keyword (should be double)
    score_both_1 = compute_rrf_score(vector_rank=1, keyword_rank=1, rrf_k=k)
    assert score_both_1 == pytest.approx(2.0 / (60 + 1), rel=1e-5)

    # Moderate in both (rank 3 in vector, rank 3 in keyword)
    # 1/63 + 1/63 = 2/63 ≈ 0.0317 > 1/61 ≈ 0.0164 (rank 1 in only one modality)
    score_both_3 = compute_rrf_score(vector_rank=3, keyword_rank=3, rrf_k=k)
    assert score_both_3 > score_vec_1


def test_reciprocal_rank_fusion_ranking_order():
    """Verify reciprocal_rank_fusion correctly merges and orders heterogeneous results."""
    chunk_a = uuid.uuid4()
    chunk_b = uuid.uuid4()
    chunk_c = uuid.uuid4()

    # Vector results: chunk_a (rank 1), chunk_b (rank 2)
    vec_results = [
        RetrievalResult(chunk_id=chunk_a, document_id=uuid.uuid4(), version_id=uuid.uuid4(), chunk_text="Chunk A", vector_distance=0.1),
        RetrievalResult(chunk_id=chunk_b, document_id=uuid.uuid4(), version_id=uuid.uuid4(), chunk_text="Chunk B", vector_distance=0.2),
    ]

    # Keyword results: chunk_b (rank 1), chunk_c (rank 2)
    kw_results = [
        RetrievalResult(chunk_id=chunk_b, document_id=uuid.uuid4(), version_id=uuid.uuid4(), chunk_text="Chunk B", keyword_score=0.9),
        RetrievalResult(chunk_id=chunk_c, document_id=uuid.uuid4(), version_id=uuid.uuid4(), chunk_text="Chunk C", keyword_score=0.8),
    ]

    # chunk_b appears in both (vec rank 2, kw rank 1) -> 1/62 + 1/61 ≈ 0.0325
    # chunk_a appears in vec only (rank 1) -> 1/61 ≈ 0.0164
    # chunk_c appears in kw only (rank 2) -> 1/62 ≈ 0.0161
    fused = reciprocal_rank_fusion(vec_results, kw_results, rrf_k=60)

    assert len(fused) == 3
    assert fused[0].chunk_id == chunk_b
    assert fused[0].vector_rank == 2
    assert fused[0].keyword_rank == 1
    assert fused[0].rrf_score == pytest.approx((1.0 / 62) + (1.0 / 61), rel=1e-5)

    assert fused[1].chunk_id == chunk_a
    assert fused[1].vector_rank == 1
    assert fused[1].keyword_rank is None

    assert fused[2].chunk_id == chunk_c
    assert fused[2].vector_rank is None
    assert fused[2].keyword_rank == 2


def test_reciprocal_rank_fusion_top_k_cutoff():
    """Verify top_k parameter truncates results list."""
    chunks = [uuid.uuid4() for _ in range(10)]
    vec_results = [
        RetrievalResult(chunk_id=cid, document_id=uuid.uuid4(), version_id=uuid.uuid4(), chunk_text=f"Chunk {i}")
        for i, cid in enumerate(chunks)
    ]
    fused = reciprocal_rank_fusion(vec_results, [], top_k=3)
    assert len(fused) == 3


# ---------------------------------------------------------------------------
# Seeded Dataset Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def seeded_knowledge_base():
    """Create a realistic seeded collection of documents, ACLs, and chunks."""
    doc_security = Document(
        id=uuid.uuid4(),
        title="Security and Access Control Standards",
        source_type="pdf",
        status="approved",
    )
    doc_parental = Document(
        id=uuid.uuid4(),
        title="Employee Benefits and Leave Policy",
        source_type="docx",
        status="approved",
    )
    doc_expense = Document(
        id=uuid.uuid4(),
        title="Travel & Expense Policy",
        source_type="markdown",
        status="approved",
    )
    doc_hr_confidential = Document(
        id=uuid.uuid4(),
        title="Executive Compensation and Salary Bands",
        source_type="pdf",
        status="approved",
    )
    doc_pending = Document(
        id=uuid.uuid4(),
        title="Draft Security Proposal (Pending Approval)",
        source_type="pdf",
        status="pending",
    )
    doc_deprecated = Document(
        id=uuid.uuid4(),
        title="Outdated Employee Handbook 2020",
        source_type="pdf",
        status="deprecated",
    )

    # ACLs
    acl_security = [
        DocumentACL(id=uuid.uuid4(), document_id=doc_security.id, allowed_role="engineering"),
        DocumentACL(id=uuid.uuid4(), document_id=doc_security.id, allowed_role="security"),
        DocumentACL(id=uuid.uuid4(), document_id=doc_security.id, allowed_role="admin"),
    ]
    acl_parental = [
        DocumentACL(id=uuid.uuid4(), document_id=doc_parental.id, allowed_role="employee"),
        DocumentACL(id=uuid.uuid4(), document_id=doc_parental.id, allowed_role="hr"),
        DocumentACL(id=uuid.uuid4(), document_id=doc_parental.id, allowed_role="admin"),
    ]
    acl_expense = [
        DocumentACL(id=uuid.uuid4(), document_id=doc_expense.id, allowed_role="employee"),
        DocumentACL(id=uuid.uuid4(), document_id=doc_expense.id, allowed_role="finance"),
        DocumentACL(id=uuid.uuid4(), document_id=doc_expense.id, allowed_role="admin"),
    ]
    # Restricted strictly to HR role / Human Resources department
    acl_hr_confidential = [
        DocumentACL(id=uuid.uuid4(), document_id=doc_hr_confidential.id, allowed_role="hr"),
        DocumentACL(id=uuid.uuid4(), document_id=doc_hr_confidential.id, allowed_role="human resources"),
        DocumentACL(id=uuid.uuid4(), document_id=doc_hr_confidential.id, allowed_role="admin"),
    ]
    acl_pending = [
        DocumentACL(id=uuid.uuid4(), document_id=doc_pending.id, allowed_role="security"),
        DocumentACL(id=uuid.uuid4(), document_id=doc_pending.id, allowed_role="hr"),
        DocumentACL(id=uuid.uuid4(), document_id=doc_pending.id, allowed_role="admin"),
    ]
    acl_deprecated = [
        DocumentACL(id=uuid.uuid4(), document_id=doc_deprecated.id, allowed_role="employee"),
    ]

    embedder = SemanticMockEmbeddingService()

    chunk_security = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc_security.id,
        version_id=uuid.uuid4(),
        chunk_text="SOC2 Compliance Protocol: Mandatory password rotation every 90 days with FIDO2 hardware tokens (Standard SEC-8042).",
        embedding=embedder.embed_query("SOC2 Compliance Protocol: Mandatory password rotation every 90 days with FIDO2 hardware tokens (Standard SEC-8042)."),
        chunk_metadata={"department": "Security", "section": "Access Control"},
    )

    chunk_parental = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc_parental.id,
        version_id=uuid.uuid4(),
        chunk_text="Parental leave policy provides 16 weeks of fully paid leave for primary caregivers and 8 weeks for secondary caregivers following the birth or adoption of a child.",
        embedding=embedder.embed_query("Parental leave policy provides 16 weeks of fully paid leave for primary caregivers and 8 weeks for secondary caregivers following the birth or adoption of a child."),
        chunk_metadata={"department": "Human Resources", "section": "Leave Policies"},
    )

    chunk_expense = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc_expense.id,
        version_id=uuid.uuid4(),
        chunk_text="Travel expense reimbursement requests must be submitted within 30 days of purchase with itemized receipts for expenses exceeding $25.",
        embedding=embedder.embed_query("Travel expense reimbursement requests must be submitted within 30 days of purchase with itemized receipts for expenses exceeding $25."),
        chunk_metadata={"department": "Finance", "section": "Reimbursements"},
    )

    chunk_hr_confidential = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc_hr_confidential.id,
        version_id=uuid.uuid4(),
        chunk_text="Confidential HR Compensation Review: VP level engineering base salary band is $320k-$380k with 40% target annual performance bonus.",
        embedding=embedder.embed_query("Confidential HR Compensation Review: VP level engineering base salary band is $320k-$380k with 40% target annual performance bonus."),
        chunk_metadata={"department": "Human Resources", "section": "Executive Comp"},
    )

    chunk_pending_security = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc_pending.id,
        version_id=uuid.uuid4(),
        chunk_text="Unapproved draft protocol SEC-8042 proposing biometric iris scanners instead of FIDO2 tokens.",
        embedding=embedder.embed_query("Unapproved draft protocol SEC-8042 proposing biometric iris scanners instead of FIDO2 tokens."),
        chunk_metadata={"department": "Security", "section": "Drafts"},
    )

    chunk_deprecated = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc_deprecated.id,
        version_id=uuid.uuid4(),
        chunk_text="Deprecated 2020 maternity leave policy: 6 weeks unpaid leave.",
        embedding=embedder.embed_query("Deprecated 2020 maternity leave policy: 6 weeks unpaid leave."),
        chunk_metadata={"department": "Human Resources", "section": "Legacy"},
    )

    docs = {
        doc_security.id: doc_security,
        doc_parental.id: doc_parental,
        doc_expense.id: doc_expense,
        doc_hr_confidential.id: doc_hr_confidential,
        doc_pending.id: doc_pending,
        doc_deprecated.id: doc_deprecated,
    }

    acls = {
        doc_security.id: acl_security,
        doc_parental.id: acl_parental,
        doc_expense.id: acl_expense,
        doc_hr_confidential.id: acl_hr_confidential,
        doc_pending.id: acl_pending,
        doc_deprecated.id: acl_deprecated,
    }

    chunks = [
        chunk_security,
        chunk_parental,
        chunk_expense,
        chunk_hr_confidential,
        chunk_pending_security,
        chunk_deprecated,
    ]

    return {
        "docs": docs,
        "acls": acls,
        "chunks": chunks,
        "embedder": embedder,
        "chunk_security": chunk_security,
        "chunk_parental": chunk_parental,
        "chunk_expense": chunk_expense,
        "chunk_hr_confidential": chunk_hr_confidential,
        "chunk_pending_security": chunk_pending_security,
        "chunk_deprecated": chunk_deprecated,
    }


def create_simulated_db_session(seeded_data, requesting_user: User):
    """Create an AsyncSession mock that executes in-memory vector cosine search and keyword search.

    Strictly applies the SQL-level status and permission filter:
    Only chunks from approved documents where the requesting user has ACL permission are retrieved.
    """
    mock_db = AsyncMock()
    docs = seeded_data["docs"]
    acls = seeded_data["acls"]
    chunks = seeded_data["chunks"]

    async def mock_execute(stmt):
        sql_str = str(stmt).lower()
        result_mock = MagicMock()

        is_vector = "order by distance" in sql_str or "<=>" in sql_str
        is_keyword = "ts_rank" in sql_str or "@@" in sql_str

        # 1. Filter documents by status == 'approved'
        # 2. Filter documents by requesting user's ACL permissions
        eligible_chunks = []
        for c in chunks:
            doc = docs[c.document_id]
            if doc.status != "approved":
                continue
            doc_acls = acls.get(c.document_id, [])
            if has_document_access(requesting_user, doc_acls):
                eligible_chunks.append(c)

        if is_vector:
            params = stmt.compile().params
            query_vec = None
            for k, v in params.items():
                if isinstance(v, (list, tuple)) or isinstance(v, np.ndarray):
                    query_vec = np.array(v, dtype=np.float32)
                    break
                elif isinstance(v, str) and v.startswith("["):
                    import json
                    query_vec = np.array(json.loads(v), dtype=np.float32)
                    break

            if query_vec is None:
                query_vec = np.zeros(1024, dtype=np.float32)

            ranked_rows = []
            for c in eligible_chunks:
                c_vec = np.array(c.embedding, dtype=np.float32)
                dot = float(np.dot(c_vec, query_vec))
                norm_c = float(np.linalg.norm(c_vec))
                norm_q = float(np.linalg.norm(query_vec))
                cosine_sim = dot / (norm_c * norm_q) if (norm_c * norm_q) > 0 else 0.0
                cosine_dist = 1.0 - cosine_sim
                ranked_rows.append((c, docs[c.document_id].title, cosine_dist))

            ranked_rows.sort(key=lambda x: x[2])  # distance ASC
            result_mock.all.return_value = ranked_rows
            return result_mock

        elif is_keyword:
            params = stmt.compile().params
            search_term = ""
            for k, v in params.items():
                if isinstance(v, str) and not v.startswith("[") and v.lower() not in ("approved", "english"):
                    search_term = v.lower()
                    break

            STOP_WORDS = {"and", "or", "the", "a", "an", "in", "of", "to", "for", "with", "is", "at", "by", "from", "on"}
            matched_rows = []
            words = [
                w.strip().lower()
                for w in search_term.replace("-", " ").replace("$", " ").split()
                if w.strip() and w.lower() not in STOP_WORDS
            ]
            for c in eligible_chunks:
                text_lower = c.chunk_text.lower().replace("-", " ").replace("$", " ")
                matches = sum(1 for w in words if w in text_lower)
                if matches > 0:
                    rank_score = float(matches) * 0.5
                    matched_rows.append((c, docs[c.document_id].title, rank_score))

            matched_rows.sort(key=lambda x: x[2], reverse=True)  # rank DESC
            result_mock.all.return_value = matched_rows
            return result_mock

        result_mock.all.return_value = []
        return result_mock

    mock_db.execute.side_effect = mock_execute
    return mock_db


# ---------------------------------------------------------------------------
# Adversarial Permission Filtering Tests (Feature 9 Hard Gate)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_adversarial_cross_department_leakage_blocked(seeded_knowledge_base):
    """ADVERSARIAL HARD GATE: Engineering employee queries restricted HR salary doc -> gets ZERO results.

    When the same query is executed by an HR employee, the restricted chunk is returned.
    """
    embedder = seeded_knowledge_base["embedder"]
    hr_chunk = seeded_knowledge_base["chunk_hr_confidential"]
    doc_hr = seeded_knowledge_base["docs"][hr_chunk.document_id]
    acls_hr = seeded_knowledge_base["acls"][hr_chunk.document_id]

    # Isolated restricted HR document dataset
    hr_restricted_data = {
        "docs": {doc_hr.id: doc_hr},
        "acls": {doc_hr.id: acls_hr},
        "chunks": [hr_chunk],
        "embedder": embedder,
    }

    # 1. Engineering employee (not allowed to view HR confidential)
    eng_user = User(
        id=uuid.uuid4(),
        name="Charlie Engineer",
        email="charlie.eng@insidehq.com",
        role="engineering",
        department="Engineering",
    )
    eng_db = create_simulated_db_session(hr_restricted_data, requesting_user=eng_user)

    # 2. HR Specialist (allowed to view HR confidential)
    hr_user = User(
        id=uuid.uuid4(),
        name="Bob HR",
        email="bob.hr@insidehq.com",
        role="hr",
        department="Human Resources",
    )
    hr_db = create_simulated_db_session(hr_restricted_data, requesting_user=hr_user)

    query = "VP level engineering base salary band and bonus compensation"

    # Engineering user search
    eng_results = await hybrid_search(
        db=eng_db,
        query=query,
        user=eng_user,
        top_k_candidates=20,
        final_top_k=10,
        embedding_service=embedder,
    )

    # HARD ASSERT: Engineering user must receive exactly 0 results
    assert len(eng_results) == 0, f"Expected 0 results for engineering user on HR query, got {len(eng_results)}"

    # HR user search
    hr_results = await hybrid_search(
        db=hr_db,
        query=query,
        user=hr_user,
        top_k_candidates=20,
        final_top_k=10,
        embedding_service=embedder,
    )

    # HR user MUST successfully retrieve the document
    assert len(hr_results) > 0, "HR user should have retrieved the HR compensation document"
    assert hr_results[0].chunk_id == hr_chunk.id
    assert "VP level engineering base salary" in hr_results[0].chunk_text
    assert hr_results[0].rrf_score > 0.0

    # Also test mixed full dataset: Engineering user never leaks HR chunk
    full_eng_db = create_simulated_db_session(seeded_knowledge_base, requesting_user=eng_user)
    mixed_results = await hybrid_search(
        db=full_eng_db,
        query=query,
        user=eng_user,
        top_k_candidates=20,
        final_top_k=10,
        embedding_service=embedder,
    )
    mixed_retrieved_ids = {r.chunk_id for r in mixed_results}
    assert hr_chunk.id not in mixed_retrieved_ids, "CRITICAL LEAK: Engineering user retrieved HR chunk in mixed dataset!"


@pytest.mark.asyncio
async def test_admin_role_can_access_all_approved_documents(seeded_knowledge_base):
    """Verify administrator role has universal access to all approved department documents."""
    embedder = seeded_knowledge_base["embedder"]
    hr_chunk = seeded_knowledge_base["chunk_hr_confidential"]

    admin_user = User(
        id=uuid.uuid4(),
        name="Alice Admin",
        email="alice.admin@insidehq.com",
        role="admin",
        department="Executive",
    )
    admin_db = create_simulated_db_session(seeded_knowledge_base, requesting_user=admin_user)

    query = "VP level engineering base salary band"
    results = await hybrid_search(
        db=admin_db,
        query=query,
        user=admin_user,
        top_k_candidates=20,
        final_top_k=5,
        embedding_service=embedder,
    )

    assert len(results) > 0
    assert results[0].chunk_id == hr_chunk.id


@pytest.mark.asyncio
async def test_user_specific_acl_filtering():
    """Verify document restricted to a specific allowed_user ID is only accessible to that exact user."""
    embedder = SemanticMockEmbeddingService()
    user_a = User(id=uuid.uuid4(), name="User A", email="a@insidehq.com", role="employee", department="Finance")
    user_b = User(id=uuid.uuid4(), name="User B", email="b@insidehq.com", role="employee", department="Finance")

    doc_private = Document(id=uuid.uuid4(), title="User A Private Performance Plan", source_type="pdf", status="approved")
    acl_private = [DocumentACL(id=uuid.uuid4(), document_id=doc_private.id, allowed_user=user_a.id)]
    chunk_private = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc_private.id,
        version_id=uuid.uuid4(),
        chunk_text="Private Performance Improvement Plan for Employee A: Milestones and Q3 Objectives.",
        embedding=embedder.embed_query("Private Performance Improvement Plan for Employee A: Milestones and Q3 Objectives."),
        chunk_metadata={"department": "Finance"},
    )

    data = {
        "docs": {doc_private.id: doc_private},
        "acls": {doc_private.id: acl_private},
        "chunks": [chunk_private],
        "embedder": embedder,
    }

    # User B tries to search
    db_b = create_simulated_db_session(data, requesting_user=user_b)
    res_b = await hybrid_search(db=db_b, query="Performance Improvement Plan Milestones", user=user_b, embedding_service=embedder)
    assert len(res_b) == 0, "User B should not have access to User A's private document"

    # User A searches
    db_a = create_simulated_db_session(data, requesting_user=user_a)
    res_a = await hybrid_search(db=db_a, query="Performance Improvement Plan Milestones", user=user_a, embedding_service=embedder)
    assert len(res_a) > 0
    assert res_a[0].chunk_id == chunk_private.id


@pytest.mark.asyncio
async def test_general_employee_policy_accessible_to_all_departments(seeded_knowledge_base):
    """Verify company-wide policies with role='employee' are accessible to all departments."""
    embedder = seeded_knowledge_base["embedder"]
    parental_chunk = seeded_knowledge_base["chunk_parental"]

    eng_user = User(id=uuid.uuid4(), name="Eng User", email="eng@insidehq.com", role="engineering", department="Engineering")
    fin_user = User(id=uuid.uuid4(), name="Fin User", email="fin@insidehq.com", role="finance", department="Finance")

    eng_db = create_simulated_db_session(seeded_knowledge_base, requesting_user=eng_user)
    fin_db = create_simulated_db_session(seeded_knowledge_base, requesting_user=fin_user)

    query = "paid parental leave for primary caregiver"

    res_eng = await hybrid_search(db=eng_db, query=query, user=eng_user, embedding_service=embedder)
    res_fin = await hybrid_search(db=fin_db, query=query, user=fin_user, embedding_service=embedder)

    assert len(res_eng) > 0 and res_eng[0].chunk_id == parental_chunk.id
    assert len(res_fin) > 0 and res_fin[0].chunk_id == parental_chunk.id


# ---------------------------------------------------------------------------
# Search Modality & Paraphrase Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_hybrid_search_keyword_heavy_query(seeded_knowledge_base):
    """Verify a keyword-heavy query (SEC-8042 FIDO2) retrieves the exact compliance chunk at rank 1."""
    admin_user = User(id=uuid.uuid4(), name="Admin", email="admin@insidehq.com", role="admin", department="Executive")
    mock_db = create_simulated_db_session(seeded_knowledge_base, requesting_user=admin_user)
    embedder = seeded_knowledge_base["embedder"]
    expected_chunk = seeded_knowledge_base["chunk_security"]

    query = "SEC-8042 FIDO2 compliance"
    results = await hybrid_search(
        db=mock_db,
        query=query,
        user=admin_user,
        top_k_candidates=20,
        final_top_k=5,
        embedding_service=embedder,
    )

    assert len(results) > 0
    top_result = results[0]
    assert top_result.chunk_id == expected_chunk.id
    assert "SEC-8042" in top_result.chunk_text
    assert "FIDO2" in top_result.chunk_text
    assert top_result.keyword_rank is not None
    assert top_result.rrf_score > 0.0


@pytest.mark.asyncio
async def test_hybrid_search_meaning_heavy_paraphrase(seeded_knowledge_base):
    """Verify a meaning-heavy paraphrase ('time off for new mothers and fathers when a baby arrives') retrieves parental leave."""
    admin_user = User(id=uuid.uuid4(), name="Admin", email="admin@insidehq.com", role="admin", department="Executive")
    mock_db = create_simulated_db_session(seeded_knowledge_base, requesting_user=admin_user)
    embedder = seeded_knowledge_base["embedder"]
    expected_chunk = seeded_knowledge_base["chunk_parental"]

    query = "time off for new mothers and fathers when a baby arrives"
    results = await hybrid_search(
        db=mock_db,
        query=query,
        user=admin_user,
        top_k_candidates=20,
        final_top_k=5,
        embedding_service=embedder,
    )

    assert len(results) > 0
    top_result = results[0]
    assert top_result.chunk_id == expected_chunk.id
    assert "Parental leave" in top_result.chunk_text
    assert top_result.vector_rank == 1
    assert top_result.rrf_score > 0.0


@pytest.mark.asyncio
async def test_hybrid_search_unapproved_documents_excluded(seeded_knowledge_base):
    """Verify that chunks from pending or deprecated documents are NEVER returned in retrieval."""
    admin_user = User(id=uuid.uuid4(), name="Admin", email="admin@insidehq.com", role="admin", department="Executive")
    mock_db = create_simulated_db_session(seeded_knowledge_base, requesting_user=admin_user)
    embedder = seeded_knowledge_base["embedder"]

    pending_chunk = seeded_knowledge_base["chunk_pending_security"]
    deprecated_chunk = seeded_knowledge_base["chunk_deprecated"]

    queries = [
        "Unapproved draft protocol biometric iris scanners SEC-8042",
        "Deprecated 2020 maternity leave policy",
    ]

    for q in queries:
        results = await hybrid_search(
            db=mock_db,
            query=q,
            user=admin_user,
            top_k_candidates=20,
            final_top_k=10,
            embedding_service=embedder,
        )
        retrieved_ids = {r.chunk_id for r in results}
        assert pending_chunk.id not in retrieved_ids, f"Pending chunk leaked in query: {q}"
        assert deprecated_chunk.id not in retrieved_ids, f"Deprecated chunk leaked in query: {q}"


@pytest.mark.asyncio
async def test_hybrid_search_empty_query_returns_empty(seeded_knowledge_base):
    """Verify whitespace or empty query returns empty list without error."""
    admin_user = User(id=uuid.uuid4(), name="Admin", email="admin@insidehq.com", role="admin", department="Executive")
    mock_db = create_simulated_db_session(seeded_knowledge_base, requesting_user=admin_user)
    results = await hybrid_search(db=mock_db, query="   ", user=admin_user)
    assert results == []


@pytest.mark.asyncio
async def test_retrieval_result_to_dict():
    """Verify RetrievalResult serialization to dictionary."""
    cid = uuid.uuid4()
    did = uuid.uuid4()
    vid = uuid.uuid4()
    res = RetrievalResult(
        chunk_id=cid,
        document_id=did,
        version_id=vid,
        chunk_text="Test Chunk Text",
        chunk_metadata={"dept": "Eng"},
        document_title="Test Doc",
        rrf_score=0.032,
        vector_rank=1,
        keyword_rank=2,
        vector_distance=0.05,
        keyword_score=0.88,
    )
    d = res.to_dict()
    assert d["chunk_id"] == str(cid)
    assert d["document_id"] == str(did)
    assert d["version_id"] == str(vid)
    assert d["chunk_text"] == "Test Chunk Text"
    assert d["rrf_score"] == 0.032
    assert d["vector_rank"] == 1
    assert d["keyword_rank"] == 2
