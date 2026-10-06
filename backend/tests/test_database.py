"""Automated tests for PostgreSQL schema, pgvector, full-text search, and ACL models."""

import uuid
import pytest
from sqlalchemy import text
from app.db.session import SyncSessionLocal
from app.models.document import Document, DocumentACL, DocumentChunk, DocumentVersion
from app.models.user import User


def test_pgvector_extension_enabled():
    """Verify that pgvector extension is installed and enabled in PostgreSQL."""
    session = SyncSessionLocal()
    try:
        result = session.execute(
            text("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'")
        ).fetchone()
        assert result is not None, "pgvector extension is not installed"
        assert result[0] == "vector"
        print(f"pgvector version: {result[1]}")
    finally:
        session.close()


def test_tables_and_indexes_exist():
    """Verify that required tables and indexes exist in the public schema."""
    session = SyncSessionLocal()
    try:
        tables_res = session.execute(
            text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            )
        ).fetchall()
        tables = {row[0] for row in tables_res}
        expected_tables = {
            "users",
            "documents",
            "document_versions",
            "document_acl",
            "document_chunks",
        }
        assert expected_tables.issubset(tables), f"Missing tables: {expected_tables - tables}"

        # Verify key indexes
        indexes_res = session.execute(
            text(
                "SELECT indexname FROM pg_indexes WHERE schemaname = 'public'"
            )
        ).fetchall()
        index_names = {row[0] for row in indexes_res}
        assert "ix_document_chunks_embedding_hnsw" in index_names, "HNSW index missing"
        assert "ix_document_chunks_content_tsvector_gin" in index_names, "GIN index missing"
        assert "ix_document_acl_role_lookup" in index_names, "ACL role lookup index missing"
        assert "ix_document_acl_user_lookup" in index_names, "ACL user lookup index missing"
    finally:
        session.close()


def test_seeded_users():
    """Verify sample users exist across departments and roles."""
    session = SyncSessionLocal()
    try:
        users = session.query(User).all()
        assert len(users) >= 5
        role_map = {u.email: (u.role, u.department) for u in users}
        assert role_map["alice.admin@insidehq.com"] == ("admin", "Executive")
        assert role_map["bob.hr@insidehq.com"] == ("hr", "Human Resources")
        assert role_map["charlie.eng@insidehq.com"] == ("engineering", "Engineering")
        assert role_map["diana.fin@insidehq.com"] == ("finance", "Finance")
        assert role_map["edward.emp@insidehq.com"] == ("employee", "Operations")
    finally:
        session.close()


def test_documents_and_acls():
    """Verify documents have valid ACL entries with roles and user permissions."""
    session = SyncSessionLocal()
    try:
        docs = session.query(Document).all()
        assert len(docs) >= 5

        # Check confidential HR doc
        hr_doc = (
            session.query(Document)
            .filter(Document.title.ilike("%Executive Compensation%"))
            .first()
        )
        assert hr_doc is not None
        assert hr_doc.status == "approved"
        acl_roles = {acl.allowed_role for acl in hr_doc.acls if acl.allowed_role}
        assert "hr" in acl_roles
        assert "admin" in acl_roles
        assert "engineering" not in acl_roles
        assert "employee" not in acl_roles

        # Check Finance Doc user-specific ACL
        fin_doc = (
            session.query(Document)
            .filter(Document.title.ilike("%Internal Audit%"))
            .first()
        )
        assert fin_doc is not None
        diana = session.query(User).filter_by(email="diana.fin@insidehq.com").first()
        assert diana is not None
        acl_users = {acl.allowed_user for acl in fin_doc.acls if acl.allowed_user}
        assert diana.id in acl_users
    finally:
        session.close()


def test_vector_similarity_query():
    """Verify that vector cosine similarity (<=> operator) queries execute using pgvector."""
    session = SyncSessionLocal()
    try:
        # Fetch an existing chunk embedding
        chunk = session.query(DocumentChunk).filter(DocumentChunk.embedding.isnot(None)).first()
        assert chunk is not None
        assert len(chunk.embedding) == 1024

        # Run vector similarity search
        query_vector = chunk.embedding
        stmt = text(
            """
            SELECT id, chunk_text, (embedding <=> :vec) AS distance
            FROM document_chunks
            ORDER BY embedding <=> :vec
            LIMIT 3
            """
        )
        results = session.execute(stmt, {"vec": str(list(query_vector))}).fetchall()
        assert len(results) > 0
        # The top result should be the chunk itself with distance near 0.0
        assert results[0][0] == chunk.id
        assert abs(results[0][2]) < 1e-4
    finally:
        session.close()


def test_full_text_search_tsvector():
    """Verify that PostgreSQL content_tsvector column and GIN index support BM25/FTS queries."""
    session = SyncSessionLocal()
    try:
        # Search for 'handbook' or 'benefits'
        stmt = text(
            """
            SELECT id, chunk_text, ts_rank_cd(content_tsvector, query) as rank
            FROM document_chunks, to_tsquery('english', 'handbook | healthcare') query
            WHERE content_tsvector @@ query
            ORDER BY rank DESC
            """
        )
        matches = session.execute(stmt).fetchall()
        assert len(matches) > 0
        assert any("handbook" in row[1].lower() or "health" in row[1].lower() for row in matches)
    finally:
        session.close()


def test_acl_validation_constraints():
    """Verify that ACL check constraint prevents entries without allowed_role AND allowed_user."""
    session = SyncSessionLocal()
    try:
        doc = session.query(Document).first()
        assert doc is not None

        invalid_acl = DocumentACL(
            document_id=doc.id,
            allowed_role=None,
            allowed_user=None,
            permission_type="read",
        )
        session.add(invalid_acl)
        with pytest.raises(Exception):
            session.commit()
        session.rollback()
    finally:
        session.close()
