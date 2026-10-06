"""Tests for Content Approval & Curation Workflow (Feature 7).

Covers:
    - GET /api/v1/curation/pending: lists pending documents awaiting approval.
    - POST /api/v1/curation/{document_id}/approve: approves document, records approved_by + approved_at, flips status to 'approved'.
    - POST /api/v1/curation/{document_id}/reject: rejects document with mandatory reason (empty reason returns 400).
    - POST /api/v1/curation/{document_id}/deprecate: deprecates document (status='deprecated').
    - Role restrictions: only 'admin' and 'curator' roles can call curation endpoints; regular employees get 403 Forbidden.
    - Retrieval exclusion gate: confirms pending, rejected, and deprecated documents are never eligible for retrieval search.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_current_user, get_db
from app.core.security import create_access_token
from app.main import app
from app.models.document import Document, DocumentACL, DocumentChunk, DocumentVersion
from app.models.user import User
from app.services.curation import (
    approve_document,
    deprecate_document,
    is_document_searchable,
    list_documents_by_status,
    list_pending_documents,
    reject_document,
)

BASE = "/api/v1/curation"


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------


def make_test_user(
    role: str = "curator",
    email: str = "curator@insidehq.com",
    name: str = "Content Curator",
    department: str = "Knowledge Ops",
) -> User:
    """Create a mock User instance."""
    return User(
        id=uuid.uuid4(),
        name=name,
        email=email,
        role=role,
        department=department,
        is_active=True,
    )


def make_test_document(
    title: str = "Security Handbook",
    status: str = "pending",
    source_type: str = "pdf",
    department: str = "Security",
    allowed_roles: Optional[List[str]] = None,
) -> Document:
    """Create a mock Document instance with version, ACLs, and chunks."""
    doc_id = uuid.uuid4()
    ver_id = uuid.uuid4()
    roles = allowed_roles or ["employee", "admin"]

    doc = Document(
        id=doc_id,
        title=title,
        source_type=source_type,
        status=status,
        created_at=datetime.now(timezone.utc),
    )
    doc.versions = [
        DocumentVersion(
            id=ver_id,
            document_id=doc_id,
            version_number=1,
            is_current=True,
        )
    ]
    doc.acls = [
        DocumentACL(
            id=uuid.uuid4(),
            document_id=doc_id,
            allowed_role=r,
            permission_type="read",
        )
        for r in roles
    ]
    doc.chunks = [
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            version_id=ver_id,
            chunk_text=f"Chunk content for {title}",
            embedding=[0.1] * 1024,
            chunk_metadata={"department": department, "filename": f"{title.lower().replace(' ', '_')}.pdf"},
        )
    ]
    return doc


# ---------------------------------------------------------------------------
# Service Unit Tests
# ---------------------------------------------------------------------------


def test_is_document_searchable_gate():
    """Verify that only 'approved' status is eligible for retrieval search."""
    # Pending documents must NEVER be searchable
    assert is_document_searchable("pending") is False
    assert is_document_searchable(make_test_document(status="pending")) is False

    # Rejected documents must NEVER be searchable
    assert is_document_searchable("rejected") is False
    assert is_document_searchable(make_test_document(status="rejected")) is False

    # Deprecated documents must NEVER be searchable
    assert is_document_searchable("deprecated") is False
    assert is_document_searchable(make_test_document(status="deprecated")) is False

    # Approved documents ARE searchable
    assert is_document_searchable("approved") is True
    assert is_document_searchable(make_test_document(status="approved")) is True


@pytest.mark.asyncio
async def test_service_approve_document():
    """Verify approve_document flips status to 'approved' and records approver metadata."""
    doc = make_test_document(status="pending")
    curator = make_test_user(role="curator")

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = doc
    mock_db.execute.return_value = mock_result

    updated = await approve_document(db=mock_db, document_id=doc.id, approver=curator)

    assert updated.status == "approved"
    assert updated.approved_by == curator.id
    assert updated.approved_at is not None
    assert is_document_searchable(updated) is True
    assert mock_db.commit.called


@pytest.mark.asyncio
async def test_service_reject_document_requires_reason():
    """Verify reject_document sets status to 'rejected' and enforces non-empty reason."""
    doc = make_test_document(status="pending")
    curator = make_test_user(role="curator")
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = doc
    mock_db.execute.return_value = mock_result

    # Empty reason should raise 400
    with pytest.raises(Exception) as exc_info:
        await reject_document(db=mock_db, document_id=doc.id, reviewer=curator, reason="")
    assert "reason" in str(exc_info.value).lower()

    # Valid reason succeeds
    updated = await reject_document(
        db=mock_db,
        document_id=doc.id,
        reviewer=curator,
        reason="Outdated policy guidelines, needs revision.",
    )
    assert updated.status == "rejected"
    assert is_document_searchable(updated) is False
    assert mock_db.commit.called


@pytest.mark.asyncio
async def test_service_deprecate_document():
    """Verify deprecate_document sets status to 'deprecated'."""
    doc = make_test_document(status="approved")
    admin = make_test_user(role="admin")
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = doc
    mock_db.execute.return_value = mock_result

    updated = await deprecate_document(db=mock_db, document_id=doc.id, reviewer=admin)
    assert updated.status == "deprecated"
    assert is_document_searchable(updated) is False
    assert mock_db.commit.called


# ---------------------------------------------------------------------------
# API Route & Role Enforcement Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_pending_documents_as_curator():
    """Curator can list pending documents."""
    curator = make_test_user(role="curator")
    pending_doc = make_test_document(title="Remote Policy 2026", status="pending")

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [pending_doc]
    mock_db.execute.return_value = mock_result

    app.dependency_overrides[get_current_user] = lambda: curator
    app.dependency_overrides[get_db] = lambda: mock_db

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get(f"{BASE}/pending", headers={"Authorization": "Bearer fake"})

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["title"] == "Remote Policy 2026"
        assert data[0]["status"] == "pending"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_approve_document_endpoint():
    """Admin or Curator can approve a document."""
    admin = make_test_user(role="admin")
    doc = make_test_document(status="pending")

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = doc
    mock_db.execute.return_value = mock_result

    app.dependency_overrides[get_current_user] = lambda: admin
    app.dependency_overrides[get_db] = lambda: mock_db

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"{BASE}/{doc.id}/approve",
                headers={"Authorization": "Bearer fake"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "approved"
        assert data["document_id"] == str(doc.id)
        assert data["approved_by"] == str(admin.id)
        assert data["approved_at"] is not None
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_reject_document_endpoint():
    """Curator can reject a document with a reason."""
    curator = make_test_user(role="curator")
    doc = make_test_document(status="pending")

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = doc
    mock_db.execute.return_value = mock_result

    app.dependency_overrides[get_current_user] = lambda: curator
    app.dependency_overrides[get_db] = lambda: mock_db

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"{BASE}/{doc.id}/reject",
                json={"reason": "Contains confidential customer PII"},
                headers={"Authorization": "Bearer fake"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "rejected"
        assert "confidential" in data["message"].lower()
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_reject_document_empty_reason_rejected():
    """Rejecting without a reason returns 422 or 400 Bad Request."""
    curator = make_test_user(role="curator")
    doc = make_test_document(status="pending")

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = doc
    mock_db.execute.return_value = mock_result

    app.dependency_overrides[get_current_user] = lambda: curator
    app.dependency_overrides[get_db] = lambda: mock_db

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"{BASE}/{doc.id}/reject",
                json={"reason": ""},
                headers={"Authorization": "Bearer fake"},
            )

        assert resp.status_code in (400, 422)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_deprecate_document_endpoint():
    """Curator can deprecate an approved document."""
    curator = make_test_user(role="curator")
    doc = make_test_document(status="approved")

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = doc
    mock_db.execute.return_value = mock_result

    app.dependency_overrides[get_current_user] = lambda: curator
    app.dependency_overrides[get_db] = lambda: mock_db

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"{BASE}/{doc.id}/deprecate",
                json={"reason": "Superceded by 2026 version"},
                headers={"Authorization": "Bearer fake"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deprecated"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_regular_employee_forbidden_from_curation():
    """Regular employee (non-admin, non-curator) must receive 403 Forbidden."""
    employee = make_test_user(role="employee", email="bob.employee@insidehq.com")
    doc_id = uuid.uuid4()

    app.dependency_overrides[get_current_user] = lambda: employee

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp_list = await client.get(f"{BASE}/pending", headers={"Authorization": "Bearer fake"})
            resp_approve = await client.post(f"{BASE}/{doc_id}/approve", headers={"Authorization": "Bearer fake"})
            resp_reject = await client.post(
                f"{BASE}/{doc_id}/reject",
                json={"reason": "test"},
                headers={"Authorization": "Bearer fake"},
            )
            resp_deprecate = await client.post(f"{BASE}/{doc_id}/deprecate", headers={"Authorization": "Bearer fake"})

        assert resp_list.status_code == 403
        assert resp_approve.status_code == 403
        assert resp_reject.status_code == 403
        assert resp_deprecate.status_code == 403
    finally:
        app.dependency_overrides.clear()
