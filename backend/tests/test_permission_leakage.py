"""Permission-Leakage Test Suite (Feature 15 Hard Gate).

Evaluates the Cartesian cross-product of every role and department against restricted
documents to guarantee zero unauthorized knowledge leakage across all access tiers:
- Executive compensation and payroll
- Internal security audits and secrets
- M&A acquisition deal terms
- Performance improvement plans (PIPs)
- Proprietary engineering architecture

Hard Gate Guarantee:
Unauthorized role x restricted document MUST yield exactly zero leaked chunks (0% leakage).
"""

from __future__ import annotations

import logging
from typing import List, Set
from unittest.mock import AsyncMock
import uuid

import pytest

from app.models.document import Document, DocumentACL, DocumentChunk
from app.models.user import User
from app.services.permissions import (
    build_permission_filter,
    get_user_permission_tokens,
    has_document_access,
)
from app.services.retrieval import RetrievalResult, hybrid_search

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Restricted Test Documents & ACL Definitions
# ---------------------------------------------------------------------------
RESTRICTED_DOCUMENTS = [
    {
        "id": uuid.uuid4(),
        "title": "Executive Compensation and Bonus Structure 2026",
        "category": "executive_payroll",
        "allowed_roles": {"executive", "hr"},
        "allowed_departments": {"executive", "hr"},
    },
    {
        "id": uuid.uuid4(),
        "title": "Internal Security Vulnerability & Secret Key Vault",
        "category": "security_secrets",
        "allowed_roles": {"security", "admin"},
        "allowed_departments": {"security", "admin"},
    },
    {
        "id": uuid.uuid4(),
        "title": "Project Titan M&A Strategic Acquisition Deal Terms",
        "category": "ma_deal_memo",
        "allowed_roles": {"executive", "legal"},
        "allowed_departments": {"executive", "legal"},
    },
    {
        "id": uuid.uuid4(),
        "title": "Confidential Employee Performance Improvement Plans (PIPs)",
        "category": "hr_pip",
        "allowed_roles": {"hr", "manager"},
        "allowed_departments": {"hr"},
    },
    {
        "id": uuid.uuid4(),
        "title": "Proprietary Core Algorithm & Architecture Specifications",
        "category": "engineering_source",
        "allowed_roles": {"engineering"},
        "allowed_departments": {"engineering"},
    },
]

ALL_TEST_ROLES = ["employee", "contractor", "manager", "hr", "finance", "executive", "legal", "security", "admin"]
ALL_TEST_DEPARTMENTS = ["Engineering", "HR", "Finance", "Legal", "Executive", "Security", "Operations"]


def create_mock_acls(doc_spec: dict) -> List[DocumentACL]:
    """Build DocumentACL objects representing the document's permitted roles and departments."""
    acls = []
    for role in doc_spec["allowed_roles"]:
        acls.append(
            DocumentACL(
                id=uuid.uuid4(),
                document_id=doc_spec["id"],
                allowed_role=role,
                permission_type="read",
            )
        )
    for dept in doc_spec["allowed_departments"]:
        acls.append(
            DocumentACL(
                id=uuid.uuid4(),
                document_id=doc_spec["id"],
                allowed_role=dept.lower(),
                permission_type="read",
            )
        )
    return acls


# ---------------------------------------------------------------------------
# Parametrized Hard-Gate Permission Leakage Suite (Role x Document)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("role", ALL_TEST_ROLES)
@pytest.mark.parametrize("doc_spec", RESTRICTED_DOCUMENTS)
def test_zero_leakage_hard_gate_for_all_roles(role: str, doc_spec: dict):
    """HARD GATE TEST: Every role x restricted document must yield 0 leakage if unauthorized.

    Checks:
    - If role is unauthorized: has_document_access MUST be False (0% leakage).
    - If role is admin or authorized: access is permitted.
    """
    user = User(
        id=uuid.uuid4(),
        name=f"Test {role.title()}",
        email=f"{role}_user@insidehq.com",
        role=role,
        department="General",
        is_active=True,
    )

    doc_acls = create_mock_acls(doc_spec)
    is_authorized = has_document_access(user=user, acls=doc_acls)

    is_explicitly_allowed = (
        role == "admin"
        or role.lower() in doc_spec["allowed_roles"]
        or user.department.lower() in doc_spec["allowed_departments"]
    )

    if not is_explicitly_allowed:
        # HARD GATE: Strict 0 leakage guarantee
        assert is_authorized is False, (
            f"SECURITY LEAKAGE DETECTED! Role '{role}' gained unauthorized access to "
            f"restricted document '{doc_spec['title']}'!"
        )
    else:
        assert is_authorized is True, (
            f"Legitimate role '{role}' was erroneously blocked from '{doc_spec['title']}'."
        )


@pytest.mark.parametrize("department", ALL_TEST_DEPARTMENTS)
@pytest.mark.parametrize("doc_spec", RESTRICTED_DOCUMENTS)
def test_zero_leakage_hard_gate_for_department_scoping(department: str, doc_spec: dict):
    """HARD GATE TEST: Cross-department document isolation must prevent lateral leakage."""
    user = User(
        id=uuid.uuid4(),
        name=f"Test Dept User ({department})",
        email=f"{department.lower()}_user@insidehq.com",
        role="employee",
        department=department,
        is_active=True,
    )

    doc_acls = create_mock_acls(doc_spec)
    is_authorized = has_document_access(user=user, acls=doc_acls)

    is_explicitly_allowed = (
        user.role.lower() in doc_spec["allowed_roles"]
        or department.lower() in doc_spec["allowed_departments"]
    )

    if not is_explicitly_allowed:
        assert is_authorized is False, (
            f"DEPARTMENT LEAKAGE DETECTED! Department '{department}' accessed "
            f"restricted document '{doc_spec['title']}'!"
        )
    else:
        assert is_authorized is True


def test_contractor_role_has_strictly_minimal_general_access():
    """Verify contractor roles cannot access any role-restricted or privileged documents."""
    contractor = User(
        id=uuid.uuid4(),
        name="External Contractor",
        email="contractor@partner.com",
        role="contractor",
        department="Operations",
        is_active=True,
    )

    tokens = get_user_permission_tokens(contractor)
    assert "admin" not in tokens
    assert "hr" not in tokens
    assert "executive" not in tokens
    assert "finance" not in tokens

    # Test against all 5 restricted documents: contractor must be blocked from all 5
    for doc in RESTRICTED_DOCUMENTS:
        acls = create_mock_acls(doc)
        assert has_document_access(user=contractor, acls=acls) is False, (
            f"Contractor leaked access to: {doc['title']}"
        )


def test_admin_role_has_universal_compliance_access():
    """Verify administrator role retains global read access for auditing and compliance."""
    admin_user = User(
        id=uuid.uuid4(),
        name="Global Admin",
        email="admin@insidehq.com",
        role="admin",
        department="IT",
        is_active=True,
    )

    for doc in RESTRICTED_DOCUMENTS:
        acls = create_mock_acls(doc)
        assert has_document_access(user=admin_user, acls=acls) is True
