"""Permission evaluation and SQL filter generation for permission-aware retrieval (Feature 9).

Ensures safety-critical access control: chunks are filtered directly within database queries
(via SQL subqueries/EXISTS clauses on document_acl) before any content reaches memory or the LLM.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Set
import uuid

from sqlalchemy import func, or_, select, true
from sqlalchemy.sql.elements import BinaryExpression, BooleanClauseList, ColumnElement

from app.models.document import Document, DocumentACL, DocumentChunk
from app.models.user import User

logger = logging.getLogger(__name__)

# Roles that are universally readable by all active authenticated enterprise employees
GENERAL_ROLES = {"public", "all", "employee", "*"}


def get_user_permission_tokens(user: User) -> Set[str]:
    """Extract normalized permission tokens for a given user.

    Includes the user's role, department name, and universal general access tokens.

    Args:
        user: The authenticated User object.

    Returns:
        Set of lowercase permission tokens matching the user's identity.
    """
    tokens = set(GENERAL_ROLES)
    if user.role:
        tokens.add(user.role.strip().lower())
    if user.department:
        tokens.add(user.department.strip().lower())
    return tokens


def build_permission_filter(
    user: User,
    document_id_col: Any = DocumentChunk.document_id,
) -> ColumnElement[bool]:
    """Construct a SQLAlchemy SQL filter enforcing ACL permissions directly in the database.

    The filter guarantees that a document chunk is only selected if:
    1. The user has the 'admin' role, OR
    2. An ACL entry exists in `document_acl` for the chunk's parent document where:
       - `allowed_user == user.id`, OR
       - `lower(allowed_role)` matches the user's role, department, or a general access role.

    Args:
        user: The current authenticated User.
        document_id_col: SQLAlchemy column representing the document ID (default DocumentChunk.document_id).

    Returns:
        SQLAlchemy boolean clause to be added to query .where() conditions.
    """
    # Administrators have global read access to all approved enterprise documents
    if user.role and user.role.strip().lower() == "admin":
        return true()

    tokens = list(get_user_permission_tokens(user))

    # EXISTS subquery against document_acl matching user ID or permitted roles/departments
    acl_subquery = (
        select(1)
        .select_from(DocumentACL)
        .where(
            DocumentACL.document_id == document_id_col,
            or_(
                DocumentACL.allowed_user == user.id,
                func.lower(DocumentACL.allowed_role).in_(tokens),
            ),
        )
        .exists()
    )

    return acl_subquery


def has_document_access(user: User, acls: Sequence[DocumentACL]) -> bool:
    """In-memory permission check for a single document against its loaded ACL collection.

    Args:
        user: The User requesting access.
        acls: Sequence of DocumentACL records associated with the document.

    Returns:
        True if the user is authorized to view the document, False otherwise.
    """
    if user.role and user.role.strip().lower() == "admin":
        return True

    tokens = get_user_permission_tokens(user)
    for acl in acls:
        if acl.allowed_user and acl.allowed_user == user.id:
            return True
        if acl.allowed_role and acl.allowed_role.strip().lower() in tokens:
            return True

    return False
