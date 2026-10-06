"""Seed script for InsideHQ — NovaSphere company knowledge base.

This script:
1. Deletes all previously seeded documents, versions, ACLs, and chunks (preserves users).
2. Reads all Markdown files from company_knowledge_base/ folder.
3. Chunks and embeds each document using the real BAAI/bge-large-en-v1.5 model.
4. Assigns department-appropriate ACLs so permission-aware retrieval works correctly.

ACL Mapping:
    Company/     → all roles (employee, engineering, hr, finance, admin)
    HR/          → hr, admin  (and employee for handbook/benefits)
    Engineering/ → engineering, admin
    Finance/     → finance, admin
    Product/     → employee, engineering, hr, finance, admin (all)
    Security/    → engineering, admin
    Support/     → employee, engineering, hr, finance, admin (all)
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Ensure backend root is in Python path
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
KNOWLEDGE_BASE_DIR = BACKEND_DIR.parent / "company_knowledge_base"
sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import delete, text as sa_text
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.db.session import SyncSessionLocal
from app.models.document import Document, DocumentACL, DocumentChunk, DocumentVersion
from app.models.user import User
from app.services.chunking import chunk_text_blocks
from app.services.embeddings import get_embedding_service


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# ACL configuration per folder
# ---------------------------------------------------------------------------

# Roles that can read each folder's documents
FOLDER_ACL_MAP: Dict[str, List[str]] = {
    "Company":     ["employee", "engineering", "hr", "finance", "admin"],
    "HR":          ["hr", "admin"],
    "Engineering": ["engineering", "admin"],
    "Finance":     ["finance", "admin"],
    "Product":     ["employee", "engineering", "hr", "finance", "admin"],
    "Security":    ["engineering", "admin"],
    "Support":     ["employee", "engineering", "hr", "finance", "admin"],
}

# HR docs that all employees should be able to read (general policies)
EMPLOYEE_READABLE_HR_DOCS = {
    "employee_handbook.md",
    "employee_benefits.md",
    "leave_policy.md",
    "attendance_policy.md",
    "work_from_home_policy.md",
}


def get_document_title(folder: str, filename: str) -> str:
    """Convert folder/filename into a human-readable document title."""
    stem = Path(filename).stem.replace("_", " ").title()
    return f"[{folder}] {stem}"


def build_acls_for_doc(folder: str, filename: str, session: Session) -> List[dict]:
    """Determine which roles can read this document."""
    base_roles = FOLDER_ACL_MAP.get(folder, ["admin"])

    # Promote general HR documents to all employees too
    if folder == "HR" and filename in EMPLOYEE_READABLE_HR_DOCS:
        roles = list(set(base_roles + ["employee", "engineering", "finance"]))
    else:
        roles = base_roles

    return [{"allowed_role": role, "permission_type": "read"} for role in roles]


def get_approved_by_user(folder: str, session: Session) -> Optional[User]:
    """Find the appropriate approver user for a document's department."""
    role_map = {
        "HR":          "hr",
        "Engineering": "engineering",
        "Finance":     "finance",
        "Company":     "admin",
        "Product":     "admin",
        "Security":    "engineering",
        "Support":     "admin",
    }
    target_role = role_map.get(folder, "admin")
    user = session.query(User).filter_by(role=target_role).first()
    if not user:
        user = session.query(User).filter_by(role="admin").first()
    return user


def delete_old_documents(session: Session) -> None:
    """Delete all seeded documents, versions, ACLs, and chunks (keep users)."""
    print("[CLEANUP] Deleting all previously seeded documents, chunks, ACLs, and versions...")

    # Cascade deletes handle versions, acls, chunks automatically (via FK ON DELETE CASCADE)
    count = session.query(Document).count()
    session.query(Document).delete(synchronize_session=False)
    session.commit()
    print(f"  ✓ Deleted {count} document(s) (and all cascaded versions/chunks/ACLs).")


def read_markdown_file(path: Path) -> str:
    """Read a markdown file and return its text content."""
    return path.read_text(encoding="utf-8", errors="replace")


def seed_knowledge_base(session: Session) -> None:
    """Seed all documents from company_knowledge_base/ into the database."""
    embed_service = get_embedding_service()
    embed_service.load_model()
    print(f"[EMBED] Embedding model loaded: {embed_service.model_name}")

    now = datetime.now(timezone.utc)

    # Collect all markdown docs from each folder
    folders = sorted([f for f in KNOWLEDGE_BASE_DIR.iterdir() if f.is_dir()])

    total_chunks = 0
    total_docs = 0

    for folder in folders:
        folder_name = folder.name
        if folder_name not in FOLDER_ACL_MAP:
            continue  # Skip unexpected folders

        md_files = sorted(folder.glob("*.md"))
        if not md_files:
            print(f"  [SKIP] No .md files in {folder_name}/")
            continue

        print(f"\n[FOLDER] {folder_name}/ ({len(md_files)} documents)")

        for md_file in md_files:
            filename = md_file.name
            title = get_document_title(folder_name, filename)
            content = read_markdown_file(md_file)

            # Create Document record
            approver = get_approved_by_user(folder_name, session)
            doc = Document(
                title=title,
                source_type="markdown",
                status="approved",
                approved_by=approver.id if approver else None,
                approved_at=now,
            )
            session.add(doc)
            session.flush()

            # Create DocumentVersion v1
            version = DocumentVersion(
                document_id=doc.id,
                version_number=1,
                is_current=True,
            )
            session.add(version)
            session.flush()

            # Create ACL entries
            acl_entries = build_acls_for_doc(folder_name, filename, session)
            for acl_info in acl_entries:
                acl = DocumentACL(
                    document_id=doc.id,
                    allowed_role=acl_info["allowed_role"],
                    allowed_user=None,
                    permission_type=acl_info["permission_type"],
                )
                session.add(acl)
            session.flush()

            # Chunk the document text (split on double-newlines for paragraphs)
            blocks = [b.strip() for b in content.split("\n\n") if b.strip()]

            # Generate real embeddings using the BAAI BGE model
            chunks = chunk_text_blocks(
                blocks=blocks,
                document_id=doc.id,
                version_id=version.id,
                source_filename=filename,
                source_type="markdown",
                chunk_size=400,
                overlap=60,
            )

            if not chunks:
                print(f"    [WARN] No chunks produced for {filename}")
                continue

            texts = [c.text for c in chunks]
            embeddings = embed_service.embed_documents(texts)

            for text_chunk, vector in zip(chunks, embeddings):
                chunk_record = DocumentChunk(
                    document_id=doc.id,
                    version_id=version.id,
                    chunk_text=text_chunk.text,
                    embedding=vector,
                    chunk_metadata={
                        "department": folder_name,
                        "filename": filename,
                        "source_type": "markdown",
                        "position": text_chunk.position,
                        "token_count": text_chunk.token_count,
                        "section_heading": text_chunk.section_heading,
                    },
                )
                session.add(chunk_record)

            session.commit()
            total_chunks += len(chunks)
            total_docs += 1
            roles_str = ", ".join(e["allowed_role"] for e in acl_entries)
            print(f"  ✓ {filename} → {len(chunks)} chunk(s) | ACLs: [{roles_str}]")

    return total_docs, total_chunks


def main() -> None:
    session = SyncSessionLocal()
    try:
        print("=" * 70)
        print("InsideHQ — NovaSphere Knowledge Base Seeder")
        print(f"Knowledge base path: {KNOWLEDGE_BASE_DIR}")
        print("=" * 70)

        if not KNOWLEDGE_BASE_DIR.exists():
            print(f"[ERROR] Knowledge base directory not found: {KNOWLEDGE_BASE_DIR}")
            sys.exit(1)

        # Step 1: Wipe all old documents (keeps users)
        delete_old_documents(session)

        # Step 2: Seed real documents with real embeddings
        total_docs, total_chunks = seed_knowledge_base(session)

        print("\n" + "=" * 70)
        print(f"[SUCCESS] Seeded {total_docs} documents with {total_chunks} embedded chunks.")
        print("=" * 70)

    except Exception as e:
        session.rollback()
        print(f"\n[ERROR] Seeding failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
