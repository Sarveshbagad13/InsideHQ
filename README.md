# InsideHQ — RAG Employee Knowledge Assistant

InsideHQ is an internal, permission-aware AI knowledge assistant that enables employees to query company documentation while strictly enforcing document-level access permissions, retrieving relevant context with hybrid search (dense embeddings + BM25 full-text search), reranking with local cross-encoders, citing sources, and offering observable request tracing.

---

## 🛠️ Fixed Tech Stack

| Layer | Technology | Rationale |
| :--- | :--- | :--- |
| **Backend Framework** | FastAPI (Python 3.11+) | Async-first, automatic OpenAPI docs, Pydantic validation, native streaming support. |
| **Database** | PostgreSQL + `pgvector` | Single relational database holding users, ACLs, documents, and vector embeddings. |
| **Keyword Search** | PostgreSQL Full-Text (`tsvector`/`tsquery`) | Integrated BM25-style keyword search within Postgres without secondary engines. |
| **Embeddings** | `BAAI/bge-large-en-v1.5` (`sentence-transformers`) | Local embedding model, zero API cost, zero rate limits. |
| **Reranker** | `BAAI/bge-reranker-large` (`sentence-transformers`) | Local cross-encoder reranker for maximum precision. |
| **LLM (Generation)** | Groq (`llama-3.3-70b-versatile` via Groq API) | High-speed generative reasoning with structured citation output. |
| **Orchestration** | Custom Hand-written Python | Transparent, debuggable step-by-step pipeline without LangChain or LlamaIndex. |
| **Auth & Permissions** | Custom JWT + Postgres ACLs | `python-jose` + `passlib`, transparent database-driven role-based access control. |
| **Document Parsing** | `unstructured` | Unified multi-format document parser (PDF, DOCX, Markdown, HTML). |
| **Evaluation** | RAGAS | Automated faithfulness, answer relevance, and context precision scoring. |
| **Observability** | Langfuse | Tracing retrieval, rerank, and generation latency and token usage. |
| **Frontend Framework** | React.js (Vite + JavaScript) | Lightweight SPA paired with Tailwind CSS and Vercel AI SDK (`useChat`). |
| **Testing** | Pytest + Permission Leakage Suite | Unit tests, API tests, and access-control validation. |

---

## 📁 Repository Structure

```text
InsideHQ/
├── .env.example              # Environment variables template
├── .env                      # Local development environment configuration
├── README.md                 # Project documentation & local setup instructions
│
├── backend/                  # FastAPI Application
│   ├── app/
│   │   ├── api/              # API routes & endpoint controllers
│   │   ├── core/             # Configuration & environment variable settings
│   │   ├── db/
│   │   │   ├── session.py    # SQLAlchemy async/sync engines & get_db dependency
│   │   │   └── seed.py       # Database seed script (users, documents, ACLs, chunks)
│   │   ├── models/
│   │   │   ├── user.py       # User model (id, name, email, role, department)
│   │   │   └── document.py   # Document, DocumentVersion, DocumentACL, DocumentChunk
│   │   ├── services/         # Pipeline services (ingestion, retrieval, LLM)
│   │   └── main.py           # FastAPI entrypoint & CORS middleware
│   ├── alembic/              # Alembic migration environment
│   │   ├── versions/         # Auto-generated migration scripts
│   │   └── env.py            # Alembic env: loads models & enables vector extension
│   ├── tests/
│   │   ├── test_health.py    # Health endpoint & config tests
│   │   └── test_database.py  # Schema, pgvector, FTS, ACL, and seed data tests
│   ├── alembic.ini           # Alembic configuration file
│   ├── pytest.ini            # Pytest configuration
│   └── requirements.txt      # Python dependencies
│
└── frontend/                 # React.js (Vite + JavaScript) Application
    ├── public/               # Public assets
    ├── src/
    │   ├── App.jsx           # Main application view with health check & blueprint
    │   ├── index.css         # Tailwind CSS styling setup
    │   └── main.jsx          # React DOM entrypoint
    ├── package.json          # Frontend dependencies & scripts
    └── vite.config.js        # Vite configuration with Tailwind CSS plugin
```

---

## 🗄️ Database Schema (F2)

| Table | Purpose |
| :---- | :------ |
| `users` | Users with `role` (admin/hr/engineering/finance/employee) and `department` |
| `documents` | Document metadata with status lifecycle: `pending → approved → deprecated/rejected` |
| `document_versions` | Version tracking per document; `is_current` marks the active revision |
| `document_acl` | Access control entries per document: role-based OR user-specific + `permission_type` |
| `document_chunks` | Chunked text with `embedding vector(1024)`, `content_tsvector` (computed), and `chunk_metadata JSONB` |

### Indexes

| Index | Table | Type | Purpose |
| :---- | :---- | :--- | :------ |
| `ix_document_chunks_embedding_hnsw` | `document_chunks` | HNSW (`vector_cosine_ops`) | ANN similarity search |
| `ix_document_chunks_content_tsvector_gin` | `document_chunks` | GIN | Full-text keyword search |
| `ix_document_acl_role_lookup` | `document_acl` | B-Tree (composite) | Fast role-based access check |
| `ix_document_acl_user_lookup` | `document_acl` | B-Tree (composite) | Fast user-based access check |

---

## 🚀 Getting Started Locally

### Prerequisites
- **Python 3.11+** installed
- **Node.js 18+** and **npm** installed

---

### 1. Environment Configuration

Copy the example environment configuration into `.env` at the root directory:

```bash
# Windows PowerShell / CMD / Bash
cp .env.example .env
```

Review and adjust variables in `.env` if necessary.

---

### 2. Start Local PostgreSQL (with pgvector)

This project uses a **portable, self-contained PostgreSQL 18 + pgvector instance** stored in `.pg_local/` that runs locally on port **5434** (avoids conflict with any existing Postgres installation).

To start it before running the backend or migrations:

```bash
# Windows PowerShell - run once per session
e:\InsideHQ\.pg_local\bin\postgres.exe -D e:\InsideHQ\.pg_local\data -p 5434
```

> **Note**: The `.pg_local/` directory is git-ignored. It was initialized once with `initdb` and the `vector` extension was installed. On first setup, Alembic migrations will recreate the schema (`alembic upgrade head`).

---

### 3. Backend Setup & Run

1. Navigate to the `backend/` directory:
   ```bash
   cd backend
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1   # Windows PowerShell
   ```

3. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Apply database migrations:
   ```bash
   alembic upgrade head
   ```

5. Seed the database with sample data:
   ```bash
   python -m app.db.seed
   ```

6. Run the backend development server:
   ```bash
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```

7. Verify backend operation:
   - **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
   - **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

8. Run all backend tests:
   ```bash
   pytest
   ```

---

### 3. Frontend Setup & Run

1. Open a new terminal and navigate to the `frontend/` directory:
   ```bash
   cd frontend
   ```

2. Install frontend dependencies:
   ```bash
   npm install
   ```

3. Start the Vite development server:
   ```bash
   npm run dev
   ```

4. Open your browser at [http://localhost:5173](http://localhost:5173). The placeholder dashboard will automatically probe the backend `GET /health` endpoint and report latency and service status in real time.

5. Test production build:
   ```bash
   npm run build
   ```

---

## 🔒 Security & Local Development Rules
- **No containerization / Docker added**: In accordance with local development requirements.
- **Local credentials**: Development JWT secret and defaults are configured in `.env`. Replace secrets in production.
