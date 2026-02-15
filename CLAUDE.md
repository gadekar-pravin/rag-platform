# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG Platform is a standalone, multi-tenant Retrieval-Augmented Generation service. It exposes a hybrid search API (vector + full-text with Reciprocal Rank Fusion) and an MCP server for VS Code Copilot integration. Originally extracted from ApexFlow v2, it shares the same AlloyDB instance but uses independent `rag_*` tables with its own Alembic migration chain.

**Why separate?** Data engineers need document search through VS Code Copilot without access to the ApexFlow app. The RAG service has its own auth (Cloud Run OIDC, not Firebase), multi-tenant Row-Level Security (not single-tenant `user_id` scoping), and no dependency on ApexFlow's `ServiceRegistry` or `settings.json`.

**Current state:** MVP complete. Schema + retrieval store + HTTP API + MCP server are implemented. GCS batch ingestion job is deferred — documents are seeded via the `POST /v1/index` endpoint or direct DB inserts.

## Common Commands

```bash
# Setup venv and install dependencies (uv preferred)
uv venv .venv && source .venv/bin/activate
uv sync --extra dev

# Alternative: pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Install MCP server dependencies
pip install -e ".[mcp]"

# Run tests
pytest tests/ -v                                  # full suite
pytest tests/unit/ -v                             # unit tests only (no DB needed)
pytest tests/integration/ -v                      # integration tests (requires AlloyDB)

# Lint and format
ruff check .                                      # lint
ruff check . --fix                                # lint with auto-fix
ruff format .                                     # format

# Type check
mypy rag_service/

# Database migrations
alembic upgrade head                              # apply all migrations
alembic downgrade -1                              # rollback one migration

# Run the RAG API server (local dev)
RAG_SHARED_TOKEN=dev-token GEMINI_API_KEY=<key> uvicorn rag_service.app:app --reload

# Run the MCP server (local dev)
RAG_SERVICE_URL=http://localhost:8000 RAG_MCP_TOKEN=dev-token python -m rag_mcp.server

# Seed sample documents (requires running DB + GEMINI_API_KEY)
python scripts/seed-dev-data.py

# Docker
docker build -f rag_service/Dockerfile -t rag-service:local .
docker build -f rag_mcp/Dockerfile -t rag-mcp:local .
```

## Architecture

### Two Services

The platform consists of two independently deployable Cloud Run services:

1. **RAG Service** (`rag_service/`) — FastAPI app that owns the database, embedding pipeline, and search logic. Authenticated via Cloud Run OIDC or a shared dev token.
2. **MCP Server** (`rag_mcp/`) — Lightweight proxy that exposes `search` and `list_documents` MCP tools over streamable HTTP transport. Calls the RAG Service over HTTP. VS Code Copilot connects here.

### Database

Same AlloyDB Omni instance as ApexFlow (`alloydb-omni-dev`), but uses independent `rag_*` tables. Schema defined in `alembic/versions/001_rag_tables.py`.

**Connection priority** (`rag_service/db.py`):
1. `DATABASE_URL` env var (explicit override)
2. `K_SERVICE` detected → Cloud Run mode using `ALLOYDB_*` vars
3. Local dev → builds from `DB_HOST`/`DB_USER`/`DB_PASSWORD`/`DB_PORT`/`DB_NAME` (defaults to `localhost:5432`, user `apexflow`)

**Connection pool:** asyncpg, min_size=1, max_size=5 (configurable via `DB_POOL_MAX`). Each new connection registers the pgvector codec.

**Tables (5 total):**

| Table | Purpose |
|---|---|
| `rag_documents` | Document metadata + content + dedup hash + soft delete |
| `rag_document_chunks` | Chunk text with offsets + generated FTS tsvector column |
| `rag_chunk_embeddings` | Vector embeddings separated from chunks (enables re-embedding without re-chunking) |
| `rag_ingestion_runs` | Batch ingestion tracking (schema ready, populated when GCS job is built) |
| `rag_ingestion_items` | Per-file ingestion status within a run |

**3-table design rationale:** Documents → chunks → embeddings separation means you can re-embed (e.g., when upgrading from `gemini-embedding-001` to a future model) without re-chunking, since chunking is the expensive LLM-driven step for semantic mode.

### Row-Level Security (RLS)

Multi-tenant isolation is enforced at the PostgreSQL level via `FORCE ROW LEVEL SECURITY` on `rag_documents`. The application sets `SET LOCAL app.tenant_id` and `SET LOCAL app.user_id` per-transaction via `rls_connection()` in `db.py`.

**Visibility rules:**
- `TEAM` docs: visible to all users in the same tenant
- `PRIVATE` docs: visible only to `owner_user_id`
- Soft-deleted docs (`deleted_at IS NOT NULL`): invisible to all

**CHECK constraint:** `TEAM` docs must have `owner_user_id IS NULL`; `PRIVATE` docs must have `owner_user_id IS NOT NULL`.

**Dedup index:** `COALESCE(owner_user_id, '')` handles NULL uniqueness so two TEAM docs with the same content_hash correctly dedup.

**Fail-closed:** `rls_connection()` raises `ValueError` if `tenant_id` or `user_id` are empty. The DB role must NOT have `SUPERUSER` or `BYPASSRLS`.

### Hybrid Search

`RagSearchStore.search_hybrid()` in `rag_service/stores/rag_search_store.py` implements Reciprocal Rank Fusion (RRF) over vector cosine similarity + full-text `ts_rank`:

1. **vector_pool CTE** — 3-table JOIN, `ORDER BY embedding <=> query`, `LIMIT chunk_limit+1`
2. **text_pool CTE** — 3-table JOIN, `fts @@ plainto_tsquery`, `LIMIT chunk_limit+1`
3. **doc_vector_rrf / doc_text_rrf** — `DISTINCT ON (document_id)` best rank per doc
4. **fused CTE** — `FULL OUTER JOIN`, `rrf_score = 1/(K+rank_v) + 1/(K+rank_t)`
5. **best_chunks** — For each top doc: top 2 vector chunks + top 2 text chunks, deduped by `chunk_id`

RLS replaces `WHERE user_id =` — no tenant filtering in application SQL.

### Embedding

`rag_service/embedding.py` — Uses `gemini-embedding-001` with `output_dimensionality=768`. Auto-detects environment: Vertex AI with ADC on GCP, or `GEMINI_API_KEY` for local dev.

**Dim guard:** Raises `ValueError` if the returned vector dimension doesn't match `RAG_EMBEDDING_DIM`. No zero-vector fallback — fails loudly to prevent corrupted index data.

**Task types:** `RETRIEVAL_DOCUMENT` for indexing, `RETRIEVAL_QUERY` for search.

### Chunking

`rag_service/chunking/chunker.py` — Ported from ApexFlow's `core/rag/chunker.py`. Two strategies:
1. **Rule-based (default):** Recursive hierarchical splitting by paragraphs → lines → sentences → words → characters, with inter-chunk overlap.
2. **Semantic:** LLM-driven topic-shift detection via Gemini.

Accepts parameters directly (no `settings.json` dependency). Defaults: 2000 chars chunk size, 200 chars overlap.

### Auth

`rag_service/auth.py` — Two modes:
1. **Cloud Run OIDC:** Verifies Google identity tokens via `google.oauth2.id_token.verify_token()`. Extracts `email` or `sub` claim as `user_id`.
2. **Shared bearer token:** `RAG_SHARED_TOKEN` env var for local dev. Disabled automatically when `K_SERVICE` is set (Cloud Run safety).

Returns an `Identity` dataclass with `tenant_id` (from `TENANT_ID` env var), `user_id`, and `principal`.

Public paths that skip auth: `/liveness`, `/readiness`, `/docs`, `/openapi.json`.

### MCP Server

`rag_mcp/server.py` — Uses `mcp` Python SDK's `FastMCP` with streamable HTTP transport. Exposes two tools:
- `search(query, limit)` → forwards to `POST /v1/search`
- `list_documents(limit, offset)` → forwards to `GET /v1/documents`

Results are formatted as concise text for LLM consumption (titles, scores, truncated chunks).

## Project Layout

```
rag_service/              # Core RAG API service
  app.py                  # FastAPI entry point with lifespan, middleware, endpoints
  auth.py                 # Cloud Run OIDC + shared dev token
  config.py               # All env-var-driven configuration
  db.py                   # Asyncpg pool + rls_connection() context manager
  embedding.py            # Gemini embedding with dim guard
  models.py               # Pydantic request/response schemas
  Dockerfile              # Multi-stage build
  chunking/
    chunker.py            # Rule-based + semantic document chunking
  stores/
    rag_document_store.py # CRUD: upsert, list, get, soft-delete
    rag_search_store.py   # Hybrid search: 3-table join, RRF, best-chunks

rag_mcp/                  # MCP server for VS Code Copilot
  server.py               # FastMCP with streamable HTTP transport
  tools.py                # rag_search + rag_list_documents tool implementations
  config.py               # RAG_SERVICE_URL, auth config
  Dockerfile              # Lightweight image

alembic/                  # Database migrations (independent chain)
  env.py                  # 3-priority connection logic (psycopg2)
  versions/
    001_rag_tables.py     # 5 tables + RLS policies + indexes

tests/
  unit/                   # Mock-based, no DB required
    test_search_store.py  # RRF math, SQL params, empty results
    test_embedding.py     # Dim guard, task types, GCP detection
    test_auth.py          # OIDC, shared token safety, public paths
    test_chunker.py       # Edge cases, overlap, splitting
  integration/            # Requires AlloyDB (gracefully skips when unavailable)
    test_rls.py           # FORCE RLS, tenant isolation, PRIVATE visibility
    test_dedup.py         # COALESCE NULL, cascade, per-owner dedup
    test_hybrid_search.py # Vector ranking, FTS, RRF fusion, best-chunks

scripts/
  create-scann-indexes.sql  # ScaNN index (AlloyDB only, run after data)
  seed-dev-data.py          # Load sample documents for local dev

.vscode/
  mcp.json                  # Example MCP config for data engineers
```

## Code Conventions

- **Python 3.12+**, strict mypy
- **Ruff rules:** E, F, I, UP, B, SIM at 120-char line length
- **Primary keys:** UUID with `gen_random_uuid()` default
- **Async:** asyncpg for all DB access; psycopg2 only for Alembic migrations
- **Stores:** Stateless classes — every method takes an `asyncpg.Connection` (with RLS already set). No `get_pool()` calls inside stores.
- **Config:** All env-var-driven via `rag_service/config.py`. No `settings.json`.
- **Embedding safety:** Fail-closed dim guard. No zero-vector fallback.
- **RLS:** All data access goes through `rls_connection()`. No `WHERE user_id = $1` in application SQL.

## Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `TENANT_ID` | Fixed tenant per environment | `default` |
| `RAG_EMBEDDING_MODEL` | Gemini embedding model | `gemini-embedding-001` |
| `RAG_EMBEDDING_DIM` | Expected embedding dimension | `768` |
| `RAG_RRF_K` | RRF fusion constant | `60` |
| `RAG_SEARCH_EXPANSION` | Pool expansion factor (chunk_limit = doc_limit * expansion) | `3` |
| `RAG_INGESTION_VERSION` | Ingestion version tag | `v1` |
| `RAG_CHUNK_SIZE` | Target chunk size in characters | `2000` |
| `RAG_CHUNK_OVERLAP` | Overlap between adjacent chunks in characters | `200` |
| `RAG_SHARED_TOKEN` | Dev-only shared bearer token (ignored on Cloud Run) | unset |
| `GEMINI_API_KEY` | Gemini API key for local dev (not needed on GCP) | -- |
| `GOOGLE_CLOUD_PROJECT` | GCP project for Vertex AI | `apexflow-ai` |
| `GOOGLE_CLOUD_LOCATION` | GCP region for Vertex AI | `us-central1` |
| `DATABASE_URL` | Full database connection string (overrides all DB_* vars) | -- |
| `DB_HOST` / `DB_PORT` / `DB_USER` / `DB_PASSWORD` / `DB_NAME` | Individual DB connection params | `localhost:5432`, user `apexflow` |
| `DB_POOL_MAX` | Max async connection pool size | `5` |
| `K_SERVICE` | Auto-set by Cloud Run; triggers production mode | -- |
| `RAG_SERVICE_URL` | RAG API URL for MCP server | `http://localhost:8000` |
| `RAG_MCP_TOKEN` | Auth token for MCP → RAG API calls | unset |
| `MCP_PORT` | MCP server listen port | `8001` |
| `DATABASE_TEST_URL` | Test database URL for integration tests | `postgresql://apexflow:apexflow@localhost:5432/apexflow` |

## API Endpoints

| Method | Path | Purpose | Auth |
|---|---|---|---|
| `POST` | `/v1/search` | Hybrid search (embed query → RLS → RRF) | Required |
| `GET` | `/v1/documents` | List visible documents (TEAM + owned PRIVATE) | Required |
| `DELETE` | `/v1/documents/{id}` | Soft delete (owner-only for PRIVATE) | Required |
| `POST` | `/v1/index` | Index a document (chunk → embed → store) | Required |
| `GET` | `/liveness` | Health check | Public |
| `GET` | `/readiness` | DB connectivity check | Public |

## Relationship to ApexFlow

This is a **separate repository**, not a monorepo. Code was ported from ApexFlow with these changes:

| Component | ApexFlow Source | Changes |
|---|---|---|
| Chunker | `core/rag/chunker.py` | Removed `settings_loader` dependency; params accepted directly |
| Embedding | `remme/utils.py` + `core/gemini_client.py` | Model → `gemini-embedding-001`; dim guard replaces zero-vector fallback; client factory inlined |
| DB pool | `core/database.py` | Added `rls_connection()` context manager with `SET LOCAL` |
| Alembic env | `alembic/env.py` | Same 3-priority logic, independent migration chain starting at 001 |
| Search store | `core/stores/document_search.py` | **Rewritten** — 3-table join, RLS, best-chunks-per-doc, debug metrics |
| Document store | `core/stores/document_store.py` | Adapted for multi-tenant with visibility + soft delete |

Ported code may diverge from ApexFlow over time. This is accepted as the trade-off of a separate repo.

## GCP Configuration

- **Project:** `apexflow-ai`
- **AlloyDB VM:** `alloydb-omni-dev` in `us-central1-a` (shared with ApexFlow)
- **Cloud Run (RAG service):** Deploy target TBD
- **Cloud Run (MCP server):** Deploy target TBD
