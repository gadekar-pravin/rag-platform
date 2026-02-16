# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG Platform is a standalone, multi-tenant Retrieval-Augmented Generation service. It exposes a hybrid search API (vector + full-text with Reciprocal Rank Fusion) and an MCP server for VS Code Copilot integration. Originally extracted from ApexFlow v2, it shares the same AlloyDB instance but uses independent `rag_*` tables with its own Alembic migration chain.

**Why separate?** Data engineers need document search through VS Code Copilot without access to the ApexFlow app. The RAG service has its own auth (Cloud Run OIDC, not Firebase), multi-tenant Row-Level Security (not single-tenant `user_id` scoping), and no dependency on ApexFlow's `ServiceRegistry` or `settings.json`.

**Current state:** MVP + ingestion + production hardening complete. Schema + retrieval store + HTTP API + MCP server + GCS batch ingestion pipeline are implemented and hardened with rate limiting, request body size limits, structured JSON logging, request ID correlation, and health check improvements. Documents can be loaded via the `POST /v1/index` endpoint, direct DB inserts, or the GCS ingestion CLI (`python -m rag_service.ingestion.main`).

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

# Install ingestion dependencies (GCS, DocAI, extractors)
pip install -e ".[ingestion]"

# Run tests
pytest tests/ -v                                  # full suite
pytest tests/unit/ -v                             # unit tests only (no DB needed)
pytest tests/integration/ -v                      # integration tests (requires AlloyDB)
pytest tests/unit/test_auth.py -v                 # single file
pytest tests/unit/test_auth.py::test_func -v      # single test function
pytest tests/unit/test_search_store.py::TestClass::test_method -v  # method in class

# Retrieval evaluation
pytest tests/eval/ -v                             # full eval suite
pytest tests/eval/test_retrieval_metrics.py -v    # metrics unit tests (no DB)
pytest tests/eval/ -v -k synthetic                # synthetic embeddings (DB, no API key)
pytest tests/eval/ -v -k real                     # real Gemini embeddings (DB + API key/ADC)

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

# Run the GCS ingestion pipeline (requires GEMINI_API_KEY + DB)
RAG_INGEST_INPUT_BUCKET=my-bucket python -m rag_service.ingestion.main --tenant=acme-corp
RAG_INGEST_INPUT_BUCKET=my-bucket python -m rag_service.ingestion.main --tenant=acme-corp --dry-run

# Docker
docker build -f rag_service/Dockerfile -t rag-service:local .
docker build -f rag_mcp/Dockerfile -t rag-mcp:local .
docker build -f rag_service/Dockerfile.ingestor -t rag-ingestor:local .
```

## Testing

- **`asyncio_mode = "auto"`** in pyproject.toml — async test functions work without `@pytest.mark.asyncio`.
- **`asyncio_default_test_loop_scope = "session"`** — all tests share a single event loop, required because `db_pool` is session-scoped. Without this, asyncpg connections created in the session loop fail with "Future attached to a different loop" in per-function loops.
- **Integration tests gracefully skip** when the database is unavailable (`pytest.skip()` on connection failure, not a hard error).
- **Fixture scoping:** `db_pool` is session-scoped (one pool per test run). `clean_tables` is function-scoped (TRUNCATE CASCADE before each test). `rls_conn` wraps each test in a transaction with `SET LOCAL` so RLS is active.
- **`DATABASE_TEST_URL`** env var overrides the test DB connection (defaults to `postgresql://apexflow:apexflow@localhost:5432/apexflow`).
- **SSH tunnel required:** AlloyDB Omni runs on a GCP VM (`alloydb-omni-dev`), not locally. Before running integration tests, open an IAP tunnel to forward port 5432:
  ```bash
  gcloud compute ssh alloydb-omni-dev --zone=us-central1-a --tunnel-through-iap -- -L 5432:localhost:5432
  ```

## Architecture

### Two Services

The platform consists of two Cloud Run services and one Cloud Run Job:

1. **RAG Service** (`rag_service/`) — FastAPI app that owns the database, embedding pipeline, and search logic. Authenticated via Cloud Run OIDC or a shared dev token.
2. **MCP Server** (`rag_mcp/`) — Lightweight proxy that exposes `search` and `list_documents` MCP tools over streamable HTTP transport. Calls the RAG Service over HTTP. VS Code Copilot connects here.
3. **Ingestion Runner** (`rag_service/ingestion/`) — Batch job that imports documents from GCS, extracts text (with OCR fallback via Document AI), chunks, embeds, and stores in AlloyDB. Runs as a Cloud Run Job or locally via CLI.

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
| `rag_ingestion_runs` | Batch ingestion tracking (run status, file counts) |
| `rag_ingestion_items` | Per-file ingestion status within a run |

**3-table design rationale:** Documents → chunks → embeddings separation means you can re-embed (e.g., when upgrading from `gemini-embedding-001` to a future model) without re-chunking, since chunking is the expensive LLM-driven step for semantic mode.

### Row-Level Security (RLS)

Multi-tenant isolation is enforced at the PostgreSQL level via `FORCE ROW LEVEL SECURITY` on `rag_documents`. The application sets `app.tenant_id` and `app.user_id` per-transaction via `set_config(name, value, is_local=true)` in `rls_connection()` (`db.py`). This is the parameterized equivalent of `SET LOCAL` — PostgreSQL's `SET` command does not support `$1` placeholders.

**Visibility rules:**
- `TEAM` docs: visible to all users in the same tenant
- `PRIVATE` docs: visible only to `owner_user_id`
- Soft-deleted docs (`deleted_at IS NOT NULL`): invisible to all

**CHECK constraint:** `TEAM` docs must have `owner_user_id IS NULL`; `PRIVATE` docs must have `owner_user_id IS NOT NULL`.

**Dedup indexes (3 partial unique indexes):**
- `ux_rag_docs_team_source_uri` — TEAM docs with `source_uri`: one canonical doc per `(tenant_id, source_uri)`
- `ux_rag_docs_team_dedup_adhoc` — TEAM ad-hoc docs (no `source_uri`): dedup by `(tenant_id, content_hash)`
- `ux_rag_docs_private_dedup` — PRIVATE docs: dedup by `(tenant_id, owner_user_id, content_hash)`

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

**Threading:** Synchronous Gemini SDK calls run via `loop.run_in_executor()` to avoid blocking the async event loop. Client is `@lru_cache`d.

**Post-processing:** Embeddings are L2-normalized before storage.

**Dim guard:** Raises `ValueError` if the returned vector dimension doesn't match `RAG_EMBEDDING_DIM`. No zero-vector fallback — fails loudly to prevent corrupted index data.

**Task types:** `RETRIEVAL_DOCUMENT` for indexing, `RETRIEVAL_QUERY` for search.

### Chunking

`rag_service/chunking/chunker.py` — Ported from ApexFlow's `core/rag/chunker.py`. Two strategies:
1. **Rule-based (default):** Recursive hierarchical splitting by paragraphs → lines → sentences → words → characters, with inter-chunk overlap.
2. **Semantic:** LLM-driven topic-shift detection via Gemini.

Accepts parameters directly (no `settings.json` dependency). Defaults: 2000 chars chunk size, 200 chars overlap.

Two entry points:
- `chunk_document()` — returns `list[str]` (text only)
- `chunk_document_with_spans()` — returns `list[tuple[str, int|None, int|None]]` (text, start_char, end_char). Offsets are computed for rule-based mode; `None` for semantic mode.

### Request Flow: Auth Middleware → Identity

`rag_service/auth.py` — Two modes:
1. **Cloud Run OIDC:** Verifies Google identity tokens via `google.oauth2.id_token.verify_token()`. Extracts `email` or `sub` claim as `user_id`.
2. **Shared bearer token:** `RAG_SHARED_TOKEN` env var for local dev. Disabled automatically when `K_SERVICE` is set (Cloud Run safety).

Token is extracted from `Authorization: Bearer <token>` header or `?token=` query param (fallback for SSE clients).

Returns an `Identity` dataclass with `tenant_id` (from `TENANT_ID` env var), `user_id`, and `principal`.

**Middleware flow:** Auth middleware in `app.py` calls `get_identity()`, stores result in `request.state.identity`. Endpoint dependencies use `_get_identity(request)` to retrieve it. Every data endpoint then uses `rls_connection(identity.tenant_id, identity.user_id)` to scope DB access.

Public paths that skip auth: `/liveness`, `/readiness`, `/docs`, `/openapi.json`.

### MCP Server

`rag_mcp/server.py` — Uses `mcp` Python SDK's `FastMCP` with streamable HTTP transport. Exposes two tools:
- `search(query, limit)` → forwards to `POST /v1/search`
- `list_documents(limit, offset)` → forwards to `GET /v1/documents`

Results are formatted as concise text for LLM consumption (titles, scores, truncated chunks).

**Auth to RAG Service** (`rag_mcp/oidc.py` + `tools.py`): Token priority in `_get_headers()`:
1. **OIDC service token** (Cloud Run) — auto-minted via GCE metadata server using the MCP service account, targeting `RAG_SERVICE_URL` as audience. Cached and refreshed 5 min before expiry. Uses stdlib `urllib.request` (no extra dependencies).
2. **Caller-forwarded token** — when `RAG_MCP_FORWARD_CALLER_TOKEN=true` (default) and a caller token is available.
3. **Static `RAG_MCP_TOKEN`** — local dev fallback.

Cloud Run detection: `K_SERVICE` env var (set automatically by Cloud Run).

### Ingestion Pipeline

`rag_service/ingestion/` — Batch import from GCS with text extraction, OCR fallback, chunking, and embedding.

**Flow:** CLI (`main.py`) → discover tenants → `IngestionRunner.run_tenant()` → per-file: download → extract → chunk → embed → `upsert_document_by_source_uri()`.

**Extractors:** Text, HTML (BeautifulSoup), DOCX (python-docx), PDF (pypdf + Document AI OCR fallback), Image (Document AI online OCR).

**Incremental mode:** `compute_source_hash()` builds a change marker from GCS metadata (generation, md5, crc32c, size, updated). If unchanged, the file is skipped.

**Document store:** `upsert_document_by_source_uri()` — canonical TEAM upsert keyed by `(tenant_id, source_uri)`. Atomically replaces chunks + embeddings on content change. Returns `unchanged` if content and settings haven't changed.

## Project Layout

```
rag_service/              # Core RAG API service
  app.py                  # FastAPI entry point with lifespan, middleware, endpoints
  auth.py                 # Cloud Run OIDC + shared dev token
  config.py               # All env-var-driven configuration
  db.py                   # Asyncpg pool + rls_connection() context manager
  embedding.py            # Gemini embedding with dim guard + health check
  logging_config.py       # Structured JSON logging (GCP severity mapping)
  models.py               # Pydantic request/response schemas (with input limits)
  Dockerfile              # Multi-stage build (API server) + HEALTHCHECK
  Dockerfile.ingestor     # Multi-stage build (batch ingestion job)
  chunking/
    chunker.py            # Rule-based + semantic chunking, with optional span offsets
  stores/
    rag_document_store.py # CRUD: upsert (ad-hoc + source_uri), list, get, soft-delete
    rag_search_store.py   # Hybrid search: 3-table join, RRF, best-chunks
  eval/
    metrics.py            # IR metrics: Precision@K, Recall@K, NDCG@K, MRR, Hit Rate
  ingestion/              # GCS batch ingestion pipeline
    main.py               # CLI entry point (python -m rag_service.ingestion.main)
    cli.py                # Argument parser (--tenant, --dry-run, --force, etc.)
    config.py             # IngestConfig from env vars
    runner.py             # IngestionRunner: orchestrates extract → chunk → embed → store
    planner.py            # discover_work_items, compute_source_hash, derive_doc_type
    gcs.py                # GCS utilities (list, download, upload)
    types.py              # WorkItem, ExtractResult, ProcessResult dataclasses
    extractors/
      base.py             # Extractor ABC + normalize_text()
      text.py             # .txt/.md extractor
      html.py             # .html/.htm extractor (BeautifulSoup)
      docx.py             # .docx extractor (python-docx)
      pdf.py              # .pdf extractor (pypdf + OCR fallback)
      image.py            # Image extractor (Document AI online OCR)
    ocr/
      document_ai.py      # Document AI client (online + batch OCR)

rag_mcp/                  # MCP server for VS Code Copilot
  server.py               # FastMCP with streamable HTTP transport
  tools.py                # rag_search + rag_list_documents (retry + sanitized errors)
  oidc.py                 # OIDC token minting via GCE metadata server (Cloud Run)
  config.py               # RAG_SERVICE_URL, auth config
  Dockerfile              # Lightweight image + HEALTHCHECK + pinned deps

alembic/                  # Database migrations (independent chain)
  alembic.ini             # Uses version_table=rag_alembic_version (avoids ApexFlow collision)
  env.py                  # 3-priority connection logic (psycopg2) + version_table
  versions/
    001_rag_tables.py     # 5 tables + RLS policies + 3 dedup indexes

docs/
  deployment.md           # Production deployment guide (Cloud Run + AlloyDB)
  alloy_db_manual_ingestion_implementation_plan_v_1.md  # Full ingestion plan

tests/
  fixtures/
    ingestion/            # Real file fixtures for extractor tests (txt, html, md, etc.)
  unit/                   # Mock-based, no DB required
    test_search_store.py  # RRF math, SQL params, empty results
    test_embedding.py     # Dim guard, task types, GCP detection
    test_auth.py          # OIDC, shared token safety, public paths
    test_chunker.py       # Edge cases, overlap, splitting
    test_ingestion.py     # Source hash, extractors, planner, config, normalize_text
    test_extractors.py    # Real-file extractor tests (text, html, docx, pdf) with fixtures
    test_endpoints.py     # FastAPI endpoints, middleware, auth, body size, rate limits
    test_mcp_oidc.py      # OIDC token minting, caching, refresh, stale fallback
    test_mcp_tools.py     # MCP token priority (OIDC > caller > static), context extraction
  integration/            # Requires AlloyDB (gracefully skips when unavailable)
    test_rls.py           # FORCE RLS, tenant isolation, PRIVATE visibility
    test_dedup.py         # COALESCE NULL, cascade, per-owner dedup
    test_hybrid_search.py # Vector ranking, FTS, RRF fusion, best-chunks
    test_ingestion_dedup.py  # GCS canonical upsert, unchanged skip, atomicity
    test_ingestion_runner.py # Runner E2E: extract→chunk→store, run tracking, incremental skip
  eval/                     # Retrieval quality evaluation
    eval_dataset.json       # 15 ground-truth queries with graded relevance (0-3)
    conftest.py             # DB pool, synthetic/real seeding fixtures
    test_retrieval_metrics.py  # Unit tests for IR metric functions (no DB)
    test_retrieval_quality.py  # Integration eval: synthetic + real Gemini embeddings

scripts/
  create-scann-indexes.sql  # ScaNN index (AlloyDB only, run after data)
  seed-dev-data.py          # Load sample documents for local dev

.github/
  workflows/
    ci.yml                  # Lint + type check + unit tests

.vscode/
  mcp.json                  # Example MCP config for data engineers

.env.example                # Template for required environment variables
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
| `RAG_INGEST_INPUT_BUCKET` | GCS bucket for ingestion source documents | (required for ingestion) |
| `RAG_INGEST_INPUT_PREFIX` | Prefix under each tenant directory | `incoming/` |
| `RAG_INGEST_TENANTS` | Comma-separated tenant allowlist | unset (all tenants) |
| `RAG_INGEST_INCREMENTAL` | Skip unchanged files by source hash | `true` |
| `RAG_INGEST_FORCE_REINDEX` | Force re-process all files | `false` |
| `RAG_INGEST_MAX_FILE_WORKERS` | Concurrent file processing workers | `3` |
| `RAG_INGEST_MAX_RETRIES_PER_FILE` | Retry attempts per failed file | `2` |
| `RAG_INGEST_OUTPUT_BUCKET` | GCS bucket for extracted text artifacts | unset |
| `RAG_INGEST_OUTPUT_PREFIX` | Prefix for output artifacts | `rag-extracted/` |
| `RAG_MAX_CONTENT_CHARS` | Truncate documents exceeding this length | `2000000` |
| `RAG_OCR_ENABLED` | Enable Document AI OCR for scanned PDFs/images | `true` |
| `RAG_DOC_AI_PROJECT` | GCP project for Document AI | unset |
| `RAG_DOC_AI_LOCATION` | GCP region for Document AI | unset |
| `RAG_DOC_AI_PROCESSOR_ID` | Document AI OCR processor ID | unset |
| `RAG_PDF_TEXT_PER_PAGE_MIN` | Chars/page threshold below which PDF falls back to OCR | `200` |

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
| DB pool | `core/database.py` | Added `rls_connection()` context manager with `set_config()` (not `SET LOCAL` — PostgreSQL `SET` doesn't support parameterized queries) |
| Alembic env | `alembic/env.py` | Same 3-priority logic, independent migration chain starting at 001, separate `rag_alembic_version` table |
| Search store | `core/stores/document_search.py` | **Rewritten** — 3-table join, RLS, best-chunks-per-doc, debug metrics |
| Document store | `core/stores/document_store.py` | Adapted for multi-tenant with visibility + soft delete |

Ported code may diverge from ApexFlow over time. This is accepted as the trade-off of a separate repo.

## GCP Configuration

- **Project:** `apexflow-ai`
- **AlloyDB VM:** `alloydb-omni-dev` in `us-central1-a` (shared with ApexFlow, private IP `10.128.0.3`)
- **Cloud Run (RAG service):** `rag-service` in `us-central1` (Direct VPC egress to AlloyDB)
- **Cloud Run (MCP server):** `rag-mcp` in `us-central1` (public HTTPS, no VPC needed)
- **Cloud Run Job (Ingestion):** `rag-ingest` in `us-central1` (Direct VPC egress to AlloyDB)
- **Artifact Registry:** `us-central1-docker.pkg.dev/apexflow-ai/rag`
- **GCS Ingestion Bucket:** `gs://rag-ingest-apexflow-ai`
- **Service Accounts:** `rag-service@`, `rag-mcp@`, `rag-ingest@` (least-privilege)
- **Secrets:** `rag-alloydb-password`, `rag-oidc-audience` in Secret Manager
- **Deployment guide:** `docs/deployment.md`
- **User guide:** `docs/vscode-mcp-setup.md`

## Security: Known Gaps

**MCP server is publicly accessible.** The `rag-mcp` Cloud Run service has `allUsers` with `roles/run.invoker`. This means anyone with the URL can call the MCP tools. The RAG service behind it is properly secured (OIDC token verification + Row-Level Security), so data access is still tenant-scoped and the MCP server cannot bypass RLS. However, the MCP endpoint itself should be restricted to team members.

**Fix (not yet applied):**
```bash
# Remove public access
gcloud run services remove-iam-policy-binding rag-mcp \
  --member="allUsers" --role="roles/run.invoker" --region=us-central1

# Grant per-user access
gcloud run services add-iam-policy-binding rag-mcp \
  --member="user:engineer@example.com" --role="roles/run.invoker" --region=us-central1
```

After this, users must generate an OIDC token with `gcloud auth print-identity-token --audiences=MCP_URL` to connect from VS Code.

## Future Roadmap

- [ ] **Lock down MCP server IAM** — Remove `allUsers`, grant `roles/run.invoker` to individual team members only.
- [ ] **Automated token refresh** — OIDC tokens expire after ~1 hour. Investigate VS Code MCP client support for automatic refresh or longer-lived credentials.
- [ ] **Per-user identity passthrough** — The MCP server currently authenticates to the RAG service using its service account (`rag-mcp@`), so all MCP users share the same tenant/user context. Pass through individual user identity for per-user RLS scoping.
