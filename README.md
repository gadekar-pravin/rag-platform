# RAG Platform

A shared, multi-tenant Retrieval-Augmented Generation (RAG) service with an MCP server for VS Code Copilot integration.

## What is this?

RAG Platform provides hybrid document search (vector similarity + full-text) as a service, with:

- **Multi-tenant isolation** via PostgreSQL Row-Level Security (RLS) — enforced at the database level, not just application code
- **Hybrid search** using Reciprocal Rank Fusion (RRF) — combines vector cosine similarity with full-text search for better relevance than either signal alone
- **MCP server** for VS Code GitHub Copilot — data engineers can search team documents directly from their editor
- **Cloud Run OIDC auth** — independent from ApexFlow's Firebase auth, so engineers don't need access to the main app

## Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL 15+ with [pgvector](https://github.com/pgvector/pgvector) extension
- A Gemini API key (for embeddings)

### Setup

```bash
# Clone and install
git clone <repo-url> && cd rag-platform
uv venv .venv && source .venv/bin/activate
uv sync --extra dev

# Apply database migrations
alembic upgrade head

# Start the RAG API server
RAG_SHARED_TOKEN=dev-token \
GEMINI_API_KEY=<your-key> \
uvicorn rag_service.app:app --reload

# (Optional) Seed sample documents
python scripts/seed-dev-data.py
```

### Try it

```bash
# Index a document
curl -X POST http://localhost:8000/v1/index \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "API Guidelines",
    "content": "Use nouns for REST URLs. Return proper HTTP status codes. Version your API with URL paths.",
    "doc_type": "guidelines"
  }'

# Search
curl -X POST http://localhost:8000/v1/search \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{"query": "REST API best practices", "limit": 5}'

# List documents
curl http://localhost:8000/v1/documents \
  -H "Authorization: Bearer dev-token"
```

## Architecture

```
                   VS Code Copilot
                        │
                   MCP Protocol
                        │
                  ┌─────▼─────┐
                  │ MCP Server │  (Cloud Run)
                  │  rag_mcp/  │
                  └─────┬─────┘
                        │ HTTP
                  ┌─────▼─────┐
                  │ RAG Service│  (Cloud Run)
                  │rag_service/│
                  └─────┬─────┘
                        │ asyncpg + RLS
                  ┌─────▼─────┐
                  │  AlloyDB   │
                  │ rag_* tbls │
                  └────────────┘
```

### Two Services

| Service | Purpose | Port | Auth |
|---|---|---|---|
| **RAG Service** (`rag_service/`) | Owns the database, embedding pipeline, and search logic | 8000 (dev) / 8080 (prod) | Cloud Run OIDC or shared dev token |
| **MCP Server** (`rag_mcp/`) | Exposes `search` and `list_documents` tools to VS Code Copilot | 8001 | On Cloud Run: auto-mints OIDC token via metadata server. Local dev: forwards caller token or `RAG_MCP_TOKEN` |

### Database Schema

Five tables with a normalized 3-table core:

```
rag_documents          # Metadata, content, dedup hash, soft delete
    │
    ├── rag_document_chunks    # Chunk text + generated FTS tsvector
    │       │
    │       └── rag_chunk_embeddings   # Vector embeddings (separate for re-embedding)
    │
rag_ingestion_runs     # Batch ingestion tracking (run status, file counts)
    │
    └── rag_ingestion_items    # Per-file status within a run
```

### Multi-Tenant Isolation

Tenant isolation uses PostgreSQL **Row-Level Security** with `FORCE ROW LEVEL SECURITY`:

```sql
-- Every request sets these per-transaction:
SET LOCAL app.tenant_id = 'acme-corp';
SET LOCAL app.user_id = 'alice@acme.com';

-- RLS policy (simplified):
-- TEAM docs: visible to all users in the tenant
-- PRIVATE docs: visible only to the owner
-- Soft-deleted docs: invisible to all
```

This is enforced at the database level — even SQL injection cannot bypass tenant boundaries.

### Hybrid Search

Search combines two signals via [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf):

1. **Vector similarity** — cosine distance between query embedding and chunk embeddings
2. **Full-text search** — PostgreSQL `ts_rank` with `plainto_tsquery`
3. **RRF fusion** — `score = 1/(K + vector_rank) + 1/(K + text_rank)`

Each result includes the **best 4 chunks** per document (2 from vector, 2 from text, deduplicated).

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/search` | Hybrid search. Body: `{query, limit?, include_debug?}` |
| `POST` | `/v1/index` | Index a document. Body: `{title, content, doc_type?, visibility?, metadata?}` |
| `GET` | `/v1/documents` | List visible documents. Query: `?limit=50&offset=0` |
| `DELETE` | `/v1/documents/{id}` | Soft-delete a document |
| `GET` | `/liveness` | Health check (no auth) |
| `GET` | `/readiness` | DB health check (no auth) |

### Search Response

```json
{
  "results": [
    {
      "document_id": "uuid",
      "title": "API Guidelines",
      "doc_type": "guidelines",
      "rrf_score": 0.032,
      "vector_score": 0.016,
      "text_score": 0.016,
      "chunks": [
        {
          "chunk_id": "uuid",
          "chunk_index": 0,
          "chunk_text": "Use nouns for REST URLs...",
          "source": "vector",
          "score": 0.95
        }
      ]
    }
  ],
  "debug": null
}
```

## VS Code Copilot Integration

Data engineers can search documents from VS Code by configuring the MCP server.

### Setup (Current: Public `rag-mcp`)

Add to `.vscode/mcp.json` in your workspace:

```json
{
  "servers": {
    "rag-search": {
      "type": "http",
      "url": "https://rag-mcp-<hash>.a.run.app/mcp"
    }
  }
}
```

No user token is required in the current public setup.

### Optional Setup (If MCP IAM Is Locked Down)

Use this config when `rag-mcp` no longer allows unauthenticated access:

```json
{
  "servers": {
    "rag-search": {
      "type": "http",
      "url": "https://rag-mcp-<hash>.a.run.app/mcp",
      "headers": {
        "Authorization": "Bearer ${input:rag-token}"
      }
    }
  },
  "inputs": [
    {
      "id": "rag-token",
      "type": "promptString",
      "description": "RAG MCP OIDC token",
      "password": true
    }
  ]
}
```

Generate the token with service account impersonation:

```bash
gcloud auth print-identity-token \
  --impersonate-service-account=SA_NAME@apexflow-ai.iam.gserviceaccount.com \
  --audiences="https://rag-mcp-<hash>.a.run.app"
```

### Available Tools

- **`search`** — Search the team's document knowledge base by natural language query
- **`list_documents`** — Browse available documents with pagination

## Manual Ingestion from GCS

The ingestion pipeline imports documents from a GCS bucket, extracts text (with OCR fallback for scanned PDFs and images), chunks, embeds, and stores them in AlloyDB.

### GCS Bucket Layout

```
gs://<bucket>/
  <tenant_id>/
    incoming/            # default prefix (configurable)
      report.pdf
      notes.docx
      guide.html
      diagram.png
```

Supported formats: `.pdf`, `.docx`, `.html`/`.htm`, `.txt`/`.md`, `.png`/`.jpg`/`.jpeg`/`.webp`/`.tiff`

### Local Usage

```bash
# Install ingestion dependencies
pip install -e ".[ingestion]"

# Run for a specific tenant
RAG_INGEST_INPUT_BUCKET=my-docs-bucket \
GEMINI_API_KEY=<key> \
python -m rag_service.ingestion.main --tenant=acme-corp

# Dry run (list files without processing)
python -m rag_service.ingestion.main --tenant=acme-corp --dry-run

# Force reindex (ignore incremental cache)
python -m rag_service.ingestion.main --tenant=acme-corp --force
```

### Cloud Run Job

```bash
# Build the ingestor image
docker build -f rag_service/Dockerfile.ingestor -t rag-ingestor:local .

# Run as a Cloud Run Job
gcloud run jobs create rag-ingest \
  --image=<region>-docker.pkg.dev/<project>/rag/rag-ingestor:latest \
  --set-env-vars="RAG_INGEST_INPUT_BUCKET=my-docs-bucket,TENANT_ID=acme-corp" \
  --task-timeout=3600s
```

### Key Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `RAG_INGEST_INPUT_BUCKET` | (required) | GCS bucket containing source documents |
| `RAG_INGEST_INPUT_PREFIX` | `incoming/` | Prefix under each tenant directory |
| `RAG_OCR_ENABLED` | `true` | Enable Document AI OCR for scanned PDFs/images |
| `RAG_DOC_AI_PROJECT` | -- | GCP project for Document AI |
| `RAG_DOC_AI_PROCESSOR_ID` | -- | Document AI processor ID |
| `RAG_INGEST_MAX_FILE_WORKERS` | `3` | Concurrent file processing workers |
| `RAG_INGEST_INCREMENTAL` | `true` | Skip unchanged files (by GCS metadata hash) |

See `docs/alloy_db_manual_ingestion_implementation_plan_v_1.md` for the full implementation plan.

## Development

### Running Tests

```bash
# Unit tests (no database required)
pytest tests/unit/ -v

# Integration tests (requires AlloyDB — gracefully skips if unavailable)
# First, open an SSH tunnel to the AlloyDB VM:
gcloud compute ssh alloydb-omni-dev --zone=us-central1-a --tunnel-through-iap -- -L 5432:localhost:5432
# Then set DATABASE_TEST_URL in .env or pass it directly:
DATABASE_TEST_URL="postgresql://apexflow:<password>@localhost:5432/apexflow" pytest tests/integration/ -v

# Retrieval quality evaluation
pytest tests/eval/test_retrieval_metrics.py -v    # metrics unit tests (no DB)
pytest tests/eval/ -v -k synthetic                # synthetic embeddings (DB, no API key)
pytest tests/eval/ -v -k real                     # real Gemini embeddings (DB + ADC/API key)

# Full suite
pytest tests/ -v
```

### Linting

```bash
ruff check .          # lint
ruff format .         # format
mypy rag_service/     # type check
```

### Running Both Services Locally

```bash
# Terminal 1: RAG API
RAG_SHARED_TOKEN=dev-token \
GEMINI_API_KEY=<key> \
uvicorn rag_service.app:app --reload --port 8000

# Terminal 2: MCP Server
RAG_SERVICE_URL=http://localhost:8000 \
RAG_MCP_TOKEN=dev-token \
python -m rag_mcp.server
```

### Docker

```bash
# Build RAG service
docker build -f rag_service/Dockerfile -t rag-service:local .

# Build MCP server
docker build -f rag_mcp/Dockerfile -t rag-mcp:local .

# Build ingestion batch job
docker build -f rag_service/Dockerfile.ingestor -t rag-ingestor:local .

# Run RAG service
docker run -p 8080:8080 \
  -e RAG_SHARED_TOKEN=dev-token \
  -e GEMINI_API_KEY=<key> \
  -e DATABASE_URL=postgresql://... \
  rag-service:local
```

## Configuration

All configuration is via environment variables (no config files):

| Variable | Default | Purpose |
|---|---|---|
| `TENANT_ID` | `default` | Fallback tenant when token claim is absent |
| `RAG_TENANT_CLAIM` | `tenant_id` | Claim name used to resolve tenant from OIDC token |
| `RAG_REQUIRE_TENANT_CLAIM` | `false` | Fail auth if tenant claim is missing |
| `RAG_SHARED_TOKEN` | -- | Dev-only auth token (ignored on Cloud Run) |
| `RAG_OIDC_AUDIENCE` | -- | Required on Cloud Run; expected audience for OIDC tokens |
| `GEMINI_API_KEY` | -- | Embedding API key (local dev only) |
| `DATABASE_URL` | -- | Full DB connection string |
| `RAG_EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model |
| `RAG_EMBEDDING_DIM` | `768` | Expected embedding dimension |
| `RAG_EMBED_MAX_CONCURRENCY` | `8` | Max concurrent embedding calls per request |
| `RAG_EMBED_MAX_RETRIES` | `2` | Retry attempts for transient embedding failures |
| `RAG_RRF_K` | `60` | RRF fusion constant |
| `RAG_SEARCH_EXPANSION` | `3` | Search pool expansion factor |
| `RAG_SEARCH_PER_DOC_CAP` | `3` | Per-document chunk cap in candidate pools |
| `RAG_SEARCH_CANDIDATE_MULTIPLIER` | `4` | Candidate oversampling multiplier before doc fusion |
| `RAG_CORS_ALLOW_ORIGINS` | `http://localhost:3000,http://localhost:5173` | Comma-separated allowed CORS origins |
| `RAG_SERVICE_URL` | `http://localhost:8000` | RAG API URL (MCP server config) |
| `RAG_MCP_FORWARD_CALLER_TOKEN` | `true` | Prefer caller token over static MCP token when available |

See `CLAUDE.md` for the full variable reference.

## Project Structure

```
rag_service/              # Core RAG API (FastAPI)
  app.py                  # Endpoints: /v1/search, /v1/documents, /v1/index
  auth.py                 # Cloud Run OIDC + shared dev token
  config.py               # Env-var-driven configuration
  db.py                   # Asyncpg pool + RLS context manager
  embedding.py            # Gemini embeddings with dimension guard
  models.py               # Pydantic schemas
  Dockerfile              # API server image
  Dockerfile.ingestor     # Batch ingestion job image
  chunking/chunker.py     # Rule-based + semantic chunking with span offsets
  stores/
    rag_document_store.py # Document CRUD with content-hash + source_uri dedup
    rag_search_store.py   # Hybrid search with RRF fusion
  eval/
    metrics.py            # IR metrics: Precision@K, Recall@K, NDCG@K, MRR, Hit Rate
  ingestion/              # GCS batch ingestion pipeline
    main.py               # CLI entry point
    runner.py             # Orchestrates extract → chunk → embed → store
    planner.py            # File discovery + source hash computation
    extractors/           # Text, HTML, DOCX, PDF, Image extractors
    ocr/                  # Document AI integration (online + batch OCR)

rag_mcp/                  # MCP server for VS Code Copilot
  server.py               # FastMCP with streamable HTTP transport
  tools.py                # search + list_documents tools
  oidc.py                 # OIDC token minting via metadata server (Cloud Run)

alembic/                  # Database migrations
  versions/001_rag_tables.py  # 5 tables, RLS policies, 3 dedup indexes

docs/                     # Implementation plans and design docs
tests/                    # Unit (mock) + integration (real DB) + eval tests
scripts/                  # ScaNN indexes, dev data seeding
```

## Security Note

The MCP server on Cloud Run currently has `allUsers` with `roles/run.invoker`, meaning it is publicly accessible. The RAG service behind it is properly secured (OIDC + RLS), so data access is still tenant-scoped. Current MCP calls use service identity (not per-user passthrough), so per-user PRIVATE-owner visibility is not yet enforced for individual VS Code users. See [Future Roadmap](#future-roadmap) for the planned fixes.

## Future Roadmap

- [ ] **Lock down MCP server Cloud Run IAM** — Remove `allUsers` from `roles/run.invoker` on `rag-mcp` and grant access only to specific team members. Currently the MCP server is publicly accessible (the RAG service behind it is secured, but the MCP endpoint itself is open). Fix: `gcloud run services remove-iam-policy-binding rag-mcp --member="allUsers" --role="roles/run.invoker"` then add individual users.
- [ ] **Automated token refresh for VS Code** — OIDC tokens expire after ~1 hour, requiring manual regeneration. Investigate VS Code MCP client support for automatic token refresh or longer-lived credentials.
- [ ] **Per-user identity passthrough** — Currently the MCP server authenticates to the RAG service using its service account identity, so all MCP users share the same tenant/user context. Pass through individual user identity for per-user RLS scoping.

## Origin

Extracted from [ApexFlow v2](https://github.com/gadekar-pravin/apexflow), with these key changes:

- **Auth:** Cloud Run OIDC (not Firebase JWT)
- **Isolation:** PostgreSQL RLS (not `WHERE user_id =`)
- **Schema:** 3-table normalized design (not co-located embeddings)
- **Embedding:** `gemini-embedding-001` with strict dim guard (not `text-embedding-004` with zero-vector fallback)
- **Config:** Env vars only (not `settings.json`)
