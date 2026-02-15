# RAG Platform — GCS → AlloyDB Manual Ingestion Implementation Plan (v1.1)

> **Purpose:** Define an **on-demand (manual)** ingestion architecture that reads multi-format files from **Google Cloud Storage (GCS)** and indexes them into the existing **AlloyDB-backed RAG Platform** (`rag_documents → rag_document_chunks → rag_chunk_embeddings`) with **TEAM-only visibility**.
>
> **Target repo:** `rag-platform`
>
> **Primary goal:** An AI agent can implement this plan end-to-end with minimal interpretation.

---

## 0) Executive Summary

We will add a **Cloud Run Job** (`rag-ingestor`) that:

1. Lists objects in a GCS bucket using **prefix-per-tenant** layout.
2. Applies **incremental ingestion** using `source_uri` + `source_hash`.
3. Extracts text (local parsing first; **Document AI OCR** only when needed).
4. Chunks + embeds using existing `rag_service.chunking` and `rag_service.embedding`.
5. Writes to AlloyDB using a new canonical upsert method:
   - **TEAM docs are canonical per `(tenant_id, source_uri)`**.
   - Updates replace chunks/embeddings **atomically**.
6. Tracks progress in `rag_ingestion_runs` and `rag_ingestion_items`.

This requires a **new Alembic migration (003)** to support **TEAM `source_uri` upsert semantics** without breaking ad-hoc content-hash dedup.

---

## 1) Requirements and Decisions

### Hard requirements
- **Source:** Files in a **GCS bucket**.
- **File types:** PDF, DOCX, HTML, images, TXT/MD (explicit v1 list below).
- **Ingestion:** **Manual only** (run on-demand; no automatic GCS triggers).
- **Visibility:** **TEAM only** (no PRIVATE ingestion in v1).
- **Scale:** ~**100 files/run**, **1–100 MB** each, **~2 runs/day**.
- **Max PDF pages:** ~**300 pages per PDF**.

### Decisions adopted in v1.1
- **Tenant mapping:** derive `tenant_id` from **GCS path prefix** (first segment after bucket).
- **Bucket layout:** **single bucket, prefix-per-tenant**.
- **Incremental ingestion:** use **GCS `generation`** as primary version marker; use `md5Hash` when present; fallback to `crc32c/size/updated`.
- **OCR:** use **Document AI Enterprise Document OCR** for:
  - scanned PDFs / image-only PDFs
  - images
  - PDFs where local extraction quality is low

---

## 2) Current System Context (what exists already)

### Services
- **RAG Service** (`rag_service/`): FastAPI + asyncpg; uses RLS; owns chunking/embedding and DB operations.
- **MCP Server** (`rag_mcp/`): forwards to RAG Service.

### Database (already present via Alembic 001/002)
- `rag_documents`
- `rag_document_chunks`
- `rag_chunk_embeddings`
- `rag_ingestion_runs`
- `rag_ingestion_items`
- RLS policies using `SET LOCAL app.tenant_id` and `SET LOCAL app.user_id`

### Known gaps to address
- `rag_documents.source_hash` exists but is not consistently populated by current store.
- Current dedup index and `upsert_document()` SQL are **incompatible** with canonical `(tenant_id, source_uri)` TEAM ingestion.
- Chunk offsets exist in schema (`chunk_start`, `chunk_end`) but store doesn’t populate them.

---

## 3) High-Level Architecture

```text
              (Manual trigger)
User/Engineer ───────────────► Cloud Run Job: rag-ingestor
                                  │
                                  │ list objects (tenant prefix)
                                  ▼
                            GCS input bucket
                                  │
                                  │ per file
                                  ▼
                        Extract text / OCR if needed
                     (Document AI Enterprise OCR optional)
                                  │
                                  ▼
                           Chunk → Embed (Gemini)
                                  │
                                  ▼
                      AlloyDB (documents/chunks/embeddings)
                                  │
                                  ▼
                      Ingestion tracking (runs/items)
```

---

## 4) GCS Layout and Tenant Mapping

### Recommended (v1): single bucket + prefix-per-tenant
- Input bucket: `gs://<rag-input-bucket>/`
- Convention:
  - `gs://<rag-input-bucket>/<tenant_id>/incoming/...`

Example:
- `gs://rag-input/acme-corp/incoming/runbooks/db-migration.pdf`

**Tenant resolution rule (authoritative):**
- `tenant_id` = **first path segment** after bucket name.

### Allowed tenant allowlist (optional)
- Env: `RAG_INGEST_TENANTS=acme-corp,default,...`
- If set, reject objects whose tenant prefix is not in allowlist.

---

## 5) Incremental Ingestion Strategy

### Source identity
For each GCS object:
- `source_uri = gs://<bucket>/<object_name>`
- `source_hash = <stable version marker>`

### `source_hash` computation (v1.1)
- Primary: `generation` (strongest “content changed” signal)
- Include when available: `md5Hash` (not always present)
- Fallback fields (in order): `crc32c`, `size`, `updated`

**Recommended string format:**
- `f"gen:{generation}|md5:{md5 or ''}|crc32c:{crc32c or ''}|size:{size}|updated:{updated_rfc3339}"`

### Incremental skip rule
Treat an object as **unchanged** if:
- a TEAM document exists with the same `(tenant_id, source_uri)`
- and stored `rag_documents.source_hash == computed source_hash`
- and stored `rag_documents.content_hash == computed content_hash` (optional fast recheck)
- and ingestion settings match (embedding model/dim/version/chunk_method)

v1 default behavior:
- If unchanged: mark ingestion item as `skipped`.
- If changed: re-extract, re-chunk, re-embed, and replace.

### Deletions
v1 behavior:
- If an object is deleted from GCS, **do not delete** from DB automatically.
- Deletion/cleanup becomes an explicit future command.

---

## 6) Canonical TEAM `source_uri` Semantics (critical)

### Problem with current schema behavior
Your current unique dedup index:
- effectively enforces **one TEAM doc per `(tenant_id, content_hash)`** (owner is always NULL for TEAM).

That prevents having two distinct TEAM docs with different `source_uri` but identical content.

### Required solution
Implement **two** dedup modes:

1) **TEAM ingested docs** (GCS/manual ingestion):
- Canonical key: **`(tenant_id, source_uri)`**
- If file changes: update the same canonical doc and replace chunks

2) **TEAM ad-hoc docs** (no `source_uri`, e.g. API indexing):
- Dedup key: **`(tenant_id, content_hash)`** only when `source_uri IS NULL`

3) **PRIVATE docs** (not in ingestion scope, but keep correct behavior):
- Dedup key: **`(tenant_id, owner_user_id, content_hash)`**

This is implemented by **Alembic migration 003** and corresponding store SQL changes.

---

## 7) Database Migration 003 (required)

### File
Create: `alembic/versions/003_source_uri_upsert_and_dedup_split.py`

### Goals
1. Add `pgcrypto` extension (needed for `gen_random_uuid()` defaults).
2. Drop the old broad TEAM/PRIVATE dedup index.
3. Create new partial unique indexes:
   - TEAM canonical per source_uri
   - TEAM ad-hoc dedup (only when source_uri IS NULL)
   - PRIVATE dedup per owner

### Migration contents (exact)

```py
"""Adjust dedup strategy for ingestion + add TEAM source_uri upsert index.

Revision ID: 003
Revises: 002
Create Date: 2026-02-15
"""

from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # gen_random_uuid() dependency safety
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    # Drop the old broad dedup index (blocks source_uri semantics)
    op.execute("DROP INDEX IF EXISTS ux_rag_docs_dedup")

    # TEAM ad-hoc dedup only (source_uri is NULL)
    op.execute("""
        CREATE UNIQUE INDEX ux_rag_docs_team_dedup_adhoc
        ON rag_documents (tenant_id, content_hash)
        WHERE deleted_at IS NULL
          AND visibility = 'TEAM'
          AND source_uri IS NULL
    """)

    # PRIVATE dedup per owner
    op.execute("""
        CREATE UNIQUE INDEX ux_rag_docs_private_dedup
        ON rag_documents (tenant_id, owner_user_id, content_hash)
        WHERE deleted_at IS NULL
          AND visibility = 'PRIVATE'
    """)

    # TEAM canonical doc per source_uri (GCS ingestion)
    op.execute("""
        CREATE UNIQUE INDEX ux_rag_docs_team_source_uri
        ON rag_documents (tenant_id, source_uri)
        WHERE deleted_at IS NULL
          AND visibility = 'TEAM'
          AND source_uri IS NOT NULL
    """)

    # Optional helper index for faster lookups
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_rag_docs_team_source_lookup
        ON rag_documents (tenant_id, source_uri)
        WHERE deleted_at IS NULL
          AND visibility = 'TEAM'
          AND source_uri IS NOT NULL
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_rag_docs_team_source_lookup")
    op.execute("DROP INDEX IF EXISTS ux_rag_docs_team_source_uri")
    op.execute("DROP INDEX IF EXISTS ux_rag_docs_private_dedup")
    op.execute("DROP INDEX IF EXISTS ux_rag_docs_team_dedup_adhoc")

    # Recreate old behavior
    op.execute("""
        CREATE UNIQUE INDEX ux_rag_docs_dedup
        ON rag_documents (tenant_id, visibility, content_hash, COALESCE(owner_user_id, ''))
        WHERE deleted_at IS NULL
    """)
```

### Important note about `pgcrypto`
- Your existing `001_rag_tables.py` uses `DEFAULT gen_random_uuid()`.
- If you create a **fresh** database from scratch, `001` may fail unless `pgcrypto` already exists.

**Recommended operational rule:**
- If the project is still pre-production: **also patch 001** to include `CREATE EXTENSION IF NOT EXISTS pgcrypto;` near the top.
- If 001 is already “published” and immutable: keep 003 and ensure `pgcrypto` exists in the target DB before running `alembic upgrade head`.

---

## 8) Code Changes — Store Layer

### 8.1 Add canonical upsert method
Modify: `rag_service/stores/rag_document_store.py`

Add a new method:
- `upsert_document_by_source_uri(...)`

Key properties:
- Requires `source_uri`.
- TEAM only: `visibility='TEAM'`, `owner_user_id=NULL`.
- Uses **transaction** to guarantee atomic replacement.
- Locks existing canonical doc row with `FOR UPDATE` when present (prevents races).
- Replaces chunks by:
  - `DELETE FROM rag_document_chunks WHERE document_id = $1` (cascades embeddings)
  - insert new chunks and embeddings

**ON CONFLICT target:**
- `ON CONFLICT (tenant_id, source_uri) WHERE visibility='TEAM' AND source_uri IS NOT NULL AND deleted_at IS NULL`

### 8.2 Refactor `upsert_document()`
Update existing `upsert_document()` to:
- If `visibility == 'TEAM'` and `source_uri` is provided, **delegate** to `upsert_document_by_source_uri()`.
- For TEAM ad-hoc (no source_uri), use the **new TEAM ad-hoc unique index**:
  - `ON CONFLICT (tenant_id, content_hash) WHERE visibility='TEAM' AND source_uri IS NULL AND deleted_at IS NULL`
- For PRIVATE, use:
  - `ON CONFLICT (tenant_id, owner_user_id, content_hash) WHERE visibility='PRIVATE' AND deleted_at IS NULL`

### 8.3 Always store JSON metadata as `{}`
Currently, `json.dumps(None)` is passed as `None`, which makes `metadata` become NULL.

Change all insert/update paths to:
- `metadata_json = json.dumps(metadata or {})`
- Always bind `$X::jsonb` with a non-null JSON string.

### 8.4 Add chunk offsets support
Update `_store_chunks_and_embeddings()` to accept:
- `chunk_offsets: list[tuple[int|None, int|None]] | None`

Then insert into `rag_document_chunks(chunk_start, chunk_end)`.

---

## 9) Code Changes — Chunking Offsets

Modify: `rag_service/chunking/chunker.py`

Add a new API that preserves backward compatibility:
- Keep `chunk_document()` as-is.
- Add `chunk_document_with_spans()` returning `[(chunk_text, start, end), ...]`.

v1 rule:
- Offsets are **reliable for rule_based** chunking.
- For semantic chunking, offsets may be `None` unless you implement a more complex mapping.

---

## 10) New Ingestion Package (inside repo)

Create:
```
rag_service/ingestion/
  __init__.py
  main.py                 # Cloud Run Job entrypoint
  cli.py                  # argparse CLI for local + job args
  config.py               # ingestion env vars
  planner.py              # list GCS objects + build worklist
  runner.py               # run tracking + worker pool
  types.py                # typed dataclasses for work items/results
  extractors/
    __init__.py
    base.py               # Extractor interface
    pdf.py                # local text + OCR heuristic
    docx.py               # python-docx
    html.py               # BeautifulSoup
    image.py              # OCR for images
    text.py               # txt/md
  ocr/
    __init__.py
    document_ai.py        # batch + online OCR + output parsing
  gcs.py                  # helpers: list/download/write artifacts
```

### Dependencies to add (`pyproject.toml`)
- `google-cloud-storage`
- `google-cloud-documentai`
- `beautifulsoup4` + `lxml`
- `python-docx`
- `pypdf` (or pdfminer; choose one)
- `Pillow` (optional)

---

## 11) Extraction + OCR Strategy (multi-format)

### Supported extensions (v1 explicit)
- PDFs: `.pdf`
- DOCX: `.docx`
- HTML: `.html`, `.htm`
- Images: `.png`, `.jpg`, `.jpeg`, `.webp`, `.tiff` (optional)
- Text: `.txt`, `.md`

Anything else in v1:
- Mark ingestion item as `failed` (or `skipped_unsupported` if you extend enum), with clear error message.

### Core principle
Prefer deterministic local parsing first; OCR only when needed.

### PDF OCR heuristic
1. Extract text locally.
2. Compute quality metrics:
   - `pages` (if available)
   - `text_len`
   - `text_per_page = text_len / max(pages, 1)`
3. If any:
   - parser throws
   - `text_per_page < RAG_PDF_TEXT_PER_PAGE_MIN` (e.g. 200)
   - extracted text mostly whitespace
   → route to OCR.

### Document AI usage model
- PDFs: batch/async OCR preferred when OCR required.
- Images: online/sync OCR is acceptable for small images; batch optional for large sets.

### Output normalization
- Normalize whitespace
- Remove null bytes
- Collapse extreme newlines

**Important:** do NOT insert artificial page separators into canonical text.
Instead store page boundary info in metadata if needed.

---

## 12) Optional: Store Extracted Text Artifacts in GCS

Because raw extracted text may be huge, store canonical extraction in a separate bucket:

- Output bucket: `RAG_INGEST_OUTPUT_BUCKET`
- Key: `<tenant_id>/runs/<run_id>/<source_hash_or_generation>.txt`

In DB:
- Keep `rag_documents.content` truncated if needed.
- Store:
  - `metadata.extracted_text_uri`
  - `metadata.content_truncated=true`

Config:
- `RAG_MAX_CONTENT_CHARS` default: `2_000_000`

---

## 13) Chunking + Embedding

### Use existing pipeline
- Chunk: `rag_service.chunking.chunker.chunk_document()` (or `chunk_document_with_spans()`)
- Embed: `rag_service.embedding.embed_chunks()`

### Recommended v1 settings
- `chunk_method = rule_based`
- Keep current env defaults unless retrieval quality suggests changes.

### Chunk offsets
- If using spans, pass `(start,end)` into store and persist in `rag_document_chunks.chunk_start/chunk_end`.

---

## 14) Atomic Write Pattern (required)

To avoid partial updates:

1. Do expensive steps OUTSIDE DB transaction:
   - download
   - parse/OCR
   - normalize
   - chunk
   - embed

2. Do a short DB transaction:
   - upsert canonical doc
   - delete old chunks
   - insert new chunks
   - insert embeddings
   - update totals

This ensures the document is never left with 0 chunks due to mid-flight failures.

---

## 15) Ingestion Run Tracking

Use existing tables.

### `rag_ingestion_runs`
- Create one row per tenant/run.
- `status: running → completed|failed`
- Store config snapshot in `metadata`.

### `rag_ingestion_items`
Per object:
- `status`: pending → processing → completed|failed|skipped
- store `source_uri`
- store `document_id` on completion

### Status rules
- `skipped`: unchanged
- `failed`: extractor/OCR/embed/store errors
- `completed`: canonical upsert + chunks/embeddings written

---

## 16) Cloud Run Job Design

### Job: `rag-ingestor`
- Single task (v1)
- Internal async worker pool

Recommended starting resources:
- CPU: 2–4 vCPU
- Memory: 8–16 GB
- Timeout: 60–120 minutes (increase if OCR-heavy)

Concurrency knobs:
- `RAG_INGEST_MAX_FILE_WORKERS=2..4`
- keep embedding concurrency controlled by existing embedding layer.

Networking:
- AlloyDB via private IP → Serverless VPC Access connector

---

## 17) IAM and Secrets

Service account: `rag-ingestor-sa`

Required permissions:
- GCS input bucket: list + read
- GCS output bucket (optional): write
- Document AI: invoke processor
- Vertex AI: embeddings calls
- Secret Manager: read DB credentials (if used)

DB credentials:
- Use existing `DATABASE_URL` logic (same as other services).

---

## 18) Ingestion Configuration (env vars)

### GCS
- `RAG_INGEST_INPUT_BUCKET` (required)
- `RAG_INGEST_INPUT_PREFIX` (default: `incoming/`)
- `RAG_INGEST_TENANT_MODE` = `prefix` (default)
- `RAG_INGEST_TENANTS` = comma allowlist (optional)

### Incremental
- `RAG_INGEST_INCREMENTAL=true`
- `RAG_INGEST_FORCE_REINDEX=false`

### Output artifacts
- `RAG_INGEST_OUTPUT_BUCKET` (recommended)
- `RAG_INGEST_OUTPUT_PREFIX=rag-extracted/`
- `RAG_MAX_CONTENT_CHARS=2000000`

### OCR
- `RAG_OCR_ENABLED=true`
- `RAG_DOC_AI_PROJECT`
- `RAG_DOC_AI_LOCATION`
- `RAG_DOC_AI_PROCESSOR_ID`

### Parsing heuristic
- `RAG_PDF_TEXT_PER_PAGE_MIN=200`

### Worker pool
- `RAG_INGEST_MAX_FILE_WORKERS=3`
- `RAG_INGEST_MAX_RETRIES_PER_FILE=2`

---

## 19) Detailed Processing Flow (per run)

### Step 1 — Initialize
1. Load config.
2. Validate bucket access.
3. Validate DB connectivity.
4. If OCR enabled: validate DocAI config.

### Step 2 — Discover candidates
For each tenant:
1. List objects under: `gs://bucket/<tenant_id>/<input_prefix>/`.
2. Filter supported extensions.
3. Build `source_uri` + `source_hash`.

### Step 3 — Create run + items
1. Insert `rag_ingestion_runs` (`status=running`).
2. Insert `rag_ingestion_items` for each object (`status=pending`).

### Step 4 — Incremental skip
For each item:
1. Query canonical doc by `(tenant_id, source_uri)`.
2. If `source_hash` matches and settings match (and optionally content_hash matches):
   - mark item `skipped`
   - continue

### Step 5 — Extract
Per file:
1. Download bytes (stream if large).
2. Route extractor:
   - PDF: local parse → OCR if needed
   - Image: OCR
   - DOCX: python-docx
   - HTML: BeautifulSoup
   - TXT/MD: decode
3. Normalize text.
4. Optional: write extracted artifact to output bucket.

### Step 6 — Chunk + Embed
1. `chunks = chunk_document_with_spans(...)
2. `embeddings = embed_chunks([chunk.text ...])`

### Step 7 — Atomic upsert
Using `rls_connection(tenant_id, user_id="ingestion-bot")`:
- call `upsert_document_by_source_uri(...)`

### Step 8 — Mark item + finalize run
- update `rag_ingestion_items` status and timestamps
- update `rag_ingestion_runs` counts/status

---

## 20) Command Line Interface

Entry:
- `python -m rag_service.ingestion.main ...`

Args:
- `--tenant=<tenant>` (repeatable)
- `--all-tenants`
- `--prefix=<prefix-under-tenant>` (optional)
- `--max-files=<N>`
- `--concurrency=<N>`
- `--force` (reindex all)
- `--dry-run`

---

## 21) Operational Runbook

Examples:
```bash
# Ingest all tenants
gcloud run jobs execute rag-ingestor --region us-central1 --args "--all-tenants"

# Ingest one tenant
gcloud run jobs execute rag-ingestor --region us-central1 --args "--tenant=acme-corp --prefix=incoming/"

# Force reindex
gcloud run jobs execute rag-ingestor --region us-central1 --args "--tenant=acme-corp --force"
```

Monitoring:
- Query `rag_ingestion_runs` and `rag_ingestion_items`.
- Logs MUST include: `run_id`, `item_id`, `tenant_id`, `source_uri`.

---

## 22) Testing Plan

### Unit tests
- `source_hash` computation:
  - generation-only
  - generation+md5
  - composite/no-md5 fallback
- Extractors:
  - HTML → text
  - DOCX → text
  - PDF heuristic triggers OCR on low text
- Store:
  - `_store_chunks_and_embeddings()` supports offsets

### Integration tests
- New dedup semantics:
  1. TEAM GCS ingest: two different `source_uri` with identical content **both succeed**.
  2. TEAM canonical: same `source_uri` + changed content **updates** and replaces chunks.
  3. TEAM ad-hoc: `source_uri IS NULL` content-hash dedup still works.
- Atomicity:
  - simulate failure after chunk delete inside transaction; assert rollback leaves old chunks intact.

---

## 23) Implementation Checklist (ordered)

1. Create Alembic `003_source_uri_upsert_and_dedup_split.py`.
2. Update store SQL for new unique indexes.
3. Implement `upsert_document_by_source_uri()` with transaction + row lock.
4. Refactor `upsert_document()`:
   - delegate to source-uri method when TEAM + source_uri
   - update ad-hoc TEAM / PRIVATE conflict targets
5. Add chunk spans API in chunker and store chunk offsets.
6. Add ingestion package (`rag_service/ingestion/*`).
7. Implement GCS planner and incremental filter.
8. Implement extractors + DocAI client.
9. Add job entrypoint + Docker wiring.
10. Add unit + integration tests.
11. Add README: “Manual ingestion from GCS”.

---

## 24) Out of Scope (explicit)
- PRIVATE ingestion.
- Auto-triggering via GCS notifications.
- Per-tenant quotas/rate limiting.
- Advanced PDF layout-aware extraction.
- PDF splitting for >500 pages (if ever needed).
- UI-based ingestion management.

