# Production Deployment Guide

Deploy the RAG platform to GCP Cloud Run with AlloyDB Omni. This guide covers all three deployable units: the RAG API service, the MCP server, and the ingestion batch job.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Architecture Overview](#2-architecture-overview)
3. [IAM & Service Accounts](#3-iam--service-accounts)
4. [Secret Manager Setup](#4-secret-manager-setup)
5. [Artifact Registry & Container Images](#5-artifact-registry--container-images)
6. [Database Setup](#6-database-setup)
7. [Deploy RAG Service](#7-deploy-rag-service)
8. [Deploy MCP Server](#8-deploy-mcp-server)
9. [Deploy Ingestion Runner](#9-deploy-ingestion-runner-cloud-run-job)
10. [Post-Deployment: ScaNN Indexes](#10-post-deployment-scann-indexes)
11. [Health Check Verification](#11-health-check-verification)
12. [VS Code Copilot Configuration](#12-vs-code-copilot-configuration)
13. [Operational Runbook](#13-operational-runbook)
14. [Security Checklist](#14-security-checklist)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Prerequisites

**Tools required:**

- `gcloud` CLI (authenticated with `gcloud auth login`)
- Docker **or** Cloud Build (see [Section 5.3](#53-build-and-push-images) for the Cloud Build alternative)
- Python 3.12+
- `uv` (recommended) or `pip`

**Enable required GCP APIs:**

```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  aiplatform.googleapis.com \
  documentai.googleapis.com \
  storage.googleapis.com \
  vpcaccess.googleapis.com \
  cloudscheduler.googleapis.com
```

**Set project defaults:**

```bash
export PROJECT_ID=apexflow-ai
export REGION=us-central1

gcloud config set project $PROJECT_ID
gcloud config set run/region $REGION
```

**Infrastructure requirements:**

- Access to the AlloyDB Omni VM `alloydb-omni-dev` in `us-central1-a` (shared with ApexFlow)
- VPC connectivity configured for Cloud Run to reach the AlloyDB private IP
- The database role used by the RAG service must NOT have `SUPERUSER` or `BYPASSRLS` privileges

---

## 2. Architecture Overview

| Component | Type | Port | Image Source |
|---|---|---|---|
| RAG Service | Cloud Run Service | 8080 | `rag_service/Dockerfile` |
| MCP Server | Cloud Run Service | 8001 | `rag_mcp/Dockerfile` |
| Ingestion Runner | Cloud Run Job | N/A (batch) | `rag_service/Dockerfile.ingestor` |

**Data flow:**

```
VS Code Copilot  →  MCP Server (HTTPS)  →  RAG Service (HTTPS)  →  AlloyDB
                                                  ↑
GCS Bucket  →  Ingestion Runner (batch)  ─────────┘
```

- The **MCP Server** proxies `search` and `list_documents` MCP tool calls to the RAG Service over HTTPS.
- The **RAG Service** owns the database, embedding pipeline, and hybrid search logic. All DB access is RLS-scoped.
- The **Ingestion Runner** reads documents from GCS, extracts text (with optional OCR), chunks, embeds, and stores in AlloyDB.

---

## 3. IAM & Service Accounts

Create three dedicated service accounts with least-privilege roles.

### 3.1 Create service accounts

```bash
for SA in rag-service rag-mcp rag-ingest; do
  gcloud iam service-accounts create $SA \
    --display-name="RAG Platform: $SA"
done
```

### 3.2 Grant roles

**RAG Service** — needs Vertex AI for embeddings and Secret Manager for DB credentials:

```bash
for ROLE in aiplatform.user secretmanager.secretAccessor; do
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:rag-service@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/${ROLE}"
done
```

**MCP Server** — needs Secret Manager and `run.invoker` to call the RAG Service:

```bash
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:rag-mcp@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

> The `run.invoker` grant is service-specific and added after deploying the RAG Service — see [Section 8](#8-deploy-mcp-server).

**Ingestion Runner** — needs Vertex AI, Secret Manager, GCS read, and optionally Document AI and GCS write:

```bash
for ROLE in aiplatform.user secretmanager.secretAccessor storage.objectViewer; do
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:rag-ingest@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/${ROLE}"
done

# Optional: Document AI OCR (if RAG_OCR_ENABLED=true)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:rag-ingest@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/documentai.apiUser"

# Optional: Write extracted text artifacts to output bucket
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:rag-ingest@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.objectCreator"
```

---

## 4. Secret Manager Setup

Store sensitive values in Secret Manager — never pass them as plain env vars.

```bash
# Database password
echo -n "YOUR_ALLOYDB_PASSWORD" | gcloud secrets create rag-alloydb-password \
  --data-file=- --replication-policy=automatic

# OIDC audience (set to the RAG Service Cloud Run URL after first deploy)
echo -n "https://rag-service-HASH.a.run.app" | gcloud secrets create rag-oidc-audience \
  --data-file=- --replication-policy=automatic
```

> **Critical:** Do NOT set `RAG_SHARED_TOKEN` in production. On Cloud Run (`K_SERVICE` is set):
> - If `RAG_SHARED_TOKEN` is present, `auth.py:117-121` logs a warning and ignores it.
> - If `RAG_OIDC_AUDIENCE` is missing, `auth.py:122-123` raises `RuntimeError("RAG_OIDC_AUDIENCE must be set on Cloud Run")` and the service crashes on startup.

---

## 5. Artifact Registry & Container Images

### 5.1 Create a Docker repository

```bash
gcloud artifacts repositories create rag \
  --repository-format=docker \
  --location=$REGION \
  --description="RAG platform container images"
```

### 5.2 Configure Docker authentication

```bash
gcloud auth configure-docker ${REGION}-docker.pkg.dev
```

### 5.3 Build and push images

Use Git SHA tags for traceability:

```bash
export IMAGE_TAG=$(git rev-parse --short HEAD)
export REGISTRY=${REGION}-docker.pkg.dev/${PROJECT_ID}/rag
```

**Option A — Local Docker build:**

```bash
# RAG Service
docker build -f rag_service/Dockerfile -t ${REGISTRY}/rag-service:${IMAGE_TAG} .
docker push ${REGISTRY}/rag-service:${IMAGE_TAG}

# MCP Server
docker build -f rag_mcp/Dockerfile -t ${REGISTRY}/rag-mcp:${IMAGE_TAG} .
docker push ${REGISTRY}/rag-mcp:${IMAGE_TAG}

# Ingestion Runner
docker build -f rag_service/Dockerfile.ingestor -t ${REGISTRY}/rag-ingestor:${IMAGE_TAG} .
docker push ${REGISTRY}/rag-ingestor:${IMAGE_TAG}
```

**Option B — Cloud Build** (no local Docker required):

```bash
gcloud services enable cloudbuild.googleapis.com

# Build and push each image via Cloud Build
for DOCKERFILE IMAGE_NAME in \
  "rag_service/Dockerfile rag-service" \
  "rag_mcp/Dockerfile rag-mcp" \
  "rag_service/Dockerfile.ingestor rag-ingestor"; do
  gcloud builds submit \
    --config=/dev/stdin \
    --project=${PROJECT_ID} \
    --timeout=600s \
    --substitutions="_IMAGE=${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}" \
    . <<EOF
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-f', '${DOCKERFILE}', '-t', '\${_IMAGE}', '.']
images:
  - '\${_IMAGE}'
EOF
done
```

> Cloud Build uploads the source, builds remotely, and pushes to Artifact Registry in one step. Each build takes ~35-50 seconds.

---

## 6. Database Setup

### 6.1 Verify database role

The DB user must NOT have `SUPERUSER` or `BYPASSRLS` — RLS enforcement depends on this:

```sql
SELECT rolname, rolsuper, rolbypassrls
FROM pg_roles
WHERE rolname = 'apexflow';
```

Both `rolsuper` and `rolbypassrls` **must** be `false`. If not:

```sql
ALTER ROLE apexflow NOSUPERUSER NOBYPASSRLS;
```

### 6.2 Run migrations

> **Shared database note:** This platform shares the AlloyDB instance with ApexFlow. Alembic is configured to use a separate version table (`rag_alembic_version` in `alembic.ini`) to avoid collisions with ApexFlow's `alembic_version` table. If you see `Can't locate revision identified by '005'` or similar, verify that `version_table = rag_alembic_version` is set in `alembic.ini` and that `alembic/env.py` passes `version_table=VERSION_TABLE` to `context.configure()`.

Alembic uses psycopg2 (sync) and the same 3-priority connection logic as `db.py` (`alembic/env.py:19-34`).

**Option A — Direct connection** (if the AlloyDB VM has a public IP or you're on the same VPC):

```bash
DATABASE_URL=postgresql://apexflow:PASSWORD@ALLOYDB_IP/apexflow \
  alembic upgrade head
```

**Option B — Via IAP tunnel** (for private AlloyDB):

```bash
# In a separate terminal, start the tunnel:
gcloud compute ssh alloydb-omni-dev --zone=us-central1-a \
  --tunnel-through-iap -- -L 5433:localhost:5432

# Then run migrations against the tunnel:
DATABASE_URL=postgresql://apexflow:PASSWORD@localhost:5433/apexflow \
  alembic upgrade head
```

### 6.3 Verify migration

Confirm the 5 `rag_*` tables exist with RLS force-enabled and required extensions:

```sql
-- Check extensions
SELECT extname FROM pg_extension WHERE extname IN ('pgcrypto', 'vector');

-- Check tables
SELECT tablename FROM pg_tables
WHERE schemaname = 'public' AND tablename LIKE 'rag_%'
ORDER BY tablename;
-- Expected: rag_chunk_embeddings, rag_document_chunks, rag_documents,
--           rag_ingestion_items, rag_ingestion_runs

-- Verify FORCE ROW LEVEL SECURITY on all tables
SELECT relname, relrowsecurity, relforcerowsecurity
FROM pg_class
WHERE relname LIKE 'rag_%' AND relkind = 'r';
-- Both columns should be true for all 5 tables
```

### 6.4 VPC connectivity

Cloud Run must reach the AlloyDB private IP. Use either:

**Direct VPC egress** (recommended for new deployments):

```bash
gcloud run services update rag-service \
  --network=default \
  --subnet=default \
  --vpc-egress=private-ranges-only
```

**VPC connector** (legacy approach):

```bash
gcloud compute networks vpc-access connectors create rag-connector \
  --region=$REGION \
  --network=default \
  --range=10.8.0.0/28 \
  --min-instances=2 \
  --max-instances=3
```

---

## 7. Deploy RAG Service

The RAG Service is the core FastAPI application. On Cloud Run, it detects `K_SERVICE` and switches to production mode: OIDC auth, Vertex AI ADC, and `ALLOYDB_*` connection variables (`db.py:47-52`).

```bash
gcloud run deploy rag-service \
  --image=${REGISTRY}/rag-service:${IMAGE_TAG} \
  --service-account=rag-service@${PROJECT_ID}.iam.gserviceaccount.com \
  --port=8080 \
  --memory=1Gi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=5 \
  --set-env-vars="\
TENANT_ID=your-tenant-id,\
GOOGLE_CLOUD_PROJECT=${PROJECT_ID},\
GOOGLE_CLOUD_LOCATION=${REGION},\
ALLOYDB_HOST=ALLOYDB_PRIVATE_IP,\
ALLOYDB_DB=apexflow,\
ALLOYDB_USER=apexflow,\
RAG_CORS_ALLOW_ORIGINS=https://your-frontend.example.com" \
  --set-secrets="ALLOYDB_PASSWORD=rag-alloydb-password:latest,RAG_OIDC_AUDIENCE=rag-oidc-audience:latest" \
  --no-allow-unauthenticated \
  --network=default --subnet=default --vpc-egress=private-ranges-only
  # OR, if using a VPC connector (legacy, see Section 6.4):
  # --vpc-connector=rag-connector
```

**Important notes:**

- Do **NOT** set `RAG_SHARED_TOKEN` — it is ignored on Cloud Run and logs a warning (`auth.py:117-121`).
- Do **NOT** set `GEMINI_API_KEY` — the embedding client auto-detects GCP and uses Vertex AI with ADC (`embedding.py:46-47`).
- `--no-allow-unauthenticated` ensures all callers must present valid OIDC tokens.

**After first deploy** — update the `rag-oidc-audience` secret with the actual Cloud Run URL:

```bash
RAG_URL=$(gcloud run services describe rag-service --format='value(status.url)')

echo -n "$RAG_URL" | gcloud secrets versions add rag-oidc-audience --data-file=-

# Redeploy to pick up the new secret version
gcloud run services update rag-service \
  --set-secrets="ALLOYDB_PASSWORD=rag-alloydb-password:latest,RAG_OIDC_AUDIENCE=rag-oidc-audience:latest"
```

---

## 8. Deploy MCP Server

The MCP Server is a lightweight proxy that forwards `search` and `list_documents` tool calls to the RAG Service. It communicates over public HTTPS and does not need a VPC connector.

```bash
gcloud run deploy rag-mcp \
  --image=${REGISTRY}/rag-mcp:${IMAGE_TAG} \
  --service-account=rag-mcp@${PROJECT_ID}.iam.gserviceaccount.com \
  --port=8001 \
  --memory=512Mi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=3 \
  --set-env-vars="\
RAG_SERVICE_URL=${RAG_URL},\
MCP_PORT=8001" \
  --no-allow-unauthenticated
```

**Grant `run.invoker`** so the MCP Server's service account can call the RAG Service:

```bash
gcloud run services add-iam-policy-binding rag-service \
  --member="serviceAccount:rag-mcp@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/run.invoker"
```

> The MCP server forwards the caller's OIDC token to the RAG Service when `RAG_MCP_FORWARD_CALLER_TOKEN=true` (default). This means the end user's identity flows through to RLS.

---

## 9. Deploy Ingestion Runner (Cloud Run Job)

The ingestion pipeline runs as a Cloud Run Job. It reads documents from GCS, extracts text, chunks, embeds, and stores in AlloyDB.

### 9.1 Create GCS bucket for source documents

```bash
export INGEST_BUCKET=rag-ingest-${PROJECT_ID}

gcloud storage buckets create gs://${INGEST_BUCKET} \
  --location=$REGION \
  --uniform-bucket-level-access
```

Upload documents following the tenant directory structure:

```
gs://BUCKET/TENANT_ID/incoming/document.pdf
gs://BUCKET/TENANT_ID/incoming/subdir/notes.md
```

### 9.2 Create the Cloud Run Job

```bash
gcloud run jobs create rag-ingest \
  --image=${REGISTRY}/rag-ingestor:${IMAGE_TAG} \
  --service-account=rag-ingest@${PROJECT_ID}.iam.gserviceaccount.com \
  --memory=2Gi \
  --cpu=1 \
  --task-timeout=3600s \
  --max-retries=1 \
  --set-env-vars="\
TENANT_ID=your-tenant-id,\
GOOGLE_CLOUD_PROJECT=${PROJECT_ID},\
GOOGLE_CLOUD_LOCATION=${REGION},\
ALLOYDB_HOST=ALLOYDB_PRIVATE_IP,\
ALLOYDB_DB=apexflow,\
ALLOYDB_USER=apexflow,\
RAG_INGEST_INPUT_BUCKET=${INGEST_BUCKET},\
RAG_INGEST_INCREMENTAL=true,\
RAG_INGEST_MAX_FILE_WORKERS=3" \
  --set-secrets="ALLOYDB_PASSWORD=rag-alloydb-password:latest" \
  --network=default --subnet=default --vpc-egress=private-ranges-only \
  --args="--tenant,your-tenant-id"
```

### 9.3 Execute the job

**Manual run:**

```bash
gcloud run jobs execute rag-ingest
```

**With CLI overrides** (passed as container args):

```bash
gcloud run jobs execute rag-ingest \
  --args="--tenant,your-tenant-id,--force,--max-files,100"
```

**Dry run** (lists work items without writing to DB):

```bash
gcloud run jobs execute rag-ingest \
  --args="--tenant,your-tenant-id,--dry-run"
```

**Check execution status:**

```bash
gcloud run jobs executions list --job=rag-ingest
gcloud run jobs executions describe EXECUTION_NAME
```

### 9.4 Scheduled ingestion (optional)

Create a Cloud Scheduler job for recurring runs:

```bash
gcloud scheduler jobs create http rag-ingest-daily \
  --schedule="0 2 * * *" \
  --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/rag-ingest:run" \
  --http-method=POST \
  --oauth-service-account-email=rag-ingest@${PROJECT_ID}.iam.gserviceaccount.com \
  --location=$REGION
```

Grant the ingestion SA permission to invoke the job (required for Cloud Scheduler to trigger it):

```bash
gcloud run jobs add-iam-policy-binding rag-ingest \
  --member="serviceAccount:rag-ingest@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/run.invoker"
```

### 9.5 Document AI OCR setup (optional)

If you need OCR for scanned PDFs or images (`RAG_OCR_ENABLED=true`), configure Document AI:

1. Create a Document AI processor in the GCP Console (type: OCR).
2. Add the env vars to the job:

```bash
gcloud run jobs update rag-ingest \
  --update-env-vars="\
RAG_OCR_ENABLED=true,\
RAG_DOC_AI_PROJECT=${PROJECT_ID},\
RAG_DOC_AI_LOCATION=${REGION},\
RAG_DOC_AI_PROCESSOR_ID=YOUR_PROCESSOR_ID"
```

If any of the three Document AI config vars are missing when OCR is enabled, `IngestConfig.validate()` raises:
`ValueError: OCR enabled but missing DocAI config: RAG_DOC_AI_PROJECT, ...` (`ingestion/config.py:96-106`).

---

## 10. Post-Deployment: ScaNN Indexes

Create ScaNN vector indexes **after** data has been populated. ScaNN indexes cannot be created on empty tables in AlloyDB.

```sql
-- ScaNN index (AlloyDB only)
CREATE INDEX IF NOT EXISTS ix_rag_emb_scann ON rag_chunk_embeddings
  USING scann (embedding cosine)
  WITH (num_leaves = 50);
```

For non-AlloyDB PostgreSQL (e.g., standard Cloud SQL or self-hosted), use IVFFlat instead:

```sql
CREATE INDEX ix_rag_emb_vector ON rag_chunk_embeddings
  USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

> The full script is at `scripts/create-scann-indexes.sql`. Run it via `psql` against the production database after indexing your first batch of documents.

---

## 11. Health Check Verification

### 11.1 Testing with Cloud Run proxy (recommended)

The simplest way to test is with `gcloud run services proxy`, which handles Cloud Run IAM automatically. For app-level auth on protected endpoints (`/v1/*`), generate an OIDC token by impersonating the rag-service SA:

```bash
RAG_URL=$(gcloud run services describe rag-service --format='value(status.url)')

# Start proxy (handles Cloud Run IAM)
gcloud run services proxy rag-service --port=9090 &

# Generate OIDC token with proper audience (requires iam.serviceAccountTokenCreator on the SA)
TOKEN=$(gcloud auth print-identity-token \
  --impersonate-service-account=rag-service@${PROJECT_ID}.iam.gserviceaccount.com \
  --audiences="$RAG_URL")

# Public endpoints (no app-level auth needed)
curl -s http://localhost:9090/liveness   # → {"status":"ok"}
curl -s http://localhost:9090/readiness  # → {"status":"ok"}

# Protected endpoints (require OIDC token)
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:9090/v1/documents
# → {"documents":[],"total":0}

curl -s -X POST http://localhost:9090/v1/search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "test search", "limit": 5}'
# → {"results":[],"debug":null}

# Stop proxy
kill %1
```

> **Note:** `gcloud auth print-identity-token --audiences=` requires a service account credential. It does not work with regular user accounts — you must use `--impersonate-service-account`. Grant yourself `roles/iam.serviceAccountTokenCreator` on the SA first:
> ```bash
> gcloud iam service-accounts add-iam-policy-binding \
>   rag-service@${PROJECT_ID}.iam.gserviceaccount.com \
>   --member="user:YOUR_EMAIL" \
>   --role="roles/iam.serviceAccountTokenCreator"
> ```

### 11.2 Testing without proxy (direct HTTPS)

You need both Cloud Run IAM access (`roles/run.invoker` on the service) and a valid OIDC token:

```bash
# Grant yourself run.invoker
gcloud run services add-iam-policy-binding rag-service \
  --member="user:YOUR_EMAIL" --role="roles/run.invoker"

# Generate token (note: IAM propagation may take 1-2 minutes)
TOKEN=$(gcloud auth print-identity-token \
  --impersonate-service-account=rag-service@${PROJECT_ID}.iam.gserviceaccount.com \
  --audiences="$RAG_URL")

curl -s -H "Authorization: Bearer $TOKEN" "$RAG_URL/liveness"
curl -s -H "Authorization: Bearer $TOKEN" "$RAG_URL/v1/documents"
```

### 11.3 Readiness details

The readiness endpoint checks both DB connectivity and the embedding service (`app.py:166-174`):

| Response | Meaning |
|---|---|
| `{"status":"ok"}` | DB and embedding service healthy |
| 503 `{"detail":"Database unavailable"}` | Cannot reach AlloyDB — check VPC/host/credentials |
| `{"status":"degraded","error":"Embedding service unavailable"}` | Vertex AI unreachable — check SA roles |

---

## 12. VS Code Copilot Configuration

Add MCP configuration to your workspace:

**.vscode/mcp.json:**

```json
{
  "servers": {
    "rag-search": {
      "type": "sse",
      "url": "https://rag-mcp-HASH.a.run.app/sse",
      "headers": {
        "Authorization": "Bearer <your-id-token>"
      }
    }
  }
}
```

**Obtain an OIDC token:**

```bash
MCP_URL=$(gcloud run services describe rag-mcp --format='value(status.url)')

gcloud auth print-identity-token --audiences="$MCP_URL"
```

Tokens expire after ~1 hour. Refresh with the same command.

**Available MCP tools:**

| Tool | Description |
|---|---|
| `search` | Hybrid search (vector + full-text with RRF fusion). Parameters: `query` (string), `limit` (int). |
| `list_documents` | List visible documents. Parameters: `limit` (int), `offset` (int). |

---

## 13. Operational Runbook

### 13.1 Updating a service

Build, push, and deploy a new revision:

```bash
export IMAGE_TAG=$(git rev-parse --short HEAD)

docker build -f rag_service/Dockerfile -t ${REGISTRY}/rag-service:${IMAGE_TAG} .
docker push ${REGISTRY}/rag-service:${IMAGE_TAG}

gcloud run services update rag-service \
  --image=${REGISTRY}/rag-service:${IMAGE_TAG}
```

### 13.2 Running migrations on schema changes

After adding a new Alembic revision, run migrations before deploying the new image:

```bash
DATABASE_URL=postgresql://apexflow:PASSWORD@ALLOYDB_IP/apexflow \
  alembic upgrade head
```

Always migrate first, then deploy — the new code may depend on schema changes.

### 13.3 Viewing logs

The RAG service emits structured JSON logs with GCP severity mapping when `K_SERVICE` is set (`logging_config.py:51-55`):

```bash
# Recent logs for RAG service
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="rag-service"' \
  --limit=50 --format=json

# Filter by severity
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="rag-service" AND severity>=ERROR' \
  --limit=20 --format=json

# Ingestion job logs
gcloud logging read \
  'resource.type="cloud_run_job" AND resource.labels.job_name="rag-ingest"' \
  --limit=50 --format=json
```

### 13.4 Rollback

```bash
# List revisions
gcloud run revisions list --service=rag-service

# Route 100% traffic to a previous revision
gcloud run services update-traffic rag-service \
  --to-revisions=rag-service-REVISION_NAME=100
```

### 13.5 Scaling notes

| Limit | Value | Source |
|---|---|---|
| Default rate limit (all endpoints) | 60 requests/minute | `app.py:81` |
| Search rate limit | 30 requests/minute | `app.py:181` |
| Index rate limit | 10 requests/minute | `app.py:295` |
| Request body size | 10 MB | `app.py:104` |
| DB connection pool max | 5 (configurable via `DB_POOL_MAX`) | `db.py:74` |
| Embedding concurrency | 8 (configurable via `RAG_EMBED_MAX_CONCURRENCY`) | `config.py:33` |
| Ingestion file workers | 3 (configurable via `RAG_INGEST_MAX_FILE_WORKERS`) | `ingestion/config.py:87` |

Rate limits are per-IP. Adjust Cloud Run `--max-instances` and `DB_POOL_MAX` together — each instance opens its own pool.

---

## 14. Security Checklist

| # | Check | How to Verify |
|---|---|---|
| 1 | `RAG_SHARED_TOKEN` is NOT set in production | `gcloud run services describe rag-service --format=yaml` — env vars should not include it |
| 2 | `RAG_OIDC_AUDIENCE` is set and matches the Cloud Run URL | Secret Manager contains the correct URL; service starts without `RuntimeError` |
| 3 | DB role has no `SUPERUSER` or `BYPASSRLS` | `SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname='apexflow'` — both `false` |
| 4 | `FORCE ROW LEVEL SECURITY` is enabled on all 5 tables | `SELECT relname, relforcerowsecurity FROM pg_class WHERE relname LIKE 'rag_%'` — all `true` |
| 5 | `--no-allow-unauthenticated` is set on both services | `gcloud run services describe SERVICE --format='value(spec.template.metadata.annotations)'` |
| 6 | Service accounts follow least privilege | Each SA has only the roles listed in [Section 3](#3-iam--service-accounts) |
| 7 | CORS origins are restricted | `RAG_CORS_ALLOW_ORIGINS` lists only your frontend domain, not `*` |
| 8 | VPC connector/Direct VPC egress is configured | `gcloud run services describe rag-service` — check network settings |
| 9 | `GEMINI_API_KEY` is NOT set on Cloud Run | ADC via Vertex AI is used instead (`embedding.py:46-47`) |
| 10 | All secrets are in Secret Manager | No plaintext passwords in env vars; `--set-secrets` used for sensitive values |

---

## 15. Troubleshooting

### `RuntimeError: RAG_OIDC_AUDIENCE must be set on Cloud Run`

**Cause:** The service detected `K_SERVICE` (Cloud Run) but `RAG_OIDC_AUDIENCE` is not set.
**Fix:** Create or update the `rag-oidc-audience` secret with the Cloud Run URL and redeploy.
**Ref:** `auth.py:122-123`

### Readiness returns 503 "Database unavailable"

**Cause:** The readiness endpoint (`app.py:169-170`) failed `check_db_connection()`.
**Fix:** Verify AlloyDB is running, VPC connector is configured, and `ALLOYDB_HOST` / credentials are correct.
**Ref:** `db.py:90-99`

### Readiness returns "degraded" (embedding)

**Cause:** The embedding health check failed (`app.py:171-173`). The service is functional but search will fail.
**Fix:** Verify the service account has `aiplatform.user` role and `GOOGLE_CLOUD_PROJECT`/`GOOGLE_CLOUD_LOCATION` are set correctly.
**Ref:** `embedding.py:138-146`

### 401 on authenticated endpoints

**Cause:** The OIDC token audience doesn't match `RAG_OIDC_AUDIENCE`, or the token issuer is not in `RAG_ALLOWED_ISSUERS`.
**Fix:** Ensure the audience in `gcloud auth print-identity-token --audiences=URL` matches the `rag-oidc-audience` secret. Default allowed issuers: `https://accounts.google.com` and `accounts.google.com`.
**Ref:** `auth.py:73-76`, `config.py:53-55`

### Embedding dimension mismatch

**Cause:** `ValueError: Embedding dimension mismatch: got X, expected 768`.
**Fix:** Ensure `RAG_EMBEDDING_DIM` (default: 768) matches the model's output dimensionality. If you changed `RAG_EMBEDDING_MODEL`, update the dim accordingly.
**Ref:** `embedding.py:80-84`

### `ValueError: RAG_INGEST_INPUT_BUCKET is required`

**Cause:** The ingestion job started without `RAG_INGEST_INPUT_BUCKET` set.
**Fix:** Set the env var in the Cloud Run Job configuration.
**Ref:** `ingestion/config.py:52-54`

### `ValueError: OCR enabled but missing DocAI config`

**Cause:** `RAG_OCR_ENABLED=true` but one or more of `RAG_DOC_AI_PROJECT`, `RAG_DOC_AI_LOCATION`, `RAG_DOC_AI_PROCESSOR_ID` are missing.
**Fix:** Either set all three Document AI config vars or set `RAG_OCR_ENABLED=false`.
**Ref:** `ingestion/config.py:95-106`

### MCP "search service temporarily unavailable"

**Cause:** The MCP server cannot reach the RAG Service.
**Fix:** Verify `RAG_SERVICE_URL` is correct (should be the full Cloud Run URL), the MCP service account has `run.invoker` on the RAG Service, and the RAG Service is healthy.
**Ref:** `rag_mcp/config.py:7`

### `ValueError: tenant_id and user_id are required (fail-closed)`

**Cause:** `rls_connection()` received empty `tenant_id` or `user_id`.
**Fix:** Ensure `TENANT_ID` env var is set and the auth middleware is correctly extracting identity from OIDC tokens.
**Ref:** `db.py:113-114`

### Alembic `Can't locate revision identified by '005'`

**Cause:** The shared AlloyDB instance has ApexFlow's `alembic_version` table with its own migration history. If the RAG platform's Alembic reads that table, it finds a revision ID it doesn't recognize.
**Fix:** Ensure `alembic.ini` contains `version_table = rag_alembic_version` and `alembic/env.py` passes `version_table=VERSION_TABLE` to both `context.configure()` calls. This was fixed in commit `2b513ab`.
**Ref:** `alembic.ini:3`, `alembic/env.py:37`

### Alembic connection errors

**Cause:** Alembic cannot reach the database. It uses psycopg2 (sync) with the same 3-priority connection logic.
**Fix:** Set `DATABASE_URL` explicitly, or set `DB_HOST`/`DB_USER`/`DB_PASSWORD`/`DB_PORT`/`DB_NAME`. If connecting through an IAP tunnel, ensure the tunnel is active and pointing to the correct local port.
**Ref:** `alembic/env.py:19-34`

### MCP server crash: `FastMCP.run() got an unexpected keyword argument 'port'`

**Cause:** MCP SDK v1.26.0 moved the `port` and `host` parameters from `FastMCP.run()` to the `FastMCP()` constructor.
**Fix:** Pass `host` and `port` to the `FastMCP()` constructor, not to `run()`. This was fixed in commit `2b513ab`.
**Ref:** `rag_mcp/server.py:21-29`

### `asyncpg.exceptions.PostgresSyntaxError: syntax error at or near "$1"` on `SET LOCAL`

**Cause:** PostgreSQL's `SET` command does not support parameterized queries (`$1`). The original `db.py` used `SET LOCAL app.tenant_id = $1`, which works in some PostgreSQL versions but fails in AlloyDB Omni.
**Fix:** Use `SELECT set_config('app.tenant_id', $1, true)` instead, which is the parameterized equivalent of `SET LOCAL` and is safe from SQL injection. This was fixed in commit `2b513ab`.
**Ref:** `db.py:118-123`

### `gcloud auth print-identity-token --audiences=` fails with `Invalid account type`

**Cause:** The `--audiences` flag requires a service account credential, not a regular user account.
**Fix:** Use `--impersonate-service-account` to generate a token with the correct audience. See [Section 11](#11-health-check-verification) for the full command. You need `roles/iam.serviceAccountTokenCreator` on the target SA.

### 403 Forbidden when calling Cloud Run with impersonated SA token

**Cause:** The service account used for impersonation does not have `roles/run.invoker` on the Cloud Run service. Cloud Run IAM rejects the request before it reaches the app.
**Fix:** Grant `run.invoker` to the SA on the specific service:
```bash
gcloud run services add-iam-policy-binding rag-service \
  --member="serviceAccount:SA_EMAIL" --role="roles/run.invoker"
```
IAM propagation can take 1-2 minutes.
