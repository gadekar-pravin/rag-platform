# RAG Platform — Architecture Document

## 1. Executive Summary

The RAG Platform is a standalone, multi-tenant Retrieval-Augmented Generation service built on Google Cloud. It enables data engineers and data architects to search proprietary internal documents directly from VS Code Copilot — without needing access to the parent application (ApexFlow). The platform exposes a hybrid search API combining vector similarity and full-text search with Reciprocal Rank Fusion (RRF), backed by AlloyDB with pgvector. Multi-tenant data isolation is enforced at the database level through PostgreSQL Row-Level Security (RLS), not application-layer `WHERE` clauses. An MCP (Model Context Protocol) server bridges VS Code Copilot to the search API, allowing engineers to query the knowledge base in natural language without leaving their IDE.

---

## 2. Why This Platform Exists

### The Problem: Tribal Knowledge Trapped in Silos

Data engineers and data architects routinely lose time to a recurring set of problems:

| Pain Point | Example |
|---|---|
| **Finding the right table or metric** | "Which table has the canonical revenue number — `fact_revenue`, `fact_orders_revenue`, or the one in the legacy schema?" |
| **Re-learning tribal knowledge** | A senior engineer leaves; their undocumented knowledge of pipeline edge cases leaves with them. |
| **Debugging recurring issues** | The same Airflow failure gets re-investigated every quarter because the runbook lives in a Slack thread from 2023. |
| **Onboarding friction** | New hires spend weeks asking "where is this documented?" — often the answer is "it isn't." |

This knowledge exists *somewhere* — scattered across Confluence pages, design docs, Slack threads, README files, and people's heads. But it is not accessible at the moment an engineer needs it: while they are writing code.

### Why Copilot Alone Is Not Enough

GitHub Copilot is effective at syntax completion, boilerplate generation, and general programming patterns. But it has no knowledge of:

- Internal database schemas and naming conventions
- Data contracts and SLA definitions
- Team-specific runbooks and incident playbooks
- Pipeline configuration patterns and guardrails
- Organizational decisions captured in design documents

When asked about these topics, Copilot either hallucinates a plausible-sounding answer or declines to respond. Neither outcome helps the engineer.

### The Solution: Give Copilot Access to Internal Knowledge

This platform bridges the gap by connecting VS Code Copilot to a curated, permissioned corpus of internal documents via the Model Context Protocol (MCP). When an engineer asks a question, Copilot retrieves relevant passages from the knowledge base and generates answers grounded in actual internal documentation — with citations back to the source.

This is **not** training a custom model or fine-tuning an LLM. The organization's documents remain in a governed database with access controls and audit trails. The LLM simply reads the retrieved passages at query time, the same way a human would read a document before answering a question.

### Why Now

| Factor | Rationale |
|---|---|
| **Already paying for Copilot** | The organization has Copilot licenses deployed. Adding RAG extends the investment rather than requiring a new tool. |
| **Fastest path to value** | No model training, no fine-tuning, no GPU infrastructure. Curate documents, ingest them, and Copilot can use them immediately. |
| **Strong governance** | Row-Level Security enforces tenant isolation at the database level. Documents are curated (not scraped). Answers include citations, reducing hallucination risk. Access is controlled via Cloud Run IAM and OIDC — every query is authenticated. |
| **Low adoption friction** | Engineers stay in VS Code. No new UI to learn, no context switching. The MCP server appears as a tool in Copilot's existing chat interface. |

---

## 3. What is RAG?

### The Problem

Large Language Models (LLMs) like Gemini are trained on public data and have no knowledge of your organization's internal documents — runbooks, design docs, data dictionaries, or pipeline configurations. When you ask an LLM a question about your own systems, it either hallucinates or says "I don't know."

### The Solution: Retrieval-Augmented Generation

**RAG** solves this by retrieving relevant documents from your knowledge base and feeding them to the LLM as context. Instead of relying on the model's training data, the LLM generates answers grounded in your actual documents.

RAG has three phases:

| Phase | What Happens | Who Does It |
|---|---|---|
| **1. Ingestion** | Documents are chunked into passages, each passage is converted to a 768-dimensional embedding vector, and both text + vectors are stored in the database | RAG Platform |
| **2. Retrieval** | A user's query is embedded and searched against stored vectors (semantic similarity) and text (keyword matching), then results are fused and ranked | RAG Platform |
| **3. Generation** | The LLM reads the retrieved passages and produces a grounded answer | VS Code Copilot |

This platform handles phases 1 and 2. Phase 3 happens in VS Code Copilot, which receives the retrieved chunks and synthesizes an answer.

### Why Hybrid Search?

- **Vector search** captures *meaning* — "database connection error" matches "AlloyDB timeout exception" even though they share no words.
- **Full-text search** captures *exact terms* — important for acronyms (`GKE`), table names (`rag_documents`), or error codes.
- **Hybrid search** combines both, so you get semantic understanding *and* keyword precision. The platform uses Reciprocal Rank Fusion (RRF) to merge the two ranked lists into a single result set.

### What Are Embeddings?

An embedding is a list of 768 floating-point numbers that represents the *meaning* of a piece of text. The platform uses Google's `gemini-embedding-001` model to generate these vectors. Texts with similar meanings produce vectors that are close together in 768-dimensional space, which allows the database to find semantically relevant passages using distance calculations (cosine similarity).

```mermaid
flowchart LR
    Q["User Question"] --> E["Embed Query<br/>(768-dim vector)"]
    E --> S["Search AlloyDB<br/>(vector + text)"]
    S --> R["Retrieve Top<br/>Chunks"]
    R --> L["LLM Synthesizes<br/>Answer"]
    L --> A["Grounded Answer<br/>with Citations"]

    style Q fill:#4285F4,color:#fff
    style E fill:#34A853,color:#fff
    style S fill:#FBBC04,color:#000
    style R fill:#EA4335,color:#fff
    style L fill:#4285F4,color:#fff
    style A fill:#34A853,color:#fff
```
*Diagram 1: How RAG Works — from question to grounded answer.*

---

## 4. System Architecture Overview

The platform consists of three independently deployable components, all running on Google Cloud:

| Component | Runtime | Purpose |
|---|---|---|
| **RAG Service** | Cloud Run (Service) | FastAPI app that owns the database, embedding pipeline, and search logic |
| **MCP Server** | Cloud Run (Service) | Lightweight proxy that exposes search + list tools to VS Code Copilot via the Model Context Protocol |
| **Ingestion Pipeline** | Cloud Run (Job) | Batch job that imports documents from GCS, extracts text (with OCR fallback), chunks, embeds, and stores |

### GCP Services Used

| Service | Role |
|---|---|
| **Cloud Run** | Serverless compute for all three components |
| **AlloyDB** (with pgvector) | PostgreSQL-compatible database with vector search (ScaNN indexes) and Row-Level Security |
| **Vertex AI** (Gemini) | Embedding generation (`gemini-embedding-001`) |
| **Cloud Storage (GCS)** | Source document storage for batch ingestion |
| **Document AI** | OCR for scanned PDFs and images |
| **Secret Manager** | Database passwords, OIDC audience configuration |
| **Artifact Registry** | Container image storage |
| **IAM** | Service account identity and least-privilege access control |

```mermaid
flowchart TB
    subgraph Clients["Clients"]
        VSCode["VS Code Copilot"]
        API["Direct API Clients"]
    end

    subgraph CloudRun["Cloud Run (us-central1)"]
        MCP["MCP Server<br/>(rag-mcp)"]
        RAG["RAG Service<br/>(rag-service)"]
        Ingest["Ingestion Job<br/>(rag-ingest)"]
    end

    subgraph GCPServices["GCP Services"]
        AlloyDB["AlloyDB + pgvector<br/>(alloydb-omni-dev)"]
        VertexAI["Vertex AI<br/>(Gemini Embeddings)"]
        GCS["Cloud Storage<br/>(Source Docs)"]
        DocAI["Document AI<br/>(OCR)"]
        SecretMgr["Secret Manager"]
    end

    VSCode -->|"Streamable HTTP<br/>(MCP Protocol)"| MCP
    API -->|"HTTPS + OIDC"| RAG
    MCP -->|"HTTPS + OIDC<br/>(service-to-service)"| RAG
    RAG -->|"asyncpg<br/>(VPC Direct)"| AlloyDB
    RAG -->|"Embed API"| VertexAI
    Ingest -->|"asyncpg<br/>(VPC Direct)"| AlloyDB
    Ingest -->|"Embed API"| VertexAI
    Ingest -->|"Download"| GCS
    Ingest -->|"OCR"| DocAI
    RAG -.->|"Secrets"| SecretMgr
    Ingest -.->|"Secrets"| SecretMgr

    style MCP fill:#4285F4,color:#fff
    style RAG fill:#34A853,color:#fff
    style Ingest fill:#FBBC04,color:#000
    style AlloyDB fill:#DB4437,color:#fff
    style VertexAI fill:#F4B400,color:#000
```
*Diagram 2: High-Level System Architecture — three Cloud Run components connected to GCP services.*

---

## 5. Search Request Flow (End-to-End)

When a data engineer asks a question in VS Code Copilot, here is the full request path:

1. **VS Code Copilot** sends the query to the MCP Server via streamable HTTP transport.
2. **MCP Server** auto-mints an OIDC token from the GCE metadata server (targeting the RAG Service URL as audience).
3. **MCP Server** forwards the query to `POST /v1/search` on the RAG Service with the OIDC token.
4. **RAG Service** verifies the OIDC token, extracts the caller identity.
5. **RAG Service** embeds the query text using Gemini (`RETRIEVAL_QUERY` task type) to produce a 768-dim vector.
6. **RAG Service** opens an RLS-scoped database connection (`set_config('app.tenant_id', ...)`) and executes the hybrid search SQL.
7. **AlloyDB** runs the 8-CTE query: ANN scan on embeddings → vector pool → text pool → RRF fusion → best chunks.
8. **RAG Service** returns structured JSON results (documents with ranked chunks).
9. **MCP Server** formats results for LLM consumption (titles, scores, truncated chunks).
10. **VS Code Copilot** uses the retrieved chunks as context to generate a grounded answer.

```mermaid
sequenceDiagram
    participant VSCode as VS Code Copilot
    participant MCP as MCP Server
    participant Meta as GCE Metadata
    participant RAG as RAG Service
    participant Gemini as Vertex AI (Gemini)
    participant DB as AlloyDB

    VSCode->>MCP: search("how does RLS work?")
    MCP->>Meta: GET /identity?audience=RAG_URL
    Meta-->>MCP: OIDC token (JWT)
    MCP->>RAG: POST /v1/search (Bearer token)
    RAG->>RAG: Verify OIDC token → Identity
    RAG->>Gemini: embed("how does RLS work?", RETRIEVAL_QUERY)
    Gemini-->>RAG: 768-dim query vector
    RAG->>DB: SET LOCAL app.tenant_id, app.user_id
    RAG->>DB: Hybrid search (vector + FTS + RRF)
    DB-->>RAG: Ranked documents + best chunks
    RAG-->>MCP: JSON: results[], rrf_scores
    MCP-->>VSCode: Formatted text for LLM context
    VSCode->>VSCode: LLM generates grounded answer
```
*Diagram 3: Search Request Sequence — from VS Code query to grounded answer.*

---

## 6. Database Design

### 3-Table Separation Rationale

The core data model uses a **documents → chunks → embeddings** separation:

```
rag_documents  →  rag_document_chunks  →  rag_chunk_embeddings
   (1)                 (many)                    (1 per chunk)
```

**Why separate embeddings from chunks?** Embedding models improve over time. When upgrading from `gemini-embedding-001` to a future model, you can re-embed all chunks without re-chunking. Since chunking (especially semantic mode) is the expensive LLM-driven step, this separation saves significant cost and time during re-embedding campaigns.

### All 5 Tables

| Table | Purpose | Key Columns |
|---|---|---|
| `rag_documents` | Document metadata, content, dedup hash, soft delete | `tenant_id`, `visibility`, `owner_user_id`, `content_hash`, `source_hash`, `embedding_model`, `deleted_at` |
| `rag_document_chunks` | Chunk text with positional offsets + auto-generated FTS tsvector | `document_id` (FK), `chunk_index`, `chunk_text`, `fts` (GENERATED tsvector) |
| `rag_chunk_embeddings` | Vector embeddings separated from chunks | `chunk_id` (FK), `embedding` (VECTOR(768)) |
| `rag_ingestion_runs` | Batch ingestion run tracking | `tenant_id`, `status`, `total_files`, `processed_files` |
| `rag_ingestion_items` | Per-file status within an ingestion run | `run_id` (FK), `document_id` (FK), `source_uri`, `status` |

### Indexes

| Index | Type | Purpose |
|---|---|---|
| `ix_rag_chunks_fts` | GIN | Full-text search on `fts` tsvector column |
| ScaNN (manual) | ANN | Approximate Nearest Neighbor on `embedding` column |
| `ux_rag_docs_team_source_uri` | Partial UNIQUE | Dedup: one TEAM doc per `(tenant_id, source_uri)` |
| `ux_rag_docs_team_dedup_adhoc` | Partial UNIQUE | Dedup: TEAM ad-hoc docs by `(tenant_id, content_hash)` |
| `ux_rag_docs_private_dedup` | Partial UNIQUE | Dedup: PRIVATE docs by `(tenant_id, owner_user_id, content_hash)` |
| `ix_rag_docs_filter` | B-tree | Fast filtering on `(tenant_id, visibility, owner_user_id, deleted_at)` |

### Extensions

- **pgvector** — Adds the `VECTOR(768)` data type and cosine distance operator (`<=>`)
- **pgcrypto** — Provides `gen_random_uuid()` for primary key generation

```mermaid
erDiagram
    rag_documents {
        uuid id PK
        text tenant_id
        text visibility
        text owner_user_id
        text title
        text doc_type
        text source_uri
        text content_hash
        text source_hash
        text embedding_model
        int embedding_dim
        timestamptz deleted_at
    }

    rag_document_chunks {
        uuid id PK
        uuid document_id FK
        int chunk_index
        text chunk_text
        int chunk_start
        int chunk_end
        tsvector fts
    }

    rag_chunk_embeddings {
        uuid id PK
        uuid chunk_id FK
        vector768 embedding
    }

    rag_ingestion_runs {
        uuid id PK
        text tenant_id
        text status
        int total_files
        int processed_files
        timestamptz completed_at
    }

    rag_ingestion_items {
        uuid id PK
        uuid run_id FK
        uuid document_id FK
        text source_uri
        text status
    }

    rag_documents ||--o{ rag_document_chunks : "has many"
    rag_document_chunks ||--|| rag_chunk_embeddings : "has one"
    rag_ingestion_runs ||--o{ rag_ingestion_items : "tracks"
    rag_ingestion_items }o--|| rag_documents : "links to"
```
*Diagram 4: Database Schema — 5 tables with the 3-table core separation (documents → chunks → embeddings).*

---

## 7. Hybrid Search Algorithm (RRF)

### The Problem

Vector search alone misses exact keyword matches (table names, error codes). Full-text search alone misses semantic similarity ("connection failed" vs "timeout exception"). The platform combines both using Reciprocal Rank Fusion.

### RRF Formula

```
rrf_score = 1/(K + rank_vector) + 1/(K + rank_text)
```

Where `K = 60` (configurable via `RAG_RRF_K`). Documents found by only one method get a contribution of `0` from the other. The constant `K` prevents top-ranked items from dominating — it smooths the score distribution so that being ranked #1 vs #3 matters less than being present in both lists.

### CTE Pipeline (Single Query)

The entire search executes as a single SQL query with 8+ Common Table Expressions (CTEs):

| CTE | Purpose |
|---|---|
| `tsquery_val` | Compute the tsquery once from the search text |
| `ann_candidates` | **ANN-first pattern**: Scan `rag_chunk_embeddings` table alone by cosine distance. This enables ScaNN index usage — joining other tables would prevent the index scan. |
| `vector_candidates` | Join ANN results back to chunks and documents for metadata |
| `vector_pool` | Apply per-document cap (max 3 chunks per doc) to improve diversity, then assign global ranks |
| `text_candidates` | Full-text search: `fts @@ plainto_tsquery(...)` with `ts_rank` scoring |
| `text_pool` | Apply per-document cap and assign global ranks |
| `doc_vector_rrf` / `doc_text_rrf` | `DISTINCT ON (document_id)` — best rank per document in each pool |
| `fused` | `FULL OUTER JOIN` on document_id, compute RRF scores |
| `best_vec` / `best_txt` / `best_chunks` | Select top 2 vector chunks + top 2 text chunks per fused document, deduplicated by `chunk_id` |

```mermaid
flowchart TB
    Query["Search Query"] --> EmbedQ["Embed Query<br/>(768-dim vector)"]
    Query --> TSQ["Build tsquery<br/>(plainto_tsquery)"]

    EmbedQ --> ANN["ANN Scan<br/>(ScaNN index on<br/>rag_chunk_embeddings)"]
    ANN --> VCand["Vector Candidates<br/>(join chunks + docs)"]
    VCand --> VPool["Vector Pool<br/>(per-doc cap, global rank)"]

    TSQ --> FTS["Full-Text Search<br/>(GIN index on<br/>rag_document_chunks)"]
    FTS --> TPool["Text Pool<br/>(per-doc cap, global rank)"]

    VPool --> DocV["Best Vector Rank<br/>per Document"]
    TPool --> DocT["Best Text Rank<br/>per Document"]

    DocV --> Fuse["RRF Fusion<br/>(FULL OUTER JOIN)<br/>score = 1/(K+rank_v) + 1/(K+rank_t)"]
    DocT --> Fuse

    Fuse --> BestC["Best Chunks<br/>(2 vector + 2 text per doc,<br/>deduped by chunk_id)"]
    BestC --> Results["Ranked Results<br/>with Chunks"]

    style Query fill:#4285F4,color:#fff
    style ANN fill:#34A853,color:#fff
    style FTS fill:#34A853,color:#fff
    style Fuse fill:#FBBC04,color:#000
    style Results fill:#EA4335,color:#fff
```
*Diagram 5: Hybrid Search Pipeline — parallel vector and text branches converge at RRF fusion.*

---

## 8. Multi-Tenant Security (Row-Level Security)

### The Problem

In a multi-tenant database, the traditional approach is adding `WHERE tenant_id = $1` to every query. This is error-prone — one missed `WHERE` clause leaks data across tenants.

### The Solution: PostgreSQL RLS

The platform uses `FORCE ROW LEVEL SECURITY` at the database level. Every table has RLS policies that automatically filter rows based on session variables. The application *cannot* bypass this filtering, even if a developer forgets a `WHERE` clause.

### How It Works

1. **On each request**, the application calls `set_config('app.tenant_id', '<tenant>', true)` and `set_config('app.user_id', '<user>', true)` — these are transaction-scoped session variables.
2. **RLS policies** on every table check `current_setting('app.tenant_id')` to filter rows.
3. **`FORCE ROW LEVEL SECURITY`** ensures policies apply even to the table owner — no backdoor.
4. **Fail-closed**: `rls_connection()` raises `ValueError` if `tenant_id` or `user_id` are empty strings.

### Visibility Rules

| Visibility | Who Can See | Rule |
|---|---|---|
| `TEAM` | All users in the same tenant | `tenant_id` match + `owner_user_id IS NULL` |
| `PRIVATE` | Only the document owner | `tenant_id` match + `owner_user_id` match |
| Soft-deleted | Nobody | `deleted_at IS NOT NULL` → hidden from all |

### RLS Cascade

- **`rag_documents`** — Direct `tenant_id` check
- **`rag_document_chunks`** — `EXISTS` subquery checking parent document's RLS
- **`rag_chunk_embeddings`** — `EXISTS` subquery through chunks → documents

This means the search SQL contains **zero `WHERE tenant_id =` clauses** — RLS handles all filtering transparently.

```mermaid
flowchart LR
    Req["HTTP Request<br/>(Bearer token)"] --> Auth["Auth Middleware<br/>verify OIDC token"]
    Auth --> ID["Extract Identity<br/>(tenant_id, user_id)"]
    ID --> RLS["rls_connection()<br/>acquire DB conn"]
    RLS --> Set["set_config()<br/>app.tenant_id<br/>app.user_id"]
    Set --> SQL["Execute Query<br/>(no WHERE tenant_id)"]
    SQL --> Pol["RLS Policies<br/>auto-filter rows"]
    Pol --> Res["Return Only<br/>Visible Rows"]

    style Req fill:#4285F4,color:#fff
    style Auth fill:#FBBC04,color:#000
    style RLS fill:#34A853,color:#fff
    style Pol fill:#EA4335,color:#fff
```
*Diagram 6: RLS Enforcement Flow — auth extracts identity, RLS policies filter rows automatically.*

---

## 9. Authentication

### Two Auth Modes

| Mode | When | How |
|---|---|---|
| **Cloud Run OIDC** | Production (`K_SERVICE` set) | Google identity tokens verified via `google.oauth2.id_token.verify_token()`. Extracts `email` or `sub` claim as `user_id`. |
| **Shared bearer token** | Local dev only | `RAG_SHARED_TOKEN` env var. **Automatically disabled on Cloud Run** for safety. |

### MCP Server Token Priority

The MCP server must authenticate to the RAG Service. Token selection follows a strict priority:

1. **OIDC service token** — Auto-minted via the GCE metadata server using the MCP service account, targeting `RAG_SERVICE_URL` as the audience. Cached with 5-minute pre-expiry refresh. Uses only stdlib (`urllib.request`), no extra dependencies.
2. **Caller-forwarded token** — When `RAG_MCP_FORWARD_CALLER_TOKEN=true` and a caller token is available in the MCP context.
3. **Static `RAG_MCP_TOKEN`** — Local dev fallback.

### Public Paths (No Auth)

`/liveness`, `/readiness`, `/docs`, `/openapi.json`

```mermaid
sequenceDiagram
    participant Client as Client
    participant MCP as MCP Server
    participant Meta as GCE Metadata
    participant RAG as RAG Service
    participant Google as Google OIDC

    Note over Client,RAG: Path A: Via MCP Server (VS Code)
    Client->>MCP: MCP tool call (search)
    MCP->>Meta: Request OIDC token<br/>(audience=RAG_SERVICE_URL)
    Meta-->>MCP: JWT (cached 55 min)
    MCP->>RAG: POST /v1/search<br/>Authorization: Bearer JWT
    RAG->>Google: Verify token signature + claims
    Google-->>RAG: Valid (email, tenant_id)
    RAG-->>MCP: Search results

    Note over Client,RAG: Path B: Direct API (OIDC)
    Client->>RAG: POST /v1/search<br/>Authorization: Bearer OIDC_TOKEN
    RAG->>Google: Verify token
    Google-->>RAG: Valid
    RAG-->>Client: Search results

    Note over Client,RAG: Path C: Local Dev (shared token)
    Client->>RAG: POST /v1/search<br/>Authorization: Bearer dev-token
    RAG->>RAG: Match RAG_SHARED_TOKEN<br/>(only if K_SERVICE not set)
    RAG-->>Client: Search results
```
*Diagram 7: Authentication Flow — three paths: MCP OIDC, direct OIDC, and local dev token.*

---

## 10. Ingestion Pipeline

### Purpose

The ingestion pipeline imports documents from Google Cloud Storage, extracts text (with OCR fallback for scanned content), chunks the text, generates embeddings, and stores everything in AlloyDB. It runs as a Cloud Run Job or locally via CLI.

### Supported File Types

| Type | Extractor | OCR Fallback |
|---|---|---|
| `.txt`, `.md` | `TextExtractor` | N/A |
| `.html`, `.htm` | `HtmlExtractor` (BeautifulSoup) | N/A |
| `.docx` | `DocxExtractor` (python-docx) | N/A |
| `.pdf` | `PdfExtractor` (pypdf) | Document AI if < 200 chars/page |
| Images (`.png`, `.jpg`, etc.) | `ImageExtractor` | Document AI online OCR |

### Pipeline Stages

1. **Discover** — List GCS objects under `gs://<bucket>/<tenant_id>/<prefix>/`
2. **Incremental Check** — Compute `source_hash` from GCS metadata (generation, md5, crc32c, size, updated). If unchanged, skip.
3. **Download** — Fetch file bytes from GCS (blocking I/O runs in thread pool)
4. **Extract** — Route to appropriate extractor. PDFs fall back to Document AI OCR if extracted text is too sparse (< 200 chars/page).
5. **Chunk** — Rule-based recursive splitting (2000 chars, 200 overlap) with character offset tracking.
6. **Embed** — Batch embed chunks via Gemini (100/batch, 8 concurrent batches, exponential backoff).
7. **Upsert** — `upsert_document_by_source_uri()` atomically replaces chunks + embeddings on content change. Returns `unchanged` if content and settings match.

### Run Tracking

Every ingestion run creates a row in `rag_ingestion_runs` and per-file rows in `rag_ingestion_items`. Status transitions: `pending` → `processing` → `completed` | `skipped` | `failed`.

```mermaid
flowchart TB
    Start["CLI / Cloud Run Job<br/>(python -m rag_service.ingestion.main)"] --> Discover["Discover GCS Objects<br/>(gs://bucket/tenant/prefix/)"]
    Discover --> Loop["For Each File<br/>(concurrent workers)"]

    Loop --> Hash["Compute Source Hash<br/>(generation + md5 + crc32c + size + updated)"]
    Hash --> Changed{"File Changed?"}
    Changed -->|"No (hash match)"| Skip["Skip<br/>(mark 'skipped')"]
    Changed -->|"Yes / Force"| Download["Download from GCS"]

    Download --> Extract["Extract Text"]
    Extract --> OCR{"Text Sparse?<br/>(< 200 chars/page)"}
    OCR -->|"Yes (PDF/Image)"| DocAI["Document AI OCR"]
    OCR -->|"No"| Chunk
    DocAI --> Chunk["Chunk Text<br/>(rule-based, 2000 chars,<br/>200 overlap)"]

    Chunk --> Embed["Embed Chunks<br/>(Gemini, 100/batch,<br/>8 concurrent)"]
    Embed --> Upsert["Atomic Upsert<br/>(upsert_document_by_source_uri)"]
    Upsert --> Done["Mark 'completed'"]

    Skip --> Track["Update rag_ingestion_items"]
    Done --> Track
    Track --> Final["Finalize rag_ingestion_runs<br/>(completed/failed status)"]

    style Start fill:#4285F4,color:#fff
    style Changed fill:#FBBC04,color:#000
    style OCR fill:#FBBC04,color:#000
    style Skip fill:#9E9E9E,color:#fff
    style Upsert fill:#34A853,color:#fff
```
*Diagram 8: Ingestion Pipeline — from GCS discovery through extraction, chunking, embedding, to atomic upsert.*

---

## 11. Embedding and Chunking

### Embedding

| Setting | Value |
|---|---|
| **Model** | `gemini-embedding-001` |
| **Dimensions** | 768 |
| **Normalization** | L2-normalized before storage |
| **Task types** | `RETRIEVAL_DOCUMENT` (indexing), `RETRIEVAL_QUERY` (search) |
| **Batch size** | 100 texts per API call |
| **Concurrency** | 8 concurrent batch calls |
| **Retries** | 2 retries with exponential backoff (0.5s base) |
| **Environment detection** | Vertex AI with ADC on GCP; `GEMINI_API_KEY` for local dev |

**Dim guard (fail-closed):** If the returned vector dimension doesn't match `RAG_EMBEDDING_DIM` (768), the embedding call raises `ValueError`. There is no zero-vector fallback — the system fails loudly to prevent corrupted index data.

**Threading:** The Gemini SDK is synchronous. All embedding calls run via `loop.run_in_executor()` to avoid blocking the async event loop. The client is `@lru_cache`d (one instance per process).

### Chunking

Two strategies are available:

| Strategy | How It Works | Default Settings |
|---|---|---|
| **Rule-based** (default) | Recursive hierarchical splitting: paragraphs → lines → sentences → words → characters, with inter-chunk overlap | 2000 chars chunk size, 200 chars overlap |
| **Semantic** | LLM-driven topic-shift detection via Gemini | N/A (model decides boundaries) |

The rule-based chunker also computes character offsets (`chunk_start`, `chunk_end`) for each chunk, enabling future search highlighting. Semantic mode returns `None` offsets since the LLM determines boundaries dynamically.

---

## 12. MCP Server (VS Code Integration)

### What is MCP?

The [Model Context Protocol](https://modelcontextprotocol.io/) is an open standard for connecting LLM applications to external data sources. VS Code Copilot supports MCP, allowing it to call custom tools during conversations.

### Architecture

The MCP server is a lightweight proxy — it has **no database access** and **no embedding logic**. It forwards all requests to the RAG Service over HTTPS with automatic OIDC authentication.

### Exposed Tools

| Tool | Parameters | Description |
|---|---|---|
| `search` | `query` (str), `limit` (int, default 10) | Search the knowledge base by natural language query. Returns formatted results with titles, scores, and truncated chunks. |
| `list_documents` | `limit` (int, default 20), `offset` (int, default 0) | Browse available documents with metadata and chunk counts. |

### Production Hardening

- **Retry on transient failures**: Retries on HTTP 502, 503, 504 with exponential backoff (0.5s base, max 2 retries).
- **Sanitized error messages**: Internal errors are mapped to user-safe messages (no stack traces or internal URLs leaked to LLM context).
- **Result formatting**: Chunks are truncated to 500 characters for LLM consumption. Scores are included for ranking transparency.
- **Transport**: Streamable HTTP (not SSE), configured via `FastMCP` from the `mcp` Python SDK.

---

## 13. Deployment Topology

### Cloud Run Services

| Component | Cloud Run Resource | VPC Access | Min/Max Instances | CPU | Memory |
|---|---|---|---|---|---|
| RAG Service | `rag-service` (Service) | Direct VPC egress to AlloyDB | 0 / 5 | 1 vCPU | 512 MiB |
| MCP Server | `rag-mcp` (Service) | None (HTTPS only) | 0 / 3 | 1 vCPU | 256 MiB |
| Ingestion | `rag-ingest` (Job) | Direct VPC egress to AlloyDB | 1 task | 2 vCPU | 2 GiB |

### Service Accounts (Least Privilege)

| Account | Roles |
|---|---|
| `rag-service@apexflow-ai.iam` | AlloyDB Client, Secret Manager Secret Accessor, Vertex AI User |
| `rag-mcp@apexflow-ai.iam` | Cloud Run Invoker (on `rag-service`), Service Account Token Creator |
| `rag-ingest@apexflow-ai.iam` | AlloyDB Client, Storage Object Viewer, Vertex AI User, Document AI User |

### Infrastructure

- **AlloyDB instance**: `alloydb-omni-dev` on GCE VM in `us-central1-a` (shared with ApexFlow, private IP `10.128.0.3`). Independent `rag_*` tables with separate Alembic migration chain (`rag_alembic_version` table).
- **Container images**: `us-central1-docker.pkg.dev/apexflow-ai/rag/{rag-service,rag-mcp,rag-ingestor}`
- **Secrets**: `rag-alloydb-password`, `rag-oidc-audience` in Secret Manager
- **GCS bucket**: `gs://rag-ingest-apexflow-ai` for source documents

```mermaid
flowchart TB
    subgraph Internet["Public Internet"]
        VSCode["VS Code Copilot"]
        APIClient["API Clients"]
    end

    subgraph GCP["GCP Project: apexflow-ai (us-central1)"]
        subgraph CloudRun["Cloud Run"]
            MCP["rag-mcp<br/>(Service)<br/>1 vCPU / 256 MiB"]
            Service["rag-service<br/>(Service)<br/>1 vCPU / 512 MiB"]
            Job["rag-ingest<br/>(Job)<br/>2 vCPU / 2 GiB"]
        end

        subgraph VPC["VPC Network"]
            AlloyDB["AlloyDB Omni<br/>(alloydb-omni-dev)<br/>10.128.0.3"]
        end

        VertexAI["Vertex AI<br/>(Gemini Embeddings)"]
        GCS["Cloud Storage<br/>(rag-ingest-apexflow-ai)"]
        DocAI["Document AI"]
        Secrets["Secret Manager"]
        AR["Artifact Registry<br/>(rag images)"]
        IAM["IAM<br/>(3 service accounts)"]
    end

    VSCode -->|"HTTPS"| MCP
    APIClient -->|"HTTPS + OIDC"| Service
    MCP -->|"HTTPS + OIDC"| Service
    Service -->|"VPC Direct Egress"| AlloyDB
    Job -->|"VPC Direct Egress"| AlloyDB
    Service -->|"API"| VertexAI
    Job -->|"API"| VertexAI
    Job -->|"Read"| GCS
    Job -->|"OCR"| DocAI
    Service -.->|"Mount"| Secrets
    Job -.->|"Mount"| Secrets
    CloudRun -.->|"Pull"| AR
    CloudRun -.->|"Auth"| IAM

    style MCP fill:#4285F4,color:#fff
    style Service fill:#34A853,color:#fff
    style Job fill:#FBBC04,color:#000
    style AlloyDB fill:#DB4437,color:#fff
```
*Diagram 9: GCP Deployment Topology — Cloud Run services and jobs, VPC boundary, and supporting GCP services.*

---

## 14. Key Advantages

- **Database-level multi-tenancy** — PostgreSQL `FORCE ROW LEVEL SECURITY` eliminates the risk of data leaks from missing `WHERE` clauses. RLS policies cascade from documents to chunks to embeddings.
- **Hybrid search with RRF** — Combines vector cosine similarity and full-text keyword matching, outperforming either method alone. A single SQL query (8+ CTEs) handles fusion, ranking, and chunk selection.
- **ANN-first pattern** — Scans the embeddings table alone during approximate nearest neighbor search, enabling ScaNN index usage. Metadata joins happen after the ANN scan, not during.
- **3-table re-embedding** — Documents → chunks → embeddings separation allows re-embedding (model upgrades) without re-chunking, saving the most expensive step.
- **Incremental ingestion** — Source hash from GCS metadata (generation, md5, crc32c, size, updated) skips unchanged files, making re-runs fast.
- **Atomic upserts** — `upsert_document_by_source_uri()` atomically replaces chunks and embeddings on content change within a single transaction.
- **Fail-closed security** — Empty `tenant_id`/`user_id` raises `ValueError`. No zero-vector embedding fallback. Shared dev token auto-disabled on Cloud Run. OIDC audience required in production.
- **IDE integration via MCP** — Data engineers search documents without leaving VS Code. The MCP server is a thin proxy with automatic OIDC auth, retry on transient failures, and sanitized error messages.
- **Production hardening** — Rate limiting (60/min global, 30/min search, 10/min index), 10 MB request body limit, structured JSON logging with request ID correlation, CORS configuration, and health checks.

---

## 15. Configuration Reference

All configuration is driven by environment variables. No `settings.json` files.

### Tenant and Auth

| Variable | Purpose | Default |
|---|---|---|
| `TENANT_ID` | Fixed tenant per environment | `default` |
| `RAG_SHARED_TOKEN` | Dev-only shared bearer token (ignored on Cloud Run) | unset |
| `RAG_OIDC_AUDIENCE` | Expected audience in OIDC tokens (required on Cloud Run) | unset |
| `RAG_ALLOWED_ISSUERS` | Comma-separated allowed OIDC issuers | `https://accounts.google.com,accounts.google.com` |
| `RAG_TENANT_CLAIM` | JWT claim key for tenant ID | `tenant_id` |
| `RAG_REQUIRE_TENANT_CLAIM` | Reject tokens missing the tenant claim | `false` |

### Embedding

| Variable | Purpose | Default |
|---|---|---|
| `RAG_EMBEDDING_MODEL` | Gemini embedding model name | `gemini-embedding-001` |
| `RAG_EMBEDDING_DIM` | Expected embedding dimension (fail-closed guard) | `768` |
| `RAG_EMBED_BATCH_SIZE` | Texts per embedding API call | `100` |
| `RAG_EMBED_MAX_CONCURRENCY` | Concurrent embedding batch calls | `8` |
| `RAG_EMBED_MAX_RETRIES` | Retry attempts for failed embedding calls | `2` |
| `RAG_EMBED_RETRY_BASE_SECONDS` | Base delay for exponential backoff | `0.5` |
| `GEMINI_API_KEY` | Gemini API key for local dev (not needed on GCP with ADC) | unset |
| `GOOGLE_CLOUD_PROJECT` | GCP project for Vertex AI | `apexflow-ai` |
| `GOOGLE_CLOUD_LOCATION` | GCP region for Vertex AI | `us-central1` |

### Search

| Variable | Purpose | Default |
|---|---|---|
| `RAG_RRF_K` | RRF fusion constant (higher = smoother ranking) | `60` |
| `RAG_SEARCH_EXPANSION` | Pool expansion factor (`chunk_limit = doc_limit * expansion`) | `3` |
| `RAG_SEARCH_PER_DOC_CAP` | Max chunks per document in each pool (diversity) | `3` |
| `RAG_SEARCH_CANDIDATE_MULTIPLIER` | ANN over-fetch multiplier (`candidate_limit = chunk_limit * multiplier`) | `4` |
| `RAG_FTS_LANGUAGE` | PostgreSQL text search language configuration | `english` |

### Chunking

| Variable | Purpose | Default |
|---|---|---|
| `RAG_CHUNK_SIZE` | Target chunk size in characters | `2000` |
| `RAG_CHUNK_OVERLAP` | Overlap between adjacent chunks | `200` |

### Database

| Variable | Purpose | Default |
|---|---|---|
| `DATABASE_URL` | Full connection string (overrides all `DB_*` vars) | unset |
| `DB_HOST` / `DB_PORT` / `DB_USER` / `DB_PASSWORD` / `DB_NAME` | Individual connection params | `localhost:5432`, user `apexflow` |
| `DB_POOL_MAX` | Max async connection pool size | `5` |
| `DATABASE_TEST_URL` | Test database URL for integration tests | `postgresql://apexflow:apexflow@localhost:5432/apexflow` |

### Ingestion

| Variable | Purpose | Default |
|---|---|---|
| `RAG_INGEST_INPUT_BUCKET` | GCS bucket for source documents | (required) |
| `RAG_INGEST_INPUT_PREFIX` | Prefix under each tenant directory | `incoming/` |
| `RAG_INGEST_TENANTS` | Comma-separated tenant allowlist | unset (all) |
| `RAG_INGEST_INCREMENTAL` | Skip unchanged files by source hash | `true` |
| `RAG_INGEST_FORCE_REINDEX` | Force re-process all files | `false` |
| `RAG_INGEST_MAX_FILE_WORKERS` | Concurrent file processing workers | `3` |
| `RAG_INGEST_MAX_RETRIES_PER_FILE` | Retry attempts per failed file | `2` |
| `RAG_INGEST_OUTPUT_BUCKET` | GCS bucket for extracted text artifacts | unset |
| `RAG_INGEST_OUTPUT_PREFIX` | Prefix for output artifacts | `rag-extracted/` |
| `RAG_MAX_CONTENT_CHARS` | Truncate documents exceeding this length | `2000000` |
| `RAG_INGESTION_VERSION` | Ingestion version tag | `v1` |

### OCR (Document AI)

| Variable | Purpose | Default |
|---|---|---|
| `RAG_OCR_ENABLED` | Enable Document AI OCR for scanned PDFs/images | `true` |
| `RAG_DOC_AI_PROJECT` | GCP project for Document AI | unset |
| `RAG_DOC_AI_LOCATION` | GCP region for Document AI | unset |
| `RAG_DOC_AI_PROCESSOR_ID` | Document AI OCR processor ID | unset |
| `RAG_PDF_TEXT_PER_PAGE_MIN` | Chars/page threshold below which PDF falls back to OCR | `200` |

### MCP Server

| Variable | Purpose | Default |
|---|---|---|
| `RAG_SERVICE_URL` | RAG API URL the MCP server forwards to | `http://localhost:8000` |
| `RAG_MCP_TOKEN` | Static auth token for MCP → RAG calls (dev) | unset |
| `RAG_MCP_FORWARD_CALLER_TOKEN` | Forward caller's bearer token to RAG service | `true` |
| `MCP_PORT` | MCP server listen port | `8001` |

See `.env.example` for a template with all required variables.

---

## 16. API and MCP Tools Reference

### REST API Endpoints

| Method | Path | Auth | Rate Limit | Purpose |
|---|---|---|---|---|
| `POST` | `/v1/search` | Required | 30/min | Hybrid search: embed query → RLS → RRF fusion |
| `GET` | `/v1/documents` | Required | 60/min | List visible documents (TEAM + owned PRIVATE) |
| `DELETE` | `/v1/documents/{id}` | Required | 60/min | Soft delete (owner-only for PRIVATE) |
| `POST` | `/v1/index` | Required | 10/min | Index a document: chunk → embed → store |
| `GET` | `/liveness` | Public | -- | Health check (always returns OK) |
| `GET` | `/readiness` | Public | -- | DB + embedding service connectivity check |

### MCP Tools

| Tool | Parameters | Description |
|---|---|---|
| `search` | `query` (str, required), `limit` (int, 1-50, default 10) | Search the team's document knowledge base by natural language query |
| `list_documents` | `limit` (int, default 20), `offset` (int, default 0) | Browse available documents with titles, types, visibility, and chunk counts |

### Further Reading

- **Deployment guide**: [`docs/deployment.md`](deployment.md)
- **VS Code MCP setup**: [`docs/vscode-mcp-setup.md`](vscode-mcp-setup.md)
- **Ingestion plan**: [`docs/alloy_db_manual_ingestion_implementation_plan_v_1.md`](alloy_db_manual_ingestion_implementation_plan_v_1.md)
