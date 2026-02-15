"""FastAPI entry point for the RAG platform service.

Endpoints:
- POST /v1/search       — Hybrid search (embed query -> RLS -> RRF)
- GET  /v1/documents    — List visible documents
- DELETE /v1/documents/{id} — Soft delete (owner-only for PRIVATE)
- POST /v1/index        — Index a document (MVP manual upload)
- GET  /liveness        — Health check
- GET  /readiness       — DB connectivity check
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, cast

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from rag_service.auth import Identity, get_identity, is_public_path, require_auth_on_cloud_run
from rag_service.chunking.chunker import chunk_document
from rag_service.config import (
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_CORS_ALLOW_CREDENTIALS,
    RAG_CORS_ALLOW_HEADERS,
    RAG_CORS_ALLOW_METHODS,
    RAG_CORS_ALLOW_ORIGINS,
)
from rag_service.db import check_db_connection, close_pool, get_pool, rls_connection
from rag_service.embedding import check_embedding_service, embed_chunks, embed_query
from rag_service.logging_config import generate_request_id, setup_logging
from rag_service.models import (
    ChunkResult,
    DeleteResponse,
    DocumentListResponse,
    DocumentSummary,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    SearchDebug,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from rag_service.stores.rag_document_store import RagDocumentStore
from rag_service.stores.rag_search_store import RagSearchStore

logger = logging.getLogger(__name__)

_doc_store = RagDocumentStore()
_search_store = RagSearchStore()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: init pool on startup, close on shutdown."""
    setup_logging()
    require_auth_on_cloud_run()
    await get_pool()
    logger.info("RAG service started")
    yield
    await close_pool()
    logger.info("RAG service stopped")


app = FastAPI(
    title="RAG Platform API",
    version="0.1.0",
    lifespan=lifespan,
)

# -- Rate limiting ------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})


if RAG_CORS_ALLOW_CREDENTIALS and "*" in RAG_CORS_ALLOW_ORIGINS:
    raise RuntimeError("Invalid CORS config: wildcard origin cannot be combined with credentials=true")

app.add_middleware(
    CORSMiddleware,
    allow_origins=RAG_CORS_ALLOW_ORIGINS,
    allow_credentials=RAG_CORS_ALLOW_CREDENTIALS,
    allow_methods=RAG_CORS_ALLOW_METHODS,
    allow_headers=RAG_CORS_ALLOW_HEADERS,
)


# -- Body size limit ----------------------------------------------------------

_MAX_BODY_BYTES = 10 * 1024 * 1024  # 10 MB


@app.middleware("http")
async def body_size_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Reject requests with bodies exceeding the size limit."""
    content_length = request.headers.get("content-length")
    if content_length is not None and int(content_length) > _MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"detail": "Request body too large"})
    return await call_next(request)


# -- Auth middleware ----------------------------------------------------------


@app.middleware("http")
async def auth_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Enforce authentication on all non-public paths."""
    if request.method == "OPTIONS" or is_public_path(request.url.path):
        return await call_next(request)

    try:
        identity = await get_identity(request)
        request.state.identity = identity
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    except Exception as e:
        logger.warning("Auth middleware error: %s", e)
        return JSONResponse(status_code=401, content={"detail": "Authentication failed"})

    return await call_next(request)


# -- Request ID middleware ----------------------------------------------------


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Attach a unique request ID for trace correlation."""
    request_id = request.headers.get("x-request-id") or generate_request_id()
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


def _get_identity(request: Request) -> Identity:
    """Dependency: extract identity from request state (set by middleware)."""
    identity = getattr(request.state, "identity", None)
    if identity is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return cast(Identity, identity)


# -- Health -------------------------------------------------------------------


@app.get("/liveness", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/readiness", response_model=HealthResponse)
async def readiness() -> HealthResponse:
    db_ok = await check_db_connection()
    if not db_ok:
        raise HTTPException(status_code=503, detail="Database unavailable")
    embedding_ok = await check_embedding_service()
    if not embedding_ok:
        return HealthResponse(status="degraded", error="Embedding service unavailable")
    return HealthResponse(status="ok")


# -- Search -------------------------------------------------------------------


@app.post("/v1/search", response_model=SearchResponse)
@limiter.limit("30/minute")
async def search(
    request: Request,
    body: SearchRequest,
    identity: Annotated[Identity, Depends(_get_identity)],
) -> SearchResponse:
    """Hybrid search: embed query -> RLS-scoped search -> RRF fusion."""
    query_text = body.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query must not be blank")

    try:
        query_vec = await embed_query(query_text)
    except Exception as e:
        logger.exception("Failed to generate query embedding")
        raise HTTPException(status_code=503, detail="Embedding service unavailable") from e

    async with rls_connection(identity.tenant_id, identity.user_id) as conn:
        raw = await _search_store.search_hybrid(
            conn,
            query_text,
            query_vec,
            doc_limit=body.limit,
            include_debug=body.include_debug,
        )

    results = [
        SearchResult(
            document_id=r["document_id"],
            title=r["title"],
            doc_type=r.get("doc_type"),
            rrf_score=r["rrf_score"],
            vector_score=r["vector_score"],
            text_score=r["text_score"],
            chunks=[
                ChunkResult(
                    chunk_id=c["chunk_id"],
                    chunk_index=c["chunk_index"],
                    chunk_text=c["chunk_text"],
                    source=c["source"],
                    score=c["score"],
                )
                for c in r.get("chunks", [])
            ],
        )
        for r in raw["results"]
    ]

    debug = None
    if raw.get("debug"):
        d = raw["debug"]
        debug = SearchDebug(
            vector_pool_size=d["vector_pool_size"],
            text_pool_size=d["text_pool_size"],
            vector_has_more=d["vector_has_more"],
            text_has_more=d["text_has_more"],
            vector_cutoff_score=d.get("vector_cutoff_score"),
            text_cutoff_score=d.get("text_cutoff_score"),
        )

    return SearchResponse(results=results, debug=debug)


# -- Documents ----------------------------------------------------------------


@app.get("/v1/documents", response_model=DocumentListResponse)
async def list_documents(
    identity: Annotated[Identity, Depends(_get_identity)],
    limit: int = 50,
    offset: int = 0,
) -> DocumentListResponse:
    """List visible documents (TEAM + owned PRIVATE, enforced by RLS)."""
    limit = min(max(limit, 1), 200)
    offset = max(offset, 0)
    async with rls_connection(identity.tenant_id, identity.user_id) as conn:
        docs, total = await _doc_store.list_documents(conn, limit=limit, offset=offset)

    return DocumentListResponse(
        documents=[
            DocumentSummary(
                id=str(d["id"]),
                title=d["title"],
                doc_type=d.get("doc_type"),
                source_uri=d.get("source_uri"),
                visibility=d.get("visibility", "TEAM"),
                total_chunks=d.get("total_chunks"),
                embedding_model=d.get("embedding_model"),
                created_at=d.get("created_at"),
                updated_at=d.get("updated_at"),
            )
            for d in docs
        ],
        total=total,
    )


@app.delete("/v1/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document(
    doc_id: str,
    identity: Annotated[Identity, Depends(_get_identity)],
) -> DeleteResponse:
    """Soft-delete a document. RLS ensures only visible/owned docs can be deleted."""
    async with rls_connection(identity.tenant_id, identity.user_id) as conn:
        deleted = await _doc_store.soft_delete(conn, doc_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found or not authorized")

    return DeleteResponse(deleted=True, document_id=doc_id)


# -- Index (MVP) --------------------------------------------------------------


@app.post("/v1/index", response_model=IndexResponse)
@limiter.limit("10/minute")
async def index_document(
    request: Request,
    body: IndexRequest,
    identity: Annotated[Identity, Depends(_get_identity)],
) -> IndexResponse:
    """Index a document: validate -> hash -> dedup -> chunk -> embed -> store."""
    content = body.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Content must not be blank")

    # Determine owner_user_id based on visibility
    owner_user_id = identity.user_id if body.visibility == "PRIVATE" else None

    # Pre-check for duplicates before expensive chunking/embedding
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    async with rls_connection(identity.tenant_id, identity.user_id) as conn:
        dedup = await _doc_store.check_dedup(
            conn,
            content_hash=content_hash,
            visibility=body.visibility,
            owner_user_id=owner_user_id,
        )
    if dedup is not None:
        return IndexResponse(
            document_id=dedup["document_id"],
            status="deduplicated",
            total_chunks=dedup["total_chunks"],
        )

    # Chunk the document
    chunks = await chunk_document(
        content,
        method="rule_based",
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
    )
    if not chunks:
        raise HTTPException(status_code=400, detail="Document produced no chunks")

    # Embed all chunks
    try:
        embeddings = await embed_chunks(chunks)
    except Exception as e:
        logger.exception("Failed to generate document embeddings")
        raise HTTPException(status_code=503, detail="Embedding service unavailable") from e

    # Store via RLS-scoped connection
    async with rls_connection(identity.tenant_id, identity.user_id) as conn:
        result = await _doc_store.upsert_document(
            conn,
            tenant_id=identity.tenant_id,
            title=body.title,
            content=content,
            chunks=chunks,
            embeddings=embeddings,
            visibility=body.visibility,
            owner_user_id=owner_user_id,
            doc_type=body.doc_type,
            metadata=body.metadata,
        )

    return IndexResponse(
        document_id=result["document_id"],
        status=result["status"],
        total_chunks=result["total_chunks"],
    )
