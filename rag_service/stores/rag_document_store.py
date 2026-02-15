"""CRUD operations for rag_documents, rag_document_chunks, and rag_chunk_embeddings.

All methods expect a connection with RLS session variables already set
(via db.rls_connection). Tenant isolation is enforced by PostgreSQL RLS
policies — no WHERE user_id = needed in application SQL.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any

import asyncpg

from rag_service.config import (
    RAG_EMBEDDING_DIM,
    RAG_EMBEDDING_MODEL,
    RAG_INGESTION_VERSION,
)

logger = logging.getLogger(__name__)


class RagDocumentStore:
    """Stateless data-access object for rag_documents and related tables."""

    async def upsert_document(
        self,
        conn: asyncpg.Connection,
        *,
        tenant_id: str,
        title: str,
        content: str,
        chunks: list[str],
        embeddings: list[list[float]],
        visibility: str = "TEAM",
        owner_user_id: str | None = None,
        doc_type: str | None = None,
        source_uri: str | None = None,
        metadata: dict[str, Any] | None = None,
        chunk_method: str = "rule_based",
    ) -> dict[str, Any]:
        """Index or update a document with chunks and embeddings.

        Uses content-hash dedup: same tenant + visibility + content_hash +
        owner means the document already exists. If the ingestion settings
        match, skip re-chunking. Otherwise, re-chunk and re-embed.

        Returns dict with document_id, status, total_chunks.
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        doc_id = str(uuid.uuid4())

        import json

        row = await conn.fetchrow(
            """
            INSERT INTO rag_documents
                (id, tenant_id, visibility, owner_user_id, title, doc_type,
                 source_uri, metadata, content, content_hash,
                 embedding_model, embedding_dim, ingestion_version,
                 chunk_method, total_chunks)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10,
                    $11, $12, $13, $14, $15)
            ON CONFLICT (tenant_id, visibility, content_hash, COALESCE(owner_user_id, ''))
                WHERE deleted_at IS NULL
            DO UPDATE SET
                title = EXCLUDED.title,
                doc_type = COALESCE(EXCLUDED.doc_type, rag_documents.doc_type),
                source_uri = COALESCE(EXCLUDED.source_uri, rag_documents.source_uri),
                metadata = COALESCE(EXCLUDED.metadata, rag_documents.metadata),
                content = EXCLUDED.content,
                updated_at = NOW()
            RETURNING id, (xmax = 0) AS is_new,
                      ingestion_version, chunk_method,
                      embedding_model, embedding_dim, total_chunks
            """,
            doc_id,
            tenant_id,
            visibility,
            owner_user_id,
            title,
            doc_type,
            source_uri,
            json.dumps(metadata) if metadata is not None else None,
            content,
            content_hash,
            RAG_EMBEDDING_MODEL,
            RAG_EMBEDDING_DIM,
            RAG_INGESTION_VERSION,
            chunk_method,
            len(chunks),
        )

        actual_id: str = str(row["id"])  # type: ignore[index]
        is_new: bool = row["is_new"]  # type: ignore[index]

        if not is_new:
            # Check if settings match — skip re-chunking if identical
            if (
                row["ingestion_version"] == RAG_INGESTION_VERSION  # type: ignore[index]
                and row["chunk_method"] == chunk_method  # type: ignore[index]
                and row["embedding_model"] == RAG_EMBEDDING_MODEL  # type: ignore[index]
                and row["embedding_dim"] == RAG_EMBEDDING_DIM  # type: ignore[index]
            ):
                return {
                    "document_id": actual_id,
                    "status": "deduplicated",
                    "total_chunks": row["total_chunks"],  # type: ignore[index]
                }

            # Settings changed — delete stale chunks (cascades to embeddings)
            await conn.execute(
                "DELETE FROM rag_document_chunks WHERE document_id = $1",
                uuid.UUID(actual_id),
            )

        # Store chunks and embeddings
        await self._store_chunks_and_embeddings(conn, uuid.UUID(actual_id), chunks, embeddings)

        # For re-indexed docs, update version fields
        if not is_new:
            await conn.execute(
                """
                UPDATE rag_documents
                SET total_chunks = $2,
                    embedding_model = $3,
                    embedding_dim = $4,
                    ingestion_version = $5,
                    chunk_method = $6,
                    updated_at = NOW()
                WHERE id = $1
                """,
                uuid.UUID(actual_id),
                len(chunks),
                RAG_EMBEDDING_MODEL,
                RAG_EMBEDDING_DIM,
                RAG_INGESTION_VERSION,
                chunk_method,
            )

        return {
            "document_id": actual_id,
            "status": "indexed",
            "total_chunks": len(chunks),
        }

    async def list_documents(
        self,
        conn: asyncpg.Connection,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List visible documents (TEAM + owned PRIVATE, enforced by RLS).

        Returns (documents, total_count).
        """
        total = await conn.fetchval("SELECT COUNT(*) FROM rag_documents")

        rows = await conn.fetch(
            """
            SELECT id, title, doc_type, source_uri, visibility,
                   total_chunks, embedding_model, created_at, updated_at
            FROM rag_documents
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )
        return [dict(r) for r in rows], total or 0

    async def get_document(
        self,
        conn: asyncpg.Connection,
        doc_id: str,
    ) -> dict[str, Any] | None:
        """Get a single document by ID (RLS enforces visibility)."""
        row = await conn.fetchrow(
            "SELECT * FROM rag_documents WHERE id = $1",
            uuid.UUID(doc_id),
        )
        return dict(row) if row else None

    async def soft_delete(
        self,
        conn: asyncpg.Connection,
        doc_id: str,
    ) -> bool:
        """Soft-delete a document by setting deleted_at.

        RLS ensures only visible docs can be updated. For PRIVATE docs,
        only the owner can delete.
        """
        tag = await conn.execute(
            """
            UPDATE rag_documents
            SET deleted_at = NOW(), updated_at = NOW()
            WHERE id = $1 AND deleted_at IS NULL
            """,
            uuid.UUID(doc_id),
        )
        return tag == "UPDATE 1"

    async def _store_chunks_and_embeddings(
        self,
        conn: asyncpg.Connection,
        doc_id: uuid.UUID,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Batch-insert chunks and their embeddings into separate tables."""
        chunk_rows = []
        embedding_rows = []

        for idx, (text, emb) in enumerate(zip(chunks, embeddings, strict=True)):
            chunk_id = uuid.uuid4()
            emb_id = uuid.uuid4()
            chunk_rows.append((chunk_id, doc_id, idx, text))
            embedding_rows.append((emb_id, chunk_id, emb))

        # Insert chunks first (FK target for embeddings)
        await conn.executemany(
            """
            INSERT INTO rag_document_chunks (id, document_id, chunk_index, chunk_text)
            VALUES ($1, $2, $3, $4)
            """,
            chunk_rows,
        )

        # Insert embeddings
        await conn.executemany(
            """
            INSERT INTO rag_chunk_embeddings (id, chunk_id, embedding)
            VALUES ($1, $2, $3::vector)
            """,
            embedding_rows,
        )
