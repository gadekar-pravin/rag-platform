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

    async def get_team_document_by_source_uri(
        self,
        conn: asyncpg.Connection,
        *,
        source_uri: str,
    ) -> dict[str, Any] | None:
        row = await conn.fetchrow(
            """
            SELECT id, source_uri, source_hash, content_hash,
                   ingestion_version, chunk_method, embedding_model, embedding_dim, total_chunks
            FROM rag_documents
            WHERE visibility = 'TEAM'
              AND source_uri = $1
              AND deleted_at IS NULL
            """,
            source_uri,
        )
        return dict(row) if row else None

    async def upsert_document_by_source_uri(
        self,
        conn: asyncpg.Connection,
        *,
        tenant_id: str,
        source_uri: str,
        source_hash: str | None,
        title: str,
        content: str,
        chunks: list[str],
        embeddings: list[list[float]],
        doc_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        chunk_method: str = "rule_based",
        chunk_offsets: list[tuple[int | None, int | None]] | None = None,
        skip_if_unchanged: bool = True,
    ) -> dict[str, Any]:
        """Canonical TEAM upsert keyed by (tenant_id, source_uri).

        Requires migration 003 unique index:
          UNIQUE (tenant_id, source_uri) WHERE deleted_at IS NULL AND visibility='TEAM' AND source_uri IS NOT NULL
        """
        if not source_uri:
            raise ValueError("source_uri is required")

        import json

        content_hash = hashlib.sha256(content.encode()).hexdigest()
        meta_json = json.dumps(metadata or {})

        async with conn.transaction():
            # Lock existing canonical doc (prevents races in parallel ingestion)
            existing = await conn.fetchrow(
                """
                SELECT id, content_hash, source_hash,
                       ingestion_version, chunk_method, embedding_model, embedding_dim, total_chunks
                FROM rag_documents
                WHERE tenant_id = $1
                  AND visibility = 'TEAM'
                  AND source_uri = $2
                  AND deleted_at IS NULL
                FOR UPDATE
                """,
                tenant_id,
                source_uri,
            )

            if existing and skip_if_unchanged:
                settings_match = (
                    existing["ingestion_version"] == RAG_INGESTION_VERSION
                    and existing["chunk_method"] == chunk_method
                    and existing["embedding_model"] == RAG_EMBEDDING_MODEL
                    and existing["embedding_dim"] == RAG_EMBEDDING_DIM
                )
                if settings_match and existing["content_hash"] == content_hash and (
                    (source_hash is None) or (existing["source_hash"] == source_hash)
                ):
                    # Still refresh metadata/title/source_hash cheaply
                    await conn.execute(
                        """
                        UPDATE rag_documents
                        SET title = $2,
                            doc_type = $3,
                            metadata = $4::jsonb,
                            content = $5,
                            source_hash = $6,
                            updated_at = NOW()
                        WHERE id = $1
                        """,
                        existing["id"],
                        title,
                        doc_type,
                        meta_json,
                        content,
                        source_hash,
                    )
                    return {
                        "document_id": str(existing["id"]),
                        "status": "unchanged",
                        "total_chunks": int(existing["total_chunks"] or 0),
                    }

            # Upsert canonical TEAM doc by source_uri
            row = await conn.fetchrow(
                """
                INSERT INTO rag_documents (
                    id, tenant_id, visibility, owner_user_id,
                    title, doc_type, source_uri, metadata, content,
                    source_hash, content_hash,
                    embedding_model, embedding_dim,
                    ingestion_version, chunk_method, total_chunks
                )
                VALUES (
                    $1, $2, 'TEAM', NULL,
                    $3, $4, $5, $6::jsonb, $7,
                    $8, $9,
                    $10, $11,
                    $12, $13, $14
                )
                ON CONFLICT (tenant_id, source_uri)
                    WHERE deleted_at IS NULL
                      AND visibility = 'TEAM'
                      AND source_uri IS NOT NULL
                DO UPDATE SET
                    title = EXCLUDED.title,
                    doc_type = EXCLUDED.doc_type,
                    metadata = EXCLUDED.metadata,
                    content = EXCLUDED.content,
                    source_hash = EXCLUDED.source_hash,
                    content_hash = EXCLUDED.content_hash,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_dim = EXCLUDED.embedding_dim,
                    ingestion_version = EXCLUDED.ingestion_version,
                    chunk_method = EXCLUDED.chunk_method,
                    updated_at = NOW()
                RETURNING id
                """,
                uuid.uuid4(),          # $1
                tenant_id,             # $2
                title,                # $3
                doc_type,             # $4
                source_uri,           # $5
                meta_json,            # $6
                content,              # $7
                source_hash,          # $8
                content_hash,         # $9
                RAG_EMBEDDING_MODEL,  # $10
                RAG_EMBEDDING_DIM,    # $11
                RAG_INGESTION_VERSION,  # $12
                chunk_method,         # $13
                len(chunks),          # $14
            )
            if row is None:
                raise RuntimeError("Failed to upsert TEAM document by source_uri")

            doc_id = row["id"]

            # Replace chunks + embeddings atomically
            await conn.execute("DELETE FROM rag_document_chunks WHERE document_id = $1", doc_id)
            await self._store_chunks_and_embeddings(
                conn, uuid.UUID(str(doc_id)), chunks, embeddings, chunk_offsets=chunk_offsets
            )

            await conn.execute(
                """
                UPDATE rag_documents
                SET total_chunks = $2,
                    updated_at = NOW()
                WHERE id = $1
                """,
                doc_id,
                len(chunks),
            )

        return {"document_id": str(doc_id), "status": "indexed", "total_chunks": len(chunks)}

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
        source_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
        chunk_method: str = "rule_based",
        chunk_offsets: list[tuple[int | None, int | None]] | None = None,
    ) -> dict[str, Any]:
        """Index or update a document with chunks and embeddings.

        After migration 003:
        - TEAM + source_uri: canonical upsert by (tenant_id, source_uri)
        - TEAM ad-hoc (source_uri is NULL): content-hash dedup by (tenant_id, content_hash)
        - PRIVATE: content-hash dedup by (tenant_id, owner_user_id, content_hash)
        """
        if visibility == "TEAM" and source_uri:
            return await self.upsert_document_by_source_uri(
                conn,
                tenant_id=tenant_id,
                source_uri=source_uri,
                source_hash=source_hash,
                title=title,
                content=content,
                chunks=chunks,
                embeddings=embeddings,
                doc_type=doc_type,
                metadata=metadata,
                chunk_method=chunk_method,
                chunk_offsets=chunk_offsets,
                skip_if_unchanged=True,
            )

        if visibility == "PRIVATE" and not owner_user_id:
            raise ValueError("owner_user_id is required for PRIVATE documents")

        import json

        content_hash = hashlib.sha256(content.encode()).hexdigest()
        meta_json = json.dumps(metadata or {})
        doc_id = uuid.uuid4()

        async with conn.transaction():
            if visibility == "TEAM":
                conflict_clause = """
                ON CONFLICT (tenant_id, content_hash)
                    WHERE deleted_at IS NULL
                      AND visibility = 'TEAM'
                      AND source_uri IS NULL
                """
                resolve_where = """
                    d.tenant_id = $2
                    AND d.visibility = 'TEAM'
                    AND d.source_uri IS NULL
                    AND d.content_hash = $10
                    AND d.deleted_at IS NULL
                """
            else:
                conflict_clause = """
                ON CONFLICT (tenant_id, owner_user_id, content_hash)
                    WHERE deleted_at IS NULL
                      AND visibility = 'PRIVATE'
                """
                resolve_where = """
                    d.tenant_id = $2
                    AND d.visibility = 'PRIVATE'
                    AND d.owner_user_id = $4
                    AND d.content_hash = $10
                    AND d.deleted_at IS NULL
                """

            row = await conn.fetchrow(
                f"""
                WITH attempted_insert AS (
                    INSERT INTO rag_documents
                        (id, tenant_id, visibility, owner_user_id, title, doc_type,
                         source_uri, metadata, content, source_hash, content_hash,
                         embedding_model, embedding_dim, ingestion_version,
                         chunk_method, total_chunks)
                    VALUES ($1, $2, $3, $4, $5, $6,
                            NULL, $8::jsonb, $9, $16, $10,
                            $11, $12, $13, $14, $15)
                    {conflict_clause}
                    DO NOTHING
                    RETURNING
                        id,
                        TRUE AS is_new,
                        ingestion_version,
                        chunk_method,
                        embedding_model,
                        embedding_dim,
                        total_chunks
                )
                SELECT id, is_new, ingestion_version, chunk_method, embedding_model, embedding_dim, total_chunks
                FROM attempted_insert
                UNION ALL
                SELECT
                    d.id,
                    FALSE AS is_new,
                    d.ingestion_version,
                    d.chunk_method,
                    d.embedding_model,
                    d.embedding_dim,
                    d.total_chunks
                FROM rag_documents d
                WHERE {resolve_where}
                LIMIT 1
                """,
                doc_id,                 # $1
                tenant_id,              # $2
                visibility,             # $3
                owner_user_id,          # $4
                title,                  # $5
                doc_type,               # $6
                None,                   # $7 (unused)
                meta_json,              # $8
                content,                # $9
                content_hash,           # $10
                RAG_EMBEDDING_MODEL,    # $11
                RAG_EMBEDDING_DIM,      # $12
                RAG_INGESTION_VERSION,  # $13
                chunk_method,           # $14
                len(chunks),            # $15
                source_hash,            # $16
            )
            if row is None:
                raise RuntimeError("Failed to insert or resolve deduplicated document row")

            actual_id: str = str(row["id"])
            is_new: bool = bool(row["is_new"])

            if not is_new:
                settings_match = (
                    row["ingestion_version"] == RAG_INGESTION_VERSION
                    and row["chunk_method"] == chunk_method
                    and row["embedding_model"] == RAG_EMBEDDING_MODEL
                    and row["embedding_dim"] == RAG_EMBEDDING_DIM
                )
                if settings_match:
                    return {
                        "document_id": actual_id,
                        "status": "deduplicated",
                        "total_chunks": int(row["total_chunks"] or 0),
                    }

                # Settings changed — delete stale chunks (cascades to embeddings)
                await conn.execute(
                    "DELETE FROM rag_document_chunks WHERE document_id = $1",
                    uuid.UUID(actual_id),
                )

            await self._store_chunks_and_embeddings(
                conn,
                uuid.UUID(actual_id),
                chunks,
                embeddings,
                chunk_offsets=chunk_offsets,
            )

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
        """List visible documents (TEAM + owned PRIVATE, enforced by RLS)."""
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
        return [dict(r) for r in rows], int(total or 0)

    async def get_document(
        self,
        conn: asyncpg.Connection,
        doc_id: str,
    ) -> dict[str, Any] | None:
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
        tag_raw = await conn.execute(
            """
            UPDATE rag_documents
            SET deleted_at = NOW(), updated_at = NOW()
            WHERE id = $1 AND deleted_at IS NULL
            """,
            uuid.UUID(doc_id),
        )
        return str(tag_raw) == "UPDATE 1"

    async def _store_chunks_and_embeddings(
        self,
        conn: asyncpg.Connection,
        doc_id: uuid.UUID,
        chunks: list[str],
        embeddings: list[list[float]],
        *,
        chunk_offsets: list[tuple[int | None, int | None]] | None = None,
    ) -> None:
        """Batch-insert chunks and their embeddings into separate tables."""
        if chunk_offsets is not None and len(chunk_offsets) != len(chunks):
            raise ValueError("chunk_offsets length must match chunks length")

        chunk_rows = []
        embedding_rows = []

        for idx, (text, emb) in enumerate(zip(chunks, embeddings, strict=True)):
            chunk_id = uuid.uuid4()
            emb_id = uuid.uuid4()
            start, end = (chunk_offsets[idx] if chunk_offsets else (None, None))
            chunk_rows.append((chunk_id, doc_id, idx, text, start, end))
            embedding_rows.append((emb_id, chunk_id, emb))

        await conn.executemany(
            """
            INSERT INTO rag_document_chunks (id, document_id, chunk_index, chunk_text, chunk_start, chunk_end)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            chunk_rows,
        )

        await conn.executemany(
            """
            INSERT INTO rag_chunk_embeddings (id, chunk_id, embedding)
            VALUES ($1, $2, $3::vector)
            """,
            embedding_rows,
        )