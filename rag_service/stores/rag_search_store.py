"""Hybrid search store with RRF fusion over 3-table join.

Key differences from ApexFlow's DocumentSearch:
- 3-table join: rag_documents -> rag_document_chunks -> rag_chunk_embeddings
- RLS enforces tenant isolation (no WHERE user_id = in SQL)
- Returns up to 4 best chunks per document (2 vector + 2 text, deduped)
- Debug metrics: pool sizes, has_more flags, cutoff scores
- Pagination signal via chunk_limit + 1
"""

from __future__ import annotations

import logging
from typing import Any

import asyncpg

from rag_service.config import (
    RAG_RRF_K,
    RAG_SEARCH_CANDIDATE_MULTIPLIER,
    RAG_SEARCH_EXPANSION,
    RAG_SEARCH_PER_DOC_CAP,
)

logger = logging.getLogger(__name__)


class RagSearchStore:
    """Stateless hybrid search over rag_documents + chunks + embeddings."""

    async def search_hybrid(
        self,
        conn: asyncpg.Connection,
        query_text: str,
        query_vec: list[float],
        *,
        doc_limit: int = 10,
        expansion: int = RAG_SEARCH_EXPANSION,
        per_doc_cap: int = RAG_SEARCH_PER_DOC_CAP,
        candidate_multiplier: int = RAG_SEARCH_CANDIDATE_MULTIPLIER,
        rrf_k: int = RAG_RRF_K,
        include_debug: bool = False,
    ) -> dict[str, Any]:
        """Reciprocal Rank Fusion of vector cosine + full-text ts_rank.

        The connection must already have SET LOCAL app.tenant_id/user_id
        applied — RLS policies handle visibility filtering.

        Returns:
            {
                "results": [
                    {
                        "document_id": str,
                        "title": str,
                        "doc_type": str | None,
                        "rrf_score": float,
                        "vector_score": float,
                        "text_score": float,
                        "chunks": [
                            {"chunk_id": str, "chunk_index": int,
                             "chunk_text": str, "source": "vector"|"text",
                             "score": float}
                        ]
                    }
                ],
                "debug": {...} | None
            }
        """
        chunk_limit = doc_limit * expansion
        per_doc_cap = max(1, per_doc_cap)
        candidate_limit = chunk_limit * max(1, candidate_multiplier)

        rows = await conn.fetch(
            """
            WITH
            -- Vector candidates (global top-N chunks)
            vector_candidates AS (
                SELECT
                    d.id AS document_id, d.title, d.doc_type,
                    c.id AS chunk_id, c.chunk_index, c.chunk_text,
                    (e.embedding <=> $1::vector) AS distance
                FROM rag_documents d
                JOIN rag_document_chunks c ON c.document_id = d.id
                JOIN rag_chunk_embeddings e ON e.chunk_id = c.id
                ORDER BY e.embedding <=> $1::vector
                LIMIT $2
            ),

            -- Vector pool with per-document cap to improve diversity
            vector_pool AS (
                SELECT
                    ranked.document_id,
                    ranked.title,
                    ranked.doc_type,
                    ranked.chunk_id,
                    ranked.chunk_index,
                    ranked.chunk_text,
                    1 - ranked.distance AS score,
                    ROW_NUMBER() OVER (ORDER BY ranked.distance) AS rank
                FROM (
                    SELECT
                        vc.*,
                        ROW_NUMBER() OVER (
                            PARTITION BY vc.document_id
                            ORDER BY vc.distance
                        ) AS per_doc_rank
                    FROM vector_candidates vc
                ) ranked
                WHERE ranked.per_doc_rank <= $6
            ),

            -- Text candidates (global top-N chunks)
            text_candidates AS (
                SELECT
                    d.id AS document_id, d.title, d.doc_type,
                    c.id AS chunk_id, c.chunk_index, c.chunk_text,
                    ts_rank(c.fts, plainto_tsquery('english', $3)) AS score
                FROM rag_documents d
                JOIN rag_document_chunks c ON c.document_id = d.id
                WHERE c.fts @@ plainto_tsquery('english', $3)
                ORDER BY ts_rank(c.fts, plainto_tsquery('english', $3)) DESC
                LIMIT $2
            ),

            -- Text pool with per-document cap to improve diversity
            text_pool AS (
                SELECT
                    ranked.document_id,
                    ranked.title,
                    ranked.doc_type,
                    ranked.chunk_id,
                    ranked.chunk_index,
                    ranked.chunk_text,
                    ranked.score,
                    ROW_NUMBER() OVER (ORDER BY ranked.score DESC) AS rank
                FROM (
                    SELECT
                        tc.*,
                        ROW_NUMBER() OVER (
                            PARTITION BY tc.document_id
                            ORDER BY tc.score DESC
                        ) AS per_doc_rank
                    FROM text_candidates tc
                ) ranked
                WHERE ranked.per_doc_rank <= $6
            ),

            -- Best vector rank per document
            doc_vector_rrf AS (
                SELECT DISTINCT ON (document_id)
                    document_id, title, doc_type, rank
                FROM vector_pool
                ORDER BY document_id, rank
            ),

            -- Best text rank per document
            doc_text_rrf AS (
                SELECT DISTINCT ON (document_id)
                    document_id, title, doc_type, rank
                FROM text_pool
                ORDER BY document_id, rank
            ),

            -- Fuse scores via RRF
            fused AS (
                SELECT
                    COALESCE(v.document_id, t.document_id) AS document_id,
                    COALESCE(v.title, t.title) AS title,
                    COALESCE(v.doc_type, t.doc_type) AS doc_type,
                    COALESCE(1.0 / ($4 + v.rank), 0) AS vector_score,
                    COALESCE(1.0 / ($4 + t.rank), 0) AS text_score,
                    COALESCE(1.0 / ($4 + v.rank), 0) +
                        COALESCE(1.0 / ($4 + t.rank), 0) AS rrf_score
                FROM doc_vector_rrf v
                FULL OUTER JOIN doc_text_rrf t ON v.document_id = t.document_id
                ORDER BY rrf_score DESC
                LIMIT $5
            )

            SELECT document_id, title, doc_type, rrf_score, vector_score, text_score
            FROM fused
            ORDER BY rrf_score DESC
            """,
            query_vec,
            candidate_limit + 1,  # +1 for has_more detection
            query_text,
            rrf_k,
            doc_limit,
            per_doc_cap,
        )

        if not rows:
            result: dict[str, Any] = {"results": []}
            if include_debug:
                result["debug"] = {
                    "vector_pool_size": 0,
                    "text_pool_size": 0,
                    "vector_has_more": False,
                    "text_has_more": False,
                }
            return result

        # Collect doc IDs for best-chunks query
        doc_ids = [r["document_id"] for r in rows]

        # Get best chunks per document (top 2 vector + top 2 text, deduped)
        best_chunks = await self._get_best_chunks(
            conn, doc_ids, query_vec, query_text
        )

        # Build results
        results = []
        for r in rows:
            doc_id = str(r["document_id"])
            results.append({
                "document_id": doc_id,
                "title": r["title"],
                "doc_type": r["doc_type"],
                "rrf_score": float(r["rrf_score"]),
                "vector_score": float(r["vector_score"]),
                "text_score": float(r["text_score"]),
                "chunks": best_chunks.get(doc_id, []),
            })

        result = {"results": results}

        if include_debug:
            stats = await self._get_pool_stats(
                query_text,
                query_vec,
                conn,
                candidate_limit + 1,
                per_doc_cap,
            )

            result["debug"] = {
                "vector_pool_size": stats["vector_pool_size"],
                "text_pool_size": stats["text_pool_size"],
                "vector_has_more": stats["vector_pool_size"] > chunk_limit,
                "text_has_more": stats["text_pool_size"] > chunk_limit,
            }

        return result

    async def _get_pool_stats(
        self,
        query_text: str,
        query_vec: list[float],
        conn: asyncpg.Connection,
        candidate_limit: int,
        per_doc_cap: int,
    ) -> dict[str, int]:
        """Compute bounded pool sizes for debug without full-table counts."""
        row = await conn.fetchrow(
            """
            WITH
            vector_candidates AS (
                SELECT
                    d.id AS document_id,
                    (e.embedding <=> $1::vector) AS distance
                FROM rag_documents d
                JOIN rag_document_chunks c ON c.document_id = d.id
                JOIN rag_chunk_embeddings e ON e.chunk_id = c.id
                ORDER BY e.embedding <=> $1::vector
                LIMIT $2
            ),
            vector_pool AS (
                SELECT ranked.document_id
                FROM (
                    SELECT
                        vc.document_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY vc.document_id
                            ORDER BY vc.distance
                        ) AS per_doc_rank
                    FROM vector_candidates vc
                ) ranked
                WHERE ranked.per_doc_rank <= $4
            ),
            text_candidates AS (
                SELECT
                    d.id AS document_id,
                    ts_rank(c.fts, plainto_tsquery('english', $3)) AS score
                FROM rag_documents d
                JOIN rag_document_chunks c ON c.document_id = d.id
                WHERE c.fts @@ plainto_tsquery('english', $3)
                ORDER BY ts_rank(c.fts, plainto_tsquery('english', $3)) DESC
                LIMIT $2
            ),
            text_pool AS (
                SELECT ranked.document_id
                FROM (
                    SELECT
                        tc.document_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY tc.document_id
                            ORDER BY tc.score DESC
                        ) AS per_doc_rank
                    FROM text_candidates tc
                ) ranked
                WHERE ranked.per_doc_rank <= $4
            )
            SELECT
                (SELECT COUNT(*) FROM vector_pool) AS vector_pool_size,
                (SELECT COUNT(*) FROM text_pool) AS text_pool_size
            """,
            query_vec,
            candidate_limit,
            query_text,
            per_doc_cap,
        )
        if row is None:
            return {"vector_pool_size": 0, "text_pool_size": 0}
        return {
            "vector_pool_size": int(row["vector_pool_size"] or 0),
            "text_pool_size": int(row["text_pool_size"] or 0),
        }

    async def _get_best_chunks(
        self,
        conn: asyncpg.Connection,
        doc_ids: list[Any],
        query_vec: list[float],
        query_text: str,
    ) -> dict[str, list[dict[str, Any]]]:
        """Get top 2 vector + top 2 text chunks per document, deduped."""
        if not doc_ids:
            return {}

        rows = await conn.fetch(
            """
            WITH
            -- Top 2 vector chunks per doc
            vec_chunks AS (
                SELECT
                    d.id AS document_id,
                    c.id AS chunk_id, c.chunk_index, c.chunk_text,
                    1 - (e.embedding <=> $1::vector) AS score,
                    'vector' AS source,
                    ROW_NUMBER() OVER (
                        PARTITION BY d.id ORDER BY e.embedding <=> $1::vector
                    ) AS rn
                FROM rag_documents d
                JOIN rag_document_chunks c ON c.document_id = d.id
                JOIN rag_chunk_embeddings e ON e.chunk_id = c.id
                WHERE d.id = ANY($3::uuid[])
            ),
            top_vec AS (
                SELECT * FROM vec_chunks WHERE rn <= 2
            ),

            -- Top 2 text chunks per doc
            txt_chunks AS (
                SELECT
                    d.id AS document_id,
                    c.id AS chunk_id, c.chunk_index, c.chunk_text,
                    ts_rank(c.fts, plainto_tsquery('english', $2)) AS score,
                    'text' AS source,
                    ROW_NUMBER() OVER (
                        PARTITION BY d.id
                        ORDER BY ts_rank(c.fts, plainto_tsquery('english', $2)) DESC
                    ) AS rn
                FROM rag_documents d
                JOIN rag_document_chunks c ON c.document_id = d.id
                WHERE d.id = ANY($3::uuid[])
                  AND c.fts @@ plainto_tsquery('english', $2)
            ),
            top_txt AS (
                SELECT * FROM txt_chunks WHERE rn <= 2
            ),

            -- Combine and dedup by chunk_id
            combined AS (
                SELECT DISTINCT ON (document_id, chunk_id)
                    document_id, chunk_id, chunk_index, chunk_text, score, source
                FROM (
                    SELECT * FROM top_vec
                    UNION ALL
                    SELECT * FROM top_txt
                ) all_chunks
                ORDER BY document_id, chunk_id, score DESC
            )

            SELECT document_id, chunk_id, chunk_index, chunk_text, score, source
            FROM combined
            ORDER BY document_id, score DESC
            """,
            query_vec,
            query_text,
            doc_ids,
        )

        result: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            doc_id = str(r["document_id"])
            if doc_id not in result:
                result[doc_id] = []
            result[doc_id].append({
                "chunk_id": str(r["chunk_id"]),
                "chunk_index": r["chunk_index"],
                "chunk_text": r["chunk_text"],
                "source": r["source"],
                "score": float(r["score"]),
            })

        return result
