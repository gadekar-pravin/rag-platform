"""Hybrid search store with RRF fusion over 3-table join.

Key differences from ApexFlow's DocumentSearch:
- 3-table join: rag_documents -> rag_document_chunks -> rag_chunk_embeddings
- RLS enforces tenant isolation (no WHERE user_id = in SQL)
- Returns up to 4 best chunks per document (2 vector + 2 text, deduped)
- Debug metrics: pool sizes, has_more flags, cutoff scores
- Pagination signal via chunk_limit + 1
- ANN-first pattern: scan embeddings first, join metadata later (enables ScaNN index)
- Single-query best-chunks: no separate DB round-trip
"""

from __future__ import annotations

import logging
from typing import Any

import asyncpg

from rag_service.config import (
    RAG_FTS_LANGUAGE,
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
        fts_language: str = RAG_FTS_LANGUAGE,
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

        # Single query: fusion + best-chunks integrated via CTEs
        # Fix 2: ANN-first — scan embeddings table alone, join metadata later
        # Fix 4: best-chunks selected from already-computed pools (no 2nd query)
        # Fix 5: tsquery computed once via CTE
        # Fix 8: FTS language parameterized ($7)
        rows = await conn.fetch(
            """
            WITH
            -- Fix 5: compute tsquery once
            tsquery_val AS (
                SELECT plainto_tsquery($7, $3) AS q
            ),

            -- Fix 2: ANN-first — scan embeddings table alone for ScaNN index
            ann_candidates AS (
                SELECT e.chunk_id, (e.embedding <=> $1::vector) AS distance
                FROM rag_chunk_embeddings e
                ORDER BY e.embedding <=> $1::vector
                LIMIT $2
            ),
            -- Join metadata after ANN scan
            vector_candidates AS (
                SELECT
                    d.id AS document_id, d.title, d.doc_type,
                    c.id AS chunk_id, c.chunk_index, c.chunk_text,
                    a.distance
                FROM ann_candidates a
                JOIN rag_document_chunks c ON c.id = a.chunk_id
                JOIN rag_documents d ON d.id = c.document_id
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
                    ts_rank(c.fts, (SELECT q FROM tsquery_val)) AS score
                FROM rag_documents d
                JOIN rag_document_chunks c ON c.document_id = d.id
                WHERE c.fts @@ (SELECT q FROM tsquery_val)
                ORDER BY ts_rank(c.fts, (SELECT q FROM tsquery_val)) DESC
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
            ),

            -- Fix 4: best chunks from already-computed pools (no 2nd query)
            -- Top 2 vector chunks per fused doc
            best_vec AS (
                SELECT vp.document_id, vp.chunk_id, vp.chunk_index,
                       vp.chunk_text, vp.score, 'vector' AS source,
                       ROW_NUMBER() OVER (
                           PARTITION BY vp.document_id ORDER BY vp.rank
                       ) AS rn
                FROM vector_pool vp
                WHERE vp.document_id IN (SELECT document_id FROM fused)
            ),
            -- Top 2 text chunks per fused doc
            best_txt AS (
                SELECT tp.document_id, tp.chunk_id, tp.chunk_index,
                       tp.chunk_text, tp.score, 'text' AS source,
                       ROW_NUMBER() OVER (
                           PARTITION BY tp.document_id ORDER BY tp.rank
                       ) AS rn
                FROM text_pool tp
                WHERE tp.document_id IN (SELECT document_id FROM fused)
            ),
            -- Combine and dedup by chunk_id
            best_chunks AS (
                SELECT DISTINCT ON (document_id, chunk_id)
                    document_id, chunk_id, chunk_index, chunk_text, score, source
                FROM (
                    SELECT document_id, chunk_id, chunk_index, chunk_text, score, source
                    FROM best_vec WHERE rn <= 2
                    UNION ALL
                    SELECT document_id, chunk_id, chunk_index, chunk_text, score, source
                    FROM best_txt WHERE rn <= 2
                ) all_chunks
                ORDER BY document_id, chunk_id, score DESC
            )

            -- Return fused docs with their best chunks as separate rows
            SELECT
                f.document_id, f.title, f.doc_type,
                f.rrf_score, f.vector_score, f.text_score,
                bc.chunk_id, bc.chunk_index, bc.chunk_text,
                bc.source AS chunk_source, bc.score AS chunk_score
            FROM fused f
            LEFT JOIN best_chunks bc ON bc.document_id = f.document_id
            ORDER BY f.rrf_score DESC, f.document_id, bc.score DESC
            """,
            query_vec,
            candidate_limit + 1,  # +1 for has_more detection
            query_text,
            rrf_k,
            doc_limit,
            per_doc_cap,
            fts_language,
        )

        if not rows:
            result: dict[str, Any] = {"results": []}
            if include_debug:
                result["debug"] = {
                    "vector_pool_size": 0,
                    "text_pool_size": 0,
                    "vector_has_more": False,
                    "text_has_more": False,
                    "vector_cutoff_score": None,
                    "text_cutoff_score": None,
                }
            return result

        # Group rows by document (one row per chunk due to LEFT JOIN)
        results: list[dict[str, Any]] = []
        seen_docs: dict[str, int] = {}  # doc_id -> index in results
        for r in rows:
            doc_id = str(r["document_id"])
            if doc_id not in seen_docs:
                seen_docs[doc_id] = len(results)
                results.append(
                    {
                        "document_id": doc_id,
                        "title": r["title"],
                        "doc_type": r["doc_type"],
                        "rrf_score": float(r["rrf_score"]),
                        "vector_score": float(r["vector_score"]),
                        "text_score": float(r["text_score"]),
                        "chunks": [],
                    }
                )
            # Append chunk if present (LEFT JOIN may produce NULL chunk_id)
            if r["chunk_id"] is not None:
                results[seen_docs[doc_id]]["chunks"].append(
                    {
                        "chunk_id": str(r["chunk_id"]),
                        "chunk_index": r["chunk_index"],
                        "chunk_text": r["chunk_text"],
                        "source": r["chunk_source"],
                        "score": float(r["chunk_score"]),
                    }
                )

        result = {"results": results}

        if include_debug:
            stats = await self._get_pool_stats(
                query_text,
                query_vec,
                conn,
                candidate_limit + 1,
                per_doc_cap,
                fts_language,
            )

            result["debug"] = {
                "vector_pool_size": stats["vector_pool_size"],
                "text_pool_size": stats["text_pool_size"],
                # Fix 6: use raw candidate counts for has_more (pre per-doc cap)
                "vector_has_more": stats["vector_raw_count"] > candidate_limit,
                "text_has_more": stats["text_raw_count"] > candidate_limit,
                "vector_cutoff_score": stats.get("vector_cutoff_score"),
                "text_cutoff_score": stats.get("text_cutoff_score"),
            }

        return result

    async def _get_pool_stats(
        self,
        query_text: str,
        query_vec: list[float],
        conn: asyncpg.Connection,
        candidate_limit: int,
        per_doc_cap: int,
        fts_language: str = RAG_FTS_LANGUAGE,
    ) -> dict[str, Any]:
        """Compute bounded pool sizes and cutoff scores for debug."""
        row = await conn.fetchrow(
            """
            WITH
            tsquery_val AS (
                SELECT plainto_tsquery($5, $3) AS q
            ),
            -- Fix 2: ANN-first pattern
            ann_candidates AS (
                SELECT e.chunk_id, (e.embedding <=> $1::vector) AS distance
                FROM rag_chunk_embeddings e
                ORDER BY e.embedding <=> $1::vector
                LIMIT $2
            ),
            vector_candidates AS (
                SELECT d.id AS document_id, a.distance
                FROM ann_candidates a
                JOIN rag_document_chunks c ON c.id = a.chunk_id
                JOIN rag_documents d ON d.id = c.document_id
            ),
            vector_pool AS (
                SELECT ranked.document_id, ranked.distance
                FROM (
                    SELECT
                        vc.document_id,
                        vc.distance,
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
                    ts_rank(c.fts, (SELECT q FROM tsquery_val)) AS score
                FROM rag_documents d
                JOIN rag_document_chunks c ON c.document_id = d.id
                WHERE c.fts @@ (SELECT q FROM tsquery_val)
                ORDER BY ts_rank(c.fts, (SELECT q FROM tsquery_val)) DESC
                LIMIT $2
            ),
            text_pool AS (
                SELECT ranked.document_id, ranked.score
                FROM (
                    SELECT
                        tc.document_id,
                        tc.score,
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
                (SELECT COUNT(*) FROM text_pool) AS text_pool_size,
                -- Fix 6: raw counts from pre-cap CTEs for accurate has_more
                (SELECT COUNT(*) FROM vector_candidates) AS vector_raw_count,
                (SELECT COUNT(*) FROM text_candidates) AS text_raw_count,
                (SELECT 1 - MAX(distance) FROM vector_pool) AS vector_cutoff_score,
                (SELECT MIN(score) FROM text_pool) AS text_cutoff_score
            """,
            query_vec,
            candidate_limit,
            query_text,
            per_doc_cap,
            fts_language,
        )
        if row is None:
            return {
                "vector_pool_size": 0,
                "text_pool_size": 0,
                "vector_raw_count": 0,
                "text_raw_count": 0,
                "vector_cutoff_score": None,
                "text_cutoff_score": None,
            }
        return {
            "vector_pool_size": int(row["vector_pool_size"] or 0),
            "text_pool_size": int(row["text_pool_size"] or 0),
            "vector_raw_count": int(row["vector_raw_count"] or 0),
            "text_raw_count": int(row["text_raw_count"] or 0),
            "vector_cutoff_score": (
                float(row["vector_cutoff_score"])
                if row["vector_cutoff_score"] is not None
                else None
            ),
            "text_cutoff_score": (
                float(row["text_cutoff_score"])
                if row["text_cutoff_score"] is not None
                else None
            ),
        }
