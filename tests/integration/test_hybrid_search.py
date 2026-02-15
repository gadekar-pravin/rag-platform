"""Integration tests for end-to-end hybrid search quality.

Uses synthetic embeddings and known text to verify vector ranking,
FTS ranking, RRF fusion, and best-chunks output.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_service.stores.rag_document_store import RagDocumentStore
from rag_service.stores.rag_search_store import RagSearchStore

pytestmark = pytest.mark.usefixtures("clean_tables")


@pytest.fixture
def doc_store() -> RagDocumentStore:
    return RagDocumentStore()


@pytest.fixture
def search_store() -> RagSearchStore:
    return RagSearchStore()


def _make_embedding(seed: int, dim: int = 768) -> list[float]:
    """Create a deterministic unit-length embedding."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


class TestHybridSearch:
    async def test_vector_search_returns_closest(self, db_pool, doc_store, search_store):
        """Vector search ranks documents by embedding similarity."""
        query_vec = _make_embedding(42)
        close_vec = _make_embedding(42)  # same seed = identical = closest
        far_vec = _make_embedding(99)

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")

            await doc_store.upsert_document(
                conn,
                tenant_id="t1",
                title="Close Doc",
                content="Close document about databases",
                chunks=["Close document about databases"],
                embeddings=[close_vec],
            )
            await doc_store.upsert_document(
                conn,
                tenant_id="t1",
                title="Far Doc",
                content="Far document about cooking recipes",
                chunks=["Far document about cooking recipes"],
                embeddings=[far_vec],
            )

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")

            result = await search_store.search_hybrid(conn, "database query", query_vec, doc_limit=10)

            assert len(result["results"]) == 2
            # Close doc should rank higher
            assert result["results"][0]["title"] == "Close Doc"

    async def test_fts_search_matches_terms(self, db_pool, doc_store, search_store):
        """Full-text search matches on keyword terms."""
        vec = _make_embedding(1)

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")

            await doc_store.upsert_document(
                conn,
                tenant_id="t1",
                title="Python Guide",
                content="Python programming language tutorial with examples",
                chunks=["Python programming language tutorial with examples"],
                embeddings=[vec],
            )
            await doc_store.upsert_document(
                conn,
                tenant_id="t1",
                title="Java Guide",
                content="Java enterprise application development guide",
                chunks=["Java enterprise application development guide"],
                embeddings=[_make_embedding(2)],
            )

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")

            result = await search_store.search_hybrid(conn, "Python programming", vec, doc_limit=10)

            assert len(result["results"]) >= 1
            # Python doc should have text_score > 0
            python_result = next(r for r in result["results"] if r["title"] == "Python Guide")
            assert python_result["text_score"] > 0

    async def test_rrf_fusion_ranks_combined(self, db_pool, doc_store, search_store):
        """RRF fusion combines vector and text signals."""
        query_vec = _make_embedding(10)

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")

            # Doc that matches both vector AND text
            await doc_store.upsert_document(
                conn,
                tenant_id="t1",
                title="Best Match",
                content="Machine learning algorithms for data science",
                chunks=["Machine learning algorithms for data science"],
                embeddings=[query_vec],  # identical to query = best vector match
            )
            # Doc that matches only vector (no FTS match)
            await doc_store.upsert_document(
                conn,
                tenant_id="t1",
                title="Vector Only",
                content="Abstract painting and modern art history",
                chunks=["Abstract painting and modern art history"],
                embeddings=[_make_embedding(999)],  # far from query
            )

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")

            result = await search_store.search_hybrid(
                conn, "machine learning algorithms", query_vec, doc_limit=10
            )

            assert len(result["results"]) >= 1
            # Best Match should rank highest (matches both vector and text signals)
            best = result["results"][0]
            assert best["title"] == "Best Match"
            assert best["vector_score"] > 0
            assert best["text_score"] > 0
            assert best["rrf_score"] > 0

    async def test_debug_metrics_returned(self, db_pool, doc_store, search_store):
        """Debug mode returns pool sizes and has_more flags."""
        vec = _make_embedding(1)

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")

            await doc_store.upsert_document(
                conn,
                tenant_id="t1",
                title="Debug Test",
                content="Testing debug metrics output",
                chunks=["Testing debug metrics output"],
                embeddings=[vec],
            )

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")

            result = await search_store.search_hybrid(conn, "debug test", vec, doc_limit=5, include_debug=True)

            assert result["debug"] is not None
            assert "vector_pool_size" in result["debug"]
            assert "text_pool_size" in result["debug"]
            assert isinstance(result["debug"]["vector_has_more"], bool)

    async def test_best_chunks_per_doc(self, db_pool, doc_store, search_store):
        """Results include best chunks per document."""
        vec = _make_embedding(1)

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")

            await doc_store.upsert_document(
                conn,
                tenant_id="t1",
                title="Multi Chunk Doc",
                content="First chunk about databases. Second chunk about APIs. Third chunk about testing.",
                chunks=[
                    "First chunk about databases",
                    "Second chunk about APIs",
                    "Third chunk about testing",
                ],
                embeddings=[vec, _make_embedding(2), _make_embedding(3)],
            )

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")

            result = await search_store.search_hybrid(conn, "databases APIs", vec, doc_limit=5)

            assert len(result["results"]) == 1
            chunks = result["results"][0]["chunks"]
            assert len(chunks) >= 1
            # Each chunk should have source and score
            for c in chunks:
                assert "chunk_text" in c
                assert "source" in c
                assert c["source"] in ("vector", "text")

    async def test_cross_tenant_search_empty(self, db_pool, doc_store, search_store):
        """Cross-tenant search returns empty results."""
        vec = _make_embedding(1)

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "tenant-a")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@a.com")

            await doc_store.upsert_document(
                conn,
                tenant_id="tenant-a",
                title="Tenant A Doc",
                content="This belongs to tenant A only",
                chunks=["This belongs to tenant A only"],
                embeddings=[vec],
            )

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "tenant-b")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@b.com")

            result = await search_store.search_hybrid(conn, "tenant A", vec, doc_limit=5)

            assert len(result["results"]) == 0
