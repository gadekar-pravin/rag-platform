"""Integration tests for ingestion dedup semantics.

Requires a real database — gracefully skips when unavailable.
Uses the 3 partial unique indexes from migration 001.
"""

from __future__ import annotations

import uuid

import pytest

from rag_service.stores.rag_document_store import RagDocumentStore

pytestmark = pytest.mark.usefixtures("clean_tables")


@pytest.fixture
def store() -> RagDocumentStore:
    return RagDocumentStore()


def _fake_embedding(dim: int = 768) -> list[float]:
    return [0.01] * dim


class TestTeamGCSIngest:
    """TEAM docs with source_uri use the canonical (tenant_id, source_uri) index."""

    async def test_different_source_uri_same_content(self, db_pool, store):
        """Two different source_uri with identical content -> both succeed as separate rows."""
        content = "Shared document content"
        chunks = ["chunk1"]
        embeddings = [_fake_embedding()]

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "bot@test.com")
            r1 = await store.upsert_document_by_source_uri(
                conn,
                tenant_id="t1",
                source_uri="gs://bucket/t1/file-a.txt",
                source_hash="hash-a",
                title="File A",
                content=content,
                chunks=chunks,
                embeddings=embeddings,
            )

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "bot@test.com")
            r2 = await store.upsert_document_by_source_uri(
                conn,
                tenant_id="t1",
                source_uri="gs://bucket/t1/file-b.txt",
                source_hash="hash-b",
                title="File B",
                content=content,
                chunks=chunks,
                embeddings=embeddings,
            )

        assert r1["status"] == "indexed"
        assert r2["status"] == "indexed"
        assert r1["document_id"] != r2["document_id"]

    async def test_canonical_update_replaces_chunks(self, db_pool, store):
        """Same source_uri with changed content -> updates doc and replaces chunks."""
        chunks_v1 = ["old chunk"]
        chunks_v2 = ["new chunk A", "new chunk B"]
        embeddings_v1 = [_fake_embedding()]
        embeddings_v2 = [_fake_embedding(), _fake_embedding()]

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "bot@test.com")
            r1 = await store.upsert_document_by_source_uri(
                conn,
                tenant_id="t1",
                source_uri="gs://bucket/t1/doc.txt",
                source_hash="hash-v1",
                title="Doc",
                content="version 1 content",
                chunks=chunks_v1,
                embeddings=embeddings_v1,
            )

        doc_id = r1["document_id"]

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "bot@test.com")
            r2 = await store.upsert_document_by_source_uri(
                conn,
                tenant_id="t1",
                source_uri="gs://bucket/t1/doc.txt",
                source_hash="hash-v2",
                title="Doc Updated",
                content="version 2 content",
                chunks=chunks_v2,
                embeddings=embeddings_v2,
            )

        assert r2["status"] == "indexed"
        assert r2["document_id"] == doc_id
        assert r2["total_chunks"] == 2

        # Verify chunks were replaced
        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "bot@test.com")
            chunk_count = await conn.fetchval(
                "SELECT COUNT(*) FROM rag_document_chunks WHERE document_id = $1",
                uuid.UUID(doc_id),
            )
            assert chunk_count == 2

    async def test_canonical_unchanged_skips(self, db_pool, store):
        """Same source_uri + same content + same source_hash -> unchanged."""
        content = "immutable content"
        chunks = ["chunk"]
        embeddings = [_fake_embedding()]

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "bot@test.com")
            r1 = await store.upsert_document_by_source_uri(
                conn,
                tenant_id="t1",
                source_uri="gs://bucket/t1/stable.txt",
                source_hash="same-hash",
                title="Stable",
                content=content,
                chunks=chunks,
                embeddings=embeddings,
            )

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "bot@test.com")
            r2 = await store.upsert_document_by_source_uri(
                conn,
                tenant_id="t1",
                source_uri="gs://bucket/t1/stable.txt",
                source_hash="same-hash",
                title="Stable",
                content=content,
                chunks=chunks,
                embeddings=embeddings,
            )

        assert r1["status"] == "indexed"
        assert r2["status"] == "unchanged"
        assert r2["document_id"] == r1["document_id"]


class TestTeamAdHocDedup:
    """TEAM docs without source_uri use content-hash dedup."""

    async def test_adhoc_content_hash_dedup(self, db_pool, store):
        """source_uri IS NULL: content-hash dedup still works."""
        content = "ad-hoc content"
        chunks = ["chunk"]
        embeddings = [_fake_embedding()]

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "bot@test.com")
            r1 = await store.upsert_document(
                conn,
                tenant_id="t1",
                title="Ad Hoc 1",
                content=content,
                chunks=chunks,
                embeddings=embeddings,
                visibility="TEAM",
            )

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "bot@test.com")
            r2 = await store.upsert_document(
                conn,
                tenant_id="t1",
                title="Ad Hoc 2",
                content=content,
                chunks=chunks,
                embeddings=embeddings,
                visibility="TEAM",
            )

        assert r1["status"] == "indexed"
        assert r2["status"] == "deduplicated"
        assert r2["document_id"] == r1["document_id"]


class TestAtomicity:
    """Verify chunk replacement is atomic."""

    async def test_old_chunks_deleted_new_present(self, db_pool, store):
        """After upsert_document_by_source_uri, old chunks should be gone, new chunks present."""
        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "bot@test.com")
            r1 = await store.upsert_document_by_source_uri(
                conn,
                tenant_id="t1",
                source_uri="gs://bucket/t1/atomic.txt",
                source_hash="v1",
                title="Atomic",
                content="version 1",
                chunks=["old-chunk-1", "old-chunk-2", "old-chunk-3"],
                embeddings=[_fake_embedding(), _fake_embedding(), _fake_embedding()],
            )

        doc_id = uuid.UUID(r1["document_id"])

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "bot@test.com")
            await store.upsert_document_by_source_uri(
                conn,
                tenant_id="t1",
                source_uri="gs://bucket/t1/atomic.txt",
                source_hash="v2",
                title="Atomic v2",
                content="version 2",
                chunks=["new-chunk-1"],
                embeddings=[_fake_embedding()],
            )

        # Verify: exactly 1 chunk, and it's the new one
        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "bot@test.com")
            rows = await conn.fetch(
                "SELECT chunk_text FROM rag_document_chunks WHERE document_id = $1 ORDER BY chunk_index",
                doc_id,
            )
            assert len(rows) == 1
            assert rows[0]["chunk_text"] == "new-chunk-1"

            # Verify embeddings also exist for the new chunk
            emb_count = await conn.fetchval(
                """
                    SELECT COUNT(*) FROM rag_chunk_embeddings e
                    JOIN rag_document_chunks c ON c.id = e.chunk_id
                    WHERE c.document_id = $1
                    """,
                doc_id,
            )
            assert emb_count == 1
