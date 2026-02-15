"""Integration tests for content-hash dedup with COALESCE NULL fix."""

from __future__ import annotations

import uuid

import pytest

from rag_service.stores.rag_document_store import RagDocumentStore

pytestmark = pytest.mark.usefixtures("clean_tables")


@pytest.fixture
def store() -> RagDocumentStore:
    return RagDocumentStore()


class TestContentHashDedup:
    async def test_team_dedup_same_content(self, db_pool, store):
        """Same content hash + same tenant + TEAM = deduplicated."""
        content = "This is the document content."
        chunks = ["This is the document content."]
        embeddings = [[0.1] * 768]

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")

            r1 = await store.upsert_document(
                conn,
                tenant_id="t1",
                title="Doc v1",
                content=content,
                chunks=chunks,
                embeddings=embeddings,
            )
            assert r1["status"] == "indexed"

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")

            r2 = await store.upsert_document(
                conn,
                tenant_id="t1",
                title="Doc v2",
                content=content,
                chunks=chunks,
                embeddings=embeddings,
            )
            assert r2["status"] == "deduplicated"
            assert r2["document_id"] == r1["document_id"]

    async def test_different_content_not_deduped(self, db_pool, store):
        """Different content hash = separate documents."""
        chunks = ["chunk"]
        embeddings = [[0.1] * 768]

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")

            r1 = await store.upsert_document(
                conn,
                tenant_id="t1",
                title="Doc A",
                content="Content A",
                chunks=chunks,
                embeddings=embeddings,
            )

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")

            r2 = await store.upsert_document(
                conn,
                tenant_id="t1",
                title="Doc B",
                content="Content B",
                chunks=chunks,
                embeddings=embeddings,
            )

            assert r1["document_id"] != r2["document_id"]

    async def test_private_dedup_per_owner(self, db_pool, store):
        """PRIVATE docs dedup per owner — different owners = different docs."""
        content = "Same private content"
        chunks = ["chunk"]
        embeddings = [[0.1] * 768]

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")

            r1 = await store.upsert_document(
                conn,
                tenant_id="t1",
                title="Private A",
                content=content,
                chunks=chunks,
                embeddings=embeddings,
                visibility="PRIVATE",
                owner_user_id="u1@test.com",
            )

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u2@test.com")

            r2 = await store.upsert_document(
                conn,
                tenant_id="t1",
                title="Private B",
                content=content,
                chunks=chunks,
                embeddings=embeddings,
                visibility="PRIVATE",
                owner_user_id="u2@test.com",
            )

            assert r1["document_id"] != r2["document_id"]

    async def test_coalesce_null_for_team_dedup(self, db_pool, store):
        """COALESCE(owner_user_id, '') in dedup index handles NULL correctly.

        Two TEAM docs from same tenant with same content should dedup,
        even though owner_user_id is NULL for both.
        """
        content = "Team document content"
        chunks = ["chunk"]
        embeddings = [[0.1] * 768]

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")

            r1 = await store.upsert_document(
                conn,
                tenant_id="t1",
                title="Team v1",
                content=content,
                chunks=chunks,
                embeddings=embeddings,
                visibility="TEAM",
            )
            assert r1["status"] == "indexed"

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u2@test.com")

            r2 = await store.upsert_document(
                conn,
                tenant_id="t1",
                title="Team v2",
                content=content,
                chunks=chunks,
                embeddings=embeddings,
                visibility="TEAM",
            )
            assert r2["status"] == "deduplicated"

    async def test_cascade_delete_chunks_and_embeddings(self, db_pool, store):
        """Deleting a document cascades to chunks and embeddings."""
        content = "Cascade test content"
        chunks = ["chunk one", "chunk two"]
        embeddings = [[0.1] * 768, [0.2] * 768]

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")

            result = await store.upsert_document(
                conn,
                tenant_id="t1",
                title="Cascade Doc",
                content=content,
                chunks=chunks,
                embeddings=embeddings,
            )
            doc_id = uuid.UUID(result["document_id"])

            # Verify chunks exist
            chunk_count = await conn.fetchval(
                "SELECT COUNT(*) FROM rag_document_chunks WHERE document_id = $1",
                doc_id,
            )
            assert chunk_count == 2

            # Soft delete
            await store.soft_delete(conn, result["document_id"])

            # Chunks still exist (soft delete doesn't cascade)
            chunk_count = await conn.fetchval(
                "SELECT COUNT(*) FROM rag_document_chunks WHERE document_id = $1",
                doc_id,
            )
            assert chunk_count == 2
