"""Integration tests for Row-Level Security enforcement.

Verifies:
- FORCE RLS blocks reads without session variables
- Correct tenant sees docs
- Wrong tenant sees nothing
- PRIVATE docs visible only to owner
"""

from __future__ import annotations

import uuid

import asyncpg
import pytest

pytestmark = pytest.mark.usefixtures("clean_tables")


class TestRLSEnforcement:
    async def test_force_rls_blocks_without_session_vars(self, db_pool):
        """SELECT without SET LOCAL app.* returns no rows (FORCE RLS)."""
        doc_id = uuid.uuid4()

        # Insert directly bypassing RLS (use a connection that sets vars)
        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")
            await conn.execute(
                """
                    INSERT INTO rag_documents (id, tenant_id, visibility, title,
                        content_hash, embedding_model, embedding_dim)
                    VALUES ($1, 't1', 'TEAM', 'Test', 'hash1', 'model', 768)
                    """,
                doc_id,
            )

        # Read without RLS vars — should return nothing
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM rag_documents")
            assert len(rows) == 0

    async def test_correct_tenant_sees_team_docs(self, db_pool):
        """Tenant can see their own TEAM documents."""
        doc_id = uuid.uuid4()

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")
            await conn.execute(
                """
                    INSERT INTO rag_documents (id, tenant_id, visibility, title,
                        content_hash, embedding_model, embedding_dim)
                    VALUES ($1, 't1', 'TEAM', 'Team Doc', 'hash2', 'model', 768)
                    """,
                doc_id,
            )

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")
            rows = await conn.fetch("SELECT * FROM rag_documents")
            assert len(rows) == 1
            assert rows[0]["title"] == "Team Doc"

    async def test_wrong_tenant_sees_nothing(self, db_pool):
        """Different tenant cannot see another tenant's documents."""
        doc_id = uuid.uuid4()

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")
            await conn.execute(
                """
                    INSERT INTO rag_documents (id, tenant_id, visibility, title,
                        content_hash, embedding_model, embedding_dim)
                    VALUES ($1, 't1', 'TEAM', 'Secret Doc', 'hash3', 'model', 768)
                    """,
                doc_id,
            )

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t2")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u2@test.com")
            rows = await conn.fetch("SELECT * FROM rag_documents")
            assert len(rows) == 0

    async def test_private_doc_visible_only_to_owner(self, db_pool):
        """PRIVATE docs are visible only to the owning user."""
        doc_id = uuid.uuid4()

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "owner@test.com")
            await conn.execute(
                """
                    INSERT INTO rag_documents (id, tenant_id, visibility,
                        owner_user_id, title, content_hash,
                        embedding_model, embedding_dim)
                    VALUES ($1, 't1', 'PRIVATE', 'owner@test.com',
                        'Private Doc', 'hash4', 'model', 768)
                    """,
                doc_id,
            )

        # Owner can see it
        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "owner@test.com")
            rows = await conn.fetch("SELECT * FROM rag_documents")
            assert len(rows) == 1

        # Other user in same tenant cannot see it
        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "other@test.com")
            rows = await conn.fetch("SELECT * FROM rag_documents")
            assert len(rows) == 0

    async def test_team_doc_visible_to_all_tenant_users(self, db_pool):
        """TEAM docs are visible to any user in the same tenant."""
        doc_id = uuid.uuid4()

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")
            await conn.execute(
                """
                    INSERT INTO rag_documents (id, tenant_id, visibility, title,
                        content_hash, embedding_model, embedding_dim)
                    VALUES ($1, 't1', 'TEAM', 'Shared Doc', 'hash5', 'model', 768)
                    """,
                doc_id,
            )

        # Different user, same tenant
        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u2@test.com")
            rows = await conn.fetch("SELECT * FROM rag_documents")
            assert len(rows) == 1

    async def test_soft_deleted_doc_invisible(self, db_pool):
        """Soft-deleted documents are not visible via RLS."""
        doc_id = uuid.uuid4()

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")
            await conn.execute(
                """
                    INSERT INTO rag_documents (id, tenant_id, visibility, title,
                        content_hash, embedding_model, embedding_dim,
                        deleted_at)
                    VALUES ($1, 't1', 'TEAM', 'Deleted Doc', 'hash6', 'model', 768,
                        NOW())
                    """,
                doc_id,
            )

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")
            rows = await conn.fetch("SELECT * FROM rag_documents")
            assert len(rows) == 0

    async def test_chunks_hidden_without_session_vars(self, db_pool):
        """Chunk rows are hidden when app.* session vars are missing."""
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        emb_id = uuid.uuid4()

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")
            await conn.execute(
                """
                    INSERT INTO rag_documents (id, tenant_id, visibility, title,
                        content_hash, embedding_model, embedding_dim)
                    VALUES ($1, 't1', 'TEAM', 'Chunk Parent', 'hash_chunks_1', 'model', 768)
                    """,
                doc_id,
            )
            await conn.execute(
                """
                    INSERT INTO rag_document_chunks (id, document_id, chunk_index, chunk_text)
                    VALUES ($1, $2, 0, 'chunk body')
                    """,
                chunk_id,
                doc_id,
            )
            await conn.execute(
                """
                    INSERT INTO rag_chunk_embeddings (id, chunk_id, embedding)
                    VALUES ($1, $2, $3::vector)
                    """,
                emb_id,
                chunk_id,
                [0.1] * 768,
            )

        async with db_pool.acquire() as conn:
            chunk_rows = await conn.fetch("SELECT * FROM rag_document_chunks")
            emb_rows = await conn.fetch("SELECT * FROM rag_chunk_embeddings")
            assert len(chunk_rows) == 0
            assert len(emb_rows) == 0

    async def test_chunks_hidden_cross_tenant(self, db_pool):
        """Different tenant cannot read chunks/embeddings from another tenant's docs."""
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        emb_id = uuid.uuid4()

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "tenant-a")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@a.com")
            await conn.execute(
                """
                    INSERT INTO rag_documents (id, tenant_id, visibility, title,
                        content_hash, embedding_model, embedding_dim)
                    VALUES ($1, 'tenant-a', 'TEAM', 'Tenant A Parent', 'hash_chunks_2', 'model', 768)
                    """,
                doc_id,
            )
            await conn.execute(
                """
                    INSERT INTO rag_document_chunks (id, document_id, chunk_index, chunk_text)
                    VALUES ($1, $2, 0, 'tenant-a chunk')
                    """,
                chunk_id,
                doc_id,
            )
            await conn.execute(
                """
                    INSERT INTO rag_chunk_embeddings (id, chunk_id, embedding)
                    VALUES ($1, $2, $3::vector)
                    """,
                emb_id,
                chunk_id,
                [0.2] * 768,
            )

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "tenant-b")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@b.com")
            chunk_rows = await conn.fetch("SELECT * FROM rag_document_chunks")
            emb_rows = await conn.fetch("SELECT * FROM rag_chunk_embeddings")
            assert len(chunk_rows) == 0
            assert len(emb_rows) == 0

    async def test_insert_policy_enforces_tenant_match(self, db_pool):
        """INSERT policy rejects rows where tenant_id doesn't match session."""
        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")
            with pytest.raises(asyncpg.PostgresError):
                await conn.execute(
                    """
                        INSERT INTO rag_documents (tenant_id, visibility, title,
                            content_hash, embedding_model, embedding_dim)
                        VALUES ('wrong-tenant', 'TEAM', 'Bad Insert', 'hash7', 'model', 768)
                        """
                )

    async def test_check_constraint_private_requires_owner(self, db_pool):
        """CHECK constraint: PRIVATE visibility requires owner_user_id."""
        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "t1")
            await conn.execute("SELECT set_config('app.user_id', $1, true)", "u1@test.com")
            with pytest.raises(asyncpg.PostgresError):
                await conn.execute(
                    """
                        INSERT INTO rag_documents (tenant_id, visibility, title,
                            content_hash, embedding_model, embedding_dim)
                        VALUES ('t1', 'PRIVATE', 'No Owner', 'hash8', 'model', 768)
                        """
                )
