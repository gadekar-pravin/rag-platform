"""Integration test fixtures — requires a real database.

Gracefully skips all tests when the database is unavailable.
Uses DATABASE_TEST_URL or falls back to localhost defaults.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

try:
    import asyncpg
    from pgvector.asyncpg import register_vector
except ImportError:
    pytest.skip("asyncpg or pgvector not installed", allow_module_level=True)


def _get_test_dsn() -> str:
    return os.environ.get(
        "DATABASE_TEST_URL",
        "postgresql://apexflow:apexflow@localhost:5432/apexflow",
    )


@pytest_asyncio.fixture(scope="session")
async def db_pool():
    """Session-scoped real asyncpg pool. Skips if DB is unavailable."""
    dsn = _get_test_dsn()
    try:
        pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=1,
            max_size=3,
            command_timeout=10,
            init=register_vector,
        )
    except Exception as e:
        pytest.skip(f"Database unavailable: {e}")
        return  # unreachable, but satisfies type checker

    yield pool
    await pool.close()


@pytest_asyncio.fixture
async def clean_tables(db_pool):
    """Truncate all rag_* tables before each test."""
    async with db_pool.acquire() as conn:
        # Disable RLS temporarily for cleanup (requires superuser or table owner)
        await conn.execute("""
            TRUNCATE
                rag_ingestion_items,
                rag_ingestion_runs,
                rag_chunk_embeddings,
                rag_document_chunks,
                rag_documents
            CASCADE
        """)
    yield


@pytest_asyncio.fixture
async def rls_conn(db_pool, test_tenant_id, test_user_id):
    """Provide a connection with RLS session variables set."""
    async with db_pool.acquire() as conn, conn.transaction():
        await conn.execute("SET LOCAL app.tenant_id = $1", test_tenant_id)
        await conn.execute("SET LOCAL app.user_id = $1", test_user_id)
        yield conn


@pytest_asyncio.fixture
async def other_rls_conn(db_pool, other_tenant_id, other_user_id):
    """Provide a connection with different tenant RLS session variables."""
    async with db_pool.acquire() as conn, conn.transaction():
        await conn.execute("SET LOCAL app.tenant_id = $1", other_tenant_id)
        await conn.execute("SET LOCAL app.user_id = $1", other_user_id)
        yield conn
