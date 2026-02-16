"""Async database connection pool with RLS context manager.

Ported from ApexFlow's core/database.py with additions:
- `rls_connection()` context manager sets SET LOCAL app.tenant_id/user_id
  per-transaction so PostgreSQL RLS policies can filter rows automatically.
- Fail-closed: raises ValueError if tenant_id or user_id are empty.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None
_pgvector_available = False

try:
    from pgvector.asyncpg import register_vector as _register_vector

    _pgvector_available = True
except ImportError:
    logger.info("pgvector not installed — vector codec will not be registered")


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Per-connection init: register pgvector codec if available."""
    if _pgvector_available:
        await _register_vector(conn)


class DatabaseConfig:
    """Builds connection strings based on detected environment."""

    @staticmethod
    def get_connection_string() -> str:
        # Priority 1: Explicit override
        if url := os.environ.get("DATABASE_URL"):
            return url

        # Priority 2: Cloud Run -> managed AlloyDB
        if os.environ.get("K_SERVICE") or os.environ.get("CLOUD_RUN_JOB"):
            host = os.environ.get("ALLOYDB_HOST")
            db = os.environ.get("ALLOYDB_DB", "apexflow")
            user = os.environ.get("ALLOYDB_USER", "apexflow")
            password = os.environ.get("ALLOYDB_PASSWORD", "")
            return f"postgresql://{user}:{password}@{host}/{db}"

        # Priority 3: Local dev
        sslmode = os.environ.get("DB_SSLMODE", "disable")
        return (
            f"postgresql://{os.environ.get('DB_USER', 'apexflow')}:"
            f"{os.environ.get('DB_PASSWORD', 'apexflow')}@"
            f"{os.environ.get('DB_HOST', 'localhost')}:"
            f"{os.environ.get('DB_PORT', '5432')}/"
            f"{os.environ.get('DB_NAME', 'apexflow')}?sslmode={sslmode}"
        )


async def get_pool() -> asyncpg.Pool:
    """Return the singleton connection pool, creating it if necessary."""
    global _pool  # noqa: PLW0603
    if _pool is None:
        dsn = DatabaseConfig.get_connection_string()
        logger.info("Creating database pool (host hidden for security)")
        _pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=1,
            max_size=int(os.environ.get("DB_POOL_MAX", "5")),
            command_timeout=30,
            init=_init_connection,
        )
    return _pool


async def close_pool() -> None:
    """Gracefully close the connection pool."""
    global _pool  # noqa: PLW0603
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Database pool closed")


async def check_db_connection() -> bool:
    """Health check: returns True if the database is reachable."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception:
        logger.exception("Database health check failed")
        return False


@asynccontextmanager
async def rls_connection(tenant_id: str, user_id: str) -> AsyncIterator[asyncpg.Connection]:
    """Acquire a connection with RLS session variables set.

    Sets `app.tenant_id` and `app.user_id` via SET LOCAL (transaction-scoped)
    so that PostgreSQL RLS policies can filter rows automatically. The connection
    is wrapped in a transaction — SET LOCAL values are discarded when the
    transaction ends.

    Raises ValueError if tenant_id or user_id are empty (fail-closed).
    """
    if not tenant_id or not user_id:
        raise ValueError("tenant_id and user_id are required (fail-closed)")

    pool = await get_pool()
    async with pool.acquire() as conn, conn.transaction():
        # SET LOCAL doesn't support parameterized queries — use
        # quote_literal via the server to prevent SQL injection.
        await conn.execute("SELECT set_config('app.tenant_id', $1, true)", tenant_id)
        await conn.execute("SELECT set_config('app.user_id', $1, true)", user_id)
        yield conn
