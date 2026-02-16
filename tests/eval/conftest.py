"""Evaluation test fixtures.

Provides two seeding modes:
- eval_seed_documents: synthetic random embeddings (CI-safe, no API key needed)
- eval_seed_documents_real: real Gemini embeddings (meaningful quality measurement)

Own DB pool (not shared with integration tests) to avoid coupling.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import pytest_asyncio

try:
    import asyncpg
    from pgvector.asyncpg import register_vector
except ImportError:
    pytest.skip("asyncpg or pgvector not installed", allow_module_level=True)

from rag_service.config import RAG_EMBEDDING_DIM
from rag_service.stores.rag_document_store import RagDocumentStore

EVAL_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"


def _get_test_dsn() -> str:
    return os.environ.get(
        "DATABASE_TEST_URL",
        "postgresql://apexflow:apexflow@localhost:5432/apexflow",
    )


@pytest_asyncio.fixture(scope="session")
async def eval_db_pool() -> Any:
    """Session-scoped asyncpg pool for eval tests. Skips if DB unavailable."""
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
        return

    yield pool
    await pool.close()


@pytest_asyncio.fixture
async def eval_clean_tables(eval_db_pool: asyncpg.Pool) -> Any:  # type: ignore[type-arg]
    """Truncate all rag_* tables before each eval test."""
    async with eval_db_pool.acquire() as conn:
        await conn.execute(
            """
            TRUNCATE
                rag_ingestion_items,
                rag_ingestion_runs,
                rag_chunk_embeddings,
                rag_document_chunks,
                rag_documents
            CASCADE
        """
        )
    yield


@pytest.fixture(scope="session")
def eval_dataset() -> dict[str, Any]:
    """Load the evaluation dataset JSON."""
    return json.loads(EVAL_DATASET_PATH.read_text())  # type: ignore[no-any-return]


def _make_synthetic_embedding(dim: int, seed: int) -> list[float]:
    """Generate a deterministic random unit vector."""
    rng = random.Random(seed)
    vec = np.array([rng.gauss(0, 1) for _ in range(dim)], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()  # type: ignore[no-any-return]


@pytest_asyncio.fixture
async def eval_seed_documents(
    eval_db_pool: asyncpg.Pool,  # type: ignore[type-arg]
    eval_clean_tables: Any,
    eval_dataset: dict[str, Any],
) -> dict[str, str]:
    """Seed eval documents with synthetic embeddings. Returns {title: doc_id}."""
    store = RagDocumentStore()
    tenant_id = "eval-test"
    user_id = "eval-user@test"
    title_to_id: dict[str, str] = {}

    for i, doc in enumerate(eval_dataset["documents"]):
        title = doc["title"]
        content = doc["content"]

        # Synthetic: one chunk per doc, random embedding
        chunks = [content]
        embeddings = [_make_synthetic_embedding(RAG_EMBEDDING_DIM, seed=i)]

        async with eval_db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", tenant_id)
            await conn.execute("SELECT set_config('app.user_id', $1, true)", user_id)
            result = await store.upsert_document(
                conn,
                tenant_id=tenant_id,
                title=title,
                content=content,
                chunks=chunks,
                embeddings=embeddings,
                doc_type=doc.get("doc_type"),
            )
            title_to_id[title] = result["document_id"]

    return title_to_id


@pytest_asyncio.fixture
async def eval_seed_documents_real(
    eval_db_pool: asyncpg.Pool,  # type: ignore[type-arg]
    eval_clean_tables: Any,
    eval_dataset: dict[str, Any],
) -> dict[str, str]:
    """Seed eval documents with real Gemini embeddings. Skips without GEMINI_API_KEY."""
    has_api_key = bool(os.getenv("GEMINI_API_KEY"))
    has_adc = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("K_SERVICE"))
    if not has_api_key and not has_adc:
        pytest.skip("No embedding credentials — set GEMINI_API_KEY or GOOGLE_APPLICATION_CREDENTIALS")

    from rag_service.chunking.chunker import chunk_document
    from rag_service.embedding import embed_chunks

    store = RagDocumentStore()
    tenant_id = "eval-test"
    user_id = "eval-user@test"
    title_to_id: dict[str, str] = {}

    for doc in eval_dataset["documents"]:
        title = doc["title"]
        content = doc["content"]

        chunks = await chunk_document(content)
        if not chunks:
            chunks = [content]
        embeddings = await embed_chunks(chunks)

        async with eval_db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", tenant_id)
            await conn.execute("SELECT set_config('app.user_id', $1, true)", user_id)
            result = await store.upsert_document(
                conn,
                tenant_id=tenant_id,
                title=title,
                content=content,
                chunks=chunks,
                embeddings=embeddings,
                doc_type=doc.get("doc_type"),
            )
            title_to_id[title] = result["document_id"]

    return title_to_id
