"""Integration tests for the ingestion runner E2E pipeline.

Requires a real database — gracefully skips when unavailable.
External services (GCS, Gemini) are mocked; extraction, chunking,
and all DB operations run for real.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest
import pytest_asyncio

from rag_service.ingestion.config import IngestConfig
from rag_service.ingestion.runner import IngestionRunner
from rag_service.ingestion.types import WorkItem

pytestmark = pytest.mark.usefixtures("clean_tables")

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "ingestion"

TENANT = "test-tenant"
USER = "ingestion-bot"
BUCKET = "test-bucket"
EMBEDDING_DIM = 768


def _fake_embedding() -> list[float]:
    return [0.01] * EMBEDDING_DIM


def _make_work_item(
    filename: str = "sample.txt",
    *,
    doc_type: str = "text",
    content_type: str | None = "text/plain",
    tenant_id: str = TENANT,
) -> WorkItem:
    name = f"{tenant_id}/incoming/{filename}"
    return WorkItem(
        tenant_id=tenant_id,
        source_uri=f"gs://{BUCKET}/{name}",
        bucket=BUCKET,
        name=name,
        content_type=content_type,
        generation="12345",
        md5_hash="abc123",
        crc32c="def456",
        size=1024,
        updated=datetime(2025, 1, 1, tzinfo=UTC),
        doc_type=doc_type,
    )


def _make_config(*, max_retries: int = 2) -> IngestConfig:
    return IngestConfig(
        input_bucket=BUCKET,
        input_prefix="incoming/",
        tenants_allowlist=None,
        incremental=True,
        force_reindex=False,
        output_bucket=None,
        output_prefix="rag-extracted/",
        max_content_chars=2_000_000,
        ocr_enabled=False,
        docai_project=None,
        docai_location=None,
        docai_processor_id=None,
        pdf_text_per_page_min=200,
        max_file_workers=3,
        max_retries_per_file=max_retries,
    )


@pytest_asyncio.fixture
async def patched_pool(db_pool: asyncpg.Pool) -> AsyncIterator[asyncpg.Pool]:
    """Patch get_pool() to return the test database pool."""
    with patch("rag_service.db.get_pool", new=AsyncMock(return_value=db_pool)):
        yield db_pool


def _download_side_effect(
    download_map: dict[str, bytes] | None = None,
) -> Callable[..., bytes]:
    """Return a side_effect function for download_bytes mock."""

    def _download(_client: Any, _bucket: Any, name: str) -> bytes:
        if download_map and name in download_map:
            return download_map[name]
        filename = name.split("/")[-1]
        fixture_path = FIXTURES_DIR / filename
        if fixture_path.exists():
            return fixture_path.read_bytes()
        return f"Fallback test content for {name}".encode()

    return _download


async def _fake_embed(chunks: list[str]) -> list[list[float]]:
    return [_fake_embedding() for _ in chunks]


class TestIngestionE2E:
    """End-to-end ingestion through the runner with real DB operations."""

    async def test_single_text_file_ingestion(self, db_pool: asyncpg.Pool, patched_pool: asyncpg.Pool) -> None:
        """Ingest 1 .txt file: doc, chunks, and embeddings all present in DB."""
        item = _make_work_item("sample.txt")
        runner = IngestionRunner(cfg=_make_config(), storage_client=MagicMock())

        with (
            patch("rag_service.ingestion.runner.discover_work_items", return_value=[item]),
            patch("rag_service.ingestion.runner.download_bytes", side_effect=_download_side_effect()),
            patch("rag_service.ingestion.runner.embed_chunks", new=AsyncMock(side_effect=_fake_embed)),
        ):
            result = await runner.run_tenant(
                tenant_id=TENANT,
                under_tenant_prefix="incoming/",
                max_files=100,
                concurrency=1,
                force=False,
                dry_run=False,
            )

        assert result["completed"] == 1
        assert result["total"] == 1
        assert result["failed"] == 0

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", TENANT)
            await conn.execute("SELECT set_config('app.user_id', $1, true)", USER)

            doc = await conn.fetchrow(
                "SELECT * FROM rag_documents WHERE source_uri = $1 AND deleted_at IS NULL",
                item.source_uri,
            )
            assert doc is not None
            assert doc["title"] == "sample.txt"
            assert doc["visibility"] == "TEAM"
            assert len(doc["content"]) > 0

            chunks = await conn.fetch(
                "SELECT * FROM rag_document_chunks WHERE document_id = $1 ORDER BY chunk_index",
                doc["id"],
            )
            assert len(chunks) >= 1

            emb_count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM rag_chunk_embeddings e
                JOIN rag_document_chunks c ON c.id = e.chunk_id
                WHERE c.document_id = $1
                """,
                doc["id"],
            )
            assert emb_count == len(chunks)

    async def test_html_file_ingestion(self, db_pool: asyncpg.Pool, patched_pool: asyncpg.Pool) -> None:
        """Ingest 1 .html file: HTML tags stripped from stored content."""
        item = _make_work_item("sample.html", doc_type="html", content_type="text/html")
        runner = IngestionRunner(cfg=_make_config(), storage_client=MagicMock())

        with (
            patch("rag_service.ingestion.runner.discover_work_items", return_value=[item]),
            patch("rag_service.ingestion.runner.download_bytes", side_effect=_download_side_effect()),
            patch("rag_service.ingestion.runner.embed_chunks", new=AsyncMock(side_effect=_fake_embed)),
        ):
            result = await runner.run_tenant(
                tenant_id=TENANT,
                under_tenant_prefix="incoming/",
                max_files=100,
                concurrency=1,
                force=False,
                dry_run=False,
            )

        assert result["completed"] == 1

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", TENANT)
            await conn.execute("SELECT set_config('app.user_id', $1, true)", USER)

            doc = await conn.fetchrow(
                "SELECT * FROM rag_documents WHERE source_uri = $1 AND deleted_at IS NULL",
                item.source_uri,
            )
            assert doc is not None
            assert "<html>" not in doc["content"]
            assert "<body>" not in doc["content"]
            assert "Welcome to RAG Platform" in doc["content"]

    async def test_multi_file_concurrent(self, db_pool: asyncpg.Pool, patched_pool: asyncpg.Pool) -> None:
        """Ingest 3 .txt files with concurrency=2: all 3 in DB."""
        items = [_make_work_item(f"file{i}.txt") for i in range(3)]
        download_map = {
            it.name: f"Content for file {i}. This is test document number {i}.".encode() for i, it in enumerate(items)
        }
        runner = IngestionRunner(cfg=_make_config(), storage_client=MagicMock())

        with (
            patch("rag_service.ingestion.runner.discover_work_items", return_value=items),
            patch(
                "rag_service.ingestion.runner.download_bytes",
                side_effect=_download_side_effect(download_map),
            ),
            patch("rag_service.ingestion.runner.embed_chunks", new=AsyncMock(side_effect=_fake_embed)),
        ):
            result = await runner.run_tenant(
                tenant_id=TENANT,
                under_tenant_prefix="incoming/",
                max_files=100,
                concurrency=2,
                force=False,
                dry_run=False,
            )

        assert result["completed"] == 3
        assert result["total"] == 3

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", TENANT)
            await conn.execute("SELECT set_config('app.user_id', $1, true)", USER)

            doc_ids = await conn.fetch(
                "SELECT id FROM rag_documents WHERE deleted_at IS NULL",
            )
            assert len(doc_ids) == 3
            assert len({row["id"] for row in doc_ids}) == 3

    async def test_dry_run_no_db_writes(self, db_pool: asyncpg.Pool, patched_pool: asyncpg.Pool) -> None:
        """dry_run=True returns total but writes nothing to DB."""
        items = [_make_work_item("sample.txt")]
        runner = IngestionRunner(cfg=_make_config(), storage_client=MagicMock())

        with (
            patch("rag_service.ingestion.runner.discover_work_items", return_value=items),
            patch("rag_service.ingestion.runner.download_bytes", side_effect=_download_side_effect()),
            patch("rag_service.ingestion.runner.embed_chunks", new=AsyncMock(side_effect=_fake_embed)),
        ):
            result = await runner.run_tenant(
                tenant_id=TENANT,
                under_tenant_prefix="incoming/",
                max_files=100,
                concurrency=1,
                force=False,
                dry_run=True,
            )

        assert result["total"] == 1
        assert result["completed"] == 0

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", TENANT)
            await conn.execute("SELECT set_config('app.user_id', $1, true)", USER)

            run_count = await conn.fetchval("SELECT COUNT(*) FROM rag_ingestion_runs")
            assert run_count == 0

            doc_count = await conn.fetchval("SELECT COUNT(*) FROM rag_documents")
            assert doc_count == 0


class TestRunTracking:
    """Verify ingestion run and item status tracking in the database."""

    async def test_completed_run_status(self, db_pool: asyncpg.Pool, patched_pool: asyncpg.Pool) -> None:
        """Successful ingestion sets run status to 'completed'."""
        item = _make_work_item("sample.txt")
        runner = IngestionRunner(cfg=_make_config(), storage_client=MagicMock())

        with (
            patch("rag_service.ingestion.runner.discover_work_items", return_value=[item]),
            patch("rag_service.ingestion.runner.download_bytes", side_effect=_download_side_effect()),
            patch("rag_service.ingestion.runner.embed_chunks", new=AsyncMock(side_effect=_fake_embed)),
        ):
            await runner.run_tenant(
                tenant_id=TENANT,
                under_tenant_prefix="incoming/",
                max_files=100,
                concurrency=1,
                force=False,
                dry_run=False,
            )

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", TENANT)
            await conn.execute("SELECT set_config('app.user_id', $1, true)", USER)

            run = await conn.fetchrow(
                "SELECT * FROM rag_ingestion_runs ORDER BY started_at DESC LIMIT 1",
            )
            assert run is not None
            assert run["status"] == "completed"
            assert run["total_files"] == 1
            assert run["processed_files"] == 1
            assert run["completed_at"] is not None

    async def test_item_status_completed(self, db_pool: asyncpg.Pool, patched_pool: asyncpg.Pool) -> None:
        """Successful item has status='completed' with document_id set."""
        item = _make_work_item("sample.txt")
        runner = IngestionRunner(cfg=_make_config(), storage_client=MagicMock())

        with (
            patch("rag_service.ingestion.runner.discover_work_items", return_value=[item]),
            patch("rag_service.ingestion.runner.download_bytes", side_effect=_download_side_effect()),
            patch("rag_service.ingestion.runner.embed_chunks", new=AsyncMock(side_effect=_fake_embed)),
        ):
            await runner.run_tenant(
                tenant_id=TENANT,
                under_tenant_prefix="incoming/",
                max_files=100,
                concurrency=1,
                force=False,
                dry_run=False,
            )

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", TENANT)
            await conn.execute("SELECT set_config('app.user_id', $1, true)", USER)

            item_row = await conn.fetchrow(
                "SELECT * FROM rag_ingestion_items WHERE source_uri = $1",
                item.source_uri,
            )
            assert item_row is not None
            assert item_row["status"] == "completed"
            assert item_row["document_id"] is not None
            assert item_row["completed_at"] is not None

    async def test_failed_item_marks_run_failed(self, db_pool: asyncpg.Pool, patched_pool: asyncpg.Pool) -> None:
        """Download failure marks item and run as 'failed'."""
        item = _make_work_item("sample.txt")
        runner = IngestionRunner(cfg=_make_config(max_retries=0), storage_client=MagicMock())

        with (
            patch("rag_service.ingestion.runner.discover_work_items", return_value=[item]),
            patch(
                "rag_service.ingestion.runner.download_bytes",
                side_effect=RuntimeError("GCS download failed"),
            ),
            patch("rag_service.ingestion.runner.embed_chunks", new=AsyncMock(side_effect=_fake_embed)),
        ):
            result = await runner.run_tenant(
                tenant_id=TENANT,
                under_tenant_prefix="incoming/",
                max_files=100,
                concurrency=1,
                force=False,
                dry_run=False,
            )

        assert result["failed"] == 1
        assert result["completed"] == 0

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", TENANT)
            await conn.execute("SELECT set_config('app.user_id', $1, true)", USER)

            item_row = await conn.fetchrow(
                "SELECT * FROM rag_ingestion_items WHERE source_uri = $1",
                item.source_uri,
            )
            assert item_row is not None
            assert item_row["status"] == "failed"
            assert item_row["error_message"] is not None
            assert "GCS download failed" in item_row["error_message"]

            run = await conn.fetchrow(
                "SELECT * FROM rag_ingestion_runs ORDER BY started_at DESC LIMIT 1",
            )
            assert run is not None
            assert run["status"] == "failed"


class TestIncrementalSkip:
    """Verify incremental mode skips unchanged files."""

    async def test_unchanged_file_skipped(self, db_pool: asyncpg.Pool, patched_pool: asyncpg.Pool) -> None:
        """Second ingestion of same file with same source hash is skipped."""
        item = _make_work_item("sample.txt")

        # First ingestion
        runner1 = IngestionRunner(cfg=_make_config(), storage_client=MagicMock())
        with (
            patch("rag_service.ingestion.runner.discover_work_items", return_value=[item]),
            patch("rag_service.ingestion.runner.download_bytes", side_effect=_download_side_effect()),
            patch("rag_service.ingestion.runner.embed_chunks", new=AsyncMock(side_effect=_fake_embed)),
        ):
            result1 = await runner1.run_tenant(
                tenant_id=TENANT,
                under_tenant_prefix="incoming/",
                max_files=100,
                concurrency=1,
                force=False,
                dry_run=False,
            )
        assert result1["completed"] == 1

        # Second ingestion — same item (same source hash)
        runner2 = IngestionRunner(cfg=_make_config(), storage_client=MagicMock())
        with (
            patch("rag_service.ingestion.runner.discover_work_items", return_value=[item]),
            patch("rag_service.ingestion.runner.download_bytes", side_effect=_download_side_effect()),
            patch("rag_service.ingestion.runner.embed_chunks", new=AsyncMock(side_effect=_fake_embed)),
        ):
            result2 = await runner2.run_tenant(
                tenant_id=TENANT,
                under_tenant_prefix="incoming/",
                max_files=100,
                concurrency=1,
                force=False,
                dry_run=False,
            )
        assert result2["skipped"] == 1
        assert result2["completed"] == 0

        # Verify item statuses across both runs
        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", TENANT)
            await conn.execute("SELECT set_config('app.user_id', $1, true)", USER)

            rows = await conn.fetch(
                """
                SELECT i.status, r.started_at AS run_started
                FROM rag_ingestion_items i
                JOIN rag_ingestion_runs r ON r.id = i.run_id
                WHERE i.source_uri = $1
                ORDER BY r.started_at DESC
                """,
                item.source_uri,
            )
            assert len(rows) == 2
            assert rows[0]["status"] == "skipped"  # most recent
            assert rows[1]["status"] == "completed"  # first run

    async def test_force_bypasses_skip(self, db_pool: asyncpg.Pool, patched_pool: asyncpg.Pool) -> None:
        """force=True re-processes even when source hash matches."""
        item = _make_work_item("sample.txt")

        # First ingestion
        runner1 = IngestionRunner(cfg=_make_config(), storage_client=MagicMock())
        with (
            patch("rag_service.ingestion.runner.discover_work_items", return_value=[item]),
            patch("rag_service.ingestion.runner.download_bytes", side_effect=_download_side_effect()),
            patch("rag_service.ingestion.runner.embed_chunks", new=AsyncMock(side_effect=_fake_embed)),
        ):
            result1 = await runner1.run_tenant(
                tenant_id=TENANT,
                under_tenant_prefix="incoming/",
                max_files=100,
                concurrency=1,
                force=False,
                dry_run=False,
            )
        assert result1["completed"] == 1

        # Second ingestion with force=True
        runner2 = IngestionRunner(cfg=_make_config(), storage_client=MagicMock())
        with (
            patch("rag_service.ingestion.runner.discover_work_items", return_value=[item]),
            patch("rag_service.ingestion.runner.download_bytes", side_effect=_download_side_effect()),
            patch("rag_service.ingestion.runner.embed_chunks", new=AsyncMock(side_effect=_fake_embed)),
        ):
            result2 = await runner2.run_tenant(
                tenant_id=TENANT,
                under_tenant_prefix="incoming/",
                max_files=100,
                concurrency=1,
                force=True,
                dry_run=False,
            )
        assert result2["completed"] == 1
        assert result2["skipped"] == 0

        async with db_pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", TENANT)
            await conn.execute("SELECT set_config('app.user_id', $1, true)", USER)

            doc = await conn.fetchrow(
                "SELECT * FROM rag_documents WHERE source_uri = $1 AND deleted_at IS NULL",
                item.source_uri,
            )
            assert doc is not None
            assert doc["updated_at"] is not None
