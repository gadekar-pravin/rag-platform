from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime

from google.cloud.storage import Client

from rag_service.chunking.chunker import chunk_document_with_spans
from rag_service.db import rls_connection
from rag_service.embedding import embed_chunks
from rag_service.ingestion.config import IngestConfig
from rag_service.ingestion.extractors.docx import DocxExtractor
from rag_service.ingestion.extractors.html import HtmlExtractor
from rag_service.ingestion.extractors.image import ImageExtractor
from rag_service.ingestion.extractors.pdf import PdfExtractor
from rag_service.ingestion.extractors.text import TextExtractor
from rag_service.ingestion.gcs import download_bytes, upload_text
from rag_service.ingestion.ocr.document_ai import DocAIConfig, DocumentAIClient
from rag_service.ingestion.planner import compute_source_hash, discover_work_items
from rag_service.ingestion.types import ProcessResult, WorkItem
from rag_service.stores.rag_document_store import RagDocumentStore

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(UTC)


class IngestionRunner:
    def __init__(self, *, cfg: IngestConfig, storage_client: Client) -> None:
        self._cfg = cfg
        self._gcs = storage_client
        self._store = RagDocumentStore()

        if cfg.ocr_enabled:
            docai_cfg = DocAIConfig(
                project=cfg.docai_project or "",
                location=cfg.docai_location or "",
                processor_id=cfg.docai_processor_id or "",
            )
            self._docai = DocumentAIClient(cfg=docai_cfg, storage_client=storage_client)
        else:
            self._docai = None  # type: ignore

    async def run_tenant(
        self,
        *,
        tenant_id: str,
        under_tenant_prefix: str,
        max_files: int,
        concurrency: int,
        force: bool,
        dry_run: bool,
        user_id: str = "ingestion-bot",
    ) -> dict[str, int]:
        # Discover objects
        items = discover_work_items(
            self._gcs,
            bucket=self._cfg.input_bucket,
            tenant_id=tenant_id,
            under_tenant_prefix=under_tenant_prefix,
            max_files=max_files,
        )

        if (
            self._cfg.tenants_allowlist is not None
            and tenant_id not in self._cfg.tenants_allowlist
        ):
            logger.warning("Tenant '%s' not in allowlist; skipping", tenant_id)
            return {"total": 0, "completed": 0, "skipped": 0, "failed": 0}

        logger.info("Tenant=%s discovered %d candidate objects", tenant_id, len(items))

        if dry_run:
            for it in items:
                logger.info("[DRY-RUN] %s", it.source_uri)
            return {"total": len(items), "completed": 0, "skipped": 0, "failed": 0}

        # Create ingestion run and item rows
        async with rls_connection(tenant_id, user_id) as conn:
            run_id = await conn.fetchval(
                """
                INSERT INTO rag_ingestion_runs (tenant_id, status, total_files, metadata)
                VALUES ($1, 'running', $2, $3::jsonb)
                RETURNING id
                """,
                tenant_id,
                len(items),
                json.dumps(
                    {
                        "input_bucket": self._cfg.input_bucket,
                        "input_prefix": under_tenant_prefix,
                        "incremental": self._cfg.incremental,
                        "force": force or self._cfg.force_reindex,
                        "concurrency": concurrency,
                        "created_at": _now().isoformat(),
                    }
                ),
            )
            assert run_id is not None

            # Insert items (pending)
            rows = []
            for it in items:
                rows.append((run_id, it.source_uri))
            await conn.executemany(
                """
                INSERT INTO rag_ingestion_items (run_id, source_uri, status)
                VALUES ($1, $2, 'pending')
                """,
                rows,
            )

        sem = asyncio.Semaphore(concurrency)

        results: list[ProcessResult] = []

        async def worker(it: WorkItem) -> None:
            async with sem:
                res = await self._process_item(
                    tenant_id=tenant_id,
                    run_id=str(run_id),
                    item=it,
                    under_tenant_prefix=under_tenant_prefix,
                    force=force,
                    user_id=user_id,
                )
                results.append(res)

        await asyncio.gather(*[worker(it) for it in items])

        # Finalize run
        completed = sum(1 for r in results if r.status == "completed")
        skipped = sum(1 for r in results if r.status == "skipped")
        failed = sum(1 for r in results if r.status == "failed")

        async with rls_connection(tenant_id, user_id) as conn:
            await conn.execute(
                """
                UPDATE rag_ingestion_runs
                SET status = CASE WHEN $2 > 0 THEN 'failed' ELSE 'completed' END,
                    processed_files = $3,
                    completed_at = NOW(),
                    error_message = CASE WHEN $2 > 0 THEN 'One or more items failed' ELSE NULL END
                WHERE id = $1
                """,
                run_id,
                failed,
                completed + skipped + failed,
            )

        return {
            "total": len(items),
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
        }

    async def _process_item(
        self,
        *,
        tenant_id: str,
        run_id: str,
        item: WorkItem,
        under_tenant_prefix: str,
        force: bool,
        user_id: str,
    ) -> ProcessResult:
        # Retry at file level
        last_err: str | None = None
        for attempt in range(self._cfg.max_retries_per_file + 1):
            try:
                return await self._process_item_once(
                    tenant_id=tenant_id,
                    run_id=run_id,
                    item=item,
                    under_tenant_prefix=under_tenant_prefix,
                    force=force,
                    user_id=user_id,
                )
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                logger.warning(
                    "Item failed (attempt %d/%d): %s :: %s",
                    attempt + 1,
                    self._cfg.max_retries_per_file + 1,
                    item.source_uri,
                    last_err,
                )
                if attempt >= self._cfg.max_retries_per_file:
                    break
                await asyncio.sleep(min(2**attempt, 10))

        # Permanent failure: mark DB status
        await self._mark_item_failed(
            tenant_id=tenant_id,
            run_id=run_id,
            source_uri=item.source_uri,
            user_id=user_id,
            error=last_err or "unknown error",
        )
        return ProcessResult(
            item=item,
            status="failed",
            document_id=None,
            error_message=last_err or "unknown error",
        )

    async def _process_item_once(
        self,
        *,
        tenant_id: str,
        run_id: str,
        item: WorkItem,
        under_tenant_prefix: str,
        force: bool,
        user_id: str,
    ) -> ProcessResult:
        # Mark processing
        await self._mark_item_processing(
            tenant_id=tenant_id,
            run_id=run_id,
            source_uri=item.source_uri,
            user_id=user_id,
        )

        # Compute source_hash
        src_hash = compute_source_hash(
            generation=item.generation,
            md5_hash=item.md5_hash,
            crc32c=item.crc32c,
            size=item.size,
            updated=item.updated,
        )

        # Incremental skip check
        if self._cfg.incremental and not (force or self._cfg.force_reindex):
            async with rls_connection(tenant_id, user_id) as conn:
                existing = await self._store.get_team_document_by_source_uri(
                    conn, source_uri=item.source_uri
                )
                if existing and (existing.get("source_hash") == src_hash):
                    await self._mark_item_skipped(
                        tenant_id=tenant_id,
                        run_id=run_id,
                        source_uri=item.source_uri,
                        user_id=user_id,
                        document_id=str(existing["id"]),
                    )
                    return ProcessResult(
                        item=item,
                        status="skipped",
                        document_id=str(existing["id"]),
                        error_message=None,
                    )

        # Download bytes (blocking I/O → run in thread to not block event loop)
        data = await asyncio.to_thread(
            download_bytes, self._gcs, item.bucket, item.name
        )

        # Build extractors for this run
        if self._cfg.ocr_enabled and self._docai is None:
            raise RuntimeError("OCR enabled but DocumentAIClient not initialized")

        extractors = [
            TextExtractor(),
            HtmlExtractor(),
            DocxExtractor(),
        ]

        if self._cfg.ocr_enabled:
            # DocAI output prefix per item if batch OCR used
            docai_out = self._docai_output_prefix(
                tenant_id=tenant_id, run_id=run_id, source_uri=item.source_uri
            )
            extractors.append(ImageExtractor(docai=self._docai))
            extractors.append(
                PdfExtractor(
                    docai=self._docai,
                    text_per_page_min=self._cfg.pdf_text_per_page_min,
                    output_prefix_for_docai=docai_out,
                )
            )
        else:
            # PDF text extraction without OCR fallback
            extractors.append(
                PdfExtractor(
                    docai=None,
                    text_per_page_min=self._cfg.pdf_text_per_page_min,
                    output_prefix_for_docai=None,
                )
            )

        # Choose extractor
        extractor = next((ex for ex in extractors if ex.can_handle(item)), None)
        if extractor is None:
            raise RuntimeError(f"Unsupported file type for: {item.name}")

        exr = await asyncio.to_thread(extractor.extract, item=item, data=data)
        text = exr.text
        if not text:
            raise RuntimeError("Extraction produced empty text")

        # Optional: store extracted artifact
        extracted_text_uri = None
        if self._cfg.output_bucket:
            key = self._extracted_key(
                tenant_id=tenant_id, run_id=run_id, source_hash=src_hash
            )
            await asyncio.to_thread(
                upload_text, self._gcs, self._cfg.output_bucket, key, text
            )
            extracted_text_uri = f"gs://{self._cfg.output_bucket}/{key}"

        # Truncate doc content if needed
        truncated = False
        stored_content = text
        if len(stored_content) > self._cfg.max_content_chars:
            stored_content = stored_content[: self._cfg.max_content_chars]
            truncated = True

        # Chunk + embed (with character offsets for search highlighting)
        chunks_with_spans = await chunk_document_with_spans(
            stored_content, method="rule_based"
        )
        if not chunks_with_spans:
            raise RuntimeError("Chunking produced zero chunks")

        chunks = [c[0] for c in chunks_with_spans]
        chunk_offsets: list[tuple[int | None, int | None]] = [
            (c[1], c[2]) for c in chunks_with_spans
        ]

        embeddings = await embed_chunks(chunks)

        # Build metadata
        meta = {
            "gcs": {
                "bucket": item.bucket,
                "name": item.name,
                "generation": item.generation,
                "md5": item.md5_hash,
                "crc32c": item.crc32c,
                "size": item.size,
                "updated": item.updated.isoformat() if item.updated else None,
                "content_type": item.content_type,
            },
            "extraction": {
                "used_ocr": exr.used_ocr,
                "pages": exr.pages,
                **exr.extraction_meta,
            },
            "ingestion": {
                "source_hash": src_hash,
                "content_truncated": truncated,
                "extracted_text_uri": extracted_text_uri,
            },
        }

        title = _title_from_name(item.name)

        # DB upsert (canonical TEAM by source_uri)
        async with rls_connection(tenant_id, user_id) as conn:
            out = await self._store.upsert_document_by_source_uri(
                conn,
                tenant_id=tenant_id,
                source_uri=item.source_uri,
                source_hash=src_hash,
                title=title,
                content=stored_content,
                chunks=chunks,
                embeddings=embeddings,
                doc_type=item.doc_type,
                metadata=meta,
                chunk_method="rule_based",
                chunk_offsets=chunk_offsets,
                skip_if_unchanged=not (force or self._cfg.force_reindex),
            )

        doc_id = out["document_id"]
        await self._mark_item_completed(
            tenant_id=tenant_id,
            run_id=run_id,
            source_uri=item.source_uri,
            user_id=user_id,
            document_id=doc_id,
        )
        return ProcessResult(
            item=item, status="completed", document_id=doc_id, error_message=None
        )

    def _docai_output_prefix(
        self, *, tenant_id: str, run_id: str, source_uri: str
    ) -> str | None:
        if not self._cfg.output_bucket:
            return None
        # Safe-ish: use run id + hashed source uri (no slashes)
        import hashlib

        h = hashlib.sha256(source_uri.encode()).hexdigest()[:16]
        key = f"{self._cfg.output_prefix}{tenant_id}/runs/{run_id}/{h}/docai/"
        return f"gs://{self._cfg.output_bucket}/{key}"

    def _extracted_key(self, *, tenant_id: str, run_id: str, source_hash: str) -> str:
        import hashlib

        h = hashlib.sha256(source_hash.encode()).hexdigest()[:16]
        return f"{self._cfg.output_prefix}{tenant_id}/runs/{run_id}/{h}.txt"

    async def _mark_item_processing(
        self, *, tenant_id: str, run_id: str, source_uri: str, user_id: str
    ) -> None:
        async with rls_connection(tenant_id, user_id) as conn:
            await conn.execute(
                """
                UPDATE rag_ingestion_items
                SET status = 'processing', started_at = NOW()
                WHERE run_id = $1::uuid AND source_uri = $2
                """,
                run_id,
                source_uri,
            )

    async def _mark_item_completed(
        self,
        *,
        tenant_id: str,
        run_id: str,
        source_uri: str,
        user_id: str,
        document_id: str,
    ) -> None:
        async with rls_connection(tenant_id, user_id) as conn:
            await conn.execute(
                """
                UPDATE rag_ingestion_items
                SET status = 'completed',
                    document_id = $3::uuid,
                    completed_at = NOW()
                WHERE run_id = $1::uuid AND source_uri = $2
                """,
                run_id,
                source_uri,
                document_id,
            )
            await conn.execute(
                "UPDATE rag_ingestion_runs SET processed_files = processed_files + 1 WHERE id = $1::uuid",
                run_id,
            )

    async def _mark_item_skipped(
        self,
        *,
        tenant_id: str,
        run_id: str,
        source_uri: str,
        user_id: str,
        document_id: str,
    ) -> None:
        async with rls_connection(tenant_id, user_id) as conn:
            await conn.execute(
                """
                UPDATE rag_ingestion_items
                SET status = 'skipped',
                    document_id = $3::uuid,
                    completed_at = NOW()
                WHERE run_id = $1::uuid AND source_uri = $2
                """,
                run_id,
                source_uri,
                document_id,
            )
            await conn.execute(
                "UPDATE rag_ingestion_runs SET processed_files = processed_files + 1 WHERE id = $1::uuid",
                run_id,
            )

    async def _mark_item_failed(
        self, *, tenant_id: str, run_id: str, source_uri: str, user_id: str, error: str
    ) -> None:
        async with rls_connection(tenant_id, user_id) as conn:
            await conn.execute(
                """
                UPDATE rag_ingestion_items
                SET status = 'failed',
                    error_message = $3,
                    completed_at = NOW()
                WHERE run_id = $1::uuid AND source_uri = $2
                """,
                run_id,
                source_uri,
                error[:4000],
            )
            await conn.execute(
                "UPDATE rag_ingestion_runs SET processed_files = processed_files + 1 WHERE id = $1::uuid",
                run_id,
            )


def _title_from_name(name: str) -> str:
    base = name.split("/")[-1]
    if not base:
        return name
    return base
