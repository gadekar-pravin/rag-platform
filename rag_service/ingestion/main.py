from __future__ import annotations

import asyncio
import logging

from google.cloud.storage import Client

from rag_service.ingestion.cli import build_parser
from rag_service.ingestion.config import IngestConfig
from rag_service.ingestion.gcs import list_tenant_prefixes
from rag_service.ingestion.runner import IngestionRunner
from rag_service.logging_config import setup_logging


async def _amain() -> int:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(level=args.log_level.upper())
    logger = logging.getLogger("rag_service.ingestion")

    cfg = IngestConfig.from_env()
    cfg.validate()

    # CLI overrides
    concurrency = args.concurrency if args.concurrency and args.concurrency > 0 else cfg.max_file_workers
    force = bool(args.force) or cfg.force_reindex

    client = Client()
    runner = IngestionRunner(cfg=cfg, storage_client=client)

    # Determine tenants
    tenants: list[str]
    tenants = list_tenant_prefixes(client, cfg.input_bucket) if args.all_tenants else list(args.tenant or [])

    if cfg.tenants_allowlist is not None:
        tenants = [t for t in tenants if t in cfg.tenants_allowlist]

    if not tenants:
        logger.warning("No tenants to ingest (after allowlist filtering). Exiting.")
        return 0

    under_prefix = args.prefix if args.prefix is not None else cfg.input_prefix

    totals = {"total": 0, "completed": 0, "skipped": 0, "failed": 0}
    for t in tenants:
        stats = await runner.run_tenant(
            tenant_id=t,
            under_tenant_prefix=under_prefix,
            max_files=int(args.max_files or 0),
            concurrency=concurrency,
            force=force,
            dry_run=bool(args.dry_run),
        )
        for k in totals:
            totals[k] += int(stats.get(k, 0))

    logger.info("DONE totals=%s", totals)
    return 0 if totals["failed"] == 0 else 2


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
