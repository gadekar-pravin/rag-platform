from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag-ingestor",
        description="Manual ingestion from GCS into AlloyDB RAG tables",
    )

    scope = p.add_mutually_exclusive_group(required=True)
    scope.add_argument("--tenant", action="append", default=[], help="Tenant id to ingest (repeatable)")
    scope.add_argument(
        "--all-tenants",
        action="store_true",
        help="Discover tenants from bucket top-level prefixes",
    )

    p.add_argument(
        "--prefix",
        default=None,
        help="Prefix under tenant to ingest (default from env RAG_INGEST_INPUT_PREFIX)",
    )
    p.add_argument("--max-files", type=int, default=0, help="Max files per tenant (0 = no cap)")
    p.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="Override RAG_INGEST_MAX_FILE_WORKERS",
    )
    p.add_argument("--force", action="store_true", help="Force reindex even if unchanged")
    p.add_argument("--dry-run", action="store_true", help="List work and exit (no DB writes)")
    p.add_argument("--log-level", default="INFO", help="Python logging level (INFO, DEBUG, ...)")
    return p
