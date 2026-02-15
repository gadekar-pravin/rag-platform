from __future__ import annotations

import os
from dataclasses import dataclass


def _get_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "t", "yes", "y", "on")


def _get_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    return int(v)


@dataclass(frozen=True)
class IngestConfig:
    # GCS
    input_bucket: str
    input_prefix: str  # under tenant, e.g. "incoming/"
    tenants_allowlist: set[str] | None

    # Incremental
    incremental: bool
    force_reindex: bool

    # Output artifacts
    output_bucket: str | None
    output_prefix: str  # e.g. "rag-extracted/"
    max_content_chars: int

    # OCR / Document AI
    ocr_enabled: bool
    docai_project: str | None
    docai_location: str | None
    docai_processor_id: str | None

    # Heuristics
    pdf_text_per_page_min: int

    # Concurrency / retries
    max_file_workers: int
    max_retries_per_file: int

    @classmethod
    def from_env(cls) -> IngestConfig:
        input_bucket = os.getenv("RAG_INGEST_INPUT_BUCKET")
        if not input_bucket:
            raise ValueError("RAG_INGEST_INPUT_BUCKET is required")

        input_prefix = os.getenv("RAG_INGEST_INPUT_PREFIX", "incoming/")
        if input_prefix and not input_prefix.endswith("/"):
            input_prefix += "/"

        tenants = os.getenv("RAG_INGEST_TENANTS")
        allowlist = {t.strip() for t in tenants.split(",") if t.strip()} if tenants else None

        output_bucket = os.getenv("RAG_INGEST_OUTPUT_BUCKET")
        output_prefix = os.getenv("RAG_INGEST_OUTPUT_PREFIX", "rag-extracted/")
        if output_prefix and not output_prefix.endswith("/"):
            output_prefix += "/"

        ocr_enabled = _get_bool("RAG_OCR_ENABLED", True)
        docai_project = os.getenv("RAG_DOC_AI_PROJECT")
        docai_location = os.getenv("RAG_DOC_AI_LOCATION")
        docai_processor_id = os.getenv("RAG_DOC_AI_PROCESSOR_ID")

        return cls(
            input_bucket=input_bucket,
            input_prefix=input_prefix,
            tenants_allowlist=allowlist,
            incremental=_get_bool("RAG_INGEST_INCREMENTAL", True),
            force_reindex=_get_bool("RAG_INGEST_FORCE_REINDEX", False),
            output_bucket=output_bucket,
            output_prefix=output_prefix,
            max_content_chars=_get_int("RAG_MAX_CONTENT_CHARS", 2_000_000),
            ocr_enabled=ocr_enabled,
            docai_project=docai_project,
            docai_location=docai_location,
            docai_processor_id=docai_processor_id,
            pdf_text_per_page_min=_get_int("RAG_PDF_TEXT_PER_PAGE_MIN", 200),
            max_file_workers=_get_int("RAG_INGEST_MAX_FILE_WORKERS", 3),
            max_retries_per_file=_get_int("RAG_INGEST_MAX_RETRIES_PER_FILE", 2),
        )

    def validate(self) -> None:
        if self.tenants_allowlist is not None and len(self.tenants_allowlist) == 0:
            raise ValueError("RAG_INGEST_TENANTS was set but parsed as empty")

        if self.ocr_enabled:
            missing = [
                k
                for k, v in {
                    "RAG_DOC_AI_PROJECT": self.docai_project,
                    "RAG_DOC_AI_LOCATION": self.docai_location,
                    "RAG_DOC_AI_PROCESSOR_ID": self.docai_processor_id,
                }.items()
                if not v
            ]
            if missing:
                raise ValueError(f"OCR enabled but missing DocAI config: {', '.join(missing)}")

        if self.max_file_workers < 1:
            raise ValueError("RAG_INGEST_MAX_FILE_WORKERS must be >= 1")
        if self.max_retries_per_file < 0:
            raise ValueError("RAG_INGEST_MAX_RETRIES_PER_FILE must be >= 0")
