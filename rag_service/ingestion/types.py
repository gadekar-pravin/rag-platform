from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class WorkItem:
    tenant_id: str
    source_uri: str  # gs://bucket/name
    bucket: str
    name: str  # object name in bucket
    content_type: str | None
    generation: str | None
    md5_hash: str | None
    crc32c: str | None
    size: int | None
    updated: datetime | None
    doc_type: str | None  # derived from extension


@dataclass(frozen=True)
class ExtractResult:
    text: str
    used_ocr: bool
    pages: int | None
    extraction_meta: dict[str, Any]


@dataclass(frozen=True)
class ProcessResult:
    item: WorkItem
    status: str  # completed|skipped|failed
    document_id: str | None
    error_message: str | None
