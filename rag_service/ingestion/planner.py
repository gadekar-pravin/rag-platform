from __future__ import annotations

from datetime import datetime

from google.cloud import storage

from rag_service.ingestion.gcs import gs_uri
from rag_service.ingestion.types import WorkItem

_SUPPORTED_EXTS: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".html": "html",
    ".htm": "html",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".tiff": "image",
    ".txt": "text",
    ".md": "text",
}


def _ext(name: str) -> str:
    base = name.lower()
    for e in _SUPPORTED_EXTS:
        if base.endswith(e):
            return e
    return ""


def derive_doc_type(name: str) -> str | None:
    e = _ext(name)
    return _SUPPORTED_EXTS.get(e)


def compute_source_hash(
    *,
    generation: str | None,
    md5_hash: str | None,
    crc32c: str | None,
    size: int | None,
    updated: datetime | None,
) -> str:
    # generation is the strongest change marker; include others when present
    upd = updated.isoformat() if updated else ""
    return f"gen:{generation or ''}|md5:{md5_hash or ''}|crc32c:{crc32c or ''}|size:{size or ''}|updated:{upd}"


def build_prefix(tenant_id: str, under_tenant_prefix: str) -> str:
    # Always "tenant_id/<under_tenant_prefix>"
    p = under_tenant_prefix or ""
    if p and not p.endswith("/"):
        p += "/"
    if p.startswith("/"):
        p = p[1:]
    return f"{tenant_id}/{p}"


def discover_work_items(
    client: storage.Client,
    *,
    bucket: str,
    tenant_id: str,
    under_tenant_prefix: str,
    max_files: int = 0,
) -> list[WorkItem]:
    prefix = build_prefix(tenant_id, under_tenant_prefix)
    items: list[WorkItem] = []
    for blob in client.list_blobs(bucket, prefix=prefix):
        if blob.name.endswith("/"):
            continue

        dt = derive_doc_type(blob.name)
        if not dt:
            continue

        items.append(
            WorkItem(
                tenant_id=tenant_id,
                source_uri=gs_uri(bucket, blob.name),
                bucket=bucket,
                name=blob.name,
                content_type=getattr(blob, "content_type", None),
                generation=str(getattr(blob, "generation", "") or "") or None,
                md5_hash=getattr(blob, "md5_hash", None),
                crc32c=getattr(blob, "crc32c", None),
                size=int(getattr(blob, "size", 0) or 0) or None,
                updated=getattr(blob, "updated", None),
                doc_type=dt,
            )
        )
        if max_files and len(items) >= max_files:
            break

    return items
