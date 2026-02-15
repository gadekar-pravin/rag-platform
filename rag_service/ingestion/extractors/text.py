from __future__ import annotations

from rag_service.ingestion.extractors.base import Extractor, normalize_text
from rag_service.ingestion.types import ExtractResult, WorkItem


class TextExtractor(Extractor):
    def can_handle(self, item: WorkItem) -> bool:
        return item.doc_type == "text"

    def extract(self, *, item: WorkItem, data: bytes) -> ExtractResult:
        text = data.decode("utf-8", errors="ignore")
        text = normalize_text(text)
        return ExtractResult(
            text=text,
            used_ocr=False,
            pages=None,
            extraction_meta={"strategy": "text"},
        )
