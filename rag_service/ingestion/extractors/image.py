from __future__ import annotations

from rag_service.ingestion.extractors.base import Extractor, normalize_text
from rag_service.ingestion.ocr.document_ai import DocumentAIClient
from rag_service.ingestion.types import ExtractResult, WorkItem


class ImageExtractor(Extractor):
    def __init__(self, *, docai: DocumentAIClient) -> None:
        self._docai = docai

    def can_handle(self, item: WorkItem) -> bool:
        return item.doc_type == "image"

    def extract(self, *, item: WorkItem, data: bytes) -> ExtractResult:
        mime = item.content_type or "image/png"
        text, meta = self._docai.ocr_online(content=data, mime_type=mime)
        text = normalize_text(text)
        return ExtractResult(
            text=text,
            used_ocr=True,
            pages=meta.get("pages"),
            extraction_meta={"strategy": "docai_online", **meta},
        )