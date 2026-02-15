from __future__ import annotations

import io

import docx  # python-docx

from rag_service.ingestion.extractors.base import Extractor, normalize_text
from rag_service.ingestion.types import ExtractResult, WorkItem


class DocxExtractor(Extractor):
    def can_handle(self, item: WorkItem) -> bool:
        return item.doc_type == "docx"

    def extract(self, *, item: WorkItem, data: bytes) -> ExtractResult:
        f = io.BytesIO(data)
        d = docx.Document(f)
        parts: list[str] = []
        for p in d.paragraphs:
            if p.text and p.text.strip():
                parts.append(p.text)
        text = normalize_text("\n".join(parts))
        return ExtractResult(
            text=text,
            used_ocr=False,
            pages=None,
            extraction_meta={"strategy": "docx"},
        )
