from __future__ import annotations

from bs4 import BeautifulSoup

from rag_service.ingestion.extractors.base import Extractor, normalize_text
from rag_service.ingestion.types import ExtractResult, WorkItem


class HtmlExtractor(Extractor):
    def can_handle(self, item: WorkItem) -> bool:
        return item.doc_type == "html"

    def extract(self, *, item: WorkItem, data: bytes) -> ExtractResult:
        raw = data.decode("utf-8", errors="ignore")
        soup = BeautifulSoup(raw, "lxml")
        # Remove script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = normalize_text(text)
        return ExtractResult(
            text=text,
            used_ocr=False,
            pages=None,
            extraction_meta={"strategy": "html"},
        )