from __future__ import annotations

import io
import logging

from pypdf import PdfReader

from rag_service.ingestion.extractors.base import Extractor, normalize_text
from rag_service.ingestion.ocr.document_ai import DocumentAIClient
from rag_service.ingestion.types import ExtractResult, WorkItem

logger = logging.getLogger(__name__)


class PdfExtractor(Extractor):
    def __init__(self, *, docai: DocumentAIClient, text_per_page_min: int, output_prefix_for_docai: str | None) -> None:
        self._docai = docai
        self._min = max(1, int(text_per_page_min))
        self._docai_output_prefix = output_prefix_for_docai  # gs://bucket/prefix/... per item

    def can_handle(self, item: WorkItem) -> bool:
        return item.doc_type == "pdf"

    def extract(self, *, item: WorkItem, data: bytes) -> ExtractResult:
        # Try local extraction first
        pages = None
        extracted = ""
        try:
            r = PdfReader(io.BytesIO(data))
            pages = len(r.pages)
            parts: list[str] = []
            for p in r.pages:
                t = p.extract_text() or ""
                if t.strip():
                    parts.append(t)
            extracted = "\n".join(parts)
        except Exception as e:
            logger.warning("PyPDF text extraction failed, falling back to OCR: %s", e)
            extracted = ""

        extracted_norm = normalize_text(extracted)
        text_len = len(extracted_norm)
        pcount = pages or 1
        tpp = int(text_len / max(pcount, 1))

        # OCR if low quality or empty
        if not extracted_norm or tpp < self._min:
            if self._docai_output_prefix is None:
                raise ValueError("RAG_INGEST_OUTPUT_BUCKET is required for PDF batch OCR")
            # Batch OCR expects GCS input; ingestion passes source_uri as GCS.
            # We do not upload bytes; we call DocAI batch on the original GCS object.
            text, meta = self._docai.ocr_pdf_batch(
                input_gcs_uri=item.source_uri,
                output_gcs_prefix=self._docai_output_prefix,
            )
            text = normalize_text(text)
            return ExtractResult(
                text=text,
                used_ocr=True,
                pages=pages,
                extraction_meta={"strategy": "docai_batch", "text_per_page": tpp, **meta},
            )

        return ExtractResult(
            text=extracted_norm,
            used_ocr=False,
            pages=pages,
            extraction_meta={"strategy": "pypdf", "text_per_page": tpp},
        )
