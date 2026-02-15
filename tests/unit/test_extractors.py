"""Unit tests for ingestion extractors with real fixture files.

Tests parse real file bytes (no mocks on the extraction libraries).
Gracefully skips if ingestion dependencies are not installed.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Guard: skip entire module if ingestion deps are not installed.
# Same pattern as test_ingestion.py.
# ---------------------------------------------------------------------------

_MISSING_DEPS: list[str] = []

for _mod_name in (
    "google.cloud",
    "google.cloud.storage",
    "bs4",
    "lxml",
    "docx",
    "pypdf",
):
    if _mod_name not in sys.modules:
        try:
            __import__(_mod_name)
        except ImportError:
            _MISSING_DEPS.append(_mod_name)

if _MISSING_DEPS:
    _storage_stub = MagicMock()
    for _m in ("google", "google.cloud", "google.cloud.storage"):
        if _m not in sys.modules:
            sys.modules[_m] = _storage_stub
    _bs4_stub = MagicMock()
    if "bs4" not in sys.modules:
        sys.modules["bs4"] = _bs4_stub
    _lxml_stub = MagicMock()
    if "lxml" not in sys.modules:
        sys.modules["lxml"] = _lxml_stub
    _docx_stub = MagicMock()
    if "docx" not in sys.modules:
        sys.modules["docx"] = _docx_stub
    _pypdf_stub = MagicMock()
    for _m in ("pypdf", "pypdf._reader"):
        if _m not in sys.modules:
            sys.modules[_m] = _pypdf_stub
    _docai_stub = MagicMock()
    for _m in (
        "google.cloud.documentai",
        "google.cloud.documentai_v1",
        "google.cloud.documentai_v1.types",
    ):
        if _m not in sys.modules:
            sys.modules[_m] = _docai_stub

from rag_service.ingestion.types import WorkItem  # noqa: E402

pytestmark = pytest.mark.skipif(bool(_MISSING_DEPS), reason=f"Missing: {_MISSING_DEPS}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(*, doc_type: str = "text", name: str = "t1/incoming/file.txt") -> WorkItem:
    return WorkItem(
        tenant_id="t1",
        source_uri=f"gs://bucket/{name}",
        bucket="bucket",
        name=name,
        content_type=None,
        generation="12345",
        md5_hash="abc123",
        crc32c="XYZ==",
        size=1024,
        updated=datetime(2026, 1, 15, tzinfo=UTC),
        doc_type=doc_type,
    )


# ===========================================================================
# TextExtractor
# ===========================================================================


class TestTextExtractorFixtures:
    def test_plain_text_happy_path(self, fixtures_dir: Path) -> None:
        from rag_service.ingestion.extractors.text import TextExtractor

        ext = TextExtractor()
        data = (fixtures_dir / "sample.txt").read_bytes()
        item = _make_item(doc_type="text", name="t1/incoming/sample.txt")
        result = ext.extract(item=item, data=data)

        assert result.used_ocr is False
        assert result.extraction_meta["strategy"] == "text"
        assert "RAG Platform" in result.text
        assert "Multi-tenant isolation" in result.text

    def test_markdown_preserved(self, fixtures_dir: Path) -> None:
        from rag_service.ingestion.extractors.text import TextExtractor

        ext = TextExtractor()
        data = (fixtures_dir / "sample.md").read_bytes()
        item = _make_item(doc_type="text", name="t1/incoming/sample.md")
        result = ext.extract(item=item, data=data)

        assert "# RAG Platform Guide" in result.text
        assert "```python" in result.text
        assert "[deployment guide]" in result.text

    def test_null_bytes_stripped(self, fixtures_dir: Path) -> None:
        from rag_service.ingestion.extractors.text import TextExtractor

        ext = TextExtractor()
        data = (fixtures_dir / "null_bytes.txt").read_bytes()
        assert b"\x00" in data, "Fixture must contain actual null bytes"
        item = _make_item(doc_type="text", name="t1/incoming/null_bytes.txt")
        result = ext.extract(item=item, data=data)

        assert "\x00" not in result.text
        assert "Hello" in result.text
        assert "World" in result.text
        assert "Test" in result.text

    def test_excessive_newlines_collapsed(self, fixtures_dir: Path) -> None:
        from rag_service.ingestion.extractors.text import TextExtractor

        ext = TextExtractor()
        data = (fixtures_dir / "excessive_newlines.txt").read_bytes()
        item = _make_item(doc_type="text", name="t1/incoming/excessive_newlines.txt")
        result = ext.extract(item=item, data=data)

        assert "\n\n\n\n" not in result.text
        assert "First paragraph" in result.text
        assert "Third paragraph" in result.text

    def test_empty_file_returns_empty_string(self, fixtures_dir: Path) -> None:
        from rag_service.ingestion.extractors.text import TextExtractor

        ext = TextExtractor()
        data = (fixtures_dir / "empty.txt").read_bytes()
        item = _make_item(doc_type="text", name="t1/incoming/empty.txt")
        result = ext.extract(item=item, data=data)

        assert result.text == ""

    def test_binary_garbage_decoded_with_ignore(self) -> None:
        from rag_service.ingestion.extractors.text import TextExtractor

        ext = TextExtractor()
        garbage = bytes(range(0x80, 0xFF))
        item = _make_item(doc_type="text", name="t1/incoming/garbage.bin")
        result = ext.extract(item=item, data=garbage)

        assert isinstance(result.text, str)


# ===========================================================================
# HtmlExtractor
# ===========================================================================


class TestHtmlExtractorFixtures:
    def test_well_formed_html_extracts_text(self, fixtures_dir: Path) -> None:
        from rag_service.ingestion.extractors.html import HtmlExtractor

        ext = HtmlExtractor()
        data = (fixtures_dir / "sample.html").read_bytes()
        item = _make_item(doc_type="html", name="t1/incoming/sample.html")
        result = ext.extract(item=item, data=data)

        assert "Welcome to RAG Platform" in result.text
        assert "Vector search" in result.text
        assert "Full-text search" in result.text
        assert "<h1>" not in result.text
        assert "<li>" not in result.text

    def test_script_and_style_stripped(self, fixtures_dir: Path) -> None:
        from rag_service.ingestion.extractors.html import HtmlExtractor

        ext = HtmlExtractor()
        data = (fixtures_dir / "script_style.html").read_bytes()
        item = _make_item(doc_type="html", name="t1/incoming/script_style.html")
        result = ext.extract(item=item, data=data)

        assert "alert" not in result.text
        assert "console.log" not in result.text
        assert "font-size" not in result.text
        assert "display: none" not in result.text
        assert "This visible paragraph should remain" in result.text
        assert "Second visible paragraph" in result.text

    def test_malformed_html_tolerant(self, fixtures_dir: Path) -> None:
        from rag_service.ingestion.extractors.html import HtmlExtractor

        ext = HtmlExtractor()
        data = (fixtures_dir / "malformed.html").read_bytes()
        item = _make_item(doc_type="html", name="t1/incoming/malformed.html")
        result = ext.extract(item=item, data=data)

        assert "First paragraph" in result.text
        assert "Nested and misnested content" in result.text

    def test_script_only_yields_minimal_text(self, fixtures_dir: Path) -> None:
        from rag_service.ingestion.extractors.html import HtmlExtractor

        ext = HtmlExtractor()
        data = (fixtures_dir / "script_only.html").read_bytes()
        item = _make_item(doc_type="html", name="t1/incoming/script_only.html")
        result = ext.extract(item=item, data=data)

        assert "var data" not in result.text
        assert "margin: 0" not in result.text
        # After stripping script/style/noscript, very little should remain
        assert len(result.text) < 50

    def test_empty_html_returns_empty_string(self, fixtures_dir: Path) -> None:
        from rag_service.ingestion.extractors.html import HtmlExtractor

        ext = HtmlExtractor()
        data = (fixtures_dir / "empty.html").read_bytes()
        item = _make_item(doc_type="html", name="t1/incoming/empty.html")
        result = ext.extract(item=item, data=data)

        assert result.text == ""


# ===========================================================================
# DocxExtractor
# ===========================================================================


class TestDocxExtractorFixtures:
    def test_docx_happy_path(self, sample_docx_bytes: bytes) -> None:
        from rag_service.ingestion.extractors.docx import DocxExtractor

        ext = DocxExtractor()
        item = _make_item(doc_type="docx", name="t1/incoming/sample.docx")
        result = ext.extract(item=item, data=sample_docx_bytes)

        assert "First paragraph" in result.text
        assert "Second paragraph" in result.text
        assert "Third and final paragraph" in result.text
        assert result.extraction_meta["strategy"] == "docx"
        assert result.used_ocr is False

    def test_empty_docx_returns_empty_string(self, empty_docx_bytes: bytes) -> None:
        from rag_service.ingestion.extractors.docx import DocxExtractor

        ext = DocxExtractor()
        item = _make_item(doc_type="docx", name="t1/incoming/empty.docx")
        result = ext.extract(item=item, data=empty_docx_bytes)

        assert result.text == ""

    def test_corrupt_docx_raises(self) -> None:
        from zipfile import BadZipFile

        from rag_service.ingestion.extractors.docx import DocxExtractor

        ext = DocxExtractor()
        item = _make_item(doc_type="docx", name="t1/incoming/corrupt.docx")

        with pytest.raises((BadZipFile, Exception)):
            ext.extract(item=item, data=b"NOT_A_VALID_ZIP")

    def test_truncated_docx_raises(self, sample_docx_bytes: bytes) -> None:
        from zipfile import BadZipFile

        from rag_service.ingestion.extractors.docx import DocxExtractor

        ext = DocxExtractor()
        item = _make_item(doc_type="docx", name="t1/incoming/truncated.docx")
        truncated = sample_docx_bytes[: len(sample_docx_bytes) // 2]

        with pytest.raises((BadZipFile, Exception)):
            ext.extract(item=item, data=truncated)


# ===========================================================================
# PdfExtractor (all with docai=None)
# ===========================================================================


class TestPdfExtractorFixtures:
    def test_pdf_happy_path(self, sample_pdf_bytes: bytes) -> None:
        from rag_service.ingestion.extractors.pdf import PdfExtractor

        ext = PdfExtractor(docai=None, text_per_page_min=1, output_prefix_for_docai=None)
        item = _make_item(doc_type="pdf", name="t1/incoming/sample.pdf")
        result = ext.extract(item=item, data=sample_pdf_bytes)

        assert "Line one" in result.text
        assert result.extraction_meta["strategy"] == "pypdf"
        assert result.pages == 1
        assert result.used_ocr is False

    def test_pdf_multi_page_count(self, multi_page_pdf_bytes: bytes) -> None:
        from rag_service.ingestion.extractors.pdf import PdfExtractor

        ext = PdfExtractor(docai=None, text_per_page_min=1, output_prefix_for_docai=None)
        item = _make_item(doc_type="pdf", name="t1/incoming/multi.pdf")
        result = ext.extract(item=item, data=multi_page_pdf_bytes)

        assert result.pages == 3
        assert "page 1" in result.text
        assert "page 2" in result.text
        assert "page 3" in result.text

    def test_empty_page_raises_no_ocr(self, empty_pdf_bytes: bytes) -> None:
        from rag_service.ingestion.extractors.pdf import PdfExtractor

        ext = PdfExtractor(docai=None, text_per_page_min=1, output_prefix_for_docai=None)
        item = _make_item(doc_type="pdf", name="t1/incoming/empty.pdf")

        with pytest.raises(RuntimeError, match="OCR is disabled"):
            ext.extract(item=item, data=empty_pdf_bytes)

    def test_low_quality_warns_ocr_skipped(self, sample_pdf_bytes: bytes) -> None:
        from rag_service.ingestion.extractors.pdf import PdfExtractor

        # Set threshold impossibly high so text is considered low quality
        ext = PdfExtractor(docai=None, text_per_page_min=999999, output_prefix_for_docai=None)
        item = _make_item(doc_type="pdf", name="t1/incoming/lowq.pdf")
        result = ext.extract(item=item, data=sample_pdf_bytes)

        assert result.extraction_meta.get("ocr_skipped") is True
        assert result.text  # Still returns the pypdf text
        assert result.used_ocr is False

    def test_corrupt_pdf_raises_no_ocr(self) -> None:
        from rag_service.ingestion.extractors.pdf import PdfExtractor

        ext = PdfExtractor(docai=None, text_per_page_min=1, output_prefix_for_docai=None)
        item = _make_item(doc_type="pdf", name="t1/incoming/corrupt.pdf")

        with pytest.raises(RuntimeError, match="OCR is disabled"):
            ext.extract(item=item, data=b"%PDF-1.4 GARBAGE")

    def test_invalid_bytes_raises(self) -> None:
        from rag_service.ingestion.extractors.pdf import PdfExtractor

        ext = PdfExtractor(docai=None, text_per_page_min=1, output_prefix_for_docai=None)
        item = _make_item(doc_type="pdf", name="t1/incoming/invalid.pdf")

        with pytest.raises(RuntimeError, match="OCR is disabled"):
            ext.extract(item=item, data=b"NOT_EVEN_PDF")

    def test_text_per_page_metadata(self, multi_page_pdf_bytes: bytes) -> None:
        from rag_service.ingestion.extractors.pdf import PdfExtractor

        ext = PdfExtractor(docai=None, text_per_page_min=1, output_prefix_for_docai=None)
        item = _make_item(doc_type="pdf", name="t1/incoming/multi.pdf")
        result = ext.extract(item=item, data=multi_page_pdf_bytes)

        tpp = result.extraction_meta["text_per_page"]
        expected_tpp = len(result.text) // result.pages
        assert tpp == expected_tpp
