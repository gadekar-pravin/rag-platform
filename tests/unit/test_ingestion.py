"""Unit tests for the ingestion pipeline.

Mock-based — no database or GCS required.
Gracefully skips if ingestion dependencies are not installed.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Guard: skip entire module if ingestion deps (google-cloud-storage, etc.)
# are not installed. We stub google.cloud.storage so planner.py can import.
# ---------------------------------------------------------------------------

_MISSING_DEPS: list[str] = []

for _mod_name in ("google.cloud", "google.cloud.storage", "bs4", "lxml", "docx", "pypdf"):
    if _mod_name not in sys.modules:
        try:
            __import__(_mod_name)
        except ImportError:
            _MISSING_DEPS.append(_mod_name)

if _MISSING_DEPS:
    # Provide lightweight stubs so module-level imports succeed
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
    # Document AI stubs
    _docai_stub = MagicMock()
    for _m in (
        "google.cloud.documentai",
        "google.cloud.documentai_v1",
        "google.cloud.documentai_v1.types",
    ):
        if _m not in sys.modules:
            sys.modules[_m] = _docai_stub

from rag_service.ingestion.config import IngestConfig  # noqa: E402
from rag_service.ingestion.extractors.base import normalize_text  # noqa: E402
from rag_service.ingestion.planner import compute_source_hash, derive_doc_type, discover_work_items  # noqa: E402
from rag_service.ingestion.types import WorkItem  # noqa: E402

# Mark extractor tests to skip if real deps aren't available
_need_ingestion_deps = pytest.mark.skipif(bool(_MISSING_DEPS), reason=f"Missing: {_MISSING_DEPS}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(*, doc_type: str = "text", content_type: str | None = None) -> WorkItem:
    return WorkItem(
        tenant_id="t1",
        source_uri="gs://bucket/t1/incoming/file.txt",
        bucket="bucket",
        name="t1/incoming/file.txt",
        content_type=content_type,
        generation="12345",
        md5_hash="abc123",
        crc32c="XYZ==",
        size=1024,
        updated=datetime(2026, 1, 15, tzinfo=UTC),
        doc_type=doc_type,
    )


# ===========================================================================
# source_hash computation (pure functions — no external deps)
# ===========================================================================


class TestSourceHash:
    def test_generation_only(self):
        """generation-only: md5 and crc32c are None."""
        h = compute_source_hash(
            generation="999",
            md5_hash=None,
            crc32c=None,
            size=None,
            updated=None,
        )
        assert "gen:999" in h
        assert "md5:" in h
        assert "crc32c:" in h

    def test_generation_and_md5(self):
        """Happy path: generation + md5 present."""
        h = compute_source_hash(
            generation="123",
            md5_hash="abc",
            crc32c=None,
            size=4096,
            updated=datetime(2026, 1, 1, tzinfo=UTC),
        )
        assert h.startswith("gen:123|")
        assert "md5:abc" in h
        assert "size:4096" in h

    def test_composite_all_fields(self):
        """All fields present — composite hash includes everything."""
        h = compute_source_hash(
            generation="42",
            md5_hash="deadbeef",
            crc32c="AAAA==",
            size=2048,
            updated=datetime(2026, 6, 15, 12, 0, tzinfo=UTC),
        )
        assert "gen:42" in h
        assert "md5:deadbeef" in h
        assert "crc32c:AAAA==" in h
        assert "size:2048" in h
        assert "2026-06-15" in h

    def test_different_generation_produces_different_hash(self):
        """Changing generation alone should produce a different hash."""
        h1 = compute_source_hash(generation="1", md5_hash=None, crc32c=None, size=None, updated=None)
        h2 = compute_source_hash(generation="2", md5_hash=None, crc32c=None, size=None, updated=None)
        assert h1 != h2


# ===========================================================================
# Extractors (require ingestion deps)
# ===========================================================================


@_need_ingestion_deps
class TestTextExtractor:
    def test_extract_utf8(self):
        from rag_service.ingestion.extractors.text import TextExtractor

        ext = TextExtractor()
        item = _make_item(doc_type="text")
        result = ext.extract(item=item, data=b"Hello world")
        assert result.text == "Hello world"
        assert result.used_ocr is False
        assert result.extraction_meta["strategy"] == "text"

    def test_extract_normalizes(self):
        from rag_service.ingestion.extractors.text import TextExtractor

        ext = TextExtractor()
        item = _make_item(doc_type="text")
        raw = "line1\x00\n\n\n\n\nline2"
        result = ext.extract(item=item, data=raw.encode())
        assert "\x00" not in result.text
        assert "\n\n\n\n" not in result.text

    def test_can_handle(self):
        from rag_service.ingestion.extractors.text import TextExtractor

        ext = TextExtractor()
        assert ext.can_handle(_make_item(doc_type="text")) is True
        assert ext.can_handle(_make_item(doc_type="html")) is False


@_need_ingestion_deps
class TestHtmlExtractor:
    def test_strips_script_and_style(self):
        from rag_service.ingestion.extractors.html import HtmlExtractor

        ext = HtmlExtractor()
        item = _make_item(doc_type="html")
        html = b"<html><head><style>body{}</style></head><body><script>alert(1)</script><p>Hello</p></body></html>"
        result = ext.extract(item=item, data=html)
        assert "alert" not in result.text
        assert "body{}" not in result.text
        assert "Hello" in result.text
        assert result.extraction_meta["strategy"] == "html"

    def test_can_handle(self):
        from rag_service.ingestion.extractors.html import HtmlExtractor

        ext = HtmlExtractor()
        assert ext.can_handle(_make_item(doc_type="html")) is True
        assert ext.can_handle(_make_item(doc_type="pdf")) is False


@_need_ingestion_deps
class TestDocxExtractor:
    def test_extract_paragraphs(self):
        from rag_service.ingestion.extractors.docx import DocxExtractor

        ext = DocxExtractor()
        item = _make_item(doc_type="docx")

        mock_para1 = MagicMock()
        mock_para1.text = "First paragraph"
        mock_para2 = MagicMock()
        mock_para2.text = ""
        mock_para3 = MagicMock()
        mock_para3.text = "Third paragraph"

        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_para1, mock_para2, mock_para3]

        with patch("rag_service.ingestion.extractors.docx.docx.Document", return_value=mock_doc):
            result = ext.extract(item=item, data=b"fake-docx-bytes")

        assert "First paragraph" in result.text
        assert "Third paragraph" in result.text
        assert result.extraction_meta["strategy"] == "docx"

    def test_can_handle(self):
        from rag_service.ingestion.extractors.docx import DocxExtractor

        ext = DocxExtractor()
        assert ext.can_handle(_make_item(doc_type="docx")) is True
        assert ext.can_handle(_make_item(doc_type="text")) is False


@_need_ingestion_deps
class TestPdfExtractor:
    def test_heuristic_triggers_ocr(self):
        """When text_per_page < threshold, OCR should be invoked."""
        from rag_service.ingestion.extractors.pdf import PdfExtractor

        mock_docai = MagicMock()
        mock_docai.ocr_pdf_batch.return_value = ("OCR extracted text", {"provider": "documentai"})

        ext = PdfExtractor(docai=mock_docai, text_per_page_min=200, output_prefix_for_docai="gs://out/prefix/")
        item = _make_item(doc_type="pdf")

        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "hi"
        mock_reader.pages = [mock_page, mock_page, mock_page]

        with patch("rag_service.ingestion.extractors.pdf.PdfReader", return_value=mock_reader):
            result = ext.extract(item=item, data=b"fake-pdf")

        assert result.used_ocr is True
        assert "OCR extracted text" in result.text
        mock_docai.ocr_pdf_batch.assert_called_once()

    def test_good_text_skips_ocr(self):
        """When text_per_page >= threshold, no OCR needed."""
        from rag_service.ingestion.extractors.pdf import PdfExtractor

        mock_docai = MagicMock()
        ext = PdfExtractor(docai=mock_docai, text_per_page_min=50, output_prefix_for_docai="gs://out/prefix/")
        item = _make_item(doc_type="pdf")

        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "A" * 300
        mock_reader.pages = [mock_page]

        with patch("rag_service.ingestion.extractors.pdf.PdfReader", return_value=mock_reader):
            result = ext.extract(item=item, data=b"fake-pdf")

        assert result.used_ocr is False
        assert result.extraction_meta["strategy"] == "pypdf"
        mock_docai.ocr_pdf_batch.assert_not_called()


@_need_ingestion_deps
class TestImageExtractor:
    def test_delegates_to_docai(self):
        from rag_service.ingestion.extractors.image import ImageExtractor

        mock_docai = MagicMock()
        mock_docai.ocr_online.return_value = ("Image text", {"provider": "documentai", "pages": 1})

        ext = ImageExtractor(docai=mock_docai)
        item = _make_item(doc_type="image", content_type="image/png")
        result = ext.extract(item=item, data=b"fake-image-bytes")

        assert result.used_ocr is True
        assert "Image text" in result.text
        mock_docai.ocr_online.assert_called_once_with(content=b"fake-image-bytes", mime_type="image/png")

    def test_default_mime_type(self):
        from rag_service.ingestion.extractors.image import ImageExtractor

        mock_docai = MagicMock()
        mock_docai.ocr_online.return_value = ("text", {})

        ext = ImageExtractor(docai=mock_docai)
        item = _make_item(doc_type="image", content_type=None)
        ext.extract(item=item, data=b"data")

        mock_docai.ocr_online.assert_called_once_with(content=b"data", mime_type="image/png")


# ===========================================================================
# Planner (pure functions — no external deps needed)
# ===========================================================================


class TestDeriveDocType:
    def test_supported_extensions(self):
        assert derive_doc_type("report.pdf") == "pdf"
        assert derive_doc_type("doc.docx") == "docx"
        assert derive_doc_type("page.html") == "html"
        assert derive_doc_type("page.htm") == "html"
        assert derive_doc_type("photo.png") == "image"
        assert derive_doc_type("photo.jpg") == "image"
        assert derive_doc_type("photo.jpeg") == "image"
        assert derive_doc_type("photo.webp") == "image"
        assert derive_doc_type("scan.tiff") == "image"
        assert derive_doc_type("readme.txt") == "text"
        assert derive_doc_type("notes.md") == "text"

    def test_unsupported_returns_none(self):
        assert derive_doc_type("archive.zip") is None
        assert derive_doc_type("data.csv") is None
        assert derive_doc_type("program.exe") is None

    def test_case_insensitive(self):
        assert derive_doc_type("FILE.PDF") == "pdf"
        assert derive_doc_type("Image.JPG") == "image"


class TestDiscoverWorkItems:
    def test_filters_by_supported_extensions(self):
        mock_client = MagicMock()

        blob_pdf = MagicMock()
        blob_pdf.name = "t1/incoming/report.pdf"
        blob_pdf.content_type = "application/pdf"
        blob_pdf.generation = 1
        blob_pdf.md5_hash = "abc"
        blob_pdf.crc32c = "xyz"
        blob_pdf.size = 100
        blob_pdf.updated = None

        blob_zip = MagicMock()
        blob_zip.name = "t1/incoming/archive.zip"

        blob_dir = MagicMock()
        blob_dir.name = "t1/incoming/subdir/"

        mock_client.list_blobs.return_value = [blob_pdf, blob_zip, blob_dir]

        items = discover_work_items(
            mock_client,
            bucket="my-bucket",
            tenant_id="t1",
            under_tenant_prefix="incoming/",
        )
        assert len(items) == 1
        assert items[0].doc_type == "pdf"

    def test_respects_max_files(self):
        mock_client = MagicMock()

        blobs = []
        for i in range(10):
            b = MagicMock()
            b.name = f"t1/incoming/doc{i}.txt"
            b.content_type = "text/plain"
            b.generation = i
            b.md5_hash = None
            b.crc32c = None
            b.size = 100
            b.updated = None
            blobs.append(b)

        mock_client.list_blobs.return_value = blobs

        items = discover_work_items(
            mock_client,
            bucket="b",
            tenant_id="t1",
            under_tenant_prefix="incoming/",
            max_files=3,
        )
        assert len(items) == 3


# ===========================================================================
# IngestConfig validation
# ===========================================================================


class TestIngestConfigValidation:
    def test_missing_bucket_raises(self):
        with pytest.raises(ValueError, match="RAG_INGEST_INPUT_BUCKET"), patch.dict("os.environ", {}, clear=True):
            IngestConfig.from_env()

    def test_ocr_enabled_without_processor_raises(self):
        cfg = IngestConfig(
            input_bucket="bucket",
            input_prefix="incoming/",
            tenants_allowlist=None,
            incremental=True,
            force_reindex=False,
            output_bucket=None,
            output_prefix="rag-extracted/",
            max_content_chars=2_000_000,
            ocr_enabled=True,
            docai_project=None,
            docai_location=None,
            docai_processor_id=None,
            pdf_text_per_page_min=200,
            max_file_workers=3,
            max_retries_per_file=2,
        )
        with pytest.raises(ValueError, match="OCR enabled but missing DocAI config"):
            cfg.validate()

    def test_valid_config_passes(self):
        cfg = IngestConfig(
            input_bucket="bucket",
            input_prefix="incoming/",
            tenants_allowlist=None,
            incremental=True,
            force_reindex=False,
            output_bucket=None,
            output_prefix="rag-extracted/",
            max_content_chars=2_000_000,
            ocr_enabled=True,
            docai_project="proj",
            docai_location="us",
            docai_processor_id="pid",
            pdf_text_per_page_min=200,
            max_file_workers=3,
            max_retries_per_file=2,
        )
        cfg.validate()  # Should not raise


# ===========================================================================
# normalize_text
# ===========================================================================


class TestNormalizeText:
    def test_removes_null_bytes(self):
        assert "\x00" not in normalize_text("hello\x00world")
        assert normalize_text("hello\x00world") == "helloworld"

    def test_collapses_blank_lines(self):
        text = "a\n\n\n\n\n\nb"
        result = normalize_text(text)
        assert "\n\n\n\n" not in result
        assert result == "a\n\n\nb"

    def test_strips_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_empty_input(self):
        assert normalize_text("") == ""
        assert normalize_text(None) == ""  # type: ignore[arg-type]
