"""Unit test conftest — no database required."""

from __future__ import annotations

import io
from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the ingestion fixtures directory."""
    return Path(__file__).resolve().parent.parent / "fixtures" / "ingestion"


@pytest.fixture
def sample_docx_bytes() -> bytes:
    """Generate a 3-paragraph DOCX file in memory."""
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_paragraph("First paragraph of the document.")
    doc.add_paragraph("Second paragraph with more detail.")
    doc.add_paragraph("Third and final paragraph.")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture
def empty_docx_bytes() -> bytes:
    """Generate a valid DOCX with no paragraphs containing text."""
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Generate a 1-page PDF with 3 lines via fpdf2."""
    fpdf = pytest.importorskip("fpdf")
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(text="Line one of the PDF document.")
    pdf.ln()
    pdf.cell(text="Line two with additional content.")
    pdf.ln()
    pdf.cell(text="Line three concludes the page.")
    return bytes(pdf.output())


@pytest.fixture
def multi_page_pdf_bytes() -> bytes:
    """Generate a 3-page PDF for page count verification."""
    fpdf = pytest.importorskip("fpdf")
    pdf = fpdf.FPDF()
    pdf.set_font("Helvetica", size=12)
    for i in range(1, 4):
        pdf.add_page()
        pdf.cell(text=f"Content on page {i}.")
    return bytes(pdf.output())


@pytest.fixture
def empty_pdf_bytes() -> bytes:
    """Generate a 1-page PDF with no text content."""
    fpdf = pytest.importorskip("fpdf")
    pdf = fpdf.FPDF()
    pdf.add_page()
    return bytes(pdf.output())
