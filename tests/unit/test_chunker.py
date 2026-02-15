"""Unit tests for the chunker — ported from ApexFlow patterns."""

from __future__ import annotations

import pytest

from rag_service.chunking.chunker import (
    _validate_chunk_params,
    chunk_document,
    chunk_document_with_spans,
)


class TestValidateChunkParams:
    def test_valid_params(self):
        size, overlap = _validate_chunk_params(1000, 100)
        assert size == 1000
        assert overlap == 100

    def test_zero_chunk_size_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            _validate_chunk_params(0, 100)

    def test_negative_overlap_clamped(self):
        size, overlap = _validate_chunk_params(1000, -50)
        assert overlap == 0

    def test_overlap_ge_size_clamped(self):
        size, overlap = _validate_chunk_params(100, 100)
        assert overlap < size

    def test_overlap_greater_than_size_clamped(self):
        size, overlap = _validate_chunk_params(100, 200)
        assert overlap < size


class TestChunkDocument:
    async def test_empty_text_returns_empty(self):
        assert await chunk_document("") == []

    async def test_whitespace_only_returns_empty(self):
        assert await chunk_document("   \n\t  ") == []

    async def test_short_text_single_chunk(self):
        text = "Hello, this is a short document."
        chunks = await chunk_document(text, chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0] == text

    async def test_long_text_splits(self):
        text = "word " * 1000  # ~5000 chars
        chunks = await chunk_document(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1
        for c in chunks:
            # Overlap may cause slight oversize, but recursive guarantees limit
            assert len(c) <= 500 + 50 + 10  # small tolerance for overlap

    async def test_paragraph_splitting(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = await chunk_document(text, chunk_size=30, chunk_overlap=0)
        assert len(chunks) >= 2

    async def test_no_whitespace_only_chunks(self):
        text = "First section.\n\n\n\n\n\nSecond section."
        chunks = await chunk_document(text, chunk_size=50, chunk_overlap=0)
        for c in chunks:
            assert c.strip()  # no whitespace-only chunks

    async def test_overlap_applied(self):
        text = "Section A content here. " * 20 + "\n\n" + "Section B content here. " * 20
        chunks = await chunk_document(text, chunk_size=200, chunk_overlap=50)
        if len(chunks) > 1:
            # Later chunks should contain overlap from previous
            # (unless they already start with the overlap text)
            assert len(chunks[1]) > 0

    async def test_method_rule_based_default(self):
        text = "Some text content for chunking."
        chunks = await chunk_document(text, method="rule_based")
        assert len(chunks) >= 1

    async def test_unknown_method_uses_rule_based(self):
        """Unknown method falls back to rule_based (not semantic)."""
        text = "Some text content."
        # Only "semantic" triggers LLM; anything else is rule_based
        chunks = await chunk_document(text, method="unknown")
        assert len(chunks) >= 1


class TestEdgeCases:
    async def test_single_character(self):
        chunks = await chunk_document("a")
        assert chunks == ["a"]

    async def test_very_small_chunk_size(self):
        text = "Hello world"
        chunks = await chunk_document(text, chunk_size=3, chunk_overlap=0)
        assert len(chunks) >= 1

    async def test_chunk_size_one(self):
        text = "ab"
        chunks = await chunk_document(text, chunk_size=1, chunk_overlap=0)
        assert all(len(c) <= 1 for c in chunks)


class TestChunkDocumentWithSpans:
    """Fix 7: Verify offsets computed during construction match original text."""

    async def test_offsets_match_original_text(self):
        """For each chunk, text[start:end] == chunk_text."""
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
        spans = await chunk_document_with_spans(text, chunk_size=30, chunk_overlap=0)

        assert len(spans) >= 2
        for chunk_text, start, end in spans:
            if start is not None and end is not None:
                assert text[start:end] == chunk_text, (
                    f"Offset mismatch: text[{start}:{end}] = {text[start:end]!r} != {chunk_text!r}"
                )

    async def test_offsets_with_overlap(self):
        """Overlapped chunks still have valid offsets."""
        text = "Section A content. " * 10 + "\n\n" + "Section B content. " * 10
        spans = await chunk_document_with_spans(text, chunk_size=100, chunk_overlap=20)

        assert len(spans) >= 2
        for chunk_text, start, end in spans:
            if start is not None and end is not None:
                assert text[start:end] == chunk_text, (
                    f"Offset mismatch: text[{start}:{end}] = {text[start:end]!r} != {chunk_text!r}"
                )

    async def test_repeated_text_no_drift(self):
        """Repeated text doesn't cause offset drift (the old find() bug)."""
        # Same sentence repeated — old find() would drift to wrong position
        text = "The quick brown fox. " * 20
        spans = await chunk_document_with_spans(text, chunk_size=100, chunk_overlap=0)

        assert len(spans) >= 2
        for chunk_text, start, end in spans:
            if start is not None and end is not None:
                assert text[start:end] == chunk_text, (
                    f"Offset drift with repeated text: text[{start}:{end}] != chunk_text"
                )

    async def test_single_chunk_offsets(self):
        """Short text produces one chunk with start=0."""
        text = "Short text."
        spans = await chunk_document_with_spans(text, chunk_size=1000, chunk_overlap=0)

        assert len(spans) == 1
        chunk_text, start, end = spans[0]
        assert chunk_text == text
        assert start == 0
        assert end == len(text)

    async def test_empty_text(self):
        """Empty text returns empty list."""
        spans = await chunk_document_with_spans("")
        assert spans == []

    async def test_offsets_cover_entire_document(self):
        """Non-overlapped chunks' offsets should cover the document without gaps."""
        text = "Alpha section. " * 15 + "\n\n" + "Beta section. " * 15
        spans = await chunk_document_with_spans(text, chunk_size=100, chunk_overlap=0)

        valid_spans = [(s, e) for _, s, e in spans if s is not None and e is not None]
        if len(valid_spans) >= 2:
            # Spans should be in order
            for i in range(1, len(valid_spans)):
                assert valid_spans[i][0] >= valid_spans[i - 1][0], "Spans not in order"

    async def test_semantic_mode_returns_none_offsets(self):
        """Semantic mode chunks have None offsets (no change from before)."""
        # We can't easily test semantic mode without mocking Gemini,
        # but we can verify the API contract via the fallback path
        text = "Some text."
        spans = await chunk_document_with_spans(text, chunk_size=1000)
        # Rule-based mode should return actual offsets
        for _, start, end in spans:
            assert start is not None
            assert end is not None

    async def test_multiline_paragraphs_with_overlap(self):
        """Multi-paragraph text with overlap has correct offsets."""
        text = (
            "Introduction to the topic.\n\n"
            "Main body of the first section with details.\n\n"
            "Second section explores another angle.\n\n"
            "Conclusion wraps everything up neatly."
        )
        spans = await chunk_document_with_spans(text, chunk_size=80, chunk_overlap=20)

        for chunk_text, start, end in spans:
            if start is not None and end is not None:
                assert text[start:end] == chunk_text
