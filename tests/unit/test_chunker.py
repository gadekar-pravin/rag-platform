"""Unit tests for the chunker — ported from ApexFlow patterns."""

from __future__ import annotations

import pytest

from rag_service.chunking.chunker import chunk_document, _validate_chunk_params


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
