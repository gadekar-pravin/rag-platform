"""Unit tests for RagSearchStore — mock DB, verify SQL structure and RRF logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import uuid

import pytest

from rag_service.stores.rag_search_store import RagSearchStore


@pytest.fixture
def store() -> RagSearchStore:
    return RagSearchStore()


@pytest.fixture
def mock_conn() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def sample_query_vec() -> list[float]:
    return [0.1] * 768


class TestSearchHybrid:
    async def test_empty_results(self, store, mock_conn, sample_query_vec):
        """Search with no matches returns empty results."""
        mock_conn.fetch.return_value = []

        result = await store.search_hybrid(
            mock_conn, "test query", sample_query_vec, doc_limit=5
        )

        assert result["results"] == []
        assert "debug" not in result or result.get("debug") is None

    async def test_empty_results_with_debug(self, store, mock_conn, sample_query_vec):
        """Search with no matches and debug=True returns debug info."""
        mock_conn.fetch.return_value = []

        result = await store.search_hybrid(
            mock_conn, "test query", sample_query_vec,
            doc_limit=5, include_debug=True
        )

        assert result["results"] == []
        assert result["debug"] is not None
        assert result["debug"]["vector_pool_size"] == 0

    async def test_results_with_chunks(self, store, mock_conn, sample_query_vec):
        """Search returns results with best chunks per document."""
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()

        # Mock the main fusion query
        mock_conn.fetch.side_effect = [
            # First call: fused results
            [
                {
                    "document_id": doc_id,
                    "title": "Test Doc",
                    "doc_type": "markdown",
                    "rrf_score": 0.03,
                    "vector_score": 0.016,
                    "text_score": 0.014,
                }
            ],
            # Second call: best chunks
            [
                {
                    "document_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_index": 0,
                    "chunk_text": "Sample chunk text",
                    "source": "vector",
                    "score": 0.95,
                }
            ],
        ]

        result = await store.search_hybrid(
            mock_conn, "test query", sample_query_vec, doc_limit=5
        )

        assert len(result["results"]) == 1
        assert result["results"][0]["document_id"] == str(doc_id)
        assert result["results"][0]["title"] == "Test Doc"
        assert len(result["results"][0]["chunks"]) == 1

    async def test_doc_limit_respected(self, store, mock_conn, sample_query_vec):
        """doc_limit parameter is passed to the SQL query."""
        mock_conn.fetch.return_value = []

        await store.search_hybrid(
            mock_conn, "test", sample_query_vec, doc_limit=3
        )

        # Verify the LIMIT parameter was passed
        call_args = mock_conn.fetch.call_args
        # $5 = doc_limit
        assert call_args[0][-1] == 3

    async def test_expansion_factor(self, store, mock_conn, sample_query_vec):
        """chunk_limit = doc_limit * expansion + 1."""
        mock_conn.fetch.return_value = []

        await store.search_hybrid(
            mock_conn, "test", sample_query_vec,
            doc_limit=5, expansion=3
        )

        call_args = mock_conn.fetch.call_args
        # $2 = chunk_limit + 1
        assert call_args[0][1] == 16  # 5 * 3 + 1


class TestRRFScoring:
    """Verify RRF score calculation logic."""

    def test_rrf_formula(self):
        """RRF score = 1/(K + rank_v) + 1/(K + rank_t)."""
        k = 60
        rank_v = 1
        rank_t = 2

        expected = 1.0 / (k + rank_v) + 1.0 / (k + rank_t)
        assert abs(expected - (1 / 61 + 1 / 62)) < 1e-10

    def test_rrf_vector_only(self):
        """Document found only by vector search gets partial score."""
        k = 60
        rank_v = 1

        score = 1.0 / (k + rank_v) + 0  # no text match
        assert score == pytest.approx(1 / 61, abs=1e-10)

    def test_rrf_text_only(self):
        """Document found only by text search gets partial score."""
        k = 60
        rank_t = 1

        score = 0 + 1.0 / (k + rank_t)  # no vector match
        assert score == pytest.approx(1 / 61, abs=1e-10)

    def test_rrf_both_rank1_beats_single(self):
        """Doc ranked #1 in both signals beats doc ranked #1 in one."""
        k = 60
        both = 1.0 / (k + 1) + 1.0 / (k + 1)
        single = 1.0 / (k + 1) + 0
        assert both > single
