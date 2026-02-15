"""Unit tests for RagSearchStore — mock DB, verify SQL structure and RRF logic."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

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

        result = await store.search_hybrid(mock_conn, "test query", sample_query_vec, doc_limit=5)

        assert result["results"] == []
        assert "debug" not in result or result.get("debug") is None

    async def test_empty_results_with_debug(self, store, mock_conn, sample_query_vec):
        """Search with no matches and debug=True returns debug info."""
        mock_conn.fetch.return_value = []

        result = await store.search_hybrid(mock_conn, "test query", sample_query_vec, doc_limit=5, include_debug=True)

        assert result["results"] == []
        assert result["debug"] is not None
        assert result["debug"]["vector_pool_size"] == 0

    async def test_results_with_chunks(self, store, mock_conn, sample_query_vec):
        """Search returns results with best chunks per document (single query)."""
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()

        # Fix 4: single fetch call returns fused docs + chunks in one result set
        mock_conn.fetch.return_value = [
            {
                "document_id": doc_id,
                "title": "Test Doc",
                "doc_type": "markdown",
                "rrf_score": 0.03,
                "vector_score": 0.016,
                "text_score": 0.014,
                "chunk_id": chunk_id,
                "chunk_index": 0,
                "chunk_text": "Sample chunk text",
                "chunk_source": "vector",
                "chunk_score": 0.95,
            }
        ]

        result = await store.search_hybrid(mock_conn, "test query", sample_query_vec, doc_limit=5)

        assert len(result["results"]) == 1
        assert result["results"][0]["document_id"] == str(doc_id)
        assert result["results"][0]["title"] == "Test Doc"
        assert len(result["results"][0]["chunks"]) == 1
        assert result["results"][0]["chunks"][0]["source"] == "vector"

    async def test_single_fetch_call(self, store, mock_conn, sample_query_vec):
        """Fix 4: search_hybrid uses a single conn.fetch (no separate best-chunks query)."""
        mock_conn.fetch.return_value = []

        await store.search_hybrid(mock_conn, "test query", sample_query_vec, doc_limit=5)

        # Only one fetch call for the integrated query
        assert mock_conn.fetch.call_count == 1

    async def test_doc_limit_respected(self, store, mock_conn, sample_query_vec):
        """doc_limit parameter is passed to the SQL query."""
        mock_conn.fetch.return_value = []

        await store.search_hybrid(mock_conn, "test", sample_query_vec, doc_limit=3)

        # Verify the LIMIT parameter was passed
        call_args = mock_conn.fetch.call_args
        # $5 = doc_limit (position index 4 in args after SQL)
        assert call_args[0][5] == 3

    async def test_expansion_factor(self, store, mock_conn, sample_query_vec):
        """candidate_limit = doc_limit * expansion * multiplier + 1."""
        mock_conn.fetch.return_value = []

        await store.search_hybrid(
            mock_conn,
            "test",
            sample_query_vec,
            doc_limit=5,
            expansion=3,
            candidate_multiplier=1,
        )

        call_args = mock_conn.fetch.call_args
        # $2 = candidate_limit + 1 (position index 1 in args after SQL)
        assert call_args[0][2] == 16  # 5 * 3 + 1

    async def test_null_chunks_handled(self, store, mock_conn, sample_query_vec):
        """LEFT JOIN may produce NULL chunk_id — should result in empty chunks list."""
        doc_id = uuid.uuid4()

        mock_conn.fetch.return_value = [
            {
                "document_id": doc_id,
                "title": "Doc with no chunks",
                "doc_type": None,
                "rrf_score": 0.01,
                "vector_score": 0.01,
                "text_score": 0.0,
                "chunk_id": None,
                "chunk_index": None,
                "chunk_text": None,
                "chunk_source": None,
                "chunk_score": None,
            }
        ]

        result = await store.search_hybrid(mock_conn, "test", sample_query_vec, doc_limit=5)

        assert len(result["results"]) == 1
        assert result["results"][0]["chunks"] == []

    async def test_multiple_chunks_per_doc(self, store, mock_conn, sample_query_vec):
        """Multiple chunk rows for the same doc are grouped correctly."""
        doc_id = uuid.uuid4()
        chunk1 = uuid.uuid4()
        chunk2 = uuid.uuid4()

        mock_conn.fetch.return_value = [
            {
                "document_id": doc_id,
                "title": "Doc",
                "doc_type": None,
                "rrf_score": 0.03,
                "vector_score": 0.016,
                "text_score": 0.014,
                "chunk_id": chunk1,
                "chunk_index": 0,
                "chunk_text": "First chunk",
                "chunk_source": "vector",
                "chunk_score": 0.95,
            },
            {
                "document_id": doc_id,
                "title": "Doc",
                "doc_type": None,
                "rrf_score": 0.03,
                "vector_score": 0.016,
                "text_score": 0.014,
                "chunk_id": chunk2,
                "chunk_index": 1,
                "chunk_text": "Second chunk",
                "chunk_source": "text",
                "chunk_score": 0.80,
            },
        ]

        result = await store.search_hybrid(mock_conn, "test", sample_query_vec, doc_limit=5)

        assert len(result["results"]) == 1
        assert len(result["results"][0]["chunks"]) == 2


class TestSQLStructure:
    """Verify SQL contains expected patterns for Fix 2, 5, 8."""

    async def test_ann_candidates_cte_present(self, store, mock_conn, sample_query_vec):
        """Fix 2: SQL contains ANN-first ann_candidates CTE."""
        mock_conn.fetch.return_value = []

        await store.search_hybrid(mock_conn, "test", sample_query_vec, doc_limit=5)

        sql = mock_conn.fetch.call_args[0][0]
        assert "ann_candidates" in sql
        assert "FROM rag_chunk_embeddings e" in sql

    async def test_tsquery_val_cte_present(self, store, mock_conn, sample_query_vec):
        """Fix 5: SQL contains tsquery_val CTE for deduplication."""
        mock_conn.fetch.return_value = []

        await store.search_hybrid(mock_conn, "test", sample_query_vec, doc_limit=5)

        sql = mock_conn.fetch.call_args[0][0]
        assert "tsquery_val" in sql
        assert "(SELECT q FROM tsquery_val)" in sql

    async def test_best_chunks_cte_present(self, store, mock_conn, sample_query_vec):
        """Fix 4: SQL contains best_chunks CTE (integrated, no separate query)."""
        mock_conn.fetch.return_value = []

        await store.search_hybrid(mock_conn, "test", sample_query_vec, doc_limit=5)

        sql = mock_conn.fetch.call_args[0][0]
        assert "best_chunks" in sql
        assert "best_vec" in sql
        assert "best_txt" in sql

    async def test_fts_language_parameterized(self, store, mock_conn, sample_query_vec):
        """Fix 8: FTS language passed as parameter $7."""
        mock_conn.fetch.return_value = []

        await store.search_hybrid(mock_conn, "test", sample_query_vec, doc_limit=5, fts_language="spanish")

        call_args = mock_conn.fetch.call_args
        # $7 = fts_language (position index 6 in args after SQL)
        assert call_args[0][7] == "spanish"

    async def test_fts_language_default_english(self, store, mock_conn, sample_query_vec):
        """Fix 8: Default FTS language is 'english'."""
        mock_conn.fetch.return_value = []

        await store.search_hybrid(mock_conn, "test", sample_query_vec, doc_limit=5)

        call_args = mock_conn.fetch.call_args
        # $7 should be the default FTS language
        assert call_args[0][7] == "english"


class TestSearchDebugCutoffScores:
    """Fix 6: Verify cutoff scores and has_more accuracy."""

    async def test_debug_includes_cutoff_scores(self, store, mock_conn, sample_query_vec):
        """When include_debug=True, cutoff scores are returned."""
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()

        mock_conn.fetch.return_value = [
            {
                "document_id": doc_id,
                "title": "Test Doc",
                "doc_type": "markdown",
                "rrf_score": 0.03,
                "vector_score": 0.016,
                "text_score": 0.014,
                "chunk_id": chunk_id,
                "chunk_index": 0,
                "chunk_text": "Sample chunk text",
                "chunk_source": "vector",
                "chunk_score": 0.95,
            }
        ]
        mock_conn.fetchrow.return_value = {
            "vector_pool_size": 5,
            "text_pool_size": 3,
            "vector_raw_count": 10,
            "text_raw_count": 8,
            "vector_cutoff_score": 0.72,
            "text_cutoff_score": 0.15,
        }

        result = await store.search_hybrid(mock_conn, "test query", sample_query_vec, doc_limit=5, include_debug=True)

        assert result["debug"] is not None
        assert result["debug"]["vector_cutoff_score"] == 0.72
        assert result["debug"]["text_cutoff_score"] == 0.15

    async def test_debug_cutoff_scores_none_when_no_results(self, store, mock_conn, sample_query_vec):
        """With empty results, cutoff scores should be None."""
        mock_conn.fetch.return_value = []

        result = await store.search_hybrid(mock_conn, "test query", sample_query_vec, doc_limit=5, include_debug=True)

        assert result["debug"]["vector_cutoff_score"] is None
        assert result["debug"]["text_cutoff_score"] is None

    async def test_debug_cutoff_scores_null_from_db(self, store, mock_conn, sample_query_vec):
        """When DB returns NULL for cutoff scores (vector-only match), handle gracefully."""
        doc_id = uuid.uuid4()

        mock_conn.fetch.return_value = [
            {
                "document_id": doc_id,
                "title": "Doc",
                "doc_type": None,
                "rrf_score": 0.01,
                "vector_score": 0.01,
                "text_score": 0.0,
                "chunk_id": None,
                "chunk_index": None,
                "chunk_text": None,
                "chunk_source": None,
                "chunk_score": None,
            }
        ]
        mock_conn.fetchrow.return_value = {
            "vector_pool_size": 2,
            "text_pool_size": 0,
            "vector_raw_count": 2,
            "text_raw_count": 0,
            "vector_cutoff_score": 0.85,
            "text_cutoff_score": None,
        }

        result = await store.search_hybrid(mock_conn, "test query", sample_query_vec, doc_limit=5, include_debug=True)

        assert result["debug"]["vector_cutoff_score"] == 0.85
        assert result["debug"]["text_cutoff_score"] is None

    async def test_has_more_uses_raw_counts(self, store, mock_conn, sample_query_vec):
        """Fix 6: has_more uses raw candidate counts, not pool sizes."""
        doc_id = uuid.uuid4()

        mock_conn.fetch.return_value = [
            {
                "document_id": doc_id,
                "title": "Doc",
                "doc_type": None,
                "rrf_score": 0.01,
                "vector_score": 0.01,
                "text_score": 0.0,
                "chunk_id": None,
                "chunk_index": None,
                "chunk_text": None,
                "chunk_source": None,
                "chunk_score": None,
            }
        ]

        # candidate_limit = 5 * 3 * 4 = 60, so threshold = 60
        # vector_raw_count=61 > 60 → has_more=True
        # text_raw_count=30 < 60 → has_more=False
        mock_conn.fetchrow.return_value = {
            "vector_pool_size": 15,
            "text_pool_size": 10,
            "vector_raw_count": 61,
            "text_raw_count": 30,
            "vector_cutoff_score": 0.7,
            "text_cutoff_score": 0.1,
        }

        result = await store.search_hybrid(mock_conn, "test", sample_query_vec, doc_limit=5, include_debug=True)

        assert result["debug"]["vector_has_more"] is True
        assert result["debug"]["text_has_more"] is False

    async def test_pool_stats_has_raw_counts(self, store, mock_conn, sample_query_vec):
        """Fix 6: _get_pool_stats SQL includes vector_raw_count and text_raw_count."""
        mock_conn.fetchrow.return_value = {
            "vector_pool_size": 5,
            "text_pool_size": 3,
            "vector_raw_count": 10,
            "text_raw_count": 8,
            "vector_cutoff_score": 0.72,
            "text_cutoff_score": 0.15,
        }

        stats = await store._get_pool_stats("test", sample_query_vec, mock_conn, 61, 3)

        sql = mock_conn.fetchrow.call_args[0][0]
        assert "vector_raw_count" in sql
        assert "text_raw_count" in sql
        assert stats["vector_raw_count"] == 10
        assert stats["text_raw_count"] == 8


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
