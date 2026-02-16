"""Integration evaluation tests for retrieval quality.

Two test classes:
- TestRetrievalQualitySynthetic: uses synthetic embeddings (validates framework)
- TestRetrievalQualityReal: uses real Gemini embeddings (meaningful quality measurement)
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from rag_service.eval.metrics import (
    aggregate_results,
    evaluate_query,
    format_report,
)
from rag_service.stores.rag_search_store import RagSearchStore
from tests.eval.conftest import EVAL_DATASET_PATH, _make_synthetic_embedding

try:
    import asyncpg
except ImportError:
    pytest.skip("asyncpg not installed", allow_module_level=True)


TENANT_ID = "eval-test"
USER_ID = "eval-user@test"


async def _run_evaluation(
    pool: asyncpg.Pool,  # type: ignore[type-arg]
    dataset: dict[str, Any],
    title_to_id: dict[str, str],
    embed_query_fn: Any,
) -> Any:
    """Run all queries through hybrid search and compute metrics."""
    search_store = RagSearchStore()
    results = []

    for q in dataset["queries"]:
        query_text = q["query"]
        query_vec = await embed_query_fn(query_text)

        async with pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", TENANT_ID)
            await conn.execute("SELECT set_config('app.user_id', $1, true)", USER_ID)
            search_result = await search_store.search_hybrid(conn, query_text, query_vec, doc_limit=5)

        retrieved_ids = [r["document_id"] for r in search_result["results"]]

        # Map title-based relevance grades to actual doc IDs
        relevance_grades: dict[str, int] = {}
        for title, grade in q["relevance"].items():
            if title in title_to_id:
                relevance_grades[title_to_id[title]] = grade

        result = evaluate_query(
            query=query_text,
            category=q["category"],
            retrieved_ids=retrieved_ids,
            relevance_grades=relevance_grades,
        )
        results.append(result)

    return aggregate_results(results)


class TestRetrievalQualitySynthetic:
    """Tests using synthetic embeddings — validates the evaluation framework."""

    async def test_eval_framework_runs(
        self,
        eval_db_pool: asyncpg.Pool,  # type: ignore[type-arg]
        eval_dataset: dict[str, Any],
        eval_seed_documents: dict[str, str],
    ) -> None:
        """Smoke test: loads dataset, runs queries, computes metrics, prints report."""
        from rag_service.config import RAG_EMBEDDING_DIM

        async def synthetic_embed(query: str) -> list[float]:
            return _make_synthetic_embedding(RAG_EMBEDDING_DIM, seed=hash(query))

        report = await _run_evaluation(eval_db_pool, eval_dataset, eval_seed_documents, synthetic_embed)
        output = format_report(report)
        print(f"\n{output}")

        # No quality thresholds — synthetic embeddings are random.
        # Just verify the framework ran without errors.
        assert report.num_queries == len(eval_dataset["queries"])
        assert report.num_queries == 15

    def test_dataset_integrity(self, eval_dataset: dict[str, Any]) -> None:
        """Validate the JSON structure of the evaluation dataset."""
        assert "documents" in eval_dataset
        assert "queries" in eval_dataset
        assert len(eval_dataset["documents"]) == 3
        assert len(eval_dataset["queries"]) == 15

        doc_titles = {d["title"] for d in eval_dataset["documents"]}
        categories = set()

        for q in eval_dataset["queries"]:
            assert "id" in q
            assert "query" in q
            assert "category" in q
            assert "relevance" in q
            categories.add(q["category"])

            # All relevance keys must reference actual documents
            for title in q["relevance"]:
                assert title in doc_titles, f"Query {q['id']} references unknown doc: {title}"

            # Relevance grades must be 0-3
            for grade in q["relevance"].values():
                assert 0 <= grade <= 3, f"Query {q['id']} has invalid grade: {grade}"

        assert categories == {"keyword_match", "semantic", "multi_topic", "single_doc", "multi_doc", "edge_case"}

    def test_all_referenced_docs_exist(
        self,
        eval_dataset: dict[str, Any],
        eval_seed_documents: dict[str, str],
    ) -> None:
        """Validate all query relevance references point to seeded documents."""
        for q in eval_dataset["queries"]:
            for title in q["relevance"]:
                assert title in eval_seed_documents, f"Query {q['id']} references '{title}' which was not seeded"

    def test_dataset_file_is_valid_json(self) -> None:
        """Ensure the dataset file parses cleanly."""
        raw = EVAL_DATASET_PATH.read_text()
        data = json.loads(raw)
        assert isinstance(data, dict)


class TestRetrievalQualityReal:
    """Tests using real Gemini embeddings — meaningful quality measurement."""

    async def test_retrieval_quality_thresholds(
        self,
        eval_db_pool: asyncpg.Pool,  # type: ignore[type-arg]
        eval_dataset: dict[str, Any],
        eval_seed_documents_real: dict[str, str],
    ) -> None:
        """Run full evaluation with real embeddings and assert quality thresholds."""
        from rag_service.embedding import embed_query

        report = await _run_evaluation(eval_db_pool, eval_dataset, eval_seed_documents_real, embed_query)
        output = format_report(report)
        print(f"\n{output}")

        assert report.num_queries == 15

        # Conservative thresholds for 3 small documents
        assert report.mean_mrr >= 0.5, f"MRR {report.mean_mrr:.3f} below 0.5 threshold"
        assert report.mean_hit_rate >= 0.7, f"Hit Rate {report.mean_hit_rate:.3f} below 0.7 threshold"
        assert report.mean_precision_at_3 >= 0.4, f"Precision@3 {report.mean_precision_at_3:.3f} below 0.4 threshold"
        assert report.mean_ndcg_at_3 >= 0.4, f"NDCG@3 {report.mean_ndcg_at_3:.3f} below 0.4 threshold"
