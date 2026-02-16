"""Unit tests for retrieval evaluation metrics.

Pure unit tests — no database or API key needed.
"""

from __future__ import annotations

import math

import pytest

from rag_service.eval.metrics import (
    EvalReport,
    QueryEvalResult,
    aggregate_results,
    evaluate_query,
    format_report,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_perfect_precision(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, 3) == 1.0

    def test_zero_precision(self) -> None:
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, 3) == 0.0

    def test_partial_precision(self) -> None:
        retrieved = ["a", "x", "b", "y"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, 4) == 0.5

    def test_k_larger_than_results(self) -> None:
        retrieved = ["a", "b"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, 5) == 1.0

    def test_k_zero(self) -> None:
        assert precision_at_k(["a"], {"a"}, 0) == 0.0

    def test_empty_results(self) -> None:
        assert precision_at_k([], {"a"}, 3) == 0.0


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert recall_at_k(retrieved, relevant, 3) == 1.0

    def test_zero_recall(self) -> None:
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 3) == 0.0

    def test_partial_recall(self) -> None:
        retrieved = ["a", "x", "y"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 3) == 0.5

    def test_empty_relevant_set(self) -> None:
        assert recall_at_k(["a", "b"], set(), 3) == 0.0

    def test_k_zero(self) -> None:
        assert recall_at_k(["a"], {"a"}, 0) == 0.0

    def test_duplicate_retrieved_docs_count_once(self) -> None:
        retrieved = ["a", "a", "x"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 3) == 0.5


class TestNDCGAtK:
    def test_ideal_ordering(self) -> None:
        # Docs ordered by descending relevance = perfect NDCG
        retrieved = ["a", "b", "c"]
        grades = {"a": 3, "b": 2, "c": 1}
        assert ndcg_at_k(retrieved, grades, 3) == 1.0

    def test_reversed_ordering(self) -> None:
        # Worst ordering: least relevant first
        retrieved = ["c", "b", "a"]
        grades = {"a": 3, "b": 2, "c": 1}
        score = ndcg_at_k(retrieved, grades, 3)
        assert 0.0 < score < 1.0

    def test_empty_grades(self) -> None:
        assert ndcg_at_k(["a", "b"], {}, 3) == 0.0

    def test_all_zero_grades(self) -> None:
        assert ndcg_at_k(["a", "b"], {"a": 0, "b": 0}, 3) == 0.0

    def test_single_relevant_at_top(self) -> None:
        retrieved = ["a", "b", "c"]
        grades = {"a": 3}
        assert ndcg_at_k(retrieved, grades, 3) == 1.0

    def test_k_zero(self) -> None:
        assert ndcg_at_k(["a"], {"a": 3}, 0) == 0.0

    def test_duplicate_retrieved_doc_does_not_inflate_score(self) -> None:
        # Duplicate hits should not create extra gain.
        assert ndcg_at_k(["a", "a"], {"a": 3}, 2) == 1.0

    def test_ndcg_math(self) -> None:
        # Manual DCG calculation
        retrieved = ["b", "a"]
        grades = {"a": 2, "b": 1}
        # DCG = (2^1 - 1)/log2(2) + (2^2 - 1)/log2(3) = 1/1 + 3/1.585 = 1 + 1.893 = 2.893
        # IDCG = (2^2 - 1)/log2(2) + (2^1 - 1)/log2(3) = 3/1 + 1/1.585 = 3 + 0.631 = 3.631
        expected_dcg = (2**1 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
        expected_idcg = (2**2 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
        expected = expected_dcg / expected_idcg
        assert abs(ndcg_at_k(retrieved, grades, 2) - expected) < 1e-10


class TestMRR:
    def test_first_position(self) -> None:
        assert mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_second_position(self) -> None:
        assert mrr(["x", "a", "c"], {"a"}) == 0.5

    def test_third_position(self) -> None:
        assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1.0 / 3)

    def test_no_relevant(self) -> None:
        assert mrr(["x", "y", "z"], {"a"}) == 0.0

    def test_empty_results(self) -> None:
        assert mrr([], {"a"}) == 0.0

    def test_multiple_relevant_returns_first(self) -> None:
        assert mrr(["x", "a", "b"], {"a", "b"}) == 0.5


class TestHitRate:
    def test_hit(self) -> None:
        assert hit_rate(["x", "a"], {"a"}) == 1.0

    def test_miss(self) -> None:
        assert hit_rate(["x", "y"], {"a"}) == 0.0

    def test_empty_results(self) -> None:
        assert hit_rate([], {"a"}) == 0.0


class TestEvaluateQuery:
    def test_computes_all_metrics(self) -> None:
        result = evaluate_query(
            query="test query",
            category="keyword_match",
            retrieved_ids=["a", "b", "c", "d", "e"],
            relevance_grades={"a": 3, "b": 2, "c": 0, "d": 0, "e": 0},
        )
        assert isinstance(result, QueryEvalResult)
        assert result.query == "test query"
        assert result.category == "keyword_match"
        assert result.precision_at_3 == pytest.approx(2.0 / 3)
        assert result.recall_at_3 == 1.0  # both relevant docs in top 3
        assert result.mrr == 1.0
        assert result.hit_rate == 1.0
        assert result.retrieved_ids == ["a", "b", "c", "d", "e"]

    def test_no_relevant_docs(self) -> None:
        result = evaluate_query(
            query="nothing",
            category="edge_case",
            retrieved_ids=["x", "y"],
            relevance_grades={"x": 0, "y": 0},
        )
        assert result.precision_at_3 == 0.0
        assert result.recall_at_3 == 0.0
        assert result.mrr == 0.0
        assert result.hit_rate == 0.0


class TestAggregateResults:
    def test_aggregates_means(self) -> None:
        results = [
            QueryEvalResult(
                query="q1",
                category="a",
                precision_at_3=1.0,
                precision_at_5=0.8,
                recall_at_3=1.0,
                recall_at_5=1.0,
                ndcg_at_3=1.0,
                ndcg_at_5=0.9,
                mrr=1.0,
                hit_rate=1.0,
            ),
            QueryEvalResult(
                query="q2",
                category="b",
                precision_at_3=0.0,
                precision_at_5=0.2,
                recall_at_3=0.0,
                recall_at_5=0.5,
                ndcg_at_3=0.0,
                ndcg_at_5=0.1,
                mrr=0.0,
                hit_rate=0.0,
            ),
        ]
        report = aggregate_results(results)
        assert isinstance(report, EvalReport)
        assert report.num_queries == 2
        assert report.mean_precision_at_3 == pytest.approx(0.5)
        assert report.mean_mrr == pytest.approx(0.5)
        assert report.mean_hit_rate == pytest.approx(0.5)
        assert "a" in report.per_category
        assert "b" in report.per_category

    def test_empty_results(self) -> None:
        report = aggregate_results([])
        assert report.num_queries == 0
        assert report.mean_mrr == 0.0


class TestFormatReport:
    def test_produces_valid_output(self) -> None:
        report = EvalReport(
            num_queries=10,
            mean_precision_at_3=0.7,
            mean_precision_at_5=0.6,
            mean_recall_at_3=0.8,
            mean_recall_at_5=0.9,
            mean_ndcg_at_3=0.75,
            mean_ndcg_at_5=0.70,
            mean_mrr=0.85,
            mean_hit_rate=0.95,
            per_category={
                "keyword_match": {
                    "precision_at_3": 0.9,
                    "recall_at_3": 1.0,
                    "ndcg_at_3": 0.95,
                    "mrr": 1.0,
                    "hit_rate": 1.0,
                    "count": 3,
                }
            },
        )
        output = format_report(report)
        assert "10 queries" in output
        assert "Precision" in output
        assert "NDCG" in output
        assert "MRR" in output
        assert "keyword_match" in output

    def test_empty_report(self) -> None:
        report = aggregate_results([])
        output = format_report(report)
        assert "0 queries" in output
