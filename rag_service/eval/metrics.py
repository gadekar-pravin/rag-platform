"""Retrieval evaluation metrics: Precision@K, Recall@K, NDCG@K, MRR, Hit Rate.

Pure Python + numpy. No database or test framework dependencies.
Reusable beyond tests (future CLI, API endpoint).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k retrieved documents that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant documents found in top-k results."""
    if k <= 0 or not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    # Recall is based on unique relevant documents retrieved, not repeated IDs.
    hits = len(set(top_k) & relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[str], relevance_grades: dict[str, int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevance_grades: Map of document ID to relevance grade (0-3).
        k: Cutoff rank.

    Returns:
        NDCG score in [0, 1]. Returns 0.0 if no relevant documents exist.
    """
    if k <= 0 or not relevance_grades:
        return 0.0

    top_k = retrieved_ids[:k]

    # DCG: sum of (2^rel - 1) / log2(rank + 1)
    dcg = 0.0
    seen_docs: set[str] = set()
    for i, doc_id in enumerate(top_k):
        # A document should only contribute gain once; duplicates are treated as zero gain.
        if doc_id in seen_docs:
            rel = 0
        else:
            seen_docs.add(doc_id)
            rel = relevance_grades.get(doc_id, 0)
        dcg += (2**rel - 1) / math.log2(i + 2)  # +2 because rank is 1-indexed

    # IDCG: best possible DCG with ideal ordering
    ideal_rels = sorted(relevance_grades.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += (2**rel - 1) / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant result.

    Returns 0.0 if no relevant document is found.
    """
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def hit_rate(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1.0 if any relevant document appears in results, else 0.0."""
    for doc_id in retrieved_ids:
        if doc_id in relevant_ids:
            return 1.0
    return 0.0


@dataclass
class QueryEvalResult:
    """Per-query evaluation metrics."""

    query: str
    category: str
    precision_at_3: float
    precision_at_5: float
    recall_at_3: float
    recall_at_5: float
    ndcg_at_3: float
    ndcg_at_5: float
    mrr: float
    hit_rate: float
    retrieved_ids: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    """Aggregated evaluation metrics across all queries."""

    num_queries: int
    mean_precision_at_3: float
    mean_precision_at_5: float
    mean_recall_at_3: float
    mean_recall_at_5: float
    mean_ndcg_at_3: float
    mean_ndcg_at_5: float
    mean_mrr: float
    mean_hit_rate: float
    per_category: dict[str, dict[str, float | int]] = field(default_factory=dict)


def evaluate_query(
    query: str,
    category: str,
    retrieved_ids: list[str],
    relevance_grades: dict[str, int],
) -> QueryEvalResult:
    """Compute all metrics for a single query."""
    relevant_ids = {doc_id for doc_id, grade in relevance_grades.items() if grade > 0}
    return QueryEvalResult(
        query=query,
        category=category,
        precision_at_3=precision_at_k(retrieved_ids, relevant_ids, 3),
        precision_at_5=precision_at_k(retrieved_ids, relevant_ids, 5),
        recall_at_3=recall_at_k(retrieved_ids, relevant_ids, 3),
        recall_at_5=recall_at_k(retrieved_ids, relevant_ids, 5),
        ndcg_at_3=ndcg_at_k(retrieved_ids, relevance_grades, 3),
        ndcg_at_5=ndcg_at_k(retrieved_ids, relevance_grades, 5),
        mrr=mrr(retrieved_ids, relevant_ids),
        hit_rate=hit_rate(retrieved_ids, relevant_ids),
        retrieved_ids=retrieved_ids,
    )


def aggregate_results(results: list[QueryEvalResult]) -> EvalReport:
    """Compute mean metrics across all query results."""
    if not results:
        return EvalReport(
            num_queries=0,
            mean_precision_at_3=0.0,
            mean_precision_at_5=0.0,
            mean_recall_at_3=0.0,
            mean_recall_at_5=0.0,
            mean_ndcg_at_3=0.0,
            mean_ndcg_at_5=0.0,
            mean_mrr=0.0,
            mean_hit_rate=0.0,
        )

    p3 = np.mean([r.precision_at_3 for r in results]).item()
    p5 = np.mean([r.precision_at_5 for r in results]).item()
    r3 = np.mean([r.recall_at_3 for r in results]).item()
    r5 = np.mean([r.recall_at_5 for r in results]).item()
    n3 = np.mean([r.ndcg_at_3 for r in results]).item()
    n5 = np.mean([r.ndcg_at_5 for r in results]).item()
    m = np.mean([r.mrr for r in results]).item()
    h = np.mean([r.hit_rate for r in results]).item()

    # Per-category breakdown
    categories: dict[str, list[QueryEvalResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    per_category: dict[str, dict[str, float]] = {}
    for cat, cat_results in sorted(categories.items()):
        per_category[cat] = {
            "precision_at_3": np.mean([r.precision_at_3 for r in cat_results]).item(),
            "recall_at_3": np.mean([r.recall_at_3 for r in cat_results]).item(),
            "ndcg_at_3": np.mean([r.ndcg_at_3 for r in cat_results]).item(),
            "mrr": np.mean([r.mrr for r in cat_results]).item(),
            "hit_rate": np.mean([r.hit_rate for r in cat_results]).item(),
            "count": len(cat_results),
        }

    return EvalReport(
        num_queries=len(results),
        mean_precision_at_3=p3,
        mean_precision_at_5=p5,
        mean_recall_at_3=r3,
        mean_recall_at_5=r5,
        mean_ndcg_at_3=n3,
        mean_ndcg_at_5=n5,
        mean_mrr=m,
        mean_hit_rate=h,
        per_category=per_category,
    )


def format_report(report: EvalReport) -> str:
    """Format an EvalReport as a human-readable table."""
    lines = [
        f"Retrieval Evaluation Report ({report.num_queries} queries)",
        "=" * 50,
        "",
        f"{'Metric':<20} {'@3':>8} {'@5':>8}",
        f"{'-' * 20} {'-' * 8} {'-' * 8}",
        f"{'Precision':<20} {report.mean_precision_at_3:>8.4f} {report.mean_precision_at_5:>8.4f}",
        f"{'Recall':<20} {report.mean_recall_at_3:>8.4f} {report.mean_recall_at_5:>8.4f}",
        f"{'NDCG':<20} {report.mean_ndcg_at_3:>8.4f} {report.mean_ndcg_at_5:>8.4f}",
        "",
        f"{'Metric':<20} {'Value':>8}",
        f"{'-' * 20} {'-' * 8}",
        f"{'MRR':<20} {report.mean_mrr:>8.4f}",
        f"{'Hit Rate':<20} {report.mean_hit_rate:>8.4f}",
    ]

    if report.per_category:
        lines.append("")
        lines.append("Per-Category Breakdown")
        lines.append("-" * 50)
        lines.append(f"{'Category':<16} {'P@3':>6} {'R@3':>6} {'NDCG@3':>8} {'MRR':>6} {'HR':>6} {'N':>4}")
        for cat, metrics in report.per_category.items():
            lines.append(
                f"{cat:<16} {metrics['precision_at_3']:>6.3f} {metrics['recall_at_3']:>6.3f}"
                f" {metrics['ndcg_at_3']:>8.3f} {metrics['mrr']:>6.3f} {metrics['hit_rate']:>6.3f}"
                f" {metrics['count']:>4.0f}"
            )

    return "\n".join(lines)
