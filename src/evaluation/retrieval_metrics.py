"""Retrieval evaluation metrics (implementation-independent)."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def _as_set(ids: Iterable[str]) -> set[str]:
    return {i for i in ids if i}


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Binary recall@k: 1.0 if any gold id appears in top-k, else 0.0."""
    if k <= 0:
        return 0.0
    gold = _as_set(gold_ids)
    if not gold:
        return 0.0
    top_k = retrieved_ids[:k]
    return 1.0 if any(uid in gold for uid in top_k) else 0.0


def reciprocal_rank(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant item (0.0 if none found)."""
    gold = _as_set(gold_ids)
    if not gold:
        return 0.0
    for i, uid in enumerate(retrieved_ids, start=1):
        if uid in gold:
            return 1.0 / i
    return 0.0


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    """Discounted cumulative gain at k (rank positions are 1-indexed)."""
    total = 0.0
    for i, rel in enumerate(relevances[:k]):
        if rel > 0:
            total += rel / math.log2(i + 2)
    return total


def ndcg_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Normalized DCG@k with binary relevance."""
    if k <= 0:
        return 0.0
    gold = _as_set(gold_ids)
    if not gold:
        return 0.0

    relevances = [1.0 if uid in gold else 0.0 for uid in retrieved_ids[:k]]
    actual = dcg_at_k(relevances, k)

    num_relevant = min(len(gold), k)
    ideal = dcg_at_k([1.0] * num_relevant, k)
    if ideal == 0.0:
        return 0.0
    return actual / ideal


def compute_retrieval_metrics(
    retrieved_ids: Sequence[str],
    gold_ids: Sequence[str],
    *,
    recall_ks: Sequence[int] = (1, 5, 10),
    ndcg_ks: Sequence[int] = (5, 10),
) -> dict[str, float]:
    """Compute all retrieval metrics for a single query."""
    metrics: dict[str, float] = {}
    for k in recall_ks:
        metrics[f"Recall@{k}"] = recall_at_k(retrieved_ids, gold_ids, k)
    metrics["MRR"] = reciprocal_rank(retrieved_ids, gold_ids)
    for k in ndcg_ks:
        metrics[f"nDCG@{k}"] = ndcg_at_k(retrieved_ids, gold_ids, k)
    return metrics


def average_metrics(per_query_metrics: Sequence[dict[str, float]]) -> dict[str, float]:
    """Macro-average metrics across queries."""
    if not per_query_metrics:
        return {}
    keys = per_query_metrics[0].keys()
    return {
        key: sum(row.get(key, 0.0) for row in per_query_metrics) / len(per_query_metrics)
        for key in keys
    }
