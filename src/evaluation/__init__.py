"""Retrieval evaluation for RAG benchmark experiments."""

from .evaluate_dense import EvaluationResult, run_retrieval_evaluation
from .report import EvaluationSummary, write_report
from .retrieval_metrics import (
    average_metrics,
    compute_retrieval_metrics,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "EvaluationResult",
    "EvaluationSummary",
    "average_metrics",
    "compute_retrieval_metrics",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "run_retrieval_evaluation",
    "write_report",
]
