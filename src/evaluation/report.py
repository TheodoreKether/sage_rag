"""Markdown report generation for retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvaluationSummary:
    retriever_name: str
    model_name: str
    index_dir: str
    qa_file: str
    top_k: int
    total_qa_pairs: int
    evaluated_pairs: int
    skipped_pairs: int
    failed_pairs: int
    elapsed_seconds: float
    average_metrics: dict[str, float] = field(default_factory=dict)
    dataset_stats: dict[str, Any] = field(default_factory=dict)

    def metric(self, name: str) -> float:
        return float(self.average_metrics.get(name, 0.0))


def render_dense_report(summary: EvaluationSummary) -> str:
    lines = [
        "# Dense Retrieval Evaluation Report",
        "",
        "## Dataset Statistics",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| QA file | `{summary.qa_file}` |",
        f"| Total QA pairs in file | {summary.total_qa_pairs} |",
        f"| QA pairs evaluated | {summary.evaluated_pairs} |",
        f"| Skipped (invalid / empty) | {summary.skipped_pairs} |",
        f"| Failed (retrieval error) | {summary.failed_pairs} |",
    ]

    if summary.dataset_stats:
        for key, value in sorted(summary.dataset_stats.items()):
            lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Retriever Configuration",
            "",
            "| Setting | Value |",
            "|---------|-------|",
            f"| Retriever | `{summary.retriever_name}` |",
            f"| Model | `{summary.model_name}` |",
            f"| Index directory | `{summary.index_dir}` |",
            f"| top_k | {summary.top_k} |",
            "",
            "## Average Metrics",
            "",
            "| Metric | Value |",
            "|--------|------:|",
            f"| Recall@1 | {summary.metric('Recall@1'):.4f} |",
            f"| Recall@5 | {summary.metric('Recall@5'):.4f} |",
            f"| Recall@10 | {summary.metric('Recall@10'):.4f} |",
            f"| MRR | {summary.metric('MRR'):.4f} |",
            f"| nDCG@5 | {summary.metric('nDCG@5'):.4f} |",
            f"| nDCG@10 | {summary.metric('nDCG@10'):.4f} |",
            "",
            "## Runtime",
            "",
            f"- Total evaluation time: **{summary.elapsed_seconds:.2f} s**",
            "",
            "## Notes",
            "",
            "- Ground truth: `supporting_evidence[].unit_id` from the QA dataset.",
            "- Per-question results (including Top-K scores) are saved in `results/retrieval/dense/retrieval_results.jsonl`.",
            "- Metrics are retriever-agnostic; swap the retriever to evaluate BM25 / Hybrid / SAGE-RAG.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(summary: EvaluationSummary, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_dense_report(summary) + "\n", encoding="utf-8")
