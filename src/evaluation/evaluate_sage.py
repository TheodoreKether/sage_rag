"""Evaluate SAGE-RAG (BM25 + Graph Expansion + Structure Ranking) on Clean Benchmark.

Reuses shared metrics from ``src.evaluation.retrieval_metrics`` (same as
BM25 / Dense / Hybrid). Does **not** modify baseline retrievers or metric code.

Usage:
  python src/evaluation/evaluate_sage.py
  python src/evaluation/evaluate_sage.py --sample 20   # smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate_dense import (  # noqa: E402
    QASample,
    load_qa_samples,
    sample_qa_pairs,
)
from src.evaluation.retrieval_metrics import (  # noqa: E402
    average_metrics,
    compute_retrieval_metrics,
    recall_at_k,
)
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.sage_rag.expansion.graph_expander import GraphExpander  # noqa: E402
from src.sage_rag.graph.graph_store import GraphStore  # noqa: E402
from src.sage_rag.ranking.structure_ranker import StructureRanker  # noqa: E402
from src.sage_rag.retrieval.sage_retriever import SageRetriever  # noqa: E402

logger = logging.getLogger(__name__)

QA_CLEAN = ROOT / "data" / "qa_dataset" / "qa_pairs_clean.jsonl"
OUT_DIR = ROOT / "results" / "retrieval" / "sage"
OUT_DIR_INITIAL10 = ROOT / "results" / "retrieval" / "sage_initial10"
SAGE_INITIAL5_METRICS = ROOT / "results" / "retrieval" / "sage" / "metrics.json"
CLEAN_BM25_RESULTS = ROOT / "results" / "retrieval" / "clean_benchmark" / "bm25_results.jsonl"
CLEAN_BM25_METRICS = ROOT / "results" / "retrieval" / "clean_benchmark" / "bm25_metrics.json"
CLEAN_DENSE_METRICS = ROOT / "results" / "retrieval" / "clean_benchmark" / "dense_metrics.json"
CLEAN_HYBRID_METRICS = ROOT / "results" / "retrieval" / "clean_benchmark" / "hybrid_metrics.json"

# Clean Benchmark numbers (also mirrored from metrics JSON when available).
FALLBACK_BASELINES: dict[str, dict[str, float]] = {
    "BM25": {
        "Recall@1": 0.5435,
        "Recall@5": 0.7652,
        "Recall@10": 0.8261,
        "MRR": 0.6439,
    },
    "Hybrid (RRF)": {
        "Recall@1": 0.4783,
        "Recall@5": 0.7152,
        "Recall@10": 0.7826,
        "MRR": 0.5777,
    },
    "Dense (BGE-M3)": {
        "Recall@1": 0.3913,
        "Recall@5": 0.5674,
        "Recall@10": 0.6543,
        "MRR": 0.4698,
    },
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate SAGE-RAG on Clean Benchmark")
    p.add_argument("--qa", type=Path, default=QA_CLEAN)
    p.add_argument("--output-dir", type=Path, default=OUT_DIR_INITIAL10)
    p.add_argument("--bm25-index", type=Path, default=ROOT / "data" / "bm25_index")
    p.add_argument("--graph-dir", type=Path, default=ROOT / "data" / "sage_graph")
    p.add_argument(
        "--initial-k",
        type=int,
        default=10,
        help="BM25 seed count before graph expansion (fair setting: 10)",
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--pool-size",
        type=int,
        default=80,
        help="Expansion pool size before ranking (keep headroom above initial_k)",
    )
    p.add_argument(
        "--compare-initial5-metrics",
        type=Path,
        default=SAGE_INITIAL5_METRICS,
        help="Prior SAGE initial_k=5 metrics.json for comparison report",
    )
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--sample", type=int, default=None, help="Optional subsample for smoke tests")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _unit_payload(unit: Any) -> dict[str, Any]:
    meta = unit.metadata or {}
    return {
        "rank": unit.rank,
        "unit_id": unit.unit_id,
        "score": float(unit.score) if unit.score is not None else None,
        "final_score": meta.get("final_score", unit.score),
        "original_score": meta.get("original_score"),
        "candidate_source": meta.get("candidate_source"),
        "expansion_relation": meta.get("expansion_relation"),
        "retrieval_score_normalized": meta.get("retrieval_score_normalized"),
        "graph_score": meta.get("graph_score"),
        "via_node_id": meta.get("via_node_id"),
        "expanded_from": meta.get("expanded_from"),
        "text": unit.text,
    }


def _load_metrics_json(path: Path, fallback_key: str) -> dict[str, float]:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics") or {}
        return {k: float(v) for k, v in metrics.items()}
    return dict(FALLBACK_BASELINES[fallback_key])


def _load_bm25_clean_results(path: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return by_id
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            by_id[str(obj.get("qa_id"))] = obj
    return by_id


def evaluate_sage(
    samples: list[QASample],
    *,
    sage: SageRetriever,
    bm25: BM25Retriever,
    initial_k: int,
    top_k: int,
    pool_size: int,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float]]:
    """Run per-query SAGE eval + candidate-recall diagnostics."""
    records: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    cand_rows: list[dict[str, float]] = []

    for sample in tqdm(samples, desc="SAGE-RAG", unit="qa"):
        # BM25 seeds / Top10 for diagnostics (same backend as SAGE base).
        bm25_topn = bm25.retrieve(sample.question, top_k=max(top_k, initial_k))
        bm25_ids = [u.unit_id for u in bm25_topn]
        bm25_seed_ids = bm25_ids[:initial_k]
        bm25_top10_ids = bm25_ids[:top_k]

        # Expansion pool then structure ranking (same as SageRetriever internals).
        pool = sage.expansion_retriever.retrieve(
            sample.question,
            top_k=pool_size,
            initial_k=initial_k,
        )
        pool_ids = [u.unit_id for u in pool]
        ranked = sage.ranker.rank(pool)
        final = ranked[:top_k]
        for rank, unit in enumerate(final, start=1):
            unit.rank = rank
        final_ids = [u.unit_id for u in final]

        metrics = compute_retrieval_metrics(
            final_ids,
            sample.gold_unit_ids,
            recall_ks=(1, 5, 10),
            ndcg_ks=(5, 10),
        )
        metric_rows.append(metrics)

        cand = {
            "bm25_seed_recall": recall_at_k(
                bm25_seed_ids, sample.gold_unit_ids, initial_k
            ),
            # Keep legacy key for initial_k=5 reports.
            "bm25_top5_recall": recall_at_k(bm25_ids[:5], sample.gold_unit_ids, 5),
            "expansion_pool_recall": recall_at_k(
                pool_ids, sample.gold_unit_ids, len(pool_ids) or 1
            ),
            "bm25_top10_recall": recall_at_k(bm25_top10_ids, sample.gold_unit_ids, top_k),
            "sage_top10_recall": metrics.get("Recall@10", 0.0),
        }
        cand_rows.append(cand)

        gold = set(sample.gold_unit_ids)
        hit_unit = next((u for u in final if u.unit_id in gold), None)
        hit_meta = (hit_unit.metadata or {}) if hit_unit else {}

        # BM25 rank of first gold in Top10 (1-indexed); None if missing.
        bm25_gold_rank = next(
            (i for i, uid in enumerate(bm25_top10_ids, start=1) if uid in gold),
            None,
        )
        sage_gold_rank = next(
            (i for i, uid in enumerate(final_ids, start=1) if uid in gold),
            None,
        )

        record: dict[str, Any] = {
            "qa_id": sample.qa_id,
            "question": sample.question,
            "gold_unit_ids": sample.gold_unit_ids,
            "gold_unit_id": sample.gold_unit_ids[0] if sample.gold_unit_ids else "",
            "question_type": sample.question_type,
            "document_id": sample.document_id,
            "retrieved_units": [_unit_payload(u) for u in final],
            "retrieved_unit_ids": final_ids,
            "bm25_seed_unit_ids": bm25_seed_ids,
            "bm25_top5_unit_ids": bm25_ids[:5],
            "bm25_top10_unit_ids": bm25_top10_ids,
            "expansion_pool_unit_ids": pool_ids,
            "candidate_recall": cand,
            "bm25_gold_rank": bm25_gold_rank,
            "sage_gold_rank": sage_gold_rank,
            "hit_candidate_source": hit_meta.get("candidate_source"),
            "hit_expansion_relation": hit_meta.get("expansion_relation"),
            **metrics,
        }
        records.append(record)

    avg = average_metrics(metric_rows)
    cand_avg = average_metrics(cand_rows)
    return records, avg, cand_avg


def write_metrics(path: Path, metrics: dict[str, float], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**meta, "metrics": {k: float(v) for k, v in metrics.items()}}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_results_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_evaluation_report(
    path: Path,
    *,
    metrics: dict[str, float],
    cand_avg: dict[str, float],
    meta: dict[str, Any],
    n: int,
    elapsed: float,
) -> None:
    lines = [
        "# SAGE-RAG Evaluation Report",
        "",
        "## Configuration",
        "",
        f"- QA: `{meta['qa_file']}` ({n} queries)",
        f"- Base retriever: BM25",
        f"- initial_k: {meta['initial_k']}",
        f"- final top_k: {meta['top_k']}",
        f"- pool_size: {meta['pool_size']}",
        f"- alpha / beta: {meta['alpha']} / {meta['beta']}",
        f"- Elapsed: {elapsed:.2f} s",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Recall@1 | {metrics.get('Recall@1', 0):.4f} |",
        f"| Recall@5 | {metrics.get('Recall@5', 0):.4f} |",
        f"| Recall@10 | {metrics.get('Recall@10', 0):.4f} |",
        f"| MRR | {metrics.get('MRR', 0):.4f} |",
        f"| nDCG@5 | {metrics.get('nDCG@5', 0):.4f} |",
        f"| nDCG@10 | {metrics.get('nDCG@10', 0):.4f} |",
        "",
        "## Candidate Recall (coverage analysis)",
        "",
        "| Stage | Candidate Recall |",
        "|-------|-----------------:|",
        f"| BM25 Top-{meta['initial_k']} (seeds) | {cand_avg.get('bm25_seed_recall', cand_avg.get('bm25_top5_recall', 0)):.4f} |",
        f"| Expansion pool | {cand_avg.get('expansion_pool_recall', 0):.4f} |",
        f"| BM25 Top-{meta['top_k']} | {cand_avg.get('bm25_top10_recall', 0):.4f} |",
        f"| SAGE Top-{meta['top_k']} | {cand_avg.get('sage_top10_recall', 0):.4f} |",
        "",
        "## Notes",
        "",
        "- Metrics use the shared `compute_retrieval_metrics` (binary Recall@k / MRR / nDCG).",
        "- Per-query results: `retrieval_results.jsonl`",
        "- Baseline comparison: `sage_vs_baselines.md`",
        "- Failure cases (BM25 miss → SAGE hit): `failure_cases.md`",
        "- Fairness check vs initial_k=5: `sage_initial10_vs_initial5.md` (when present)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_comparison(
    path: Path,
    sage_metrics: dict[str, float],
    baselines: dict[str, dict[str, float]],
) -> None:
    rows = [
        ("BM25", baselines["BM25"]),
        ("Dense (BGE-M3)", baselines["Dense (BGE-M3)"]),
        ("Hybrid (RRF)", baselines["Hybrid (RRF)"]),
        ("SAGE-RAG (BM25+Graph)", sage_metrics),
    ]
    lines = [
        "# SAGE-RAG vs Baselines (Clean Benchmark)",
        "",
        "Benchmark: `data/qa_dataset/qa_pairs_clean.jsonl` (460 queries), top_k=10.",
        "",
        "| Method | Recall@1 | Recall@5 | Recall@10 | MRR |",
        "|--------|---------:|---------:|----------:|----:|",
    ]
    for name, m in rows:
        r10 = m.get("Recall@10", 0.0)
        lines.append(
            f"| {name} | {m.get('Recall@1', 0):.4f} | {m.get('Recall@5', 0):.4f} | "
            f"{r10:.4f} ({r10 * 100:.2f}%) | {m.get('MRR', 0):.4f} |"
        )
    bm25_r10 = baselines["BM25"].get("Recall@10", 0.0)
    sage_r10 = sage_metrics.get("Recall@10", 0.0)
    lines += [
        "",
        "## Delta vs BM25",
        "",
        f"- Recall@10: {(sage_r10 - bm25_r10) * 100:+.2f} pp",
        f"- MRR: {sage_metrics.get('MRR', 0) - baselines['BM25'].get('MRR', 0):+.4f}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _infer_success_reason(
    gold_ids: list[str],
    bm25_ids: list[str],
    sage_units: list[dict[str, Any]],
) -> str:
    gold = set(gold_ids)
    hit = next((u for u in sage_units if u.get("unit_id") in gold), None)
    if hit is None:
        return "unknown (gold not in SAGE Top10 — unexpected)"
    src = hit.get("candidate_source")
    rel = hit.get("expansion_relation")
    if src == "initial":
        return "initial BM25 seed retained / re-ranked into Top10"
    if rel == "parent_of":
        return "parent-child structure (parent_of expansion)"
    if rel == "next_to":
        return "sibling clause (next_to expansion)"
    if rel == "refers_to":
        return "cross-reference jump (refers_to expansion)"
    if src == "expanded":
        return "standard structure expansion (expanded candidate)"
    # Heuristic: gold clause is prefix/parent of a BM25 hit
    for bid in bm25_ids:
        for gid in gold_ids:
            if bid != gid and (bid.startswith(gid) or gid in bid):
                return "parent-child numbering relation (heuristic)"
    return "structure-aware ranking of expanded pool"


def write_failure_cases(
    path: Path,
    records: list[dict[str, Any]],
    bm25_by_id: dict[str, dict[str, Any]],
    *,
    min_cases: int = 5,
) -> int:
    """BM25 R@10 fail but SAGE R@10 success."""
    cases: list[dict[str, Any]] = []
    for rec in records:
        sage_hit = float(rec.get("Recall@10") or 0.0) >= 1.0
        bm25_ids = rec.get("bm25_top10_unit_ids") or []
        # Prefer live BM25 ids from this run; fall back to clean_benchmark dump.
        if not bm25_ids and rec["qa_id"] in bm25_by_id:
            bm25_ids = [
                x.get("unit_id")
                for x in (bm25_by_id[rec["qa_id"]].get("retrieved") or [])
                if x.get("unit_id")
            ]
        bm25_hit = any(uid in set(rec["gold_unit_ids"]) for uid in bm25_ids[:10])
        if (not bm25_hit) and sage_hit:
            cases.append({**rec, "bm25_top10_unit_ids": bm25_ids[:10]})

    lines = [
        "# Failure Analysis: BM25 miss → SAGE hit",
        "",
        f"Selected cases where BM25 Recall@10 = 0 but SAGE Recall@10 = 1 "
        f"(showing up to {max(min_cases, len(cases))} / found {len(cases)}).",
        "",
    ]
    shown = cases[: max(min_cases, 5)]
    for i, rec in enumerate(shown, start=1):
        reason = _infer_success_reason(
            rec["gold_unit_ids"],
            rec.get("bm25_top10_unit_ids") or [],
            rec.get("retrieved_units") or [],
        )
        lines += [
            f"## Case {i}",
            "",
            f"**qa_id:** `{rec['qa_id']}`",
            "",
            f"**Question:** {rec['question']}",
            "",
            f"**Gold Evidence:** `{rec.get('gold_unit_id') or (rec['gold_unit_ids'][:1] or [''])[0]}`",
            "",
            "### BM25 Top10",
            "",
        ]
        for j, uid in enumerate(rec.get("bm25_top10_unit_ids") or [], start=1):
            mark = " ← gold" if uid in set(rec["gold_unit_ids"]) else ""
            lines.append(f"{j}. `{uid}`{mark}")
        lines += ["", "### SAGE Top10", ""]
        for u in rec.get("retrieved_units") or []:
            mark = " ← gold" if u.get("unit_id") in set(rec["gold_unit_ids"]) else ""
            lines.append(
                f"{u.get('rank')}. `{u.get('unit_id')}` "
                f"(source={u.get('candidate_source')}, relation={u.get('expansion_relation')}, "
                f"final={u.get('final_score')}){mark}"
            )
        lines += ["", f"**Analysis:** {reason}", "", "---", ""]

    if not shown:
        lines.append("_No BM25-miss / SAGE-hit cases found in this run._\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    return len(cases)


def write_initial_k_comparison(
    path: Path,
    *,
    metrics_k10: dict[str, float],
    metrics_k5: dict[str, float],
    meta_k10: dict[str, Any],
    cand_avg_k10: dict[str, float],
) -> None:
    keys = ["Recall@1", "Recall@5", "Recall@10", "MRR", "nDCG@5", "nDCG@10"]
    lines = [
        "# SAGE initial_k=10 vs initial_k=5",
        "",
        "Fairness check: keep BM25 Top-10 seeds before Graph Expansion.",
        "",
        f"- initial_k=10 run: `results/retrieval/sage_initial10/` "
        f"(α={meta_k10.get('alpha')}, β={meta_k10.get('beta')})",
        "- initial_k=5 run: `results/retrieval/sage/`",
        "",
        "| Metric | SAGE initial_k=5 | SAGE initial_k=10 | Δ |",
        "|--------|-----------------:|------------------:|--:|",
    ]
    for key in keys:
        a = float(metrics_k5.get(key, 0.0))
        b = float(metrics_k10.get(key, 0.0))
        delta = b - a
        if key.startswith("Recall") or key.startswith("nDCG"):
            lines.append(f"| {key} | {a:.4f} | {b:.4f} | {delta * 100:+.2f} pp |")
        else:
            lines.append(f"| {key} | {a:.4f} | {b:.4f} | {delta:+.4f} |")

    seed_rec = float(
        cand_avg_k10.get("bm25_seed_recall", cand_avg_k10.get("bm25_top10_recall", 0.0))
    )
    pool_rec = float(cand_avg_k10.get("expansion_pool_recall", 0.0))
    final_rec = float(cand_avg_k10.get("sage_top10_recall", 0.0))
    lines += [
        "",
        "## Candidate recall (initial_k=10 run)",
        "",
        "| Stage | Candidate Recall |",
        "|-------|-----------------:|",
        f"| BM25 Top-10 (seeds) | {seed_rec:.4f} |",
        f"| Expansion pool | {pool_rec:.4f} |",
        f"| SAGE final Top-10 | {final_rec:.4f} |",
        "",
        f"- Expansion lift over BM25 Top-10 seeds: **{(pool_rec - seed_rec) * 100:+.2f} pp**",
        "",
        "## Conclusion notes",
        "",
        "- With `initial_k=10`, SAGE starts from the same BM25 Top-10 pool as the baseline,",
        "  so rank 6–10 evidence is no longer dropped before expansion.",
        "- Remaining gaps vs BM25 then come from ranking / expanded-candidate competition,",
        "  not from an undersized seed set.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_rank6_10_promotion_cases(
    path: Path,
    records: list[dict[str, Any]],
    *,
    min_cases: int = 5,
) -> tuple[int, int]:
    """Cases where gold was BM25 rank 6–10 and is retained / promoted by SAGE."""
    retained: list[dict[str, Any]] = []
    promoted: list[dict[str, Any]] = []
    for rec in records:
        b_rank = rec.get("bm25_gold_rank")
        s_rank = rec.get("sage_gold_rank")
        if not isinstance(b_rank, int) or not isinstance(s_rank, int):
            continue
        if 6 <= b_rank <= 10:
            retained.append(rec)
            if s_rank < b_rank:
                promoted.append(rec)

    shown = promoted[:min_cases]
    if len(shown) < min_cases:
        for rec in retained:
            if rec in shown:
                continue
            shown.append(rec)
            if len(shown) >= min_cases:
                break

    lines = [
        "# BM25 rank 6–10 retention / promotion under SAGE (initial_k=10)",
        "",
        "Focus: gold evidence that already sits in BM25 positions 6–10 "
        "(would have been dropped when `initial_k=5`).",
        "",
        f"- Retained in SAGE Top10: **{len(retained)}**",
        f"- Strictly promoted (SAGE rank < BM25 rank): **{len(promoted)}**",
        f"- Showing **{len(shown)}** cases below.",
        "",
    ]
    for i, rec in enumerate(shown, start=1):
        b_rank = rec.get("bm25_gold_rank")
        s_rank = rec.get("sage_gold_rank")
        moved = (
            "promoted"
            if isinstance(b_rank, int) and isinstance(s_rank, int) and s_rank < b_rank
            else "retained"
        )
        hit = next(
            (
                u
                for u in (rec.get("retrieved_units") or [])
                if u.get("unit_id") in set(rec.get("gold_unit_ids") or [])
            ),
            {},
        )
        lines += [
            f"## Case {i} ({moved})",
            "",
            f"**qa_id:** `{rec.get('qa_id')}`",
            "",
            f"**Question:** {rec.get('question')}",
            "",
            f"**Gold Evidence:** `{rec.get('gold_unit_id')}`",
            "",
            f"- BM25 Top10 position: **#{b_rank}**",
            f"- SAGE ranking position: **#{s_rank}**",
            f"- candidate_source: `{hit.get('candidate_source')}`",
            f"- expansion_relation: `{hit.get('expansion_relation')}`",
            f"- final_score: `{hit.get('final_score')}`",
            "",
            "### BM25 Top10 (context)",
            "",
        ]
        gold = set(rec.get("gold_unit_ids") or [])
        for j, uid in enumerate(rec.get("bm25_top10_unit_ids") or [], start=1):
            mark = " ← gold" if uid in gold else ""
            lines.append(f"{j}. `{uid}`{mark}")
        lines += ["", "### SAGE Top10 (context)", ""]
        for u in rec.get("retrieved_units") or []:
            mark = " ← gold" if u.get("unit_id") in gold else ""
            lines.append(
                f"{u.get('rank')}. `{u.get('unit_id')}` "
                f"(source={u.get('candidate_source')}, relation={u.get('expansion_relation')})"
                f"{mark}"
            )
        lines += ["", "---", ""]

    if not shown:
        lines.append("_No BM25 rank 6–10 gold cases found in this run._\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    return len(retained), len(promoted)


def write_candidate_recall_report(
    path: Path,
    cand_avg: dict[str, float],
    n: int,
    *,
    initial_k: int = 10,
    top_k: int = 10,
) -> None:
    seed_key_val = cand_avg.get("bm25_seed_recall", cand_avg.get("bm25_top5_recall", 0.0))
    lines = [
        "# Candidate Recall Analysis",
        "",
        f"Macro-average over **{n}** Clean queries (initial_k={initial_k}).",
        "",
        "| Stage | Candidate Recall |",
        "|-------|-----------------:|",
        f"| BM25 Top-{initial_k} (seeds) | {seed_key_val:.4f} |",
        f"| After Graph Expansion (pool) | {cand_avg.get('expansion_pool_recall', 0):.4f} |",
        f"| BM25 Top-{top_k} | {cand_avg.get('bm25_top10_recall', 0):.4f} |",
        f"| SAGE final Top-{top_k} | {cand_avg.get('sage_top10_recall', 0):.4f} |",
        "",
        "## Interpretation",
        "",
        f"- If **Expansion pool > BM25 Top-{initial_k}**, graph edges recovered gold evidence "
        "missing from the BM25 seed set.",
        f"- If **SAGE Top-{top_k} > BM25 Top-{top_k}**, structure ranking surfaced recovered "
        "evidence into the final list (or reordered seeds favorably).",
        "- If pool rises but SAGE Top-10 does not, ranking / pool truncation is the bottleneck.",
        "",
    ]
    delta = float(cand_avg.get("expansion_pool_recall", 0)) - float(seed_key_val)
    lines.append(
        f"- Expansion lift over BM25 Top-{initial_k}: **{delta * 100:+.2f} pp**\n"
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.qa.is_file():
        logger.error("QA file not found: %s", args.qa)
        return 1

    samples, skipped = load_qa_samples(args.qa)
    if args.sample is not None:
        samples = sample_qa_pairs(samples, args.sample, seed=args.seed)
    if not samples:
        logger.error("No QA samples to evaluate")
        return 1

    store = GraphStore.from_dir(args.graph_dir)
    expander = GraphExpander(store)
    bm25 = BM25Retriever(index_dir=args.bm25_index)
    ranker = StructureRanker(alpha=args.alpha, beta=args.beta)
    sage = SageRetriever(
        bm25,
        expander,
        ranker=ranker,
        pool_size=args.pool_size,
    )

    meta = {
        "retriever": "SAGE-RAG (BM25 + Graph Expansion + Structure Ranking)",
        "qa_file": str(args.qa),
        "initial_k": args.initial_k,
        "top_k": args.top_k,
        "pool_size": args.pool_size,
        "alpha": args.alpha,
        "beta": args.beta,
        "evaluated_pairs": len(samples),
        "skipped_pairs": skipped,
    }

    t0 = time.perf_counter()
    records, metrics, cand_avg = evaluate_sage(
        samples,
        sage=sage,
        bm25=bm25,
        initial_k=args.initial_k,
        top_k=args.top_k,
        pool_size=args.pool_size,
    )
    elapsed = time.perf_counter() - t0
    meta["elapsed_seconds"] = elapsed

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_metrics(out / "metrics.json", metrics, meta)
    write_results_jsonl(out / "retrieval_results.jsonl", records)
    write_evaluation_report(
        out / "evaluation_report.md",
        metrics=metrics,
        cand_avg=cand_avg,
        meta=meta,
        n=len(samples),
        elapsed=elapsed,
    )

    baselines = {
        "BM25": _load_metrics_json(CLEAN_BM25_METRICS, "BM25"),
        "Dense (BGE-M3)": _load_metrics_json(CLEAN_DENSE_METRICS, "Dense (BGE-M3)"),
        "Hybrid (RRF)": _load_metrics_json(CLEAN_HYBRID_METRICS, "Hybrid (RRF)"),
    }
    write_comparison(out / "sage_vs_baselines.md", metrics, baselines)

    bm25_by_id = _load_bm25_clean_results(CLEAN_BM25_RESULTS)
    n_fail_success = write_failure_cases(out / "failure_cases.md", records, bm25_by_id)
    write_candidate_recall_report(
        out / "candidate_recall.md",
        cand_avg,
        len(samples),
        initial_k=args.initial_k,
        top_k=args.top_k,
    )

    # Compare against prior initial_k=5 run when available.
    compare_path = args.compare_initial5_metrics
    if compare_path.is_file() and args.initial_k == 10:
        prev = json.loads(compare_path.read_text(encoding="utf-8"))
        prev_metrics = {k: float(v) for k, v in (prev.get("metrics") or {}).items()}
        write_initial_k_comparison(
            out / "sage_initial10_vs_initial5.md",
            metrics_k10=metrics,
            metrics_k5=prev_metrics,
            meta_k10=meta,
            cand_avg_k10=cand_avg,
        )

    n_retained, n_promoted = write_rank6_10_promotion_cases(
        out / "rank6_10_promotion_cases.md",
        records,
    )

    hit_src = Counter(
        (r.get("hit_candidate_source") or "none")
        for r in records
        if float(r.get("Recall@10") or 0) >= 1.0
    )
    hit_rel = Counter(
        (r.get("hit_expansion_relation") or "none")
        for r in records
        if float(r.get("Recall@10") or 0) >= 1.0
    )

    print(json.dumps({"metrics": metrics, "candidate_recall": cand_avg}, ensure_ascii=False, indent=2))
    print(f"\nWrote outputs under {out}")
    print(f"BM25-miss→SAGE-hit cases: {n_fail_success}")
    print(f"BM25 rank6-10 retained/promoted: {n_retained}/{n_promoted}")
    print(f"SAGE hit sources: {dict(hit_src)}")
    print(f"SAGE hit relations: {dict(hit_rel)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
