"""Evaluate SAGE-RAG v4 (Risk-aware Evidence Competition) on Clean Benchmark.

Usage:
  python src/evaluation/evaluate_sage_v4.py
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
from src.sage_rag.ranking.risk_aware_ranker import RiskAwareEvidenceSelector  # noqa: E402
from src.sage_rag.retrieval.sage_expansion_retriever import SageExpansionRetriever  # noqa: E402
from src.sage_rag.retrieval.sage_retriever_v4 import SageRetrieverV4  # noqa: E402

logger = logging.getLogger(__name__)

QA_CLEAN = ROOT / "data" / "qa_dataset" / "qa_pairs_clean.jsonl"
OUT_DIR = ROOT / "results" / "retrieval" / "sage_v4"
SAGE_V2_METRICS = ROOT / "results" / "retrieval" / "sage_v2" / "metrics.json"
SAGE_V3_METRICS = ROOT / "results" / "retrieval" / "sage_v3" / "metrics.json"
CLEAN_BM25_METRICS = ROOT / "results" / "retrieval" / "clean_benchmark" / "bm25_metrics.json"
CLEAN_DENSE_METRICS = ROOT / "results" / "retrieval" / "clean_benchmark" / "dense_metrics.json"
CLEAN_HYBRID_METRICS = ROOT / "results" / "retrieval" / "clean_benchmark" / "hybrid_metrics.json"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate SAGE-RAG v4 on Clean Benchmark")
    p.add_argument("--qa", type=Path, default=QA_CLEAN)
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    p.add_argument("--bm25-index", type=Path, default=ROOT / "data" / "bm25_index")
    p.add_argument("--graph-dir", type=Path, default=ROOT / "data" / "sage_graph")
    p.add_argument("--initial-k", type=int, default=10)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--pool-size", type=int, default=80)
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=0.30)
    p.add_argument("--lam", type=float, default=0.20)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _preview(text: str, n: int = 120) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + "..."


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
        "expanded_from": meta.get("expanded_from"),
        "retrieval_score_normalized": meta.get("retrieval_score_normalized"),
        "structure_score": meta.get("structure_score"),
        "query_coverage_score": meta.get("query_coverage_score"),
        "risk_penalty": meta.get("risk_penalty"),
        "graph_distance": meta.get("graph_distance"),
        "text": unit.text,
    }


def _load_metrics(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {k: float(v) for k, v in (payload.get("metrics") or {}).items()}


def _load_v3_best(path: Path) -> dict[str, float]:
    """Prefer Fixed strategy metrics from v3 ablation payload."""
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    by = payload.get("metrics_by_strategy") or {}
    fixed = by.get("sage_v3_fixed") or {}
    if fixed:
        return {k: float(v) for k, v in fixed.items()}
    return {k: float(v) for k, v in (payload.get("metrics") or {}).items()}


def evaluate(
    samples: list[QASample],
    *,
    expansion: SageExpansionRetriever,
    selector: RiskAwareEvidenceSelector,
    bm25: BM25Retriever,
    initial_k: int,
    top_k: int,
    pool_size: int,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    cand_rows: list[dict[str, float]] = []
    case_bank: list[dict[str, Any]] = []

    for sample in tqdm(samples, desc="SAGE-RAG-v4", unit="qa"):
        gold = set(sample.gold_unit_ids)

        bm25_hits = bm25.retrieve(sample.question, top_k=top_k)
        bm25_ids = [u.unit_id for u in bm25_hits]
        bm25_hit = bool(gold & set(bm25_ids))

        pool = expansion.retrieve(
            sample.question, top_k=pool_size, initial_k=initial_k
        )
        pool_ids = [u.unit_id for u in pool]
        scored_all = selector.score_candidates(pool, query=sample.question)
        final = scored_all[:top_k]
        for rank, u in enumerate(final, start=1):
            u.rank = rank
        final_ids = [u.unit_id for u in final]
        v4_hit = bool(gold & set(final_ids))

        metrics = compute_retrieval_metrics(
            final_ids,
            sample.gold_unit_ids,
            recall_ks=(1, 5, 10),
            ndcg_ks=(5, 10),
        )
        metric_rows.append(metrics)

        expand_gold_ids = [
            uid for uid in sample.gold_unit_ids if uid in set(pool_ids) and uid not in set(bm25_ids)
        ]
        cand = {
            "bm25_top10_recall": recall_at_k(bm25_ids, sample.gold_unit_ids, top_k),
            "expansion_pool_recall": recall_at_k(
                pool_ids, sample.gold_unit_ids, len(pool_ids) or 1
            ),
            "sage_v4_top10_recall": metrics.get("Recall@10", 0.0),
            "gold_only_via_expand": 1.0 if expand_gold_ids else 0.0,
            "expand_gold_in_v4": (
                1.0 if expand_gold_ids and any(g in set(final_ids) for g in expand_gold_ids) else 0.0
            ),
        }
        cand_rows.append(cand)

        hit = next((u for u in final if u.unit_id in gold), None)
        hit_meta = (hit.metadata or {}) if hit else {}

        # Score map for analysis.
        score_by_id = {
            u.unit_id: (u.metadata or {}) for u in scored_all
        }
        expanded_scored = [
            u
            for u in scored_all
            if (u.metadata or {}).get("candidate_source") == "expanded"
        ]
        promoted = [
            u
            for u in final
            if (u.metadata or {}).get("candidate_source") == "expanded"
        ]
        # High-risk expanded that did NOT enter Top-k (filtered by risk / score).
        filtered_risky = [
            u
            for u in expanded_scored
            if u.unit_id not in set(final_ids)
            and float((u.metadata or {}).get("risk_penalty") or 0) >= 0.35
        ]

        case_type = None
        if (not bm25_hit) and v4_hit:
            case_type = "bm25_miss_v4_hit"
        elif expand_gold_ids and any(g in set(final_ids) for g in expand_gold_ids):
            case_type = "graph_promoted_gold"
        elif hit_meta.get("candidate_source") == "expanded":
            case_type = "graph_promoted_hit"
        elif (
            filtered_risky
            and len(filtered_risky) >= 3
            and not promoted
            and float(metrics.get("Recall@10") or 0) >= 1.0
        ):
            # Successful BM25/v4 hit where noisy neighbors were risk-filtered.
            case_type = "risk_filtered_noise"
        elif expand_gold_ids and not any(g in set(final_ids) for g in expand_gold_ids):
            # Expand found gold but v4 still missed — near-miss for analysis.
            case_type = "expand_gold_near_miss"

        if case_type:
            gold_scores = []
            for gid in sample.gold_unit_ids:
                m = score_by_id.get(gid) or {}
                # pool rank among scored_all
                pool_rank = next(
                    (i + 1 for i, u in enumerate(scored_all) if u.unit_id == gid),
                    None,
                )
                gold_scores.append(
                    {
                        "unit_id": gid,
                        "in_bm25": gid in set(bm25_ids),
                        "in_pool": gid in set(pool_ids),
                        "v4_rank": pool_rank,
                        "final_score": m.get("final_score"),
                        "retrieval_score_normalized": m.get("retrieval_score_normalized"),
                        "structure_score": m.get("structure_score"),
                        "query_coverage_score": m.get("query_coverage_score"),
                        "risk_penalty": m.get("risk_penalty"),
                        "candidate_source": m.get("candidate_source"),
                        "expansion_relation": m.get("expansion_relation"),
                    }
                )
            case_bank.append(
                {
                    "case_type": case_type,
                    "qa_id": sample.qa_id,
                    "question": sample.question,
                    "gold_unit_ids": sample.gold_unit_ids,
                    "bm25_top10": [
                        {"unit_id": u.unit_id, "score": u.score, "text": _preview(u.text)}
                        for u in bm25_hits
                    ],
                    "expanded_top": [
                        {
                            "unit_id": u.unit_id,
                            "relation": (u.metadata or {}).get("expansion_relation"),
                            "risk": (u.metadata or {}).get("risk_penalty"),
                            "final_score": (u.metadata or {}).get("final_score"),
                            "text": _preview(u.text, 80),
                        }
                        for u in expanded_scored[:8]
                    ],
                    "promoted_graph": [
                        {
                            "rank": u.rank,
                            "unit_id": u.unit_id,
                            "relation": (u.metadata or {}).get("expansion_relation"),
                            "risk": (u.metadata or {}).get("risk_penalty"),
                            "final_score": (u.metadata or {}).get("final_score"),
                            "coverage": (u.metadata or {}).get("query_coverage_score"),
                            "structure": (u.metadata or {}).get("structure_score"),
                            "text": _preview(u.text, 80),
                        }
                        for u in promoted
                    ],
                    "filtered_risky": [
                        {
                            "unit_id": u.unit_id,
                            "relation": (u.metadata or {}).get("expansion_relation"),
                            "risk": (u.metadata or {}).get("risk_penalty"),
                            "final_score": (u.metadata or {}).get("final_score"),
                            "v4_pool_rank": next(
                                (
                                    i + 1
                                    for i, x in enumerate(scored_all)
                                    if x.unit_id == u.unit_id
                                ),
                                None,
                            ),
                            "text": _preview(u.text, 80),
                        }
                        for u in filtered_risky[:5]
                    ],
                    "gold_score_trace": gold_scores,
                    "v4_top10": [_unit_payload(u) for u in final],
                }
            )

        records.append(
            {
                "qa_id": sample.qa_id,
                "question": sample.question,
                "gold_unit_ids": sample.gold_unit_ids,
                "gold_unit_id": sample.gold_unit_ids[0] if sample.gold_unit_ids else "",
                "question_type": sample.question_type,
                "document_id": sample.document_id,
                "retrieved_units": [_unit_payload(u) for u in final],
                "retrieved_unit_ids": final_ids,
                "bm25_top10_unit_ids": bm25_ids,
                "expansion_pool_unit_ids": pool_ids,
                "candidate_recall": cand,
                "hit_candidate_source": hit_meta.get("candidate_source"),
                "hit_expansion_relation": hit_meta.get("expansion_relation"),
                "hit_risk_penalty": hit_meta.get("risk_penalty"),
                "hit_structure_score": hit_meta.get("structure_score"),
                "hit_query_coverage": hit_meta.get("query_coverage_score"),
                **metrics,
            }
        )

    return records, average_metrics(metric_rows), average_metrics(cand_rows), case_bank


def write_ablation(
    path: Path,
    *,
    bm25: dict[str, float],
    dense: dict[str, float],
    hybrid: dict[str, float],
    sage_v2: dict[str, float],
    sage_v3: dict[str, float],
    sage_v4: dict[str, float],
    cand_avg: dict[str, float],
    hit_src: Counter,
    n: int,
) -> None:
    pool_r = cand_avg.get("expansion_pool_recall", 0.0)
    bm25_r = cand_avg.get("bm25_top10_recall", bm25.get("Recall@10", 0.0))
    lines = [
        "# SAGE-RAG v4 Ablation (Clean Benchmark)",
        "",
        f"n={n}, initial_k=10, top_k=10.",
        "",
        "| Method | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |",
        "|--------|---------:|---------:|----------:|----:|--------:|",
    ]

    def row(name: str, m: dict[str, float]) -> None:
        lines.append(
            f"| {name} | {m.get('Recall@1', 0):.4f} | {m.get('Recall@5', 0):.4f} | "
            f"{m.get('Recall@10', 0):.4f} | {m.get('MRR', 0):.4f} | "
            f"{m.get('nDCG@10', 0):.4f} |"
        )

    row("BM25", bm25)
    row("Dense (BGE-M3)", dense)
    row("Hybrid (RRF)", hybrid)
    lines.append(
        f"| BM25 + Graph Expansion (candidate recall) | — | — | "
        f"{pool_r:.4f} | — | — |"
    )
    row("SAGE v2 (greedy selection)", sage_v2)
    row("SAGE v3 Fixed Allocation", sage_v3)
    row("SAGE v4 Risk-aware Competition", sage_v4)

    r10_bm25 = bm25.get("Recall@10", bm25_r)
    r10_v2 = sage_v2.get("Recall@10", 0.0)
    r10_v3 = sage_v3.get("Recall@10", 0.0)
    r10_v4 = sage_v4.get("Recall@10", 0.0)

    lines += [
        "",
        "## Pipeline contrast",
        "",
        "- **v2**: expand → greedy structure selection (no explicit risk / no allocation)",
        "- **v3**: expand → hard graph slot allocation → v2 selection",
        "- **v4**: expand → unified competition with RiskPenalty (no forced slots)",
        "",
        "## Candidate → Final",
        "",
        f"- BM25 Top10 recall: **{bm25_r:.4f}**",
        f"- Expansion pool recall: **{pool_r:.4f}** (+{(pool_r - bm25_r) * 100:.2f} pp)",
        f"- SAGE v2 R@10: **{r10_v2:.4f}**",
        f"- SAGE v3 Fixed R@10: **{r10_v3:.4f}**",
        f"- SAGE v4 R@10: **{r10_v4:.4f}**",
        f"- Expand-only gold recovered by v4: "
        f"**{cand_avg.get('expand_gold_in_v4', 0) * 100:.2f}%** of queries",
        "",
        f"- v4 hit sources (R@10 successes): `{dict(hit_src)}`",
        "",
        "## Takeaway",
        "",
    ]
    if r10_v4 > r10_bm25 + 1e-9:
        lines.append(
            f"- **R@10 lift:** v4 {r10_v4:.4f} > BM25 {r10_bm25:.4f} "
            f"(+{(r10_v4 - r10_bm25) * 100:.2f} pp)."
        )
    elif abs(r10_v4 - r10_bm25) < 1e-9:
        lines.append(
            f"- **R@10 tied with BM25** at {r10_v4:.4f}; check MRR/nDCG and "
            "expanded-hit counts for ranking quality gains."
        )
    else:
        lines.append(
            f"- R@10 not yet above BM25 ({r10_v4:.4f} vs {r10_bm25:.4f})."
        )

    if r10_v4 > r10_v3 + 1e-9:
        lines.append(
            f"- **Beats v3 hard allocation** by {(r10_v4 - r10_v3) * 100:+.2f} pp R@10 "
            "(risk-aware competition avoids forced displacement)."
        )
    if sage_v4.get("MRR", 0) > sage_v2.get("MRR", 0) + 1e-9:
        lines.append(
            f"- MRR improves vs v2: {sage_v4.get('MRR', 0):.4f} vs "
            f"{sage_v2.get('MRR', 0):.4f}."
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vs_baselines(
    path: Path,
    *,
    bm25: dict[str, float],
    dense: dict[str, float],
    hybrid: dict[str, float],
    sage_v2: dict[str, float],
    sage_v3: dict[str, float],
    sage_v4: dict[str, float],
    cand_avg: dict[str, float],
) -> None:
    pool_r = cand_avg.get("expansion_pool_recall", 0.0)
    lines = [
        "# SAGE v4 vs Baselines",
        "",
        "| Method | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |",
        "|--------|---------:|---------:|----------:|----:|--------:|",
    ]

    def row(name: str, m: dict[str, float]) -> None:
        lines.append(
            f"| {name} | {m.get('Recall@1', 0):.4f} | {m.get('Recall@5', 0):.4f} | "
            f"{m.get('Recall@10', 0):.4f} | {m.get('MRR', 0):.4f} | "
            f"{m.get('nDCG@10', 0):.4f} |"
        )

    row("BM25", bm25)
    row("Dense", dense)
    row("Hybrid", hybrid)
    lines.append(
        f"| BM25+Expansion (pool) | — | — | {pool_r:.4f} | — | — |"
    )
    row("SAGE v2", sage_v2)
    row("SAGE v3 Fixed", sage_v3)
    row("**SAGE v4**", sage_v4)
    lines += [
        "",
        "## Deltas vs BM25",
        "",
        f"- ΔR@1: {(sage_v4.get('Recall@1', 0) - bm25.get('Recall@1', 0)) * 100:+.2f} pp",
        f"- ΔR@5: {(sage_v4.get('Recall@5', 0) - bm25.get('Recall@5', 0)) * 100:+.2f} pp",
        f"- ΔR@10: {(sage_v4.get('Recall@10', 0) - bm25.get('Recall@10', 0)) * 100:+.2f} pp",
        f"- ΔMRR: {sage_v4.get('MRR', 0) - bm25.get('MRR', 0):+.4f}",
        f"- ΔnDCG@10: {sage_v4.get('nDCG@10', 0) - bm25.get('nDCG@10', 0):+.4f}",
        "",
        "## Deltas vs SAGE v2",
        "",
        f"- ΔR@10: {(sage_v4.get('Recall@10', 0) - sage_v2.get('Recall@10', 0)) * 100:+.2f} pp",
        f"- ΔMRR: {sage_v4.get('MRR', 0) - sage_v2.get('MRR', 0):+.4f}",
        "",
        "## Deltas vs SAGE v3 Fixed",
        "",
        f"- ΔR@10: {(sage_v4.get('Recall@10', 0) - sage_v3.get('Recall@10', 0)) * 100:+.2f} pp",
        f"- ΔMRR: {sage_v4.get('MRR', 0) - sage_v3.get('MRR', 0):+.4f}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_failure_cases(path: Path, case_bank: list[dict[str, Any]], *, min_n: int = 10) -> None:
    preferred_order = [
        "bm25_miss_v4_hit",
        "graph_promoted_gold",
        "graph_promoted_hit",
        "risk_filtered_noise",
        "expand_gold_near_miss",
    ]
    per_type_cap = {
        "bm25_miss_v4_hit": 3,
        "graph_promoted_gold": 3,
        "graph_promoted_hit": 2,
        "risk_filtered_noise": 3,
        "expand_gold_near_miss": 3,
    }
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    counts: dict[str, int] = {k: 0 for k in preferred_order}

    for ctype in preferred_order:
        for c in case_bank:
            if c["case_type"] != ctype or c["qa_id"] in used_ids:
                continue
            if counts[ctype] >= per_type_cap.get(ctype, 2):
                break
            selected.append(c)
            used_ids.add(c["qa_id"])
            counts[ctype] += 1

    if len(selected) < min_n:
        for c in case_bank:
            if c["qa_id"] in used_ids:
                continue
            selected.append(c)
            used_ids.add(c["qa_id"])
            if len(selected) >= min_n:
                break

    lines = [
        "# SAGE v4 Failure / Success Case Analysis",
        "",
        f"Selected **{len(selected)}** illustrative cases "
        f"(from {len(case_bank)} tagged queries). Type counts: `{counts}`.",
        "",
        "Case types:",
        "- `bm25_miss_v4_hit`: BM25 Top10 miss, SAGE v4 hit",
        "- `graph_promoted_gold`: expanded gold enters final Top-k",
        "- `graph_promoted_hit`: final hit is an expanded candidate",
        "- `risk_filtered_noise`: high-risk expanded neighbors stay out of Top-k",
        "- `expand_gold_near_miss`: expansion found gold but v4 Top10 still missed",
        "",
    ]

    for i, c in enumerate(selected, start=1):
        lines += [
            f"## Case {i}: `{c['case_type']}` — {c['qa_id']}",
            "",
            f"**Query:** {c['question']}",
            "",
            f"**Gold:** `{c['gold_unit_ids']}`",
            "",
            "### Gold score trace (v4)",
            "",
        ]
        for g in c.get("gold_score_trace") or []:
            lines.append(
                f"- `{g['unit_id']}` in_bm25={g['in_bm25']} in_pool={g['in_pool']} "
                f"v4_rank={g['v4_rank']} src={g.get('candidate_source')} "
                f"rel={g.get('expansion_relation')} "
                f"final={g.get('final_score')} retr={g.get('retrieval_score_normalized')} "
                f"struct={g.get('structure_score')} cov={g.get('query_coverage_score')} "
                f"risk={g.get('risk_penalty')}"
            )
        lines += ["", "### BM25 Top10", ""]
        for j, u in enumerate(c.get("bm25_top10") or [], start=1):
            lines.append(f"{j}. `{u['unit_id']}` score={u.get('score')} — {u.get('text')}")
        lines += ["", "### Expanded candidates (scored, top)", ""]
        for u in c.get("expanded_top") or []:
            lines.append(
                f"- `{u['unit_id']}` rel={u.get('relation')} risk={u.get('risk')} "
                f"final={u.get('final_score')} — {u.get('text')}"
            )
        if c.get("promoted_graph"):
            lines += ["", "### Graph evidence promoted into Top-k", ""]
            for u in c["promoted_graph"]:
                lines.append(
                    f"- #{u.get('rank')} `{u['unit_id']}` rel={u.get('relation')} "
                    f"risk={u.get('risk')} struct={u.get('structure')} "
                    f"cov={u.get('coverage')} final={u.get('final_score')} — {u.get('text')}"
                )
        if c.get("filtered_risky"):
            lines += ["", "### High-risk graph candidates filtered out of Top-k", ""]
            for u in c["filtered_risky"]:
                lines.append(
                    f"- `{u['unit_id']}` rel={u.get('relation')} risk={u.get('risk')} "
                    f"final={u.get('final_score')} pool_rank={u.get('v4_pool_rank')} "
                    f"— {u.get('text')}"
                )
        lines += ["", "### SAGE v4 final Top10", ""]
        for u in c.get("v4_top10") or []:
            lines.append(
                f"- #{u.get('rank')} `{u['unit_id']}` src={u.get('candidate_source')} "
                f"rel={u.get('expansion_relation')} risk={u.get('risk_penalty')} "
                f"final={u.get('final_score')}"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    samples, skipped = load_qa_samples(args.qa)
    if args.sample is not None:
        samples = sample_qa_pairs(samples, args.sample, seed=args.seed)

    store = GraphStore.from_dir(args.graph_dir)
    expander = GraphExpander(store)
    bm25 = BM25Retriever(index_dir=args.bm25_index)
    expansion = SageExpansionRetriever(bm25, expander)
    selector = RiskAwareEvidenceSelector(
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, lam=args.lam
    )
    # Keep SageRetrieverV4 constructed for API parity / smoke.
    _ = SageRetrieverV4(bm25, expander, ranker=selector, pool_size=args.pool_size)

    t0 = time.perf_counter()
    records, metrics, cand_avg, case_bank = evaluate(
        samples,
        expansion=expansion,
        selector=selector,
        bm25=bm25,
        initial_k=args.initial_k,
        top_k=args.top_k,
        pool_size=args.pool_size,
    )
    elapsed = time.perf_counter() - t0

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    bm25_m = _load_metrics(CLEAN_BM25_METRICS)
    dense_m = _load_metrics(CLEAN_DENSE_METRICS)
    hybrid_m = _load_metrics(CLEAN_HYBRID_METRICS)
    v2_m = _load_metrics(SAGE_V2_METRICS)
    v3_m = _load_v3_best(SAGE_V3_METRICS)

    meta = {
        "retriever": "SAGE-RAG v4 (Risk-aware Evidence Competition)",
        "qa_file": str(args.qa),
        "initial_k": args.initial_k,
        "top_k": args.top_k,
        "pool_size": args.pool_size,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "lam": args.lam,
        "evaluated_pairs": len(samples),
        "skipped_pairs": skipped,
        "elapsed_seconds": elapsed,
        "metrics": metrics,
        "candidate_recall": cand_avg,
    }
    (out / "metrics.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out / "retrieval_results.jsonl").open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    hit_src = Counter(
        (r.get("hit_candidate_source") or "none")
        for r in records
        if float(r.get("Recall@10") or 0) >= 1.0
    )

    report = [
        "# SAGE-RAG v4 Evaluation Report",
        "",
        f"- QA: `{args.qa}` ({len(samples)} queries)",
        f"- initial_k={args.initial_k}, top_k={args.top_k}, pool_size={args.pool_size}",
        f"- Score = α·Retr + β·Struct + γ·Coverage − λ·Risk",
        f"- weights: α={args.alpha}, β={args.beta}, γ={args.gamma}, λ={args.lam}",
        f"- elapsed: {elapsed:.2f}s",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|--------|------:|",
    ]
    for k in ("Recall@1", "Recall@5", "Recall@10", "MRR", "nDCG@5", "nDCG@10"):
        report.append(f"| {k} | {metrics.get(k, 0):.4f} |")
    report += [
        "",
        "## Candidate Recall",
        "",
        f"- BM25 Top10: {cand_avg.get('bm25_top10_recall', 0):.4f}",
        f"- Expansion pool: {cand_avg.get('expansion_pool_recall', 0):.4f}",
        f"- SAGE v4 Top10: {cand_avg.get('sage_v4_top10_recall', 0):.4f}",
        f"- Expand-only gold in v4 Top10: {cand_avg.get('expand_gold_in_v4', 0):.4f}",
        "",
        f"- Hit sources: `{dict(hit_src)}`",
        "",
        "See also `ablation.md`, `sage_v4_vs_baselines.md`, `failure_cases.md`.",
        "",
    ]
    (out / "evaluation_report.md").write_text("\n".join(report), encoding="utf-8")

    write_ablation(
        out / "ablation.md",
        bm25=bm25_m,
        dense=dense_m,
        hybrid=hybrid_m,
        sage_v2=v2_m,
        sage_v3=v3_m,
        sage_v4=metrics,
        cand_avg=cand_avg,
        hit_src=hit_src,
        n=len(samples),
    )
    write_vs_baselines(
        out / "sage_v4_vs_baselines.md",
        bm25=bm25_m,
        dense=dense_m,
        hybrid=hybrid_m,
        sage_v2=v2_m,
        sage_v3=v3_m,
        sage_v4=metrics,
        cand_avg=cand_avg,
    )
    write_failure_cases(out / "failure_cases.md", case_bank, min_n=10)

    print(
        json.dumps(
            {"metrics": metrics, "candidate_recall": cand_avg, "hit_sources": dict(hit_src)},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Wrote {out}")
    print(f"case_bank size: {len(case_bank)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
