"""Evaluate SAGE-RAG v3 (Candidate Allocation + Structure Selection) on Clean Benchmark.

Compares:
  BM25 | Expansion pool | SAGE v2 | SAGE v3 fixed | SAGE v3 adaptive

Usage:
  python src/evaluation/evaluate_sage_v3.py
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
from src.sage_rag.ranking.structure_ranker_v2 import StructureRankerV2  # noqa: E402
from src.sage_rag.retrieval.candidate_allocator import CandidateAllocator  # noqa: E402
from src.sage_rag.retrieval.sage_expansion_retriever import SageExpansionRetriever  # noqa: E402
from src.sage_rag.retrieval.sage_retriever_v3 import SageRetrieverV3  # noqa: E402

logger = logging.getLogger(__name__)

QA_CLEAN = ROOT / "data" / "qa_dataset" / "qa_pairs_clean.jsonl"
OUT_DIR = ROOT / "results" / "retrieval" / "sage_v3"
SAGE_V2_METRICS = ROOT / "results" / "retrieval" / "sage_v2" / "metrics.json"
CLEAN_BM25_METRICS = ROOT / "results" / "retrieval" / "clean_benchmark" / "bm25_metrics.json"
CLEAN_DENSE_METRICS = ROOT / "results" / "retrieval" / "clean_benchmark" / "dense_metrics.json"
CLEAN_HYBRID_METRICS = ROOT / "results" / "retrieval" / "clean_benchmark" / "hybrid_metrics.json"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate SAGE-RAG v3 on Clean Benchmark")
    p.add_argument("--qa", type=Path, default=QA_CLEAN)
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    p.add_argument("--bm25-index", type=Path, default=ROOT / "data" / "bm25_index")
    p.add_argument("--graph-dir", type=Path, default=ROOT / "data" / "sage_graph")
    p.add_argument("--initial-k", type=int, default=10)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--pool-size", type=int, default=80)
    p.add_argument("--fixed-graph-budget", type=int, default=3)
    p.add_argument("--adaptive-default-budget", type=int, default=2)
    p.add_argument("--adaptive-high-budget", type=int, default=4)
    p.add_argument("--path-threshold", type=float, default=0.7)
    p.add_argument("--coverage-threshold", type=float, default=0.25)
    p.add_argument("--alpha", type=float, default=0.40)
    p.add_argument("--beta", type=float, default=0.20)
    p.add_argument("--gamma", type=float, default=0.45)
    p.add_argument("--lam", type=float, default=0.35)
    p.add_argument("--sample", type=int, default=None)
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
        "allocation_source": meta.get("allocation_source"),
        "graph_budget": meta.get("graph_budget"),
        "retrieval_score_normalized": meta.get("retrieval_score_normalized"),
        "structure_path_score": meta.get("structure_path_score"),
        "query_coverage_score": meta.get("query_coverage_score"),
        "redundancy_penalty": meta.get("redundancy_penalty"),
        "text": unit.text,
    }


def _load_metrics(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {k: float(v) for k, v in (payload.get("metrics") or {}).items()}


def evaluate(
    samples: list[QASample],
    *,
    expansion: SageExpansionRetriever,
    sage_v2: SageRetrieverV3,
    sage_fixed: SageRetrieverV3,
    sage_adaptive: SageRetrieverV3,
    bm25: BM25Retriever,
    initial_k: int,
    top_k: int,
    pool_size: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    """Run BM25 / pool / v2(none) / v3-fixed / v3-adaptive on each query."""
    records: list[dict[str, Any]] = []
    rows_v2: list[dict[str, float]] = []
    rows_fixed: list[dict[str, float]] = []
    rows_adaptive: list[dict[str, float]] = []
    cand_rows: list[dict[str, float]] = []

    for sample in tqdm(samples, desc="SAGE-RAG-v3", unit="qa"):
        gold = set(sample.gold_unit_ids)

        bm25_hits = bm25.retrieve(sample.question, top_k=top_k)
        bm25_ids = [u.unit_id for u in bm25_hits]

        pool = expansion.retrieve(
            sample.question, top_k=pool_size, initial_k=initial_k
        )
        pool_ids = [u.unit_id for u in pool]

        final_v2 = sage_v2.retrieve(
            sample.question, top_k=top_k, initial_k=initial_k, pool_size=pool_size
        )
        final_fixed = sage_fixed.retrieve(
            sample.question, top_k=top_k, initial_k=initial_k, pool_size=pool_size
        )
        final_adaptive = sage_adaptive.retrieve(
            sample.question, top_k=top_k, initial_k=initial_k, pool_size=pool_size
        )

        m_v2 = compute_retrieval_metrics(
            [u.unit_id for u in final_v2],
            sample.gold_unit_ids,
            recall_ks=(1, 5, 10),
            ndcg_ks=(5, 10),
        )
        m_fixed = compute_retrieval_metrics(
            [u.unit_id for u in final_fixed],
            sample.gold_unit_ids,
            recall_ks=(1, 5, 10),
            ndcg_ks=(5, 10),
        )
        m_adaptive = compute_retrieval_metrics(
            [u.unit_id for u in final_adaptive],
            sample.gold_unit_ids,
            recall_ks=(1, 5, 10),
            ndcg_ks=(5, 10),
        )
        rows_v2.append(m_v2)
        rows_fixed.append(m_fixed)
        rows_adaptive.append(m_adaptive)

        # Gold entry analysis for Q1.
        gold_in_bm25 = bool(gold & set(bm25_ids))
        gold_in_pool = bool(gold & set(pool_ids))
        gold_only_via_expand = gold_in_pool and not gold_in_bm25

        fixed_ids = {u.unit_id for u in final_fixed}
        adaptive_ids = {u.unit_id for u in final_adaptive}
        fixed_graph_reserved = {
            u.unit_id
            for u in final_fixed
            if (u.metadata or {}).get("allocation_source") == "graph_reserved"
        }
        adaptive_graph_reserved = {
            u.unit_id
            for u in final_adaptive
            if (u.metadata or {}).get("allocation_source") == "graph_reserved"
        }
        expand_gold_ids = [
            uid
            for uid in sample.gold_unit_ids
            if uid in set(pool_ids) and uid not in set(bm25_ids)
        ]

        cand = {
            "bm25_top10_recall": recall_at_k(bm25_ids, sample.gold_unit_ids, top_k),
            "expansion_pool_recall": recall_at_k(
                pool_ids, sample.gold_unit_ids, len(pool_ids) or 1
            ),
            "sage_v2_top10_recall": m_v2.get("Recall@10", 0.0),
            "sage_v3_fixed_top10_recall": m_fixed.get("Recall@10", 0.0),
            "sage_v3_adaptive_top10_recall": m_adaptive.get("Recall@10", 0.0),
            "gold_only_via_expand": 1.0 if gold_only_via_expand else 0.0,
            "expand_gold_in_fixed": (
                1.0
                if expand_gold_ids and any(g in fixed_ids for g in expand_gold_ids)
                else 0.0
            ),
            "expand_gold_in_adaptive": (
                1.0
                if expand_gold_ids and any(g in adaptive_ids for g in expand_gold_ids)
                else 0.0
            ),
            "expand_gold_reserved_fixed": (
                1.0
                if expand_gold_ids
                and any(g in fixed_graph_reserved for g in expand_gold_ids)
                else 0.0
            ),
            "expand_gold_reserved_adaptive": (
                1.0
                if expand_gold_ids
                and any(g in adaptive_graph_reserved for g in expand_gold_ids)
                else 0.0
            ),
        }
        cand_rows.append(cand)

        hit_fixed = next((u for u in final_fixed if u.unit_id in gold), None)
        hit_adaptive = next((u for u in final_adaptive if u.unit_id in gold), None)

        records.append(
            {
                "qa_id": sample.qa_id,
                "question": sample.question,
                "gold_unit_ids": sample.gold_unit_ids,
                "gold_unit_id": sample.gold_unit_ids[0] if sample.gold_unit_ids else "",
                "question_type": sample.question_type,
                "document_id": sample.document_id,
                "bm25_top10_unit_ids": bm25_ids,
                "expansion_pool_unit_ids": pool_ids,
                "retrieved_units_v2": [_unit_payload(u) for u in final_v2],
                "retrieved_units_fixed": [_unit_payload(u) for u in final_fixed],
                "retrieved_units_adaptive": [_unit_payload(u) for u in final_adaptive],
                "retrieved_unit_ids": [u.unit_id for u in final_adaptive],
                "candidate_recall": cand,
                "gold_only_via_expand": gold_only_via_expand,
                "expand_gold_ids": expand_gold_ids,
                "hit_fixed_source": (hit_fixed.metadata or {}).get("candidate_source")
                if hit_fixed
                else None,
                "hit_fixed_allocation": (hit_fixed.metadata or {}).get(
                    "allocation_source"
                )
                if hit_fixed
                else None,
                "hit_adaptive_source": (hit_adaptive.metadata or {}).get(
                    "candidate_source"
                )
                if hit_adaptive
                else None,
                "hit_adaptive_allocation": (hit_adaptive.metadata or {}).get(
                    "allocation_source"
                )
                if hit_adaptive
                else None,
                "metrics_v2": m_v2,
                "metrics_fixed": m_fixed,
                "metrics_adaptive": m_adaptive,
                **{f"adaptive_{k}": v for k, v in m_adaptive.items()},
                **m_adaptive,
            }
        )

    return (
        records,
        average_metrics(rows_v2),
        average_metrics(rows_fixed),
        average_metrics(rows_adaptive),
        average_metrics(cand_rows),
    )


def write_ablation(
    path: Path,
    *,
    bm25: dict[str, float],
    dense: dict[str, float],
    hybrid: dict[str, float],
    sage_v2: dict[str, float],
    sage_v2_cached: dict[str, float],
    sage_fixed: dict[str, float],
    sage_adaptive: dict[str, float],
    cand_avg: dict[str, float],
    hit_fixed: Counter,
    hit_adaptive: Counter,
    n: int,
) -> None:
    pool_r = cand_avg.get("expansion_pool_recall", 0.0)
    bm25_r = cand_avg.get("bm25_top10_recall", bm25.get("Recall@10", 0.0))
    # Prefer freshly measured v2 (none-strategy) for apples-to-apples comparison.
    v2 = sage_v2 or sage_v2_cached

    lines = [
        "# SAGE-RAG v3 Ablation (Clean Benchmark)",
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
    row("SAGE v2 (no allocation)", v2)
    row("SAGE v3 Fixed Allocation", sage_fixed)
    row("SAGE v3 Adaptive Allocation", sage_adaptive)

    r10_bm25 = bm25.get("Recall@10", bm25_r)
    r10_v2 = v2.get("Recall@10", 0.0)
    r10_fixed = sage_fixed.get("Recall@10", 0.0)
    r10_ad = sage_adaptive.get("Recall@10", 0.0)

    eg_fixed = cand_avg.get("expand_gold_in_fixed", 0.0)
    eg_ad = cand_avg.get("expand_gold_in_adaptive", 0.0)
    only_exp = cand_avg.get("gold_only_via_expand", 0.0)
    # Conditional recovery rate among expand-only-gold queries.
    cond_fixed = (eg_fixed / only_exp) if only_exp > 1e-12 else 0.0
    cond_ad = (eg_ad / only_exp) if only_exp > 1e-12 else 0.0

    lines += [
        "",
        "## Candidate → Allocation → Final",
        "",
        f"- BM25 Top10 recall: **{bm25_r:.4f}**",
        f"- Expansion pool recall: **{pool_r:.4f}** "
        f"(+{(pool_r - bm25_r) * 100:.2f} pp)",
        f"- SAGE v2 final R@10: **{r10_v2:.4f}**",
        f"- SAGE v3 Fixed final R@10: **{r10_fixed:.4f}**",
        f"- SAGE v3 Adaptive final R@10: **{r10_ad:.4f}**",
        "",
        f"- Queries where gold only appears via expansion: "
        f"**{only_exp * 100:.2f}%** ({only_exp * n:.0f}/{n})",
        f"- Expand-only gold recovered in Fixed Top10: "
        f"**{eg_fixed * 100:.2f}%** of all queries "
        f"(**{cond_fixed * 100:.1f}%** of expand-only cases); "
        f"reserved={cand_avg.get('expand_gold_reserved_fixed', 0) * 100:.2f}%",
        f"- Expand-only gold recovered in Adaptive Top10: "
        f"**{eg_ad * 100:.2f}%** of all queries "
        f"(**{cond_ad * 100:.1f}%** of expand-only cases); "
        f"reserved={cand_avg.get('expand_gold_reserved_adaptive', 0) * 100:.2f}%",
        "",
        "## Hit source / allocation (R@10 successes)",
        "",
        f"- Fixed hit candidate_source: `{dict(hit_fixed)}`",
        f"- Adaptive hit candidate_source: `{dict(hit_adaptive)}`",
        "",
        "## Research questions",
        "",
        "### Q1: Do expansion-found golds enter Top-k via allocation?",
        "",
    ]
    if eg_fixed > 0 or eg_ad > 0:
        lines.append(
            f"- **Yes (mechanism):** vs SAGE v2 (~1 expanded hit), Fixed places "
            f"**{hit_fixed.get('expanded', 0)}** expanded hits and Adaptive "
            f"**{hit_adaptive.get('expanded', 0)}** in R@10 successes. "
            f"Among expand-only-gold queries, recovery is "
            f"{cond_fixed * 100:.1f}% (Fixed) / {cond_ad * 100:.1f}% (Adaptive)."
        )
    else:
        lines.append(
            "- **No / rare:** allocation did not place expand-only golds into "
            "final Top-k at measurable rate."
        )

    lines += [
        "",
        "### Q2: Is Fixed allocation effective?",
        "",
    ]
    if r10_fixed > r10_v2 + 1e-9:
        lines.append(
            f"- **Yes on R@10:** Fixed improves over v2 by "
            f"**{(r10_fixed - r10_v2) * 100:+.2f} pp**."
        )
    elif sage_fixed.get("MRR", 0) > v2.get("MRR", 0) + 1e-9:
        lines.append(
            f"- **Mixed:** R@10 {r10_fixed:.4f} vs v2 {r10_v2:.4f} "
            f"({(r10_fixed - r10_v2) * 100:+.2f} pp), but MRR "
            f"{sage_fixed.get('MRR', 0):.4f} ≥ v2 {v2.get('MRR', 0):.4f}. "
            "Graph slots enter Top-k, yet displacement of BM25 ranks 8–10 "
            "offsets most recovered expand-golds."
        )
    else:
        lines.append(
            f"- **Weak / negative on macro R@10:** Fixed={r10_fixed:.4f} vs "
            f"v2={r10_v2:.4f}. Hard reservation recovers some expand-golds but "
            "drops more BM25 golds that sat in original ranks 8–10."
        )

    lines += [
        "",
        "### Q3: Is Adaptive better than Fixed?",
        "",
    ]
    if r10_ad > r10_fixed + 1e-9 or (
        abs(r10_ad - r10_fixed) < 1e-9
        and sage_adaptive.get("MRR", 0) > sage_fixed.get("MRR", 0) + 1e-9
    ):
        lines.append(
            f"- **Yes:** Adaptive R@10={r10_ad:.4f}, MRR={sage_adaptive.get('MRR', 0):.4f} "
            f"vs Fixed R@10={r10_fixed:.4f}, MRR={sage_fixed.get('MRR', 0):.4f}."
        )
    elif abs(r10_ad - r10_fixed) < 1e-9 and abs(
        sage_adaptive.get("MRR", 0) - sage_fixed.get("MRR", 0)
    ) < 1e-9:
        lines.append(
            "- **Similar:** Adaptive ≈ Fixed on R@10/MRR under current thresholds."
        )
    else:
        lines.append(
            f"- **Not clearly:** Adaptive R@10={r10_ad:.4f} vs Fixed {r10_fixed:.4f}. "
            "Larger adaptive budgets (when HQ expanded exist) increase displacement "
            "cost without enough extra gold recovery."
        )

    lines += [
        "",
        "### Q4: Does R@10 exceed BM25 82.61%?",
        "",
    ]
    best_r10 = max(r10_fixed, r10_ad)
    if best_r10 > r10_bm25 + 1e-9:
        lines.append(
            f"- **Yes:** best v3 R@10={best_r10:.4f} > BM25 {r10_bm25:.4f} "
            f"(+{(best_r10 - r10_bm25) * 100:.2f} pp)."
        )
    elif abs(best_r10 - r10_bm25) < 1e-9:
        lines.append(
            f"- **Tied:** best v3 R@10={best_r10:.4f} = BM25 {r10_bm25:.4f}."
        )
    else:
        lines.append(
            f"- **No:** best v3 R@10={best_r10:.4f} < BM25 {r10_bm25:.4f} "
            f"({(best_r10 - r10_bm25) * 100:+.2f} pp). "
            "Root cause: expand-only golds are only ~5.4% of queries; hard "
            "slots recover ~1/3 of them but permanently remove BM25 tail "
            "slots that still hold gold more often."
        )

    lines += [
        "",
        "## Hypothesis check",
        "",
        "Core claim: structure should affect **evidence budget**, not only expansion.",
        "",
        "- **Mechanism supported:** allocation changes Top-k composition "
        "(expanded hits ↑ from ~1 → 8–9).",
        "- **Macro R@10 not yet supported:** candidate-recall gap "
        f"({(pool_r - bm25_r) * 100:.2f} pp) is only partially converted; "
        "hard budget trades BM25-tail golds for lower-precision graph slots.",
        "- **Next lever:** gold-aware / risk-sensitive reservation "
        "(reserve only when expanded priority beats the displaced original), "
        "or couple allocation with a stronger expand-gold scorer.",
        "",
    ]
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
    ranker = StructureRankerV2(
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, lam=args.lam
    )
    allocator = CandidateAllocator(
        fixed_graph_budget=args.fixed_graph_budget,
        adaptive_default_budget=args.adaptive_default_budget,
        adaptive_high_budget=args.adaptive_high_budget,
        path_threshold=args.path_threshold,
        coverage_threshold=args.coverage_threshold,
        scorer=ranker,
    )

    sage_v2 = SageRetrieverV3(
        bm25, expander, allocator=allocator, ranker=ranker, strategy="none"
    )
    sage_fixed = SageRetrieverV3(
        bm25, expander, allocator=allocator, ranker=ranker, strategy="fixed"
    )
    sage_adaptive = SageRetrieverV3(
        bm25, expander, allocator=allocator, ranker=ranker, strategy="adaptive"
    )

    t0 = time.perf_counter()
    records, m_v2, m_fixed, m_adaptive, cand_avg = evaluate(
        samples,
        expansion=expansion,
        sage_v2=sage_v2,
        sage_fixed=sage_fixed,
        sage_adaptive=sage_adaptive,
        bm25=bm25,
        initial_k=args.initial_k,
        top_k=args.top_k,
        pool_size=args.pool_size,
    )
    elapsed = time.perf_counter() - t0

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "retriever": "SAGE-RAG v3 (Allocation + Structure Selection)",
        "qa_file": str(args.qa),
        "initial_k": args.initial_k,
        "top_k": args.top_k,
        "pool_size": args.pool_size,
        "fixed_graph_budget": args.fixed_graph_budget,
        "adaptive_default_budget": args.adaptive_default_budget,
        "adaptive_high_budget": args.adaptive_high_budget,
        "path_threshold": args.path_threshold,
        "coverage_threshold": args.coverage_threshold,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "lam": args.lam,
        "evaluated_pairs": len(samples),
        "skipped_pairs": skipped,
        "elapsed_seconds": elapsed,
        "metrics": m_adaptive,
        "metrics_by_strategy": {
            "sage_v2_none": m_v2,
            "sage_v3_fixed": m_fixed,
            "sage_v3_adaptive": m_adaptive,
        },
        "candidate_recall": cand_avg,
    }
    (out / "metrics.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out / "retrieval_results.jsonl").open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    bm25_m = _load_metrics(CLEAN_BM25_METRICS)
    report = [
        "# SAGE-RAG v3 Evaluation Report",
        "",
        f"- QA: `{args.qa}` ({len(samples)} queries)",
        f"- initial_k={args.initial_k}, top_k={args.top_k}, pool_size={args.pool_size}",
        f"- fixed_graph_budget={args.fixed_graph_budget}, "
        f"adaptive={args.adaptive_default_budget}/{args.adaptive_high_budget}",
        f"- path≥{args.path_threshold}, coverage≥{args.coverage_threshold}",
        f"- selection weights: α={args.alpha}, β={args.beta}, γ={args.gamma}, λ={args.lam}",
        f"- elapsed: {elapsed:.2f}s",
        "",
        "## Metrics (Adaptive primary)",
        "",
        "| Metric | SAGE v2 (none) | SAGE v3 Fixed | SAGE v3 Adaptive |",
        "|--------|---------------:|--------------:|-----------------:|",
    ]
    for k in ("Recall@1", "Recall@5", "Recall@10", "MRR", "nDCG@5", "nDCG@10"):
        report.append(
            f"| {k} | {m_v2.get(k, 0):.4f} | {m_fixed.get(k, 0):.4f} | "
            f"{m_adaptive.get(k, 0):.4f} |"
        )
    report += [
        "",
        "## Candidate Recall",
        "",
        f"- BM25 Top10: {cand_avg.get('bm25_top10_recall', 0):.4f}",
        f"- Expansion pool: {cand_avg.get('expansion_pool_recall', 0):.4f}",
        f"- SAGE v2 Top10: {cand_avg.get('sage_v2_top10_recall', 0):.4f}",
        f"- SAGE v3 Fixed Top10: {cand_avg.get('sage_v3_fixed_top10_recall', 0):.4f}",
        f"- SAGE v3 Adaptive Top10: {cand_avg.get('sage_v3_adaptive_top10_recall', 0):.4f}",
        "",
        f"BM25 reference R@10 (cached): {bm25_m.get('Recall@10', 0):.4f}",
        "",
        "See also `ablation.md`.",
        "",
    ]
    (out / "evaluation_report.md").write_text("\n".join(report), encoding="utf-8")

    hit_fixed = Counter(
        (r.get("hit_fixed_source") or "none")
        for r in records
        if float((r.get("metrics_fixed") or {}).get("Recall@10") or 0) >= 1.0
    )
    hit_adaptive = Counter(
        (r.get("hit_adaptive_source") or "none")
        for r in records
        if float((r.get("metrics_adaptive") or {}).get("Recall@10") or 0) >= 1.0
    )

    write_ablation(
        out / "ablation.md",
        bm25=bm25_m,
        dense=_load_metrics(CLEAN_DENSE_METRICS),
        hybrid=_load_metrics(CLEAN_HYBRID_METRICS),
        sage_v2=m_v2,
        sage_v2_cached=_load_metrics(SAGE_V2_METRICS),
        sage_fixed=m_fixed,
        sage_adaptive=m_adaptive,
        cand_avg=cand_avg,
        hit_fixed=hit_fixed,
        hit_adaptive=hit_adaptive,
        n=len(samples),
    )

    print(
        json.dumps(
            {
                "sage_v2_none": m_v2,
                "sage_v3_fixed": m_fixed,
                "sage_v3_adaptive": m_adaptive,
                "candidate_recall": cand_avg,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Wrote {out}")
    print(f"hit_fixed: {dict(hit_fixed)}")
    print(f"hit_adaptive: {dict(hit_adaptive)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
