"""Evaluate SAGE-RAG v2 (greedy Structure-aware Evidence Selection) on Clean Benchmark.

Does not modify baseline retrievers, shared metrics, or SAGE v1 modules.

Usage:
  python src/evaluation/evaluate_sage_v2.py
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
from src.sage_rag.retrieval.sage_expansion_retriever import SageExpansionRetriever  # noqa: E402

logger = logging.getLogger(__name__)

QA_CLEAN = ROOT / "data" / "qa_dataset" / "qa_pairs_clean.jsonl"
OUT_DIR = ROOT / "results" / "retrieval" / "sage_v2"
SAGE_V1_METRICS = ROOT / "results" / "retrieval" / "sage_initial10" / "metrics.json"
CLEAN_BM25_METRICS = ROOT / "results" / "retrieval" / "clean_benchmark" / "bm25_metrics.json"
CLEAN_DENSE_METRICS = ROOT / "results" / "retrieval" / "clean_benchmark" / "dense_metrics.json"
CLEAN_HYBRID_METRICS = ROOT / "results" / "retrieval" / "clean_benchmark" / "hybrid_metrics.json"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate SAGE-RAG v2 on Clean Benchmark")
    p.add_argument("--qa", type=Path, default=QA_CLEAN)
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    p.add_argument("--bm25-index", type=Path, default=ROOT / "data" / "bm25_index")
    p.add_argument("--graph-dir", type=Path, default=ROOT / "data" / "sage_graph")
    p.add_argument("--initial-k", type=int, default=10)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--pool-size", type=int, default=80)
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
    ranker: StructureRankerV2,
    bm25: BM25Retriever,
    initial_k: int,
    top_k: int,
    pool_size: int,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float]]:
    records: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    cand_rows: list[dict[str, float]] = []

    for sample in tqdm(samples, desc="SAGE-RAG-v2", unit="qa"):
        bm25_hits = bm25.retrieve(sample.question, top_k=top_k)
        bm25_ids = [u.unit_id for u in bm25_hits]

        pool = expansion.retrieve(
            sample.question, top_k=pool_size, initial_k=initial_k
        )
        pool_ids = [u.unit_id for u in pool]
        final = ranker.rank(pool, query=sample.question, top_k=top_k)
        final_ids = [u.unit_id for u in final]

        metrics = compute_retrieval_metrics(
            final_ids,
            sample.gold_unit_ids,
            recall_ks=(1, 5, 10),
            ndcg_ks=(5, 10),
        )
        metric_rows.append(metrics)

        cand = {
            "bm25_top10_recall": recall_at_k(bm25_ids, sample.gold_unit_ids, top_k),
            "expansion_pool_recall": recall_at_k(
                pool_ids, sample.gold_unit_ids, len(pool_ids) or 1
            ),
            "sage_v2_top10_recall": metrics.get("Recall@10", 0.0),
        }
        cand_rows.append(cand)

        gold = set(sample.gold_unit_ids)
        hit = next((u for u in final if u.unit_id in gold), None)
        hit_meta = (hit.metadata or {}) if hit else {}

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
                "hit_query_coverage": hit_meta.get("query_coverage_score"),
                **metrics,
            }
        )

    return records, average_metrics(metric_rows), average_metrics(cand_rows)


def write_ablation(
    path: Path,
    *,
    sage_v2: dict[str, float],
    sage_v1: dict[str, float],
    bm25: dict[str, float],
    dense: dict[str, float],
    hybrid: dict[str, float],
    cand_avg: dict[str, float],
    hit_src: Counter,
    n: int,
) -> None:
    pool_r = cand_avg.get("expansion_pool_recall", 0.0)
    bm25_r = cand_avg.get("bm25_top10_recall", bm25.get("Recall@10", 0.0))
    lines = [
        "# SAGE-RAG v2 Ablation (Clean Benchmark)",
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
    # Candidate-only stage (coverage upper bound from expansion).
    lines.append(
        f"| BM25 + Graph Expansion (candidate recall) | — | — | "
        f"{pool_r:.4f} | — | — |"
    )
    row("SAGE v1 (initial_k=10)", sage_v1)
    row("SAGE v2 (greedy selection)", sage_v2)

    lines += [
        "",
        "## Candidate coverage",
        "",
        f"- BM25 Top10 recall: **{bm25_r:.4f}**",
        f"- Expansion pool recall: **{pool_r:.4f}** "
        f"(+{(pool_r - bm25_r) * 100:.2f} pp)",
        f"- SAGE v1 final R@10: **{sage_v1.get('Recall@10', 0):.4f}**",
        f"- SAGE v2 final R@10: **{sage_v2.get('Recall@10', 0):.4f}**",
        "",
        "## Did v2 utilize graph-found evidence?",
        "",
        f"- SAGE v2 hit sources among R@10 successes: `{dict(hit_src)}`",
        "",
    ]
    v2_r = sage_v2.get("Recall@10", 0.0)
    v1_r = sage_v1.get("Recall@10", 0.0)
    if v2_r > v1_r + 1e-9:
        lines.append(
            f"- **Yes (metric lift):** v2 improves R@10 by "
            f"**{(v2_r - v1_r) * 100:+.2f} pp** over v1."
        )
    elif hit_src.get("expanded", 0) > 0:
        lines.append(
            "- **Partially:** v2 places some expanded evidence in Top10 "
            "(see hit sources), even if macro R@10 is flat vs v1/BM25."
        )
    else:
        lines.append(
            "- **Not yet on R@10:** expanded hits are rare in final Top10; "
            "pool still shows recoverable gold — selection weights may need tuning."
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
    ranker = StructureRankerV2(
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, lam=args.lam
    )

    t0 = time.perf_counter()
    records, metrics, cand_avg = evaluate(
        samples,
        expansion=expansion,
        ranker=ranker,
        bm25=bm25,
        initial_k=args.initial_k,
        top_k=args.top_k,
        pool_size=args.pool_size,
    )
    elapsed = time.perf_counter() - t0

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "retriever": "SAGE-RAG v2 (BM25 + Expansion + Greedy Selection)",
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
    }
    (out / "metrics.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out / "retrieval_results.jsonl").open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = [
        "# SAGE-RAG v2 Evaluation Report",
        "",
        f"- QA: `{args.qa}` ({len(samples)} queries)",
        f"- initial_k={args.initial_k}, top_k={args.top_k}, pool_size={args.pool_size}",
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
        f"- SAGE v2 Top10: {cand_avg.get('sage_v2_top10_recall', 0):.4f}",
        "",
        "See also `ablation.md`.",
        "",
    ]
    (out / "evaluation_report.md").write_text("\n".join(report), encoding="utf-8")

    hit_src = Counter(
        (r.get("hit_candidate_source") or "none")
        for r in records
        if float(r.get("Recall@10") or 0) >= 1.0
    )
    write_ablation(
        out / "ablation.md",
        sage_v2=metrics,
        sage_v1=_load_metrics(SAGE_V1_METRICS),
        bm25=_load_metrics(CLEAN_BM25_METRICS),
        dense=_load_metrics(CLEAN_DENSE_METRICS),
        hybrid=_load_metrics(CLEAN_HYBRID_METRICS),
        cand_avg=cand_avg,
        hit_src=hit_src,
        n=len(samples),
    )

    print(json.dumps({"metrics": metrics, "candidate_recall": cand_avg}, ensure_ascii=False, indent=2))
    print(f"Wrote {out}")
    print(f"hit sources: {dict(hit_src)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
