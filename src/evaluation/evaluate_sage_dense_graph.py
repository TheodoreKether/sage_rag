"""Evaluate Dense+Graph reranking experiment on Clean Benchmark.

Ablations:
  A: BM25 + Graph + Dense ranking
  B: SAGE v5 (cached or re-run reference metrics)
  C: Dense-only baseline (cached)
  D: Without Graph (BM25 pool + Dense ranking)
  H: Hybrid BM25+Dense+Graph+Coverage (analysis)

Usage:
  python src/evaluation/evaluate_sage_dense_graph.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

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
from src.sage_rag.ranking.dense_graph_ranker import DenseGraphRanker  # noqa: E402
from src.sage_rag.ranking.dense_rescorer import DenseRescorer  # noqa: E402
from src.sage_rag.ranking.semantic_rescorer import SemanticRescorer  # noqa: E402
from src.sage_rag.retrieval.sage_dense_graph_retriever import (  # noqa: E402
    SageDenseGraphRetriever,
)
from src.sage_rag.retrieval.sage_expansion_retriever import SageExpansionRetriever  # noqa: E402

logger = logging.getLogger(__name__)

QA_CLEAN = ROOT / "data" / "qa_dataset" / "qa_pairs_clean.jsonl"
OUT_DIR = ROOT / "results" / "retrieval" / "sage_dense_graph"
SAGE_V4_METRICS = ROOT / "results" / "retrieval" / "sage_v4" / "metrics.json"
SAGE_V5_METRICS = ROOT / "results" / "retrieval" / "sage_v5" / "metrics.json"
CLEAN_BM25_METRICS = ROOT / "results" / "retrieval" / "clean_benchmark" / "bm25_metrics.json"
CLEAN_DENSE_METRICS = ROOT / "results" / "retrieval" / "clean_benchmark" / "dense_metrics.json"
CLEAN_HYBRID_METRICS = ROOT / "results" / "retrieval" / "clean_benchmark" / "hybrid_metrics.json"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate Dense+Graph experiment")
    p.add_argument("--qa", type=Path, default=QA_CLEAN)
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    p.add_argument("--bm25-index", type=Path, default=ROOT / "data" / "bm25_index")
    p.add_argument("--vector-store", type=Path, default=ROOT / "data" / "vector_store")
    p.add_argument("--graph-dir", type=Path, default=ROOT / "data" / "sage_graph")
    p.add_argument("--initial-k", type=int, default=20)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--pool-size", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.50)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=0.25)
    p.add_argument("--lam", type=float, default=0.20)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-hybrid", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _preview(text: str, n: int = 100) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + "..."


def _load_metrics(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {k: float(v) for k, v in (payload.get("metrics") or {}).items()}


def _unit_payload(unit: Any) -> dict[str, Any]:
    meta = unit.metadata or {}
    return {
        "rank": unit.rank,
        "unit_id": unit.unit_id,
        "final_score": meta.get("final_score", unit.score),
        "dense_score": meta.get("dense_score"),
        "dense_score_raw": meta.get("dense_score_raw"),
        "semantic_score": meta.get("semantic_score"),
        "bm25_score": meta.get("bm25_score"),
        "candidate_source": meta.get("candidate_source"),
        "expansion_relation": meta.get("expansion_relation"),
        "expanded_from": meta.get("expanded_from"),
        "structure_score": meta.get("structure_score"),
        "query_coverage_score": meta.get("query_coverage_score"),
        "risk_penalty": meta.get("risk_penalty"),
        "text": unit.text,
    }


def _run_variant(
    samples: list[QASample],
    *,
    name: str,
    retrieve_fn: Callable[
        [QASample], tuple[list[Any], list[Any], list[str], list[str]]
    ],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    cand_rows: list[dict[str, float]] = []
    cases: list[dict[str, Any]] = []
    rel_promoted: Counter = Counter()
    rel_expand_gold: Counter = Counter()

    for sample in tqdm(samples, desc=name, unit="qa"):
        gold = set(sample.gold_unit_ids)
        final, pool_units, pool_ids, bm25_ids = retrieve_fn(sample)
        final_ids = [u.unit_id for u in final]
        pool_by_id = {u.unit_id: u for u in pool_units}

        metrics = compute_retrieval_metrics(
            final_ids,
            sample.gold_unit_ids,
            recall_ks=(1, 5, 10),
            ndcg_ks=(5, 10),
        )
        metric_rows.append(metrics)

        expand_gold_ids: list[str] = []
        deeper_bm25_gold_ids: list[str] = []
        for uid in sample.gold_unit_ids:
            if uid in set(bm25_ids):
                continue
            if uid not in set(pool_ids):
                continue
            src = (
                (pool_by_id[uid].metadata or {}).get("candidate_source")
                if uid in pool_by_id
                else None
            )
            if src == "expanded":
                expand_gold_ids.append(uid)
                rel = (pool_by_id[uid].metadata or {}).get("expansion_relation") or "?"
                rel_expand_gold[str(rel)] += 1
            elif src == "initial":
                deeper_bm25_gold_ids.append(uid)

        expand_in_final = [uid for uid in expand_gold_ids if uid in set(final_ids)]
        for uid in expand_in_final:
            rel = (pool_by_id[uid].metadata or {}).get("expansion_relation") or "?"
            rel_promoted[str(rel)] += 1

        cand = {
            "bm25_top10_recall": recall_at_k(bm25_ids[:top_k], sample.gold_unit_ids, top_k),
            "expansion_pool_recall": recall_at_k(
                pool_ids, sample.gold_unit_ids, len(pool_ids) or 1
            ),
            "final_top10_recall": metrics.get("Recall@10", 0.0),
            "gold_only_via_expand": 1.0 if expand_gold_ids else 0.0,
            "expand_gold_in_final": 1.0 if expand_in_final else 0.0,
            "n_expand_gold": float(len(expand_gold_ids)),
            "n_expand_gold_promoted": float(len(expand_in_final)),
            "n_deeper_bm25_gold": float(len(deeper_bm25_gold_ids)),
            "n_deeper_bm25_gold_promoted": float(
                sum(1 for u in deeper_bm25_gold_ids if u in set(final_ids))
            ),
        }
        cand_rows.append(cand)

        hit = next((u for u in final if u.unit_id in gold), None)
        hit_meta = (hit.metadata or {}) if hit else {}
        bm25_hit = bool(gold & set(bm25_ids[:top_k]))
        final_hit = bool(gold & set(final_ids))

        if (not bm25_hit) and final_hit:
            cases.append(
                {
                    "case_type": "bm25_miss_dense_graph_hit",
                    "qa_id": sample.qa_id,
                    "question": sample.question,
                    "gold_unit_ids": sample.gold_unit_ids,
                    "expand_gold_ids": expand_gold_ids,
                    "deeper_bm25_gold_ids": deeper_bm25_gold_ids,
                    "hit": _unit_payload(hit) if hit else None,
                    "promoted_graph": [
                        _unit_payload(u)
                        for u in final
                        if (u.metadata or {}).get("candidate_source") == "expanded"
                    ],
                    "top10": [_unit_payload(u) for u in final],
                }
            )

        records.append(
            {
                "qa_id": sample.qa_id,
                "question": sample.question,
                "gold_unit_ids": sample.gold_unit_ids,
                "retrieved_units": [_unit_payload(u) for u in final],
                "retrieved_unit_ids": final_ids,
                "bm25_top10_unit_ids": bm25_ids[:top_k],
                "expansion_pool_unit_ids": pool_ids,
                "candidate_recall": cand,
                "hit_candidate_source": hit_meta.get("candidate_source"),
                "hit_expansion_relation": hit_meta.get("expansion_relation"),
                "hit_dense_score": hit_meta.get("dense_score"),
                **metrics,
            }
        )

    cand_avg = average_metrics(cand_rows)
    cand_avg["rel_expand_gold"] = dict(rel_expand_gold)
    cand_avg["rel_promoted"] = dict(rel_promoted)
    return records, average_metrics(metric_rows), cand_avg, cases


def write_comparison(
    path: Path,
    *,
    bm25: dict[str, float],
    dense: dict[str, float],
    hybrid: dict[str, float],
    sage_v4: dict[str, float],
    sage_v5: dict[str, float],
    dense_graph: dict[str, float],
    hybrid_graph: dict[str, float] | None,
    pool_r: float,
) -> None:
    lines = [
        "# Dense+Graph vs Baselines / SAGE",
        "",
        "| Method | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@5 | nDCG@10 |",
        "|--------|---------:|---------:|----------:|----:|-------:|--------:|",
    ]

    def row(name: str, m: dict[str, float]) -> None:
        lines.append(
            f"| {name} | {m.get('Recall@1', 0):.4f} | {m.get('Recall@5', 0):.4f} | "
            f"{m.get('Recall@10', 0):.4f} | {m.get('MRR', 0):.4f} | "
            f"{m.get('nDCG@5', 0):.4f} | {m.get('nDCG@10', 0):.4f} |"
        )

    row("BM25", bm25)
    row("Dense", dense)
    row("Hybrid", hybrid)
    lines.append(f"| BM25+Graph pool | — | — | {pool_r:.4f} | — | — | — |")
    row("SAGE v4", sage_v4)
    row("SAGE v5 (BM25 semantic)", sage_v5)
    row("**A: Dense+Graph**", dense_graph)
    if hybrid_graph:
        row("H: Hybrid BM25+Dense+Graph", hybrid_graph)
    lines += [
        "",
        "## Deltas (Dense+Graph vs others)",
        "",
        f"- vs BM25 R@10: {(dense_graph.get('Recall@10', 0) - bm25.get('Recall@10', 0)) * 100:+.2f} pp",
        f"- vs Dense R@10: {(dense_graph.get('Recall@10', 0) - dense.get('Recall@10', 0)) * 100:+.2f} pp",
        f"- vs SAGE v5 R@10: {(dense_graph.get('Recall@10', 0) - sage_v5.get('Recall@10', 0)) * 100:+.2f} pp",
        f"- vs SAGE v4 R@10: {(dense_graph.get('Recall@10', 0) - sage_v4.get('Recall@10', 0)) * 100:+.2f} pp",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ablation(
    path: Path,
    *,
    variants: dict[str, dict[str, float]],
    cand: dict[str, dict[str, float]],
    bm25: dict[str, float],
    dense: dict[str, float],
    sage_v5: dict[str, float],
    n: int,
) -> None:
    lines = [
        "# Dense+Graph Ablation",
        "",
        f"n={n}.",
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
    row("C: Dense only baseline", dense)
    row("B: SAGE v5", sage_v5)
    for key, label in [
        ("A_dense_graph", "A: BM25+Graph+Dense ranking"),
        ("D_no_graph", "D: Without Graph (Dense rerank)"),
        ("H_hybrid", "H: Hybrid BM25+Dense+Graph+Cov"),
    ]:
        if key in variants:
            row(label, variants[key])

    a = variants.get("A_dense_graph") or {}
    d = variants.get("D_no_graph") or {}
    ca = cand.get("A_dense_graph") or {}
    cd = cand.get("D_no_graph") or {}
    cv5 = cand.get("B_sage_v5") or {}

    lines += [
        "",
        "## Research questions",
        "",
        "### Q1. Dense 是否帮助 Graph evidence 排序？",
        "",
        f"- A expand-gold promoted: **{ca.get('expand_gold_in_final', 0) * 100:.2f}%** "
        f"of queries (n_gold={ca.get('n_expand_gold', 0):.0f}, "
        f"promoted={ca.get('n_expand_gold_promoted', 0):.0f})",
        f"- SAGE v5 expand-gold promoted (cached/ref): "
        f"{cv5.get('expand_gold_in_final', 0) * 100:.2f}%",
        f"- Relation expand-gold counts A: `{ca.get('rel_expand_gold', {})}`",
        f"- Relation promoted A: `{ca.get('rel_promoted', {})}`",
        f"- A R@10={a.get('Recall@10', 0):.4f} vs v5={sage_v5.get('Recall@10', 0):.4f}",
        "",
        "### Q2. Graph 是否仍提供额外价值？",
        "",
        f"- A R@10={a.get('Recall@10', 0):.4f} vs D (no graph) "
        f"R@10={d.get('Recall@10', 0):.4f} "
        f"(Δ={(a.get('Recall@10', 0) - d.get('Recall@10', 0)) * 100:+.2f} pp)",
        f"- Pool recall A={ca.get('expansion_pool_recall', 0):.4f} vs "
        f"D={cd.get('expansion_pool_recall', 0):.4f}",
        "",
        "### Q3. Dense+Graph 是否优于 BM25+Graph (v5)？",
        "",
        f"- ΔR@10 vs v5: {(a.get('Recall@10', 0) - sage_v5.get('Recall@10', 0)) * 100:+.2f} pp",
        f"- ΔMRR vs v5: {a.get('MRR', 0) - sage_v5.get('MRR', 0):+.4f}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_failure_cases(path: Path, cases: list[dict[str, Any]], *, min_n: int = 5) -> None:
    # Prefer cases where expand gold was promoted
    expand_hits = [c for c in cases if c.get("expand_gold_ids")]
    deeper = [c for c in cases if c.get("deeper_bm25_gold_ids") and not c.get("expand_gold_ids")]
    selected = (expand_hits + deeper)[: max(min_n, 8)]
    lines = [
        "# Dense+Graph Failure / Success Cases",
        "",
        f"Selected **{len(selected)}** cases (from {len(cases)} BM25-miss hits).",
        "",
    ]
    for i, c in enumerate(selected, start=1):
        hit = c.get("hit") or {}
        lines += [
            f"## Case {i}: `{c['qa_id']}`",
            "",
            f"**Query:** {c['question']}",
            "",
            f"**Gold:** `{c['gold_unit_ids']}`",
            f"- expand golds: `{c.get('expand_gold_ids')}`",
            f"- deeper BM25 golds: `{c.get('deeper_bm25_gold_ids')}`",
            "",
            f"- Hit: `{hit.get('unit_id')}` src={hit.get('candidate_source')} "
            f"rel={hit.get('expansion_relation')} dense={hit.get('dense_score')} "
            f"struct={hit.get('structure_score')} cov={hit.get('query_coverage_score')} "
            f"risk={hit.get('risk_penalty')} final={hit.get('final_score')}",
            "",
        ]
        if c.get("promoted_graph"):
            lines.append("### Promoted graph in Top10")
            lines.append("")
            for u in c["promoted_graph"]:
                lines.append(
                    f"- #{u.get('rank')} `{u.get('unit_id')}` "
                    f"rel={u.get('expansion_relation')} dense={u.get('dense_score')} "
                    f"final={u.get('final_score')} — {_preview(str(u.get('text') or ''))}"
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
    dense_rescorer = DenseRescorer(index_dir=args.vector_store)
    bm25_rescorer = SemanticRescorer(bm25)

    ranker_dense = DenseGraphRanker(
        mode="dense",
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        lam=args.lam,
    )
    ranker_hybrid = DenseGraphRanker(mode="hybrid")

    # Keep constructors for API parity / smoke.
    _ = SageDenseGraphRetriever(
        bm25, expander, dense_rescorer=dense_rescorer, mode="dense"
    )

    def retrieve_dense_graph(sample: QASample, *, use_graph: bool):
        bm25_ids = [u.unit_id for u in bm25.retrieve(sample.question, top_k=args.top_k)]
        if use_graph:
            pool = expansion.retrieve(
                sample.question, top_k=args.pool_size, initial_k=args.initial_k
            )
        else:
            hits = bm25.retrieve(sample.question, top_k=args.initial_k)
            pool = [SageExpansionRetriever._mark_initial(u) for u in hits]
        pool_ids = [u.unit_id for u in pool]
        pool = bm25_rescorer.rescore(sample.question, pool)
        pool = dense_rescorer.rescore(sample.question, pool)
        final = ranker_dense.rank(pool, query=sample.question, top_k=args.top_k)
        return final, pool, pool_ids, bm25_ids

    def retrieve_hybrid(sample: QASample):
        bm25_ids = [u.unit_id for u in bm25.retrieve(sample.question, top_k=args.top_k)]
        pool = expansion.retrieve(
            sample.question, top_k=args.pool_size, initial_k=args.initial_k
        )
        pool_ids = [u.unit_id for u in pool]
        pool = bm25_rescorer.rescore(sample.question, pool)
        pool = dense_rescorer.rescore(sample.question, pool)
        final = ranker_hybrid.rank(pool, query=sample.question, top_k=args.top_k)
        return final, pool, pool_ids, bm25_ids

    t0 = time.perf_counter()
    variants: dict[str, dict[str, float]] = {}
    cand_by: dict[str, dict[str, float]] = {}

    rec_a, m_a, c_a, cases_a = _run_variant(
        samples,
        name="A-Dense+Graph",
        retrieve_fn=lambda s: retrieve_dense_graph(s, use_graph=True),
        top_k=args.top_k,
    )
    variants["A_dense_graph"] = m_a
    cand_by["A_dense_graph"] = c_a

    _, m_d, c_d, _ = _run_variant(
        samples,
        name="D-NoGraph-DenseRerank",
        retrieve_fn=lambda s: retrieve_dense_graph(s, use_graph=False),
        top_k=args.top_k,
    )
    variants["D_no_graph"] = m_d
    cand_by["D_no_graph"] = c_d

    m_h: dict[str, float] = {}
    if not args.skip_hybrid:
        _, m_h, c_h, _ = _run_variant(
            samples,
            name="H-Hybrid",
            retrieve_fn=retrieve_hybrid,
            top_k=args.top_k,
        )
        variants["H_hybrid"] = m_h
        cand_by["H_hybrid"] = c_h

    elapsed = time.perf_counter() - t0
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    bm25_m = _load_metrics(CLEAN_BM25_METRICS)
    dense_m = _load_metrics(CLEAN_DENSE_METRICS)
    hybrid_m = _load_metrics(CLEAN_HYBRID_METRICS)
    v4_m = _load_metrics(SAGE_V4_METRICS)
    v5_m = _load_metrics(SAGE_V5_METRICS)
    # Attach v5 promotion stats from cached metrics if present.
    v5_payload = {}
    if SAGE_V5_METRICS.is_file():
        v5_payload = json.loads(SAGE_V5_METRICS.read_text(encoding="utf-8"))
    v5_promo = v5_payload.get("expanded_gold_promotion") or {}
    cand_by["B_sage_v5"] = {
        "expand_gold_in_final": float(
            v5_promo.get("promotion_rate_among_expand_only_queries") or 0.0
        ),
        "n_expand_gold": float(v5_promo.get("total_expand_only_golds") or 0.0),
        "n_expand_gold_promoted": float(
            v5_promo.get("total_promoted_expand_golds") or 0.0
        ),
        "expansion_pool_recall": float(
            (v5_payload.get("candidate_recall") or {}).get("expansion_pool_recall") or 0.0
        ),
    }

    hit_src = Counter(
        (r.get("hit_candidate_source") or "none")
        for r in rec_a
        if float(r.get("Recall@10") or 0) >= 1.0
    )

    n_expand_q = sum(
        1 for r in rec_a if (r.get("candidate_recall") or {}).get("gold_only_via_expand")
    )
    n_promoted_q = sum(
        1 for r in rec_a if (r.get("candidate_recall") or {}).get("expand_gold_in_final")
    )
    n_expand_gold = sum(
        float((r.get("candidate_recall") or {}).get("n_expand_gold") or 0) for r in rec_a
    )
    n_promoted_gold = sum(
        float((r.get("candidate_recall") or {}).get("n_expand_gold_promoted") or 0)
        for r in rec_a
    )

    meta = {
        "retriever": "Dense+Graph (BM25 generate + Dense structure ranking)",
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
        "metrics": m_a,
        "metrics_by_ablation": variants,
        "candidate_recall": c_a,
        "candidate_recall_by_ablation": {
            k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}
            for k, v in cand_by.items()
        },
        "expanded_gold_promotion": {
            "expand_only_gold_queries": n_expand_q,
            "queries_with_promoted_expand_gold": n_promoted_q,
            "promotion_rate_among_expand_only_queries": (
                n_promoted_q / n_expand_q if n_expand_q else 0.0
            ),
            "total_expand_only_golds": n_expand_gold,
            "total_promoted_expand_golds": n_promoted_gold,
            "gold_promotion_rate": (
                n_promoted_gold / n_expand_gold if n_expand_gold else 0.0
            ),
            "rel_expand_gold": c_a.get("rel_expand_gold"),
            "rel_promoted": c_a.get("rel_promoted"),
            "sage_v5_promotion_rate": v5_promo.get(
                "promotion_rate_among_expand_only_queries"
            ),
            "sage_v5_promoted_golds": v5_promo.get("total_promoted_expand_golds"),
        },
        "hit_sources": dict(hit_src),
    }
    (out / "metrics.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out / "retrieval_results.jsonl").open("w", encoding="utf-8") as fh:
        for rec in rec_a:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = [
        "# Dense+Graph Evaluation Report",
        "",
        f"- QA: `{args.qa}` ({len(samples)} queries)",
        "- Pipeline: BM25(initial_k=20) → Graph Expansion → Dense rescore "
        "→ Structure/Risk ranking",
        f"- Score = α·Dense + β·Graph + γ·Coverage − λ·Risk "
        f"(α={args.alpha}, β={args.beta}, γ={args.gamma}, λ={args.lam})",
        f"- elapsed: {elapsed:.2f}s",
        "",
        "## Metrics (A)",
        "",
        "| Metric | Value |",
        "|--------|------:|",
    ]
    for k in ("Recall@1", "Recall@5", "Recall@10", "MRR", "nDCG@5", "nDCG@10"):
        report.append(f"| {k} | {m_a.get(k, 0):.4f} |")
    report += [
        "",
        "## Expanded gold promotion",
        "",
        f"- Expand-only gold queries: {n_expand_q}",
        f"- Promoted queries: {n_promoted_q} "
        f"({(n_promoted_q / n_expand_q * 100) if n_expand_q else 0:.1f}%)",
        f"- Expand-only golds: {n_expand_gold:.0f} → promoted {n_promoted_gold:.0f} "
        f"({(n_promoted_gold / n_expand_gold * 100) if n_expand_gold else 0:.1f}%)",
        f"- By relation (pool golds): `{c_a.get('rel_expand_gold')}`",
        f"- By relation (promoted): `{c_a.get('rel_promoted')}`",
        f"- SAGE v5 promotion rate (ref): "
        f"{v5_promo.get('gold_promotion_rate', v5_promo.get('promotion_rate_among_expand_only_queries'))}",
        "",
        f"- Hit sources: `{dict(hit_src)}`",
        "",
        "See `comparison.md`, ablation in report folder, `failure_cases.md`.",
        "",
    ]
    (out / "evaluation_report.md").write_text("\n".join(report), encoding="utf-8")

    write_comparison(
        out / "comparison.md",
        bm25=bm25_m,
        dense=dense_m,
        hybrid=hybrid_m,
        sage_v4=v4_m,
        sage_v5=v5_m,
        dense_graph=m_a,
        hybrid_graph=m_h or None,
        pool_r=float(c_a.get("expansion_pool_recall", 0.0)),
    )
    write_ablation(
        out / "ablation.md",
        variants=variants,
        cand=cand_by,
        bm25=bm25_m,
        dense=dense_m,
        sage_v5=v5_m,
        n=len(samples),
    )
    # Also write ablation.md name expected; user asked comparison.md + failure_cases
    write_failure_cases(out / "failure_cases.md", cases_a, min_n=5)

    print(
        json.dumps(
            {
                "metrics": m_a,
                "ablation": variants,
                "promotion": meta["expanded_gold_promotion"],
                "hit_sources": dict(hit_src),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
