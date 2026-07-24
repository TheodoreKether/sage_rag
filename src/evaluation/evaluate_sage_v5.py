"""Evaluate SAGE-RAG v5 (Semantic Re-scoring + Structure Ranking) on Clean Benchmark.

Ablations:
  A: full v5 (initial_k=20, pool=100, graph + semantic)
  B: without graph expansion
  C: without semantic rescoring (≈ v4 selector on expansion pool)
  D: initial_k=10 vs initial_k=20 (full pipeline otherwise)

Usage:
  python src/evaluation/evaluate_sage_v5.py
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
from src.sage_rag.ranking.risk_aware_ranker import RiskAwareEvidenceSelector  # noqa: E402
from src.sage_rag.ranking.risk_aware_ranker_v5 import RiskAwareEvidenceSelectorV5  # noqa: E402
from src.sage_rag.ranking.semantic_rescorer import SemanticRescorer  # noqa: E402
from src.sage_rag.retrieval.sage_expansion_retriever import SageExpansionRetriever  # noqa: E402
from src.sage_rag.retrieval.sage_retriever_v5 import SageRetrieverV5  # noqa: E402

logger = logging.getLogger(__name__)

QA_CLEAN = ROOT / "data" / "qa_dataset" / "qa_pairs_clean.jsonl"
OUT_DIR = ROOT / "results" / "retrieval" / "sage_v5"
SAGE_V2_METRICS = ROOT / "results" / "retrieval" / "sage_v2" / "metrics.json"
SAGE_V4_METRICS = ROOT / "results" / "retrieval" / "sage_v4" / "metrics.json"
CLEAN_BM25_METRICS = ROOT / "results" / "retrieval" / "clean_benchmark" / "bm25_metrics.json"
CLEAN_DENSE_METRICS = ROOT / "results" / "retrieval" / "clean_benchmark" / "dense_metrics.json"
CLEAN_HYBRID_METRICS = ROOT / "results" / "retrieval" / "clean_benchmark" / "hybrid_metrics.json"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate SAGE-RAG v5 on Clean Benchmark")
    p.add_argument("--qa", type=Path, default=QA_CLEAN)
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    p.add_argument("--bm25-index", type=Path, default=ROOT / "data" / "bm25_index")
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
    p.add_argument("--skip-ablation", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _preview(text: str, n: int = 120) -> str:
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
        "score": float(unit.score) if unit.score is not None else None,
        "final_score": meta.get("final_score", unit.score),
        "semantic_score": meta.get("semantic_score"),
        "semantic_score_raw": meta.get("semantic_score_raw"),
        "original_score": meta.get("original_score"),
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
    bm25: BM25Retriever,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float], list[dict[str, Any]]]:
    """retrieve_fn returns (final_units, pool_units, pool_ids, bm25_top10_ids)."""
    records: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    cand_rows: list[dict[str, float]] = []
    cases: list[dict[str, Any]] = []

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

        # True graph-only golds: in pool as expanded, not in BM25 Top10.
        expand_gold_ids = []
        deeper_bm25_gold_ids = []  # in BM25 ranks 11..initial_k, not Top10
        for uid in sample.gold_unit_ids:
            if uid in set(bm25_ids):
                continue
            if uid not in set(pool_ids):
                continue
            src = (pool_by_id.get(uid).metadata or {}).get("candidate_source") if uid in pool_by_id else None
            if src == "expanded":
                expand_gold_ids.append(uid)
            elif src == "initial":
                deeper_bm25_gold_ids.append(uid)

        expand_in_final = [uid for uid in expand_gold_ids if uid in set(final_ids)]
        deeper_in_final = [uid for uid in deeper_bm25_gold_ids if uid in set(final_ids)]
        cand = {
            "bm25_top10_recall": recall_at_k(bm25_ids[:top_k], sample.gold_unit_ids, top_k),
            "expansion_pool_recall": recall_at_k(
                pool_ids, sample.gold_unit_ids, len(pool_ids) or 1
            ),
            "final_top10_recall": metrics.get("Recall@10", 0.0),
            "gold_only_via_expand": 1.0 if expand_gold_ids else 0.0,
            "gold_from_deeper_bm25": 1.0 if deeper_bm25_gold_ids else 0.0,
            "expand_gold_in_final": 1.0 if expand_in_final else 0.0,
            "deeper_bm25_gold_in_final": 1.0 if deeper_in_final else 0.0,
            "n_expand_gold": float(len(expand_gold_ids)),
            "n_expand_gold_promoted": float(len(expand_in_final)),
            "n_deeper_bm25_gold": float(len(deeper_bm25_gold_ids)),
            "n_deeper_bm25_gold_promoted": float(len(deeper_in_final)),
        }
        cand_rows.append(cand)

        hit = next((u for u in final if u.unit_id in gold), None)
        hit_meta = (hit.metadata or {}) if hit else {}
        bm25_hit = bool(gold & set(bm25_ids[:top_k]))
        v5_hit = bool(gold & set(final_ids))

        if (not bm25_hit) and v5_hit:
            promoted = [
                u
                for u in final
                if (u.metadata or {}).get("candidate_source") == "expanded"
            ]
            cases.append(
                {
                    "case_type": "bm25_miss_v5_hit",
                    "qa_id": sample.qa_id,
                    "question": sample.question,
                    "gold_unit_ids": sample.gold_unit_ids,
                    "bm25_top10": bm25_ids[:top_k],
                    "expand_gold_ids": expand_gold_ids,
                    "deeper_bm25_gold_ids": deeper_bm25_gold_ids,
                    "promoted_graph": [_unit_payload(u) for u in promoted],
                    "hit": _unit_payload(hit) if hit else None,
                    "v5_top10": [_unit_payload(u) for u in final],
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
                "bm25_top10_unit_ids": bm25_ids[:top_k],
                "expansion_pool_unit_ids": pool_ids,
                "candidate_recall": cand,
                "hit_candidate_source": hit_meta.get("candidate_source"),
                "hit_expansion_relation": hit_meta.get("expansion_relation"),
                "hit_semantic_score": hit_meta.get("semantic_score"),
                "hit_risk_penalty": hit_meta.get("risk_penalty"),
                **metrics,
            }
        )

    return records, average_metrics(metric_rows), average_metrics(cand_rows), cases


def write_ablation(
    path: Path,
    *,
    variants: dict[str, dict[str, float]],
    cand: dict[str, dict[str, float]],
    bm25: dict[str, float],
    dense: dict[str, float],
    hybrid: dict[str, float],
    sage_v2: dict[str, float],
    sage_v4: dict[str, float],
    n: int,
) -> None:
    lines = [
        "# SAGE-RAG v5 Ablation (Clean Benchmark)",
        "",
        f"n={n}.",
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
    pool_r = (cand.get("A_full") or {}).get("expansion_pool_recall", 0.0)
    lines.append(
        f"| BM25 + Graph Expansion (pool) | — | — | {pool_r:.4f} | — | — | — |"
    )
    row("SAGE v2", sage_v2)
    row("SAGE v4", sage_v4)
    for key, label in [
        ("A_full", "A: SAGE v5 full (k20/pool100)"),
        ("B_no_graph", "B: without graph expansion"),
        ("C_no_semantic", "C: without semantic rescoring (≈v4)"),
        ("D_k10", "D: initial_k=10"),
        ("D_k20", "D: initial_k=20 (=A)"),
    ]:
        if key in variants:
            row(label, variants[key])

    a = variants.get("A_full") or {}
    b = variants.get("B_no_graph") or {}
    c = variants.get("C_no_semantic") or {}
    d10 = variants.get("D_k10") or {}
    d20 = variants.get("D_k20") or a

    ca = cand.get("A_full") or {}
    lines += [
        "",
        "## Ablation takeaways",
        "",
        "### 1. Semantic rescoring 是否解决 expanded 排序？",
        "",
        f"- A (with semantic) R@10={a.get('Recall@10', 0):.4f} vs "
        f"C (no semantic) R@10={c.get('Recall@10', 0):.4f} "
        f"(Δ={(a.get('Recall@10', 0) - c.get('Recall@10', 0)) * 100:+.2f} pp)",
        f"- Expand-gold promoted: A={ca.get('expand_gold_in_final', 0) * 100:.2f}% "
        f"of queries; C={(cand.get('C_no_semantic') or {}).get('expand_gold_in_final', 0) * 100:.2f}%",
        "",
        "### 2. Graph expansion 是否提供额外候选？",
        "",
        f"- A R@10={a.get('Recall@10', 0):.4f} vs B (no graph) "
        f"R@10={b.get('Recall@10', 0):.4f} "
        f"(Δ={(a.get('Recall@10', 0) - b.get('Recall@10', 0)) * 100:+.2f} pp)",
        f"- Pool recall A={ca.get('expansion_pool_recall', 0):.4f} vs "
        f"B={(cand.get('B_no_graph') or {}).get('expansion_pool_recall', 0):.4f}",
        "",
        "### 3. initial_k=10 vs 20？",
        "",
        f"- k10 R@10={d10.get('Recall@10', 0):.4f} vs k20 "
        f"R@10={d20.get('Recall@10', 0):.4f} "
        f"(Δ={(d20.get('Recall@10', 0) - d10.get('Recall@10', 0)) * 100:+.2f} pp)",
        "",
        "## Candidate → Final conversion (A)",
        "",
        f"- BM25 Top10 recall: {ca.get('bm25_top10_recall', 0):.4f}",
        f"- Expansion pool recall: {ca.get('expansion_pool_recall', 0):.4f}",
        f"- SAGE v5 Top10 recall: {ca.get('final_top10_recall', 0):.4f}",
        f"- Expand-only gold queries: {ca.get('gold_only_via_expand', 0) * 100:.2f}%",
        f"- Expand-gold promoted into Top10: {ca.get('expand_gold_in_final', 0) * 100:.2f}%",
        f"- Deeper-BM25 gold queries: {ca.get('gold_from_deeper_bm25', 0) * 100:.2f}% "
        f"(promoted {ca.get('deeper_bm25_gold_in_final', 0) * 100:.2f}%)",
        f"- Mean expand-gold count / query: {ca.get('n_expand_gold', 0):.4f}",
        f"- Mean expand-gold promoted / query: {ca.get('n_expand_gold_promoted', 0):.4f}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vs_baselines(
    path: Path,
    *,
    bm25: dict[str, float],
    dense: dict[str, float],
    hybrid: dict[str, float],
    sage_v2: dict[str, float],
    sage_v4: dict[str, float],
    sage_v5: dict[str, float],
    pool_r: float,
) -> None:
    lines = [
        "# SAGE v5 vs Baselines",
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
    lines.append(f"| BM25+Expansion (pool) | — | — | {pool_r:.4f} | — | — | — |")
    row("SAGE v2", sage_v2)
    row("SAGE v4", sage_v4)
    row("**SAGE v5**", sage_v5)
    lines += [
        "",
        "## Deltas vs BM25",
        "",
        f"- ΔR@1: {(sage_v5.get('Recall@1', 0) - bm25.get('Recall@1', 0)) * 100:+.2f} pp",
        f"- ΔR@5: {(sage_v5.get('Recall@5', 0) - bm25.get('Recall@5', 0)) * 100:+.2f} pp",
        f"- ΔR@10: {(sage_v5.get('Recall@10', 0) - bm25.get('Recall@10', 0)) * 100:+.2f} pp",
        f"- ΔMRR: {sage_v5.get('MRR', 0) - bm25.get('MRR', 0):+.4f}",
        "",
        "## Deltas vs SAGE v4",
        "",
        f"- ΔR@10: {(sage_v5.get('Recall@10', 0) - sage_v4.get('Recall@10', 0)) * 100:+.2f} pp",
        f"- ΔMRR: {sage_v5.get('MRR', 0) - sage_v4.get('MRR', 0):+.4f}",
        f"- ΔR@5: {(sage_v5.get('Recall@5', 0) - sage_v4.get('Recall@5', 0)) * 100:+.2f} pp",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_failure_cases(path: Path, cases: list[dict[str, Any]], *, min_n: int = 5) -> None:
    selected = cases[: max(min_n, 5)]
    lines = [
        "# SAGE v5 Case Study / Failure Analysis",
        "",
        f"Selected **{len(selected)}** `bm25_miss_v5_hit` cases "
        f"(from {len(cases)} total).",
        "",
        "Hypothesis check: Graph discovers candidates; semantic rescoring "
        "calibrates relevance; structure-aware ranking selects evidence.",
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
            "",
            "### Why BM25 failed",
            "",
            f"- Gold not in BM25 Top10: `{c.get('bm25_top10')}`",
            f"- Graph expand-only gold ids: `{c.get('expand_gold_ids')}`",
            f"- Deeper BM25 (11..k) gold ids: `{c.get('deeper_bm25_gold_ids')}`",
            "",
            "### How Graph / deeper retrieval found the candidate",
            "",
        ]
        if c.get("expand_gold_ids"):
            lines.append(
                "- Gold appears via **graph expansion** (not in BM25 Top10)."
            )
        if c.get("deeper_bm25_gold_ids"):
            lines.append(
                "- Gold appears in **BM25 ranks 11..initial_k** and survives "
                "semantic re-scoring into Top10."
            )
        if not c.get("expand_gold_ids") and not c.get("deeper_bm25_gold_ids"):
            lines.append(
                "- Hit path unclear from pool tags; inspect Top10 payloads below."
            )
        lines += ["", "### Why semantic rescoring / ranking helped", ""]
        lines.append(
            f"- Final hit: `{hit.get('unit_id')}` src={hit.get('candidate_source')} "
            f"rel={hit.get('expansion_relation')} "
            f"semantic={hit.get('semantic_score')} "
            f"struct={hit.get('structure_score')} "
            f"cov={hit.get('query_coverage_score')} "
            f"risk={hit.get('risk_penalty')} "
            f"final={hit.get('final_score')}"
        )
        if c.get("promoted_graph"):
            lines += ["", "### Promoted graph evidence in Top10", ""]
            for u in c["promoted_graph"]:
                lines.append(
                    f"- #{u.get('rank')} `{u.get('unit_id')}` "
                    f"rel={u.get('expansion_relation')} "
                    f"sem={u.get('semantic_score')} final={u.get('final_score')} "
                    f"— {_preview(str(u.get('text') or ''), 80)}"
                )
        lines += ["", "### SAGE v5 Top10", ""]
        for u in c.get("v5_top10") or []:
            lines.append(
                f"- #{u.get('rank')} `{u.get('unit_id')}` "
                f"src={u.get('candidate_source')} rel={u.get('expansion_relation')} "
                f"sem={u.get('semantic_score')} final={u.get('final_score')}"
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
    rescorer = SemanticRescorer(bm25)
    ranker_v5 = RiskAwareEvidenceSelectorV5(
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, lam=args.lam
    )
    ranker_v4 = RiskAwareEvidenceSelector()

    sage_full = SageRetrieverV5(
        bm25,
        expander,
        semantic_rescorer=rescorer,
        ranker=ranker_v5,
        pool_size=args.pool_size,
        use_graph_expansion=True,
        use_semantic_rescoring=True,
    )
    sage_no_graph = SageRetrieverV5(
        bm25,
        expander,
        semantic_rescorer=rescorer,
        ranker=ranker_v5,
        pool_size=args.pool_size,
        use_graph_expansion=False,
        use_semantic_rescoring=True,
    )

    def retrieve_full(sample: QASample, initial_k: int):
        bm25_ids = [u.unit_id for u in bm25.retrieve(sample.question, top_k=args.top_k)]
        pool = expansion.retrieve(
            sample.question, top_k=args.pool_size, initial_k=initial_k
        )
        pool_ids = [u.unit_id for u in pool]
        rescored = rescorer.rescore(sample.question, pool)
        final = ranker_v5.rank(rescored, query=sample.question, top_k=args.top_k)
        return final, pool, pool_ids, bm25_ids

    def retrieve_no_graph(sample: QASample):
        bm25_ids = [u.unit_id for u in bm25.retrieve(sample.question, top_k=args.top_k)]
        hits = bm25.retrieve(sample.question, top_k=args.initial_k)
        pool = [SageExpansionRetriever._mark_initial(u) for u in hits]
        pool_ids = [u.unit_id for u in pool]
        rescored = rescorer.rescore(sample.question, pool)
        final = ranker_v5.rank(rescored, query=sample.question, top_k=args.top_k)
        return final, pool, pool_ids, bm25_ids

    def retrieve_no_semantic(sample: QASample):
        """Ablation C: expansion + v4 ranker (no semantic rescoring)."""
        bm25_ids = [u.unit_id for u in bm25.retrieve(sample.question, top_k=args.top_k)]
        pool = expansion.retrieve(
            sample.question, top_k=args.pool_size, initial_k=args.initial_k
        )
        pool_ids = [u.unit_id for u in pool]
        final = ranker_v4.rank(pool, query=sample.question, top_k=args.top_k)
        return final, pool, pool_ids, bm25_ids

    t0 = time.perf_counter()
    variants: dict[str, dict[str, float]] = {}
    cand_by: dict[str, dict[str, float]] = {}
    case_bank: list[dict[str, Any]] = []

    # A: full
    rec_a, m_a, c_a, cases_a = _run_variant(
        samples,
        name="SAGE-v5-A-full",
        retrieve_fn=lambda s: retrieve_full(s, args.initial_k),
        bm25=bm25,
        top_k=args.top_k,
    )
    variants["A_full"] = m_a
    variants["D_k20"] = m_a
    cand_by["A_full"] = c_a
    cand_by["D_k20"] = c_a
    case_bank = cases_a

    if not args.skip_ablation:
        _, m_b, c_b, _ = _run_variant(
            samples,
            name="SAGE-v5-B-no-graph",
            retrieve_fn=retrieve_no_graph,
            bm25=bm25,
            top_k=args.top_k,
        )
        variants["B_no_graph"] = m_b
        cand_by["B_no_graph"] = c_b

        _, m_c, c_c, _ = _run_variant(
            samples,
            name="SAGE-v5-C-no-semantic",
            retrieve_fn=retrieve_no_semantic,
            bm25=bm25,
            top_k=args.top_k,
        )
        variants["C_no_semantic"] = m_c
        cand_by["C_no_semantic"] = c_c

        _, m_d10, c_d10, _ = _run_variant(
            samples,
            name="SAGE-v5-D-k10",
            retrieve_fn=lambda s: retrieve_full(s, 10),
            bm25=bm25,
            top_k=args.top_k,
        )
        variants["D_k10"] = m_d10
        cand_by["D_k10"] = c_d10

    elapsed = time.perf_counter() - t0
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    bm25_m = _load_metrics(CLEAN_BM25_METRICS)
    dense_m = _load_metrics(CLEAN_DENSE_METRICS)
    hybrid_m = _load_metrics(CLEAN_HYBRID_METRICS)
    v2_m = _load_metrics(SAGE_V2_METRICS)
    v4_m = _load_metrics(SAGE_V4_METRICS)

    hit_src = Counter(
        (r.get("hit_candidate_source") or "none")
        for r in rec_a
        if float(r.get("Recall@10") or 0) >= 1.0
    )

    # Promotion stats (true graph-expanded golds only)
    n_expand_queries = sum(
        1 for r in rec_a if (r.get("candidate_recall") or {}).get("gold_only_via_expand")
    )
    n_promoted = sum(
        1 for r in rec_a if (r.get("candidate_recall") or {}).get("expand_gold_in_final")
    )
    n_expand_gold = sum(
        float((r.get("candidate_recall") or {}).get("n_expand_gold") or 0) for r in rec_a
    )
    n_promoted_gold = sum(
        float((r.get("candidate_recall") or {}).get("n_expand_gold_promoted") or 0)
        for r in rec_a
    )
    n_deeper_q = sum(
        1 for r in rec_a if (r.get("candidate_recall") or {}).get("gold_from_deeper_bm25")
    )
    n_deeper_promoted = sum(
        1
        for r in rec_a
        if (r.get("candidate_recall") or {}).get("deeper_bm25_gold_in_final")
    )

    meta = {
        "retriever": "SAGE-RAG v5 (Semantic Re-scoring + Risk-aware Ranking)",
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
        "candidate_recall_by_ablation": cand_by,
        "expanded_gold_promotion": {
            "expand_only_gold_queries": n_expand_queries,
            "queries_with_promoted_expand_gold": n_promoted,
            "promotion_rate_among_expand_only_queries": (
                n_promoted / n_expand_queries if n_expand_queries else 0.0
            ),
            "total_expand_only_golds": n_expand_gold,
            "total_promoted_expand_golds": n_promoted_gold,
            "gold_promotion_rate": (
                n_promoted_gold / n_expand_gold if n_expand_gold else 0.0
            ),
            "deeper_bm25_gold_queries": n_deeper_q,
            "queries_with_promoted_deeper_bm25_gold": n_deeper_promoted,
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
        "# SAGE-RAG v5 Evaluation Report",
        "",
        f"- QA: `{args.qa}` ({len(samples)} queries)",
        f"- Pipeline: BM25(initial_k={args.initial_k}) → Graph Expansion "
        f"→ Semantic Re-score (BM25) → Risk-aware Ranking",
        f"- pool_size={args.pool_size}, top_k={args.top_k}",
        f"- Score = α·Semantic + β·Struct + γ·Coverage − λ·Risk",
        f"- weights: α={args.alpha}, β={args.beta}, γ={args.gamma}, λ={args.lam}",
        f"- structure priors: parent_of=1.0, refers_to=0.9, next_to=0.3",
        f"- elapsed: {elapsed:.2f}s",
        "",
        "## Metrics (A: full)",
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
        f"- True graph expand-only gold queries: {n_expand_queries}",
        f"- Queries with promoted expand-gold: {n_promoted} "
        f"({(n_promoted / n_expand_queries * 100) if n_expand_queries else 0:.1f}%)",
        f"- Expand-only gold units: {n_expand_gold:.0f}",
        f"- Promoted into Top10: {n_promoted_gold:.0f} "
        f"({(n_promoted_gold / n_expand_gold * 100) if n_expand_gold else 0:.1f}%)",
        f"- Deeper BM25 (rank 11..initial_k) gold queries: {n_deeper_q} "
        f"(promoted: {n_deeper_promoted})",
        "",
        f"- Hit sources: `{dict(hit_src)}`",
        "",
        "See `ablation.md`, `sage_v5_vs_baselines.md`, `failure_cases.md`.",
        "",
    ]
    (out / "evaluation_report.md").write_text("\n".join(report), encoding="utf-8")

    write_ablation(
        out / "ablation.md",
        variants=variants,
        cand=cand_by,
        bm25=bm25_m,
        dense=dense_m,
        hybrid=hybrid_m,
        sage_v2=v2_m,
        sage_v4=v4_m,
        n=len(samples),
    )
    write_vs_baselines(
        out / "sage_v5_vs_baselines.md",
        bm25=bm25_m,
        dense=dense_m,
        hybrid=hybrid_m,
        sage_v2=v2_m,
        sage_v4=v4_m,
        sage_v5=m_a,
        pool_r=float(c_a.get("expansion_pool_recall", 0.0)),
    )
    write_failure_cases(out / "failure_cases.md", case_bank, min_n=5)

    print(
        json.dumps(
            {
                "metrics": m_a,
                "ablation": {k: v for k, v in variants.items()},
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
