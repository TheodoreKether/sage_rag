"""Compare StructureRanker v1 vs v2 on a single query.

Usage:
  python src/sage_rag/test_structure_ranking_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.retrieval.bm25 import BM25Retriever
from src.sage_rag.expansion.graph_expander import GraphExpander
from src.sage_rag.graph.graph_store import GraphStore
from src.sage_rag.ranking.structure_ranker import StructureRanker
from src.sage_rag.ranking.structure_ranker_v2 import StructureRankerV2
from src.sage_rag.retrieval.sage_expansion_retriever import SageExpansionRetriever

QUERY = "除零错误审计要求是什么"
GRAPH_DIR = _REPO_ROOT / "data" / "sage_graph"
BM25_DIR = _REPO_ROOT / "data" / "bm25_index"


def _preview(text: str, n: int = 70) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + "..."


def _print_list(title: str, units, *, limit: int = 15) -> None:
    print("\n" + title)
    for u in units[:limit]:
        meta = u.metadata or {}
        src = meta.get("candidate_source", "?")
        rel = meta.get("expansion_relation")
        cov = meta.get("query_coverage_score")
        path = meta.get("structure_path_score", meta.get("graph_score"))
        red = meta.get("redundancy_penalty")
        final = meta.get("final_score", u.score)
        extra = ""
        if cov is not None:
            extra = f" cov={cov:.3f} path={path} red={red}"
        print(
            f"  #{u.rank} [{src}/{rel}] score={final}{extra}  "
            f"{u.unit_id}  {_preview(u.text)}"
        )


def main() -> int:
    store = GraphStore.from_dir(GRAPH_DIR)
    expander = GraphExpander(store)
    bm25 = BM25Retriever(index_dir=BM25_DIR)
    expansion = SageExpansionRetriever(bm25, expander)
    v1 = StructureRanker(alpha=0.7, beta=0.3)
    v2 = StructureRankerV2()

    print("=" * 72)
    print(f"Query: {QUERY}")
    print("=" * 72)

    bm25_top = bm25.retrieve(QUERY, top_k=10)
    _print_list("1. BM25 Top10", bm25_top)

    pool = expansion.retrieve(QUERY, top_k=80, initial_k=10)
    expanded_only = [
        u for u in pool if (u.metadata or {}).get("candidate_source") == "expanded"
    ]
    print(f"\n2. Expansion pool size={len(pool)} (expanded={len(expanded_only)})")
    _print_list("   Expanded candidates (sample)", expanded_only, limit=12)

    ranked_v1 = v1.rank(pool)[:10]
    _print_list("3. SAGE v1 Top10 (flat structure rank)", ranked_v1)

    ranked_v2 = v2.rank(pool, query=QUERY, top_k=10)
    _print_list("4. SAGE v2 Top10 (greedy evidence selection)", ranked_v2)

    v1_ids = {u.unit_id for u in ranked_v1}
    v2_ids = {u.unit_id for u in ranked_v2}
    bm25_ids = {u.unit_id for u in bm25_top}
    promoted = [
        u
        for u in ranked_v2
        if (u.metadata or {}).get("candidate_source") == "expanded"
        and u.unit_id not in bm25_ids
    ]
    print("\n5. Graph-found evidence lifted into v2 Top10 (not in BM25 Top10):")
    if not promoted:
        print("  (none in this query)")
    for u in promoted:
        meta = u.metadata or {}
        print(
            f"  #{u.rank} {u.unit_id} rel={meta.get('expansion_relation')} "
            f"cov={meta.get('query_coverage_score')} final={meta.get('final_score')}"
        )

    only_v2 = v2_ids - v1_ids
    print(f"\n  v2-only unit_ids vs v1: {sorted(only_v2)}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
