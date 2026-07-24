"""Tests for StructureRanker + SageRetriever.

Usage:
  python src/sage_rag/test_structure_ranking.py
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
from src.sage_rag.retrieval.sage_expansion_retriever import SageExpansionRetriever
from src.sage_rag.retrieval.sage_retriever import SageRetriever

QUERY = "除零错误审计要求是什么"
PARENT_CLAUSE_UNIT = "GB_T_39412-2020::6::6.1.1::1"
GRAPH_DIR = _REPO_ROOT / "data" / "sage_graph"
BM25_DIR = _REPO_ROOT / "data" / "bm25_index"


def _meta(u, key, default=None):
    return (u.metadata or {}).get(key, default)


def _print_row(u) -> None:
    src = _meta(u, "candidate_source", "?")
    rel = _meta(u, "expansion_relation")
    retr = _meta(u, "original_score", u.score)
    g = _meta(u, "graph_score")
    f = _meta(u, "final_score", u.score)
    rn = _meta(u, "retrieval_score_normalized")
    print(
        f"  rank={u.rank}  unit_id={u.unit_id}\n"
        f"         source={src}  relation={rel}\n"
        f"         retrieval_score={retr}  retr_norm={rn}  "
        f"graph_score={g}  final_score={f}"
    )


def main() -> int:
    if not (GRAPH_DIR / "nodes.jsonl").is_file():
        print(f"Graph not found: {GRAPH_DIR}")
        return 1
    if not BM25_DIR.is_dir():
        print(f"BM25 index not found: {BM25_DIR}")
        return 1

    store = GraphStore.from_dir(GRAPH_DIR)
    expander = GraphExpander(store)
    bm25 = BM25Retriever(index_dir=BM25_DIR)
    expansion = SageExpansionRetriever(bm25, expander)
    ranker = StructureRanker(alpha=0.7, beta=0.3)
    sage = SageRetriever(bm25, expander, ranker=ranker)

    print("=" * 72)
    print(f"Query: {QUERY}")
    print("=" * 72)

    # 1) BM25 Top5 (before expansion)
    print("\n1. Expansion前：BM25 Top5")
    bm25_top5 = bm25.retrieve(QUERY, top_k=5)
    for u in bm25_top5:
        print(f"  rank={u.rank}  score={u.score:.4f}  unit_id={u.unit_id}")

    # 2) Expansion candidate pool
    print("\n2. Expansion candidate pool (initial_k=5, pool_size=40)")
    pool = expansion.retrieve(QUERY, top_k=40, initial_k=5)
    for u in pool:
        src = _meta(u, "candidate_source")
        rel = _meta(u, "expansion_relation")
        score = u.score if u.score is not None else 0.0
        print(
            f"  #{u.rank}  source={src}  score={score:.4f}  "
            f"relation={rel}  unit_id={u.unit_id}"
        )
    pool_pos = next(
        (i for i, u in enumerate(pool, start=1) if u.unit_id == PARENT_CLAUSE_UNIT),
        None,
    )
    print(f"\n  parent clause {PARENT_CLAUSE_UNIT} pool position: {pool_pos}")

    # 3) Ranking Top10
    print("\n3. Ranking之后：Top10 (SageRetriever)")
    ranked_pool = ranker.rank(pool)
    final = sage.retrieve(QUERY, top_k=10, initial_k=5, pool_size=40)
    for u in final:
        _print_row(u)

    ranked_pos = next(
        (i for i, u in enumerate(ranked_pool, start=1) if u.unit_id == PARENT_CLAUSE_UNIT),
        None,
    )
    final_pos = next(
        (i for i, u in enumerate(final, start=1) if u.unit_id == PARENT_CLAUSE_UNIT),
        None,
    )

    print("\n4. 排序变化观察")
    print(f"  parent clause pool position (pre-rank):  {pool_pos}")
    print(f"  parent clause ranked position (full):    {ranked_pos}")
    print(f"  parent clause in final Top10:            {final_pos}")

    # Sanity
    assert len(final) <= 10
    assert all(_meta(u, "final_score") is not None for u in final)
    assert all(_meta(u, "graph_score") is not None for u in final)
    # Ordering by final_score descending
    finals = [float(_meta(u, "final_score")) for u in final]
    assert finals == sorted(finals, reverse=True)

    if pool_pos is not None and ranked_pos is not None:
        if ranked_pos < pool_pos:
            print(
                f"  → parent_of candidate 被提升: pool #{pool_pos} → ranked #{ranked_pos}"
            )
        else:
            print(
                f"  → parent_of candidate 位置: pool #{pool_pos} → ranked #{ranked_pos}"
            )

    print("\nSUMMARY: StructureRanker + SageRetriever = PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
