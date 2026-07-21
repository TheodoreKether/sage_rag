"""CLI: Evidence Units → Standard Evidence Graph.

Usage:
  python src/sage_rag/build_graph.py \\
      --input data/evidence_units/evidence_units.jsonl \\
      --output data/sage_graph
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow `python src/sage_rag/build_graph.py` without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.sage_rag.graph.graph_builder import StandardEvidenceGraphBuilder  # noqa: E402
from src.sage_rag.graph.graph_store import (  # noqa: E402
    GraphStore,
    load_evidence_units,
    write_statistics_markdown,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build SAGE-RAG Standard Evidence Graph from Evidence Units"
    )
    p.add_argument(
        "--input",
        type=Path,
        default=_REPO_ROOT / "data" / "evidence_units" / "evidence_units.jsonl",
        help="Evidence Units JSONL path",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "data" / "sage_graph",
        help="Output directory for nodes.jsonl / edges.jsonl / graph_statistics.md",
    )
    p.add_argument(
        "--stats-out",
        type=Path,
        default=_REPO_ROOT / "results" / "sage_graph_statistics.md",
        help="Also write a copy of statistics under results/",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.input.is_file():
        logger.error("Input not found: %s", args.input)
        return 1

    units = load_evidence_units(args.input)
    if not units:
        logger.error("No evidence units loaded from %s", args.input)
        return 1

    builder = StandardEvidenceGraphBuilder()
    nodes, edges = builder.build(units)
    stats = builder.compute_statistics(nodes, edges)

    store = GraphStore(args.output)
    store.save(nodes, edges, stats=stats)

    # Mirror statistics into results/ for paper-facing artifacts.
    write_statistics_markdown(stats, args.stats_out)

    print(
        "\n".join(
            [
                "Standard Evidence Graph build complete.",
                f"  nodes: {args.output / 'nodes.jsonl'}",
                f"  edges: {args.output / 'edges.jsonl'}",
                f"  stats: {args.output / 'graph_statistics.md'}",
                f"  stats (results): {args.stats_out}",
                "",
                "Nodes:",
                f"  document: {stats.node_counts.get('document', 0)}",
                f"  chapter:  {stats.node_counts.get('chapter', 0)}",
                f"  clause:   {stats.node_counts.get('clause', 0)}",
                f"  evidence: {stats.node_counts.get('evidence', 0)}",
                "",
                "Edges:",
                f"  parent_of: {stats.edge_counts.get('parent_of', 0)}",
                f"  next_to:   {stats.edge_counts.get('next_to', 0)}",
                f"  refers_to: {stats.edge_counts.get('refers_to', 0)}",
                f"  refers_to resolved/unresolved: "
                f"{stats.refers_to_resolved}/{stats.refers_to_unresolved}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
