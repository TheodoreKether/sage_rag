"""Expand an Evidence Unit to related evidence via structural graph edges.

v1 scope (depth=1 only):
- ``parent_of``: hierarchical parent clause evidence (+ chapter metadata)
- ``next_to``: evidence under adjacent clauses
- ``refers_to``: evidence under resolved reference targets

No GNN / Neo4j / LLM / embedding similarity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.sage_rag.graph.graph_builder import make_clause_node_id
from src.sage_rag.graph.graph_schema import Node
from src.sage_rag.graph.graph_store import GraphStore

logger = logging.getLogger(__name__)

DEFAULT_EDGE_TYPES: tuple[str, ...] = ("parent_of", "next_to", "refers_to")


@dataclass
class ExpandedEvidence:
    """One expanded evidence hit (always an evidence node / unit_id)."""

    unit_id: str
    text: str
    relation: str
    via_node_id: str = ""
    page: Any = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "text": self.text,
            "relation": self.relation,
            "via_node_id": self.via_node_id,
            "page": self.page,
            "attributes": self.attributes,
        }


class GraphExpander:
    """Structure-aware 1-hop expansion from an evidence ``unit_id``."""

    def __init__(
        self,
        store: GraphStore,
        *,
        max_refers_evidence: int = 30,
    ) -> None:
        self.store = store
        self.max_refers_evidence = max_refers_evidence

    def expand(
        self,
        unit_id: str,
        depth: int = 1,
        edge_types: Sequence[str] | None = None,
    ) -> list[ExpandedEvidence]:
        """Expand related evidence around ``unit_id``.

        Parameters
        ----------
        unit_id:
            Seed evidence node id (must equal the original Evidence Unit id).
        depth:
            Only ``1`` is supported in v1.
        edge_types:
            Subset of ``{parent_of, next_to, refers_to}``. ``None`` = all three.
        """
        if depth != 1:
            raise ValueError("GraphExpander v1 only supports depth=1")

        seed = self.store.get_node(unit_id)
        if seed is None or seed.type != "evidence":
            logger.warning("expand: seed evidence not found: %s", unit_id)
            return []

        allowed = list(edge_types) if edge_types is not None else list(DEFAULT_EDGE_TYPES)
        hits: list[ExpandedEvidence] = []
        seen: set[tuple[str, str]] = set()  # (unit_id, relation)

        if "parent_of" in allowed:
            for hit in self._expand_parent_of(seed):
                self._add_hit(hits, seen, hit, exclude_unit=unit_id)

        if "next_to" in allowed:
            for hit in self._expand_next_to(seed):
                self._add_hit(hits, seen, hit, exclude_unit=unit_id)

        if "refers_to" in allowed:
            for hit in self._expand_refers_to(seed):
                self._add_hit(hits, seen, hit, exclude_unit=unit_id)

        return hits

    # ------------------------------------------------------------------ parent

    def _expand_parent_of(self, seed: Node) -> list[ExpandedEvidence]:
        """evidence ← clause; lift to dotted parent clause; record chapter.

        Example: ``6.1.1.13`` → parent clause ``6.1.1`` evidence, with
        ``parent_chapter_id`` pointing at chapter ``6``.
        """
        hits: list[ExpandedEvidence] = []

        parent_clause = self._immediate_parent_clause(seed.id)
        if parent_clause is None:
            return hits

        chapter = self._parent_chapter(parent_clause.id)
        hier = self._hierarchical_parent_clause(parent_clause)

        meta = {
            "immediate_clause_id": parent_clause.id,
            "parent_chapter_id": chapter.id if chapter is not None else "",
            "parent_clause_id": hier.id if hier is not None else "",
        }

        if hier is not None:
            for ev in self.store.get_evidence_children(hier.id):
                hits.append(
                    self._from_evidence_node(
                        ev,
                        "parent_of",
                        via_node_id=hier.id,
                        extra=meta,
                    )
                )

        return hits

    # ------------------------------------------------------------------ next_to

    def _expand_next_to(self, seed: Node) -> list[ExpandedEvidence]:
        """Adjacent clauses (bidirectional ``next_to``) → their evidence children."""
        hits: list[ExpandedEvidence] = []
        parent_clause = self._immediate_parent_clause(seed.id)
        if parent_clause is None:
            return hits

        neighbors = self.store.get_neighbors(
            parent_clause.id,
            edge_types=["next_to"],
            direction="both",
        )
        for nb in neighbors:
            if nb.node is None or nb.node.type != "clause":
                continue
            for ev in self.store.get_evidence_children(nb.node_id):
                hits.append(
                    self._from_evidence_node(ev, "next_to", via_node_id=nb.node_id)
                )
        return hits

    # ------------------------------------------------------------------ refers

    def _expand_refers_to(self, seed: Node) -> list[ExpandedEvidence]:
        """Follow outgoing ``refers_to`` edges; collect evidence under targets."""
        hits: list[ExpandedEvidence] = []
        refs = self.store.get_neighbors(
            seed.id, edge_types=["refers_to"], direction="out"
        )
        for nb in refs:
            target = nb.node
            if target is None:
                logger.debug(
                    "refers_to unresolved from %s → %s", seed.id, nb.node_id
                )
                continue
            collected = self._collect_evidence_under(target)
            for ev in collected[: self.max_refers_evidence]:
                hits.append(
                    self._from_evidence_node(
                        ev,
                        "refers_to",
                        via_node_id=target.id,
                        extra={
                            "reference": (nb.edge.attributes or {}).get("reference"),
                            "target_type": target.type,
                        },
                    )
                )
        return hits

    def _collect_evidence_under(self, node: Node) -> list[Node]:
        """Map a structural / evidence target onto evidence nodes."""
        if node.type == "evidence":
            return [node]
        if node.type == "clause":
            return self.store.get_evidence_children(node.id)
        if node.type == "chapter":
            out: list[Node] = []
            for clause in self.store.get_clause_children(node.id):
                out.extend(self.store.get_evidence_children(clause.id))
                if len(out) >= self.max_refers_evidence:
                    break
            return out[: self.max_refers_evidence]
        if node.type == "document":
            doc_id = node.id
            return [
                n
                for n in self.store.iter_nodes("evidence")
                if (n.attributes or {}).get("document_id") == doc_id
            ][: self.max_refers_evidence]
        return []

    # ------------------------------------------------------------------ helpers

    def _immediate_parent_clause(self, evidence_id: str) -> Node | None:
        for nb in self.store.get_neighbors(
            evidence_id, edge_types=["parent_of"], direction="in"
        ):
            if nb.node is not None and nb.node.type == "clause":
                return nb.node
        return None

    def _parent_chapter(self, clause_node_id: str) -> Node | None:
        for nb in self.store.get_neighbors(
            clause_node_id, edge_types=["parent_of"], direction="in"
        ):
            if nb.node is not None and nb.node.type == "chapter":
                return nb.node
        return None

    def _hierarchical_parent_clause(self, clause: Node) -> Node | None:
        """Infer dotted parent clause, e.g. ``6.1.1.13`` → ``6.1.1`` if present."""
        clause_id = str((clause.attributes or {}).get("clause_id") or "")
        document_id = str((clause.attributes or {}).get("document_id") or "")
        if not clause_id or not document_id or "." not in clause_id:
            return None
        parent_clause_id = clause_id.rsplit(".", 1)[0]
        if not parent_clause_id:
            return None
        node_id = make_clause_node_id(document_id, parent_clause_id)
        node = self.store.get_node(node_id)
        if node is not None and node.type == "clause":
            return node
        return None

    @staticmethod
    def _from_evidence_node(
        node: Node,
        relation: str,
        *,
        via_node_id: str = "",
        extra: dict[str, Any] | None = None,
    ) -> ExpandedEvidence:
        attrs = dict(node.attributes or {})
        if extra:
            attrs.update(extra)
        return ExpandedEvidence(
            unit_id=node.id,
            text=str(attrs.get("text") or ""),
            relation=relation,
            via_node_id=via_node_id,
            page=attrs.get("page"),
            attributes=attrs,
        )

    @staticmethod
    def _add_hit(
        hits: list[ExpandedEvidence],
        seen: set[tuple[str, str]],
        hit: ExpandedEvidence,
        *,
        exclude_unit: str,
    ) -> None:
        if hit.unit_id == exclude_unit:
            return
        key = (hit.unit_id, hit.relation)
        if key in seen:
            return
        seen.add(key)
        hits.append(hit)
