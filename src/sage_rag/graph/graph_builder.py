"""Build Standard Evidence Graph from Evidence Units (structure-only, no LLM)."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Iterable

from .graph_schema import (
    NEXT_TO_WEIGHT,
    PARENT_OF_WEIGHT,
    REFERS_TO_WEIGHT,
    Edge,
    EdgeType,
    GraphStatistics,
    Node,
    NodeType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stable node ids
# ---------------------------------------------------------------------------
# Chapter / clause both naturally look like ``doc::1``. Without a type
# namespace those collide and break the document→chapter→clause→evidence chain.
# Document / evidence ids stay exactly as in the Evidence Unit schema.


def make_chapter_node_id(document_id: str, chapter_id: str) -> str:
    return f"{document_id}::chapter::{chapter_id}"


def make_clause_node_id(document_id: str, parent_clause: str) -> str:
    return f"{document_id}::clause::{parent_clause}"


# ---------------------------------------------------------------------------
# Clause ordering (for next_to)
# ---------------------------------------------------------------------------


def clause_sort_key(clause_id: str) -> tuple:
    """Sort key for clause / table / annex identifiers within a chapter."""
    text = (clause_id or "").strip()
    if not text:
        return (99, ())

    lower = text.lower()
    # Table N / 表N
    m = re.match(r"^(?:table\s*|表)\s*(\d+)\s*(.*)$", text, re.IGNORECASE)
    if m:
        return (2, (int(m.group(1)), m.group(2)))

    # 附录A / Annex A / A.1 / B.3.3
    m = re.match(r"^(?:附录|annex)\s*([A-Za-z])\s*(.*)$", text, re.IGNORECASE)
    if m:
        rest = m.group(2).strip(" .")
        return (1, (m.group(1).upper(), _numeric_tuple(rest) if rest else ()))

    m = re.match(r"^([A-Za-z])(?:\.(\d+(?:\.\d+)*))?$", text)
    if m and not text[0].isdigit():
        rest = m.group(2) or ""
        return (1, (m.group(1).upper(), _numeric_tuple(rest) if rest else ()))

    # Dotted numeric: 6.1.1.10
    if re.match(r"^\d+(?:\.\d+)*$", text):
        return (0, _numeric_tuple(text))

    return (3, (lower,))


def _numeric_tuple(dotted: str) -> tuple[int, ...]:
    parts: list[int] = []
    for p in dotted.split("."):
        p = p.strip()
        if not p:
            continue
        parts.append(int(p) if p.isdigit() else hash(p) % 10_000)
    return tuple(parts)


# ---------------------------------------------------------------------------
# Reference extraction (rule-based refers_to)
# ---------------------------------------------------------------------------

# Explicit cross-ref cues required by v1 design.
_REF_CLAUSE = re.compile(
    r"(?:见|参见|参照|按)\s*(?:第)?(\d+(?:\.\d+)*)(?:\s*[条款节项])?",
)
_REF_APPENDIX = re.compile(
    r"(?:见|参见|参照|按)?\s*(附录\s*[A-Za-z]|Annex\s+[A-Z])\b",
    re.IGNORECASE,
)
_REF_GBT = re.compile(
    r"(GB/?T\s*\d+(?:\.\s*\d+)*(?:\s*[—\u2014\u2013\-－]\s*\d{4})?)",
    re.IGNORECASE,
)
_REF_ISO = re.compile(
    r"(ISO\s*\d+(?:\s*-\s*\d+)*(?:\s*:\s*\d{4})?)",
    re.IGNORECASE,
)


def extract_reference_strings(text: str) -> list[str]:
    """Extract explicit reference surface forms from evidence text."""
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        cleaned = re.sub(r"\s+", " ", raw.strip())
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            found.append(cleaned)

    for m in _REF_APPENDIX.finditer(text):
        _add(m.group(1))
    for m in _REF_CLAUSE.finditer(text):
        _add(m.group(1))
    for m in _REF_GBT.finditer(text):
        _add(m.group(1))
    for m in _REF_ISO.finditer(text):
        _add(m.group(1))
    return found


def normalize_standard_doc_id(ref: str) -> str | None:
    """Best-effort map of ``GB/T 39401—2020`` / ``ISO 10303-1:2021`` → document_id."""
    text = re.sub(r"\s+", "", ref)
    text = text.replace("—", "-").replace("–", "-").replace("－", "-")

    m = re.match(r"GB/?T(\d+(?:\.\d+)*)(?:-(\d{4}))?", text, re.IGNORECASE)
    if m:
        # Corpus uses underscores: GB_T_39401-2020 (part numbers rarely in document_id).
        base = m.group(1).split(".")[0]
        year = m.group(2)
        if year:
            return f"GB_T_{base}-{year}"
        return f"GB_T_{base}"

    m = re.match(r"ISO(\d+(?:-\d+)*)(?::(\d{4}))?", text, re.IGNORECASE)
    if m:
        body = m.group(1)
        year = m.group(2)
        if year:
            return f"ISO_{body}-{year}"
        return f"ISO_{body}"

    return None


def normalize_appendix_id(ref: str) -> str:
    """Normalize ``附录 A`` / ``Annex A`` → ``附录A`` or ``Annex A`` variants for lookup."""
    m = re.search(r"(?:附录|Annex)\s*([A-Za-z])", ref, re.IGNORECASE)
    if not m:
        return ref.strip()
    letter = m.group(1).upper()
    if "annex" in ref.lower():
        return f"Annex {letter}"
    return f"附录{letter}"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class StandardEvidenceGraphBuilder:
    """Construct document→chapter→clause→evidence structural graph + refs."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self._edge_keys: set[tuple[str, str, str]] = set()
        # Lookup indexes for reference resolution
        self._clause_ids: set[str] = set()
        self._chapter_ids: set[str] = set()
        self._document_ids: set[str] = set()

    # -- public API ---------------------------------------------------------

    def build(self, evidence_units: Iterable[dict[str, Any]]) -> tuple[list[Node], list[Edge]]:
        units = list(evidence_units)
        logger.info("Building Standard Evidence Graph from %d evidence units", len(units))

        self._add_hierarchy_nodes_and_parent_edges(units)
        self._add_next_to_edges(units)
        self._add_refers_to_edges(units)

        nodes = list(self.nodes.values())
        logger.info(
            "Graph built: %d nodes, %d edges",
            len(nodes),
            len(self.edges),
        )
        return nodes, self.edges

    def compute_statistics(
        self,
        nodes: list[Node] | None = None,
        edges: list[Edge] | None = None,
    ) -> GraphStatistics:
        nodes = nodes if nodes is not None else list(self.nodes.values())
        edges = edges if edges is not None else self.edges

        node_counts: dict[str, int] = defaultdict(int)
        for n in nodes:
            node_counts[n.type] += 1

        edge_counts: dict[str, int] = defaultdict(int)
        resolved = unresolved = 0
        for e in edges:
            edge_counts[e.type] += 1
            if e.type == EdgeType.REFERS_TO.value:
                if e.attributes.get("resolved"):
                    resolved += 1
                else:
                    unresolved += 1

        return GraphStatistics(
            node_counts=dict(node_counts),
            edge_counts=dict(edge_counts),
            num_documents=node_counts.get(NodeType.DOCUMENT.value, 0),
            num_evidence_units=node_counts.get(NodeType.EVIDENCE.value, 0),
            refers_to_resolved=resolved,
            refers_to_unresolved=unresolved,
        )

    # -- hierarchy ----------------------------------------------------------

    def _add_hierarchy_nodes_and_parent_edges(self, units: list[dict[str, Any]]) -> None:
        for unit in units:
            document_id = str(unit["document_id"])
            chapter_id = str(unit["chapter_id"])
            parent_clause = str(unit["parent_clause"])
            unit_id = str(unit["unit_id"])

            doc_node_id = document_id
            chapter_node_id = make_chapter_node_id(document_id, chapter_id)
            clause_node_id = make_clause_node_id(document_id, parent_clause)

            self._ensure_document_node(unit, doc_node_id)
            self._ensure_chapter_node(unit, chapter_node_id, chapter_id)
            self._ensure_clause_node(unit, clause_node_id, parent_clause, chapter_id)
            self._ensure_evidence_node(unit, unit_id)

            self._add_edge(doc_node_id, chapter_node_id, EdgeType.PARENT_OF.value, PARENT_OF_WEIGHT)
            self._add_edge(
                chapter_node_id, clause_node_id, EdgeType.PARENT_OF.value, PARENT_OF_WEIGHT
            )
            self._add_edge(clause_node_id, unit_id, EdgeType.PARENT_OF.value, PARENT_OF_WEIGHT)

    def _ensure_document_node(self, unit: dict[str, Any], node_id: str) -> None:
        if node_id in self.nodes:
            return
        self.nodes[node_id] = Node(
            id=node_id,
            type=NodeType.DOCUMENT.value,
            attributes={
                "title": unit.get("title", ""),
                "document_type": unit.get("document_type", ""),
            },
        )
        self._document_ids.add(node_id)

    def _ensure_chapter_node(
        self, unit: dict[str, Any], node_id: str, chapter_id: str
    ) -> None:
        if node_id in self.nodes:
            return
        self.nodes[node_id] = Node(
            id=node_id,
            type=NodeType.CHAPTER.value,
            attributes={
                "chapter_id": chapter_id,
                "chapter_title": unit.get("chapter_title", ""),
                "document_id": unit.get("document_id", ""),
            },
        )
        self._chapter_ids.add(node_id)

    def _ensure_clause_node(
        self,
        unit: dict[str, Any],
        node_id: str,
        clause_id: str,
        chapter_id: str,
    ) -> None:
        if node_id in self.nodes:
            return
        self.nodes[node_id] = Node(
            id=node_id,
            type=NodeType.CLAUSE.value,
            attributes={
                "clause_id": clause_id,
                "chapter_id": chapter_id,
                "document_id": unit.get("document_id", ""),
            },
        )
        self._clause_ids.add(node_id)

    def _ensure_evidence_node(self, unit: dict[str, Any], unit_id: str) -> None:
        if unit_id in self.nodes:
            return
        meta = unit.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        self.nodes[unit_id] = Node(
            id=unit_id,
            type=NodeType.EVIDENCE.value,
            attributes={
                "page": unit.get("page"),
                "text": unit.get("text", ""),
                "token_length": unit.get("token_length"),
                "contains_table": bool(meta.get("contains_table", False)),
                "contains_appendix": bool(meta.get("contains_appendix", False)),
                "document_id": unit.get("document_id", ""),
                "chapter_id": unit.get("chapter_id", ""),
                "parent_clause": unit.get("parent_clause", ""),
                "char_length": unit.get("char_length"),
                "split_index": unit.get("split_index"),
                "split_total": unit.get("split_total"),
            },
        )

    # -- next_to ------------------------------------------------------------

    def _add_next_to_edges(self, units: list[dict[str, Any]]) -> None:
        """Adjacent clauses under the same (document, chapter), sorted by clause id."""
        groups: dict[tuple[str, str], set[str]] = defaultdict(set)
        for unit in units:
            document_id = str(unit["document_id"])
            chapter_id = str(unit["chapter_id"])
            parent_clause = str(unit["parent_clause"])
            groups[(document_id, chapter_id)].add(parent_clause)

        for (document_id, _chapter_id), clause_ids in groups.items():
            ordered = sorted(clause_ids, key=clause_sort_key)
            for left, right in zip(ordered, ordered[1:]):
                src = make_clause_node_id(document_id, left)
                tgt = make_clause_node_id(document_id, right)
                self._add_edge(src, tgt, EdgeType.NEXT_TO.value, NEXT_TO_WEIGHT)

    # -- refers_to ----------------------------------------------------------

    def _add_refers_to_edges(self, units: list[dict[str, Any]]) -> None:
        for unit in units:
            unit_id = str(unit["unit_id"])
            document_id = str(unit["document_id"])
            text = unit.get("text") or ""
            for ref in extract_reference_strings(text):
                for target, resolved in self._resolve_reference_targets(document_id, ref):
                    self._add_edge(
                        unit_id,
                        target,
                        EdgeType.REFERS_TO.value,
                        REFERS_TO_WEIGHT,
                        attributes={
                            "resolved": resolved,
                            "reference": ref,
                        },
                    )

    def _resolve_reference_targets(
        self, document_id: str, ref: str
    ) -> list[tuple[str, bool]]:
        """Map a surface reference onto zero or more existing node ids."""
        # Appendix / Annex → chapter or clause in same document
        if re.search(r"附录|Annex", ref, re.IGNORECASE):
            for candidate in self._appendix_candidates(document_id, ref):
                if candidate in self._chapter_ids or candidate in self._clause_ids:
                    return [(candidate, True)]
            return [(normalize_appendix_id(ref), False)]

        # Numeric clause in same document
        if re.fullmatch(r"\d+(?:\.\d+)*", ref.strip()):
            ref_id = ref.strip()
            clause_node = make_clause_node_id(document_id, ref_id)
            if clause_node in self._clause_ids:
                return [(clause_node, True)]
            chapter_node = make_chapter_node_id(document_id, ref_id)
            if chapter_node in self._chapter_ids:
                return [(chapter_node, True)]
            return [(ref_id, False)]

        # External standard id (may match multiple corpus documents)
        doc_candidate = normalize_standard_doc_id(ref)
        if doc_candidate:
            if doc_candidate in self._document_ids:
                return [(doc_candidate, True)]
            matches = sorted(
                did
                for did in self._document_ids
                if did == doc_candidate
                or did.startswith(doc_candidate + "-")
                or did.startswith(doc_candidate + "_")
            )
            if matches:
                return [(m, True) for m in matches]
            return [(ref, False)]

        return [(ref, False)]

    def _appendix_candidates(self, document_id: str, ref: str) -> list[str]:
        letter_m = re.search(r"(?:附录|Annex)\s*([A-Za-z])", ref, re.IGNORECASE)
        if not letter_m:
            return []
        letter = letter_m.group(1).upper()
        variants = [f"附录{letter}", f"Annex {letter}", f"Annex{letter}", letter]
        out: list[str] = []
        for v in variants:
            out.append(make_chapter_node_id(document_id, v))
            out.append(make_clause_node_id(document_id, v))
        return out

    # -- edge helper --------------------------------------------------------

    def _add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        weight: float,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        key = (source, target, edge_type)
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        self.edges.append(
            Edge(
                source=source,
                target=target,
                type=edge_type,
                weight=weight,
                attributes=attributes or {},
            )
        )
