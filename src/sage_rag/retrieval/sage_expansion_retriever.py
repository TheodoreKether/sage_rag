"""Graph-guided Candidate Expansion over any baseline retriever.

``SageExpansionRetriever`` is a **structure-aware candidate expansion** module:

1. Call an existing baseline retriever (BM25 / Dense / Hybrid / …) for seeds
2. Expand each seed via ``GraphExpander`` on the Standard Evidence Graph
3. Merge + deduplicate (``unit_id`` unique)
4. Return candidates **without** structure-aware re-ranking

Ordering (v1):
- Keep baseline hits first, preserving their original scores
- Append expanded hits with ``score = 0.0``

This module intentionally does **not** implement Structure-aware Evidence Ranking.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, Sequence, runtime_checkable

from src.retrieval.retriever_base import EvidenceUnit, RetrieverBase
from src.sage_rag.expansion.graph_expander import ExpandedEvidence, GraphExpander

logger = logging.getLogger(__name__)


@runtime_checkable
class SupportsRetrieve(Protocol):
    """Duck-typed baseline retriever — no dependency on concrete class names."""

    def retrieve(self, query: str, top_k: int) -> list[EvidenceUnit]:
        ...


class SageExpansionRetriever(RetrieverBase):
    """Baseline retrieval + graph expansion (candidate generation only)."""

    def __init__(
        self,
        base_retriever: SupportsRetrieve,
        graph_expander: GraphExpander,
        *,
        expansion_edge_types: Sequence[str] | None = None,
        expansion_depth: int = 1,
    ) -> None:
        if not isinstance(base_retriever, SupportsRetrieve):
            # Still allow objects that implement retrieve but fail isinstance
            # when not decorated; require attribute presence.
            if not callable(getattr(base_retriever, "retrieve", None)):
                raise TypeError(
                    "base_retriever must provide retrieve(query, top_k) -> list[EvidenceUnit]"
                )
        if expansion_depth != 1:
            raise ValueError("SageExpansionRetriever v1 only supports expansion_depth=1")

        self.base_retriever = base_retriever
        self.graph_expander = graph_expander
        self.expansion_edge_types = (
            list(expansion_edge_types) if expansion_edge_types is not None else None
        )
        self.expansion_depth = expansion_depth

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        initial_k: int = 5,
    ) -> list[EvidenceUnit]:
        """Retrieve then expand; originals first, expanded fill remaining slots.

        Parameters
        ----------
        query:
            Natural-language query.
        top_k:
            Final list length after merge (originals preferred).
        initial_k:
            How many seeds to pull from the baseline retriever.
        """
        if not query or not str(query).strip():
            raise ValueError("Query must be a non-empty string")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if initial_k <= 0:
            raise ValueError("initial_k must be a positive integer")

        # Step 1 — initial retrieval
        seed_k = min(initial_k, top_k)
        initial = self.base_retriever.retrieve(str(query).strip(), top_k=seed_k)
        initial_marked = [self._mark_initial(u) for u in initial]

        # Step 2 — graph expansion per seed
        expanded_raw: list[tuple[ExpandedEvidence, str]] = []
        for seed in initial_marked:
            try:
                hits = self.graph_expander.expand(
                    seed.unit_id,
                    depth=self.expansion_depth,
                    edge_types=self.expansion_edge_types,
                )
            except Exception as exc:
                logger.warning("Expansion failed for %s: %s", seed.unit_id, exc)
                continue
            for hit in hits:
                expanded_raw.append((hit, seed.unit_id))

        # Step 3 — merge + dedup (unit_id unique); originals win
        seen: set[str] = set()
        merged: list[EvidenceUnit] = []

        for unit in initial_marked:
            if unit.unit_id in seen:
                continue
            seen.add(unit.unit_id)
            merged.append(unit)

        for hit, seed_unit_id in expanded_raw:
            if hit.unit_id in seen:
                continue
            seen.add(hit.unit_id)
            merged.append(self._from_expanded(hit, seed_unit_id=seed_unit_id))

        # Step 4 — truncate to top_k (no re-ranking)
        final = merged[:top_k]
        for rank, unit in enumerate(final, start=1):
            unit.rank = rank
        return final

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _mark_initial(unit: EvidenceUnit) -> EvidenceUnit:
        meta = dict(unit.metadata or {})
        meta["candidate_source"] = "initial"
        meta["original_score"] = float(unit.score) if unit.score is not None else 0.0
        meta["expansion_relation"] = None
        return EvidenceUnit(
            unit_id=unit.unit_id,
            document_id=unit.document_id,
            parent_clause=unit.parent_clause,
            text=unit.text,
            metadata=meta,
            rank=unit.rank,
            score=unit.score,
            document_type=unit.document_type,
            title=unit.title,
            chapter_id=unit.chapter_id,
            chapter_title=unit.chapter_title,
            page=unit.page,
            token_length=unit.token_length,
            char_length=unit.char_length,
            split_index=unit.split_index,
            split_total=unit.split_total,
        )

    @staticmethod
    def _from_expanded(
        hit: ExpandedEvidence,
        *,
        seed_unit_id: str,
    ) -> EvidenceUnit:
        attrs = dict(hit.attributes or {})
        # Evidence-node payload from the graph; strip expander bookkeeping keys
        # into metadata while keeping RetrieverBase.EvidenceUnit fields.
        bookkeeping = {
            "parent_clause_id",
            "parent_chapter_id",
            "immediate_clause_id",
            "reference",
            "target_type",
        }
        eu_meta = {
            k: v
            for k, v in attrs.items()
            if k
            not in {
                "text",
                "page",
                "document_id",
                "chapter_id",
                "parent_clause",
                "token_length",
                "char_length",
                "split_index",
                "split_total",
                *bookkeeping,
            }
        }
        # Preserve structured flags if present on the evidence node.
        if "contains_table" in attrs:
            eu_meta["contains_table"] = attrs["contains_table"]
        if "contains_appendix" in attrs:
            eu_meta["contains_appendix"] = attrs["contains_appendix"]

        eu_meta["candidate_source"] = "expanded"
        eu_meta["original_score"] = 0.0
        eu_meta["expansion_relation"] = hit.relation
        eu_meta["via_node_id"] = hit.via_node_id
        eu_meta["expanded_from"] = seed_unit_id
        for key in bookkeeping:
            if attrs.get(key) not in (None, ""):
                eu_meta[key] = attrs[key]

        return EvidenceUnit(
            unit_id=hit.unit_id,
            document_id=str(attrs.get("document_id") or ""),
            parent_clause=str(attrs.get("parent_clause") or ""),
            text=hit.text or str(attrs.get("text") or ""),
            metadata=eu_meta,
            rank=None,
            score=0.0,
            document_type=str(attrs.get("document_type") or ""),
            title=str(attrs.get("title") or ""),
            chapter_id=str(attrs.get("chapter_id") or ""),
            chapter_title=str(attrs.get("chapter_title") or ""),
            page=int(hit.page or attrs.get("page") or 0),
            token_length=int(attrs.get("token_length") or 0),
            char_length=int(attrs.get("char_length") or 0),
            split_index=int(attrs.get("split_index") or 1),
            split_total=int(attrs.get("split_total") or 1),
        )
