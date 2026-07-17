"""Reciprocal Rank Fusion utilities for hybrid retrieval."""

from __future__ import annotations

from collections import defaultdict

from .retriever_base import EvidenceUnit

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[EvidenceUnit]],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[tuple[str, float, EvidenceUnit]]:
    """Fuse multiple ranked lists with RRF.

    score(d) = sum_i 1 / (k + rank_i(d))

    Only rank positions from each list contribute; raw retriever scores are ignored.
    """
    if k < 0:
        raise ValueError("RRF constant k must be non-negative")

    fused_scores: dict[str, float] = defaultdict(float)
    representative: dict[str, EvidenceUnit] = {}

    for hits in ranked_lists:
        for hit in hits:
            if hit.rank is None or hit.rank <= 0:
                continue
            fused_scores[hit.unit_id] += 1.0 / (k + hit.rank)
            representative.setdefault(hit.unit_id, hit)

    ordered = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [(unit_id, score, representative[unit_id]) for unit_id, score in ordered]
