"""Structure-aware Evidence Selection (v2).

Score(e | q, Selected) =
    α · RetrievalScore(e)
  + β · StructurePathScore(e)
  + γ · QueryCoverageScore(e)
  − λ · RedundancyPenalty(e | Selected)

Selection is **greedy**: repeatedly pick the candidate maximizing the conditional
score given already selected evidence (true Evidence Selection, not flat sort).

v1 (``StructureRanker``) is kept for ablation.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any, Sequence

from src.retrieval.retriever_base import EvidenceUnit
from src.retrieval.text_tokenizer import tokenize

logger = logging.getLogger(__name__)

# Relation priors (v2; next_to raised vs v1 to allow sibling recovery).
PATH_SCORE_INITIAL = 1.0
PATH_SCORE_BY_RELATION: dict[str, float] = {
    "parent_of": 0.8,
    "refers_to": 0.7,
    "next_to": 0.5,
}

_CLAUSE_NUM = re.compile(r"\b\d+(?:\.\d+)+\b")
_STD_REF = re.compile(
    r"(?:GB/?T|ISO|IEC)\s*[\d\-—–．.]+",
    re.IGNORECASE,
)
_STOPWORDS = {
    "的",
    "是",
    "什么",
    "如何",
    "怎么",
    "哪些",
    "一个",
    "以及",
    "或者",
    "对于",
    "进行",
    "要求",
    "规定",
    "相关",
    "有关",
    "是否",
    "可以",
    "需要",
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "for",
    "and",
    "or",
    "is",
    "are",
    "what",
    "how",
    "which",
}


_GENERIC_TERMS = {
    "审计",
    "安全",
    "要求",
    "规定",
    "检查",
    "测试",
    "方法",
    "系统",
    "数据",
    "信息",
    "功能",
    "处理",
    "使用",
    "实现",
    "相关",
    "标准",
    "文档",
}


class StructureRankerV2:
    """Greedy structure-aware evidence selection."""

    def __init__(
        self,
        alpha: float = 0.40,
        beta: float = 0.20,
        gamma: float = 0.45,
        lam: float = 0.35,
    ) -> None:
        if min(alpha, beta, gamma, lam) < 0:
            raise ValueError("weights must be non-negative")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.lam = float(lam)

    def rank(
        self,
        candidates: Sequence[EvidenceUnit],
        query: str = "",
        top_k: int | None = None,
    ) -> list[EvidenceUnit]:
        """Greedy select up to ``top_k`` (default: all) from the candidate pool."""
        if not candidates:
            return []

        pool = list(candidates)
        k = len(pool) if top_k is None else max(1, min(int(top_k), len(pool)))

        raw = [self._retrieval_score(c) for c in pool]
        retr_norm = self._minmax_normalize(raw)
        query_terms = self._extract_query_terms(query)

        # Precompute independent components.
        path_scores = [self._structure_path_score(c) for c in pool]
        cov_scores = [self._query_coverage_score(c, query_terms) for c in pool]

        selected: list[EvidenceUnit] = []
        selected_idx: list[int] = []
        remaining = set(range(len(pool)))

        while remaining and len(selected) < k:
            best_i = -1
            best_score = float("-inf")
            best_parts: dict[str, float] = {}
            for i in remaining:
                red = self._redundancy_penalty(pool[i], [pool[j] for j in selected_idx])
                score = (
                    self.alpha * retr_norm[i]
                    + self.beta * path_scores[i]
                    + self.gamma * cov_scores[i]
                    - self.lam * red
                )
                if score > best_score or (
                    abs(score - best_score) < 1e-12
                    and best_i >= 0
                    and pool[i].unit_id < pool[best_i].unit_id
                ):
                    best_score = score
                    best_i = i
                    best_parts = {
                        "retrieval_score_normalized": retr_norm[i],
                        "structure_path_score": path_scores[i],
                        "query_coverage_score": cov_scores[i],
                        "redundancy_penalty": red,
                        "final_score": score,
                    }
            assert best_i >= 0
            annotated = self._annotate(pool[best_i], **best_parts)
            selected.append(annotated)
            selected_idx.append(best_i)
            remaining.remove(best_i)

        for rank, unit in enumerate(selected, start=1):
            unit.rank = rank
        return selected

    # ------------------------------------------------------------------ scores

    @staticmethod
    def _retrieval_score(unit: EvidenceUnit) -> float:
        meta = unit.metadata or {}
        if meta.get("original_score") is not None:
            try:
                return float(meta["original_score"])
            except (TypeError, ValueError):
                pass
        if unit.score is not None:
            try:
                return float(unit.score)
            except (TypeError, ValueError):
                pass
        return 0.0

    @staticmethod
    def _minmax_normalize(scores: Sequence[float]) -> list[float]:
        if not scores:
            return []
        if len(scores) == 1:
            return [1.0]
        lo = min(scores)
        hi = max(scores)
        if hi <= lo:
            return [1.0 for _ in scores]
        return [(s - lo) / (hi - lo) for s in scores]

    def _structure_path_score(self, unit: EvidenceUnit) -> float:
        meta = unit.metadata or {}
        if meta.get("candidate_source") == "initial":
            return PATH_SCORE_INITIAL
        relation = str(meta.get("expansion_relation") or "")
        base = float(PATH_SCORE_BY_RELATION.get(relation, 0.0))
        if base <= 0:
            return 0.0
        dist = self._graph_distance(unit)
        if dist <= 1:
            return base
        # Soft decay: parent_of@2 ≈ 0.6 when base=0.8
        return max(0.0, base * (0.75 ** (dist - 1)))

    @staticmethod
    def _graph_distance(unit: EvidenceUnit) -> int:
        """Estimate hop distance from expansion seed (default 1 for depth-1 edges)."""
        meta = unit.metadata or {}
        if meta.get("candidate_source") == "initial":
            return 0
        # Prefer explicit attribute if ever present.
        if meta.get("graph_distance") is not None:
            try:
                return max(1, int(meta["graph_distance"]))
            except (TypeError, ValueError):
                pass
        # Infer from clause numbering vs expanded_from seed clause.
        seed = str(meta.get("expanded_from") or "")
        cand_clause = str(unit.parent_clause or "")
        seed_clause = StructureRankerV2._clause_from_unit_id(seed)
        if seed_clause and cand_clause:
            if cand_clause == seed_clause:
                return 1
            if seed_clause.startswith(cand_clause + ".") or cand_clause.startswith(
                seed_clause + "."
            ):
                # parent/child along numbering → 1 hop in our expander
                depth_gap = abs(
                    seed_clause.count(".") - cand_clause.count(".")
                )
                return max(1, depth_gap)
        return 1

    @staticmethod
    def _clause_from_unit_id(unit_id: str) -> str:
        # unit_id: doc::chapter::parent_clause::split
        parts = unit_id.split("::")
        if len(parts) >= 3:
            return parts[2]
        return ""

    def _query_coverage_score(
        self, unit: EvidenceUnit, query_terms: dict[str, float]
    ) -> float:
        """Weighted overlap in [0, 1]; multi-hit & specific terms score higher."""
        if not query_terms:
            return 0.0
        blob = " ".join(
            [
                unit.text or "",
                unit.parent_clause or "",
                unit.chapter_id or "",
                unit.chapter_title or "",
                unit.title or "",
                unit.document_id or "",
            ]
        )
        blob_l = blob.lower()
        blob_toks = set(tokenize(blob))
        hits = 0.0
        weight_sum = 0.0
        hit_count = 0
        for term, w in query_terms.items():
            weight_sum += w
            if term in blob_toks or term in blob_l:
                hits += w
                hit_count += 1
        if weight_sum <= 0 or hit_count == 0:
            return 0.0
        raw = hits / weight_sum
        # Prefer covering multiple distinct query cues (not a single generic token).
        multi = min(1.0, hit_count / max(2.0, min(len(query_terms), 4.0)))
        return max(0.0, min(1.0, 0.65 * raw + 0.35 * multi))

    @staticmethod
    def _extract_query_terms(query: str) -> dict[str, float]:
        """Return term → weight (clause/std/phrases heavier than generic words)."""
        if not query or not query.strip():
            return {}
        weights: dict[str, float] = {}

        def add(term: str, w: float) -> None:
            term = term.strip().lower()
            if not term or term in _STOPWORDS:
                return
            weights[term] = max(weights.get(term, 0.0), w)

        for m in _CLAUSE_NUM.findall(query):
            add(m, 3.0)
        for m in _STD_REF.findall(query):
            add(re.sub(r"\s+", "", m), 3.0)

        # CJK character n-grams keep phrases like 除零 / 除零错误.
        for span in re.findall(r"[\u4e00-\u9fff]+", query):
            if len(span) >= 2:
                for i in range(len(span) - 1):
                    add(span[i : i + 2], 2.0)
            if len(span) >= 3:
                for i in range(len(span) - 2):
                    add(span[i : i + 3], 2.5)
            if len(span) >= 4:
                for i in range(len(span) - 3):
                    add(span[i : i + 4], 3.0)

        for tok in tokenize(query):
            t = tok.strip().lower()
            if not t or t in _STOPWORDS or len(t) < 2:
                continue
            if t in _GENERIC_TERMS:
                add(t, 0.4)
            else:
                add(t, 1.5)
        return weights

    def _redundancy_penalty(
        self, unit: EvidenceUnit, selected: Sequence[EvidenceUnit]
    ) -> float:
        if not selected:
            return 0.0
        penalty = 0.0
        u_clause = (unit.parent_clause or "").strip()
        u_chapter = (unit.chapter_id or "").strip()
        u_doc = (unit.document_id or "").strip()
        for s in selected:
            s_clause = (s.parent_clause or "").strip()
            s_chapter = (s.chapter_id or "").strip()
            s_doc = (s.document_id or "").strip()
            if u_doc and u_doc == s_doc and u_clause and u_clause == s_clause:
                penalty += 0.55  # same clause / split siblings
                continue
            if (
                u_doc
                and u_doc == s_doc
                and u_chapter
                and u_chapter == s_chapter
                and u_clause
                and s_clause
                and self._near_clauses(u_clause, s_clause)
            ):
                penalty += 0.35  # adjacent / parent-child clutter
                continue
            if u_doc and u_doc == s_doc and u_chapter and u_chapter == s_chapter:
                penalty += 0.12
        return min(penalty, 0.95)

    @staticmethod
    def _near_clauses(a: str, b: str) -> bool:
        if a == b:
            return True
        if a.startswith(b + ".") or b.startswith(a + "."):
            return True
        # Sibling leaves: 6.1.1.12 vs 6.1.1.13
        if "." in a and "." in b:
            ap, a_last = a.rsplit(".", 1)
            bp, b_last = b.rsplit(".", 1)
            if ap == bp and a_last.isdigit() and b_last.isdigit():
                return abs(int(a_last) - int(b_last)) <= 2
        return False

    @staticmethod
    def _annotate(
        unit: EvidenceUnit,
        *,
        retrieval_score_normalized: float,
        structure_path_score: float,
        query_coverage_score: float,
        redundancy_penalty: float,
        final_score: float,
    ) -> EvidenceUnit:
        meta = copy.deepcopy(unit.metadata or {})
        meta["retrieval_score_normalized"] = float(retrieval_score_normalized)
        meta["structure_path_score"] = float(structure_path_score)
        meta["graph_score"] = float(structure_path_score)  # alias for report compat
        meta["query_coverage_score"] = float(query_coverage_score)
        meta["redundancy_penalty"] = float(redundancy_penalty)
        meta["final_score"] = float(final_score)
        meta["ranker"] = "structure_ranker_v2"
        if "original_score" not in meta:
            meta["original_score"] = float(unit.score) if unit.score is not None else 0.0

        return EvidenceUnit(
            unit_id=unit.unit_id,
            document_id=unit.document_id,
            parent_clause=unit.parent_clause,
            text=unit.text,
            metadata=meta,
            rank=unit.rank,
            score=float(final_score),
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
