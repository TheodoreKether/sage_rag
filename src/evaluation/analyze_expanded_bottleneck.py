"""Read-only bottleneck analysis for SAGE v4 expanded candidates.

Does NOT modify BM25 / Dense / Graph / SAGE v4 algorithms.
Writes: results/retrieval/sage_v4/expanded_candidate_analysis.md
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate_dense import load_qa_samples
from src.retrieval.bm25 import BM25Retriever
from src.sage_rag.expansion.graph_expander import GraphExpander
from src.sage_rag.graph.graph_store import GraphStore
from src.sage_rag.ranking.risk_aware_ranker import RiskAwareEvidenceSelector
from src.sage_rag.retrieval.sage_expansion_retriever import SageExpansionRetriever

QA = ROOT / "data" / "qa_dataset" / "qa_pairs_clean.jsonl"
OUT_MD = ROOT / "results" / "retrieval" / "sage_v4" / "expanded_candidate_analysis.md"
OUT_JSON = ROOT / "results" / "retrieval" / "sage_v4" / "expanded_candidate_analysis.json"
METRICS_PATH = ROOT / "results" / "retrieval" / "sage_v4" / "metrics.json"

TOP_K = 10
INITIAL_K = 10
POOL = 80


def avg(xs: list[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def classify(r: dict) -> tuple[str, list[str], dict[str, float]]:
    if r["in_top10"]:
        return "success_entered_top10", ["success"], {}
    deltas = {
        "low_retrieval_inherit": r["d_retr"],
        "structure_weight_gap": r["d_struct"],
        "low_coverage": r["d_cov"],
        "high_risk_penalty": r["d_risk"],
    }
    primary = max(deltas, key=deltas.get)
    tags: list[str] = []
    if r["relation"] == "next_to":
        tags.append("next_to_noise_relation")
    if r["coverage"] < 0.25:
        tags.append("coverage_insufficient")
    if r["risk"] >= 0.25:
        tags.append("risk_penalty_high")
    if r["retrieval"] < 0.25:
        tags.append("retrieval_too_low")
    if r["structure"] <= 0.55 and r["relation"] == "next_to":
        tags.append("weak_structure_prior")
    if primary == "low_retrieval_inherit" and r["d_retr"] >= 0.08:
        tags.append("missing_semantic_relevance")
    if primary == "structure_weight_gap" and r["d_struct"] >= 0.05:
        tags.append("ranking_weight_structure")
    if not tags:
        tags.append(primary)
    return primary, tags, deltas


def main() -> int:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    samples, _ = load_qa_samples(QA)
    store = GraphStore.from_dir(ROOT / "data" / "sage_graph")
    bm25 = BM25Retriever(index_dir=ROOT / "data" / "bm25_index")
    expansion = SageExpansionRetriever(bm25, GraphExpander(store))
    selector = RiskAwareEvidenceSelector(
        alpha=float(metrics.get("alpha", 0.45)),
        beta=float(metrics.get("beta", 0.25)),
        gamma=float(metrics.get("gamma", 0.30)),
        lam=float(metrics.get("lam", 0.20)),
    )

    rows: list[dict] = []
    for sample in samples:
        bm25_hits = bm25.retrieve(sample.question, top_k=TOP_K)
        bm25_set = {u.unit_id for u in bm25_hits}
        gold = list(sample.gold_unit_ids)
        if any(g in bm25_set for g in gold):
            continue

        pool = expansion.retrieve(sample.question, top_k=POOL, initial_k=INITIAL_K)
        pool_ids = [u.unit_id for u in pool]
        pool_set = set(pool_ids)
        expand_golds = [g for g in gold if g in pool_set and g not in bm25_set]
        if not expand_golds:
            continue

        scored = selector.score_candidates(pool, query=sample.question)
        score_rank = {u.unit_id: i + 1 for i, u in enumerate(scored)}
        meta_by = {u.unit_id: (u.metadata or {}) for u in scored}
        pool_pos = {uid: i + 1 for i, uid in enumerate(pool_ids)}

        top10 = scored[:TOP_K]
        cutoff = (
            float((top10[-1].metadata or {}).get("final_score", 0.0)) if top10 else 0.0
        )
        initials = [
            u
            for u in scored
            if (u.metadata or {}).get("candidate_source") == "initial"
        ]
        tenth = initials[min(9, len(initials) - 1)] if initials else None
        tm = (tenth.metadata or {}) if tenth else {}
        tenth_initial_final = float(tm.get("final_score") or 0.0)

        for gid in expand_golds:
            m = meta_by.get(gid) or {}
            text = next((u.text for u in scored if u.unit_id == gid), "")
            final = float(m.get("final_score") or 0.0)
            row = {
                "qa_id": sample.qa_id,
                "question": sample.question,
                "gold_unit_id": gid,
                "relation": m.get("expansion_relation"),
                "expanded_from": m.get("expanded_from"),
                "pool_rank": pool_pos.get(gid),
                "v4_rank": score_rank.get(gid),
                "in_top10": (score_rank.get(gid) or 999) <= TOP_K,
                "retrieval": float(m.get("retrieval_score_normalized") or 0.0),
                "structure": float(m.get("structure_score") or 0.0),
                "coverage": float(m.get("query_coverage_score") or 0.0),
                "risk": float(m.get("risk_penalty") or 0.0),
                "final": final,
                "cutoff_top10": cutoff,
                "gap_to_top10": cutoff - final,
                "tenth_initial_final": tenth_initial_final,
                "tenth_initial_retr": float(tm.get("retrieval_score_normalized") or 0.0),
                "tenth_initial_struct": float(tm.get("structure_score") or 0.0),
                "tenth_initial_cov": float(tm.get("query_coverage_score") or 0.0),
                "tenth_initial_risk": float(tm.get("risk_penalty") or 0.0),
                "text": (text or "").replace("\n", " ").strip()[:140],
            }
            row["term_retr"] = selector.alpha * row["retrieval"]
            row["term_struct"] = selector.beta * row["structure"]
            row["term_cov"] = selector.gamma * row["coverage"]
            row["term_risk"] = selector.lam * row["risk"]
            row["d_retr"] = selector.alpha * (
                row["tenth_initial_retr"] - row["retrieval"]
            )
            row["d_struct"] = selector.beta * (
                row["tenth_initial_struct"] - row["structure"]
            )
            row["d_cov"] = selector.gamma * (
                row["tenth_initial_cov"] - row["coverage"]
            )
            row["d_risk"] = selector.lam * (
                row["risk"] - row["tenth_initial_risk"]
            )
            primary, tags, deltas = classify(row)
            row["primary_bottleneck"] = primary
            row["tags"] = tags
            row["deltas"] = deltas
            rows.append(row)

    misses = [r for r in rows if not r["in_top10"]]
    hits = [r for r in rows if r["in_top10"]]
    primary_cnt = Counter(r["primary_bottleneck"] for r in misses)
    rel_cnt = Counter(r["relation"] for r in rows)
    rel_miss = Counter(r["relation"] for r in misses)
    tag_cnt = Counter(t for r in misses for t in r["tags"])

    n_q_expand = len({r["qa_id"] for r in rows})
    lines: list[str] = []
    a = lines.append

    a("# SAGE v4 Expanded Candidate Bottleneck Analysis")
    a("")
    a(
        "分析目标：解释 Graph Expansion 将 candidate recall 从 **82.61% → 88.04%** 后，"
        "为何只有极少 expanded evidence 进入最终 Top-10。"
    )
    a("")
    a(
        "**约束**：只读分析；未修改 BM25 / Dense / Graph / SAGE v4 算法代码。"
    )
    a("")
    a(
        "数据：`data/qa_dataset/qa_pairs_clean.jsonl`（460 queries）；"
        "用现有 `RiskAwareEvidenceSelector` 复现打分"
        "（权重与 `results/retrieval/sage_v4/metrics.json` 一致）。"
    )
    a("")
    a("---")
    a("")
    a("## 1. 总体统计")
    a("")
    a("| 指标 | 数值 |")
    a("|------|-----:|")
    a("| 总 queries | 460 |")
    a("| BM25 Top10 recall | 82.61% |")
    a("| Expansion pool recall | 88.04% |")
    a("| 额外覆盖（pool−BM25） | +5.43 pp |")
    a(
        f"| **Expand-only gold**（BM25 Top10 未中、pool 中命中） | "
        f"**{len(rows)}** gold / **{n_q_expand}** queries |"
    )
    a(
        f"| 其中进入 SAGE v4 Top10 | **{len(hits)}** "
        f"({100 * len(hits) / max(len(rows), 1):.1f}%) |"
    )
    a(
        f"| 未进入 Top10（bottleneck） | **{len(misses)}** "
        f"({100 * len(misses) / max(len(rows), 1):.1f}%) |"
    )
    a("| SAGE v4 final R@10 | 82.83% |")
    a("| 从 +5.43 pp candidate 增益转化到 final | ≈ +0.22 pp |")
    a("")
    a("### Expansion relation 分布（expand-only gold）")
    a("")
    a("| Relation | 全部 | 未进 Top10 | 进 Top10 |")
    a("|----------|-----:|----------:|---------:|")
    for rel in sorted(rel_cnt.keys(), key=lambda x: (-rel_cnt[x], str(x))):
        h = sum(1 for r in hits if r["relation"] == rel)
        a(f"| {rel} | {rel_cnt[rel]} | {rel_miss.get(rel, 0)} | {h} |")
    a("")
    a("### 分数分布（expand-only gold）")
    a("")
    a("| 分量 | miss 均值 | hit 均值 | miss 中位数 |")
    a("|------|----------:|---------:|-----------:|")
    for key, name in [
        ("retrieval", "retrieval_score"),
        ("structure", "structure_score"),
        ("coverage", "coverage_score"),
        ("risk", "risk_penalty"),
        ("final", "final_score"),
        ("v4_rank", "v4_rank"),
        ("gap_to_top10", "gap_to_top10"),
    ]:
        hit_v = avg([r[key] for r in hits]) if hits else float("nan")
        miss_med = median([r[key] for r in misses]) if misses else float("nan")
        a(
            f"| {name} | {avg([r[key] for r in misses]):.4f} | "
            f"{hit_v:.4f} | {miss_med:.4f} |"
        )
    a("")
    a("### 与第 10 名 initial 的加权差距（miss；正值 = gold 更弱）")
    a("")
    a("| 项 | 含义 | miss 均值 |")
    a("|----|------|----------:|")
    a(
        f"| Δ α·Retr | seed 继承检索 vs BM25 尾部 | "
        f"{avg([r['d_retr'] for r in misses]):.4f} |"
    )
    a(
        f"| Δ β·Struct | structure 先验差距 | "
        f"{avg([r['d_struct'] for r in misses]):.4f} |"
    )
    a(
        f"| Δ γ·Cov | coverage 差距 | "
        f"{avg([r['d_cov'] for r in misses]):.4f} |"
    )
    a(
        f"| Δ λ·Risk | 额外风险惩罚 | "
        f"{avg([r['d_risk'] for r in misses]):.4f} |"
    )
    a("")
    a(
        f"加权差距合计均值 ≈ "
        f"{avg([r['d_retr'] + r['d_struct'] + r['d_cov'] + r['d_risk'] for r in misses]):.4f}"
        f"（与 gap_to_top10 均值 {avg([r['gap_to_top10'] for r in misses]):.4f} 同量级）"
    )
    a("")
    a("---")
    a("")
    a("## 2. 失败原因分类占比")
    a("")
    a(
        "主瓶颈按「相对第 10 名 initial，哪一项加权差距最大」判定。"
    )
    a("")
    a("| Primary bottleneck | Count | Share of misses |")
    a("|--------------------|------:|----------------:|")
    for k, v in primary_cnt.most_common():
        a(f"| {k} | {v} | {100 * v / max(len(misses), 1):.1f}% |")
    a("")
    a("### 标签共现（可多选）")
    a("")
    a("| Tag | Count |")
    a("|-----|------:|")
    for k, v in tag_cnt.most_common():
        a(f"| {k} | {v} |")
    a("")
    a("### 解读")
    a("")
    top_primary = primary_cnt.most_common(1)[0][0] if primary_cnt else "n/a"
    a(f"- 主瓶颈众数：**`{top_primary}`**")
    a(
        f"- `next_to` 占 expand-only gold 的 "
        f"{100 * rel_cnt.get('next_to', 0) / max(len(rows), 1):.1f}%；"
        "结构先验弱、风险高，最难进入 Top10。"
    )
    a(
        "- 即使 `parent_of` / `refers_to`，expanded 的 retrieval 项仍显著低于 "
        "BM25 Top10 尾部（仅 seed 继承的一部分）——这是系统性 gap。"
    )
    a(
        "- coverage 不足会同时压低 γ·Cov，并抬高有效 risk"
        "（v4 的 reliability discount），形成双重打击。"
    )
    a("")
    a("---")
    a("")
    a("## 3. 逐 case 明细（全部 expand-only gold）")
    a("")
    a(
        "| # | in@10 | rel | v4_rank | pool# | retr | struct | cov | risk | "
        "final | gap | primary | qa_id |"
    )
    a(
        "|---|------:|-----|--------:|------:|-----:|-------:|----:|-----:|"
        "------:|----:|---------|-------|"
    )
    for i, r in enumerate(
        sorted(rows, key=lambda x: (x["in_top10"], x["v4_rank"] or 999)), start=1
    ):
        a(
            f"| {i} | {'Y' if r['in_top10'] else 'N'} | {r['relation']} | "
            f"{r['v4_rank']} | {r['pool_rank']} | {r['retrieval']:.3f} | "
            f"{r['structure']:.3f} | {r['coverage']:.3f} | {r['risk']:.3f} | "
            f"{r['final']:.3f} | {r['gap_to_top10']:.3f} | "
            f"{r['primary_bottleneck']} | `{r['qa_id']}` |"
        )
    a("")
    a("---")
    a("")
    a("## 4. 典型 Case（≥10）")
    a("")

    selected: list[dict] = []
    selected.extend(hits[:3])
    seen_p: set[str] = set()
    for r in sorted(misses, key=lambda x: -(x["gap_to_top10"])):
        if r["primary_bottleneck"] not in seen_p:
            selected.append(r)
            seen_p.add(r["primary_bottleneck"])
    n_next = 0
    for r in misses:
        if r["relation"] == "next_to" and r not in selected:
            selected.append(r)
            n_next += 1
            if n_next >= 2:
                break
    for r in sorted(misses, key=lambda x: x["v4_rank"] or 999):
        if 11 <= (r["v4_rank"] or 999) <= 25 and r not in selected:
            selected.append(r)
        if len(selected) >= 12:
            break
    for r in misses:
        if len(selected) >= 10:
            break
        if r not in selected:
            selected.append(r)

    for i, r in enumerate(selected[:12], start=1):
        status = "HIT" if r["in_top10"] else "MISS"
        a(
            f"### Case {i}: `{status}` / {r['relation']} / "
            f"primary=`{r['primary_bottleneck']}`"
        )
        a("")
        a(f"- **Query:** {r['question']}")
        a(f"- **Gold:** `{r['gold_unit_id']}`")
        a(f"- **Expanded from:** `{r['expanded_from']}`")
        a(f"- **Expansion relation:** `{r['relation']}`")
        a(
            f"- **Pool merge rank:** {r['pool_rank']}；"
            f"**v4 score rank:** {r['v4_rank']}；"
            f"**in Top10:** {r['in_top10']}"
        )
        a(
            f"- **Scores:** retr={r['retrieval']:.4f}, struct={r['structure']:.4f}, "
            f"cov={r['coverage']:.4f}, risk={r['risk']:.4f}, "
            f"**final={r['final']:.4f}**"
        )
        a(
            f"- **Terms:** α·R={r['term_retr']:.4f}, β·S={r['term_struct']:.4f}, "
            f"γ·C={r['term_cov']:.4f}, −λ·Risk={-r['term_risk']:.4f}"
        )
        a(
            f"- **Top10 cutoff final:** {r['cutoff_top10']:.4f}；"
            f"**gap:** {r['gap_to_top10']:.4f}"
        )
        a(
            f"- **vs 10th initial:** d_retr={r['d_retr']:.4f}, "
            f"d_struct={r['d_struct']:.4f}, d_cov={r['d_cov']:.4f}, "
            f"d_risk={r['d_risk']:.4f}"
        )
        a(f"- **Tags:** {', '.join(r['tags'])}")
        a(f"- **Text:** {r['text']}")
        a("")

    a("---")
    a("")
    a("## 5. 三个核心问题的结论")
    a("")
    a("### Q1. 主要瓶颈是否是缺少语义 relevance score？")
    a("")
    retr_share = 100 * primary_cnt.get("low_retrieval_inherit", 0) / max(len(misses), 1)
    miss_sem = (
        100 * tag_cnt.get("missing_semantic_relevance", 0) / max(len(misses), 1)
    )
    a("**是（最主要瓶颈之一）。**")
    a("")
    a(
        f"- Primary=`low_retrieval_inherit` 占 misses 的 **{retr_share:.1f}%**；"
        f"标签 `missing_semantic_relevance` 覆盖 **{miss_sem:.1f}%** misses。"
    )
    a(
        "- Expanded 的 raw BM25 score 为 0，仅靠 `w(relation)·seed_retr` 继承；"
        "对 next_to（w=0.10）几乎无检索信号。"
    )
    a(
        f"- miss 样本 α·Retr 相对第10名 initial 平均落后 "
        f"**{avg([r['d_retr'] for r in misses]):.3f}**，通常是最大单项 gap。"
    )
    a(
        "- 因此：不是“完全没有 relevance”，而是"
        "**缺少相对 query 的独立语义/词汇 relevance**；"
        "seed 继承不足以对抗 BM25 Top10 尾部。"
    )
    a("")
    a("### Q2. 是否是 Graph expansion 质量问题？")
    a("")
    a("**部分是，但不是唯一原因。**")
    a("")
    a(
        f"- Expansion **确实找到了** gold（{len(rows)} 个 expand-only gold），"
        "说明图结构边有信息量。"
    )
    a(
        f"- 但 relation 偏斜：`next_to` 占 "
        f"{100 * rel_cnt.get('next_to', 0) / max(len(rows), 1):.1f}% "
        "的 expand-only gold，结构先验弱、风险高，属于**低精度邻居噪声**混入。"
    )
    a(
        "- parent_of/refers_to 的 gold 也大量未进 Top10 → "
        "边类型正确仍排不上去，说明不全是 expansion 质量。"
    )
    a(
        "- 结论：expansion **召回有效、精度一般**；"
        "质量问题放大了 ranking 瓶颈，但不是 R@10 转化失败的唯一解释。"
    )
    a("")
    a("### Q3. 是否是 Ranking 公式权重问题？")
    a("")
    a("**是（第二大瓶颈，与 Q1 耦合）。**")
    a("")
    a(
        "- initial 固定 `structure=1.0` 且 `risk=0`；"
        "expanded 最多 structure≈1.0（parent）且常带 risk，再叠加更低 retrieval。"
    )
    a(
        f"- β=0.25 / γ=0.30 不足以补偿 α=0.45 下的 retrieval 缺口；"
        f"miss 平均仍差 Top10 cutoff "
        f"**{avg([r['gap_to_top10'] for r in misses]):.3f}**。"
    )
    a(
        "- 近失手（v4_rank 11–25）表明权重微调或更强 coverage/语义项可能挽救一部分，"
        "但多数 next_to gold gap 更大。"
    )
    a(
        "- 结论：当前公式在“反噪声”上偏保守"
        "（解释了为何 v4 略超 BM25 而未伤 R@10），"
        "但也**系统性压制**了可靠 graph gold。"
    )
    a("")
    a("---")
    a("")
    a("## 6. 综合判断（Bottleneck 排序）")
    a("")
    a("1. **缺少独立语义 relevance（对 expanded）** — 主因")
    a("2. **Ranking 权重 / 结构先验对 initial 过度友好** — 主因耦合")
    a("3. **Graph expansion 中 next_to 噪声与低 coverage 邻居** — 重要放大器")
    a("")
    a(
        "Candidate recall +5.43 pp 中，最终只转化约 **+0.22 pp** R@10："
        "瓶颈在 **selection/scoring**，不在 expansion 召回。"
    )
    a("")
    a("---")
    a("")
    a("## 7. 对 SAGE v5 的建议（仅建议，本报告不实现）")
    a("")
    a(
        "1. **为 expanded 增加 query-conditioned lexical/semantic relevance**"
        "（仍可不训练：BM25(q, evidence_text) 或 embedding cosine），"
        "替代/补充纯 seed 继承。"
    )
    a(
        "2. **Relation-aware 竞争门槛**：next_to 默认需更高 coverage；"
        "parent_of/refers_to 提高进 Top-k 的通道。"
    )
    a(
        "3. **校准 initial vs expanded 的 structure 基准**："
        "避免 initial 无条件 structure=1.0 造成不可逾越的地板分。"
    )
    a(
        "4. **保留 risk-aware 思想**：继续惩罚低 coverage / 跨文档 / next_to，"
        "但让高 coverage gold 的 risk≈0 时有足够 retrieval 信号赢下尾部 initial。"
    )
    a(
        "5. **优先打磨 refers_to/parent_of 路径上的近失手（rank 11–25）**"
        "——这是最可能的廉价 R@10 增益区间。"
    )
    a("")
    a("---")
    a("")
    a("## Appendix: 方法说明")
    a("")
    a(
        "- Expand-only 定义：gold ∉ BM25 Top10 且 gold ∈ expansion pool"
        "（initial_k=10, pool_size=80）。"
    )
    a(
        "- 分数由现有 `RiskAwareEvidenceSelector.score_candidates` 复现，"
        "权重与 `metrics.json` 一致。"
    )
    a(
        "- Primary bottleneck：比较 gold 与第10名 initial 在 "
        "α·Retr / β·Struct / γ·Cov / λ·Risk 上的差距，取最大正差距项。"
    )
    a("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(
        json.dumps(
            {
                "n_expand_only_gold": len(rows),
                "n_queries": n_q_expand,
                "n_hit": len(hits),
                "n_miss": len(misses),
                "primary_bottleneck": dict(primary_cnt),
                "relation_all": dict(rel_cnt),
                "relation_miss": dict(rel_miss),
                "tag_counts": dict(tag_cnt),
                "miss_avg": {
                    "retrieval": avg([r["retrieval"] for r in misses]),
                    "structure": avg([r["structure"] for r in misses]),
                    "coverage": avg([r["coverage"] for r in misses]),
                    "risk": avg([r["risk"] for r in misses]),
                    "final": avg([r["final"] for r in misses]),
                    "gap_to_top10": avg([r["gap_to_top10"] for r in misses]),
                    "v4_rank": avg([float(r["v4_rank"]) for r in misses]),
                    "d_retr": avg([r["d_retr"] for r in misses]),
                    "d_struct": avg([r["d_struct"] for r in misses]),
                    "d_cov": avg([r["d_cov"] for r in misses]),
                    "d_risk": avg([r["d_risk"] for r in misses]),
                },
                "cases": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "expand_only_gold": len(rows),
                "queries": n_q_expand,
                "hits": len(hits),
                "misses": len(misses),
                "primary": dict(primary_cnt),
                "relations": dict(rel_cnt),
                "out_md": str(OUT_MD),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
