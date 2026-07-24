# Sage RAG

面向**结构化技术标准文档**（GB / ISO / IEC）的研究型 RAG 基准与方法验证平台。

与将 PDF 当作纯文本切块的通用 RAG 不同，本项目在解析、切分与评测中保留**条款级结构**（章节 / 条款 / 附录 / 表格 / 交叉引用），用于回答一个核心问题：

> 在标准文档检索上，BM25 / Dense / Hybrid 还差在哪里？这些差距是否需要结构信息（SAGE-RAG）？

---

## 当前进度（论文阶段）

| 阶段 | 状态 | 说明 |
|------|------|------|
| 1. 数据构建 | ✅ | 13 份 PDF → 565 Evidence Units + Dense/BM25 索引 |
| 2. QA 评测集 | ✅ | V2 自然语言 QA（492）→ 对齐改写 → **Clean（460）** |
| 3. 检索基线 | ✅ | Dense / BM25 / Hybrid（RRF） |
| 4. Root Cause Analysis | ✅ | 过滤 Dataset Issue 后的失败归因 |
| 5. Design Motivation | ✅ | 结构模块动机初稿 |
| 6. SAGE-RAG 图构建 | ✅ | Evidence Unit → Standard Evidence Graph |
| 7. Graph Store + Expansion | ✅ | 内存图索引 + depth=1 结构扩展（pool recall ≈91.5%） |
| 8. SAGE v1 Ranking | ✅ | retrieval ⊕ graph 规则打分 |
| 9. SAGE v2 Selection | ✅ | Greedy structure-aware evidence selection |
| 10. SAGE v3 Allocation | ✅ | 硬槽位把 expanded 塞进 Top10（宏观指标回退） |
| 11. SAGE v4 Risk-aware | ✅ | Retr + Struct + Cov − Risk；首次 R@10 略超 BM25 |
| 12. SAGE v5 Semantic | ✅ | BM25 semantic rescoring + initial_k=20；R@10=**85.00%** |
| 13. Dense+Graph 实验 | ✅ | Dense 校准 expanded 排序；Hybrid 融合 R@10=**86.09%** |

**论文主评测集：** `data/qa_dataset/qa_pairs_clean.jsonl`（460 queries）  
**主基线结果：** `results/retrieval/clean_benchmark/`  
**当前最优 SAGE：** Hybrid BM25+Dense+Graph → `results/retrieval/sage_dense_graph/`  
**主失败分析：** `results/root_cause_analysis_clean/`  
**结构图：** `data/sage_graph/`（统计见 `results/sage_graph_statistics.md`）

---

## 主实验结果（Clean Benchmark，460 queries）

### 基线

| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR |
|-----------|---------:|---------:|----------:|----:|
| BM25 | 0.5435 | 0.7652 | 0.8261 | 0.6439 |
| Hybrid (RRF) | 0.4783 | 0.7152 | 0.7826 | 0.5777 |
| Dense (BGE-M3) | 0.3913 | 0.5674 | 0.6543 | 0.4698 |

相对排序：**BM25 > Hybrid > Dense**。详见 [results/retrieval/clean_benchmark/comparison.md](results/retrieval/clean_benchmark/comparison.md)。

### SAGE 演进（BM25 作候选生成）

| Method | R@1 | R@5 | R@10 | MRR | nDCG@10 | 要点 |
|--------|----:|----:|-----:|----:|--------:|------|
| BM25 baseline | 0.5435 | 0.7652 | 0.8261 | 0.6439 | 0.6884 | 强 lexical 基线 |
| Expansion pool (k=20) | — | — | **0.9152** | — | — | Graph 扩大候选 recall |
| SAGE v4 Risk-aware | 0.5957 | 0.8065 | 0.8283 | 0.6809 | 0.7173 | 首次略超 BM25 |
| SAGE v5 BM25 semantic | 0.5913 | 0.8022 | 0.8500 | 0.6801 | 0.7214 | initial_k=20 为主因；expand-gold 进 Top10=**0%** |
| A: Dense ranking + Graph | 0.4739 | 0.7348 | 0.8043 | 0.5877 | 0.6405 | expand 提升 30%，整体 R@10 下降 |
| **H: BM25+Dense+Graph** | **0.6000** | **0.8130** | **0.8609** | **0.6883** | **0.7302** | **当前最优** |

### 关键结论（Dense+Graph 实验）

假设：*Graph expansion improves candidate recall, while dense semantic relevance helps rank structurally expanded evidence.*

| Claim | 结果 |
|-------|------|
| Graph 提升候选 recall | ✅ 成立（pool R@10 91.52%） |
| Dense 帮助 expanded 排序 | ✅ 机制成立（v5: 0% → A: **30%** promotion） |
| Dense **替换** BM25 终排 | ❌ 不成立（A R@10 80.43% < v5） |
| BM25 + Dense + Graph 融合 | ✅ 成立（H 最优 R@10 **86.09%**） |

Expanded gold 按边类型（A）：`parent_of` 2/2，`refers_to` 3/8，`next_to` 1/10。

→ **Dense 解决的是「扩出来的证据排不上去」，不是「替代 BM25」。**  
完整报告：`results/retrieval/sage_dense_graph/{comparison,ablation,evaluation_report,failure_cases}.md`

---

## 项目目录

```
sage_rag/
├── README.md
├── requirements.txt
├── build_qa_dataset.py / evaluate_dense.py / evaluate_bm25.py
│
├── src/
│   ├── parsing/ chunking/ embedding/ generation/
│   ├── retrieval/               # Dense / BM25 / Hybrid / RRF
│   ├── evaluation/              # 指标、报告、SAGE 各版本评测入口
│   │   ├── evaluate_sage.py … evaluate_sage_v5.py
│   │   ├── evaluate_sage_dense_graph.py   # ★ Dense+Graph 实验
│   │   └── analyze_expanded_bottleneck.py
│   ├── analysis/
│   └── sage_rag/                # ★ SAGE-RAG
│       ├── build_graph.py
│       ├── graph/               # schema / builder / store
│       ├── expansion/           # GraphExpander (depth=1)
│       ├── ranking/
│       │   ├── structure_ranker.py / structure_ranker_v2.py
│       │   ├── risk_aware_ranker.py / risk_aware_ranker_v5.py
│       │   ├── semantic_rescorer.py       # BM25 semantic (v5)
│       │   ├── dense_rescorer.py          # ★ BGE-M3 cosine
│       │   └── dense_graph_ranker.py      # ★ Dense / Hybrid 终排
│       └── retrieval/
│           ├── sage_expansion_retriever.py
│           ├── sage_retriever.py … sage_retriever_v5.py
│           ├── candidate_allocator.py     # v3
│           └── sage_dense_graph_retriever.py  # ★ Dense+Graph
│
├── scripts/  docs/  configs/  baselines/  tests/
│
├── data/                        # 本地数据（默认不入 Git）
│   ├── raw_pdf/ parsed_json/ evidence_units/
│   ├── sage_graph/              # Standard Evidence Graph
│   ├── vector_store/            # Dense FAISS + embeddings.npy
│   ├── bm25_index/
│   └── qa_dataset/qa_pairs_clean.jsonl   # ★ 460
│
└── results/retrieval/
    ├── clean_benchmark/         # ★ 三基线
    ├── sage/ … sage_v5/         # SAGE 各版本
    └── sage_dense_graph/        # ★ Dense+Graph 实验
```

### 关键脚本一览

| 脚本 | 用途 |
|------|------|
| `src/sage_rag/build_graph.py` | Evidence Unit → Standard Evidence Graph |
| `src/evaluation/evaluate_sage_v5.py` | SAGE v5（BM25 semantic + risk-aware） |
| `src/evaluation/evaluate_sage_dense_graph.py` | ★ Dense+Graph / Hybrid / Ablation |
| `src/evaluation/analyze_expanded_bottleneck.py` | Expanded gold 排序瓶颈分析 |
| `scripts/evaluate_clean_benchmark.py` | Clean 集 Dense/BM25/Hybrid |
| `scripts/run_root_cause_analysis_clean.py` | Clean RCA + Design Motivation |
| `scripts/build_qa_clean.py` | 剔除 Dataset Issue → Clean QA |

---

## SAGE 流水线演进

```
query
  → BM25 initial retrieval (initial_k=20)
  → Graph Expansion (depth=1: parent_of / refers_to / next_to)
  → Candidate merge + dedup          # pool recall ≈ 91.5%
  → Relevance rescoring
        · v5: BM25 semantic
        · Dense+Graph: BGE-M3 cosine（复用 embeddings.npy，不重建索引）
  → Structure-aware ranking
        Final = α·Rel + β·Graph + γ·Coverage − λ·Risk
        Hybrid 分析变体: 0.25·BM25 + 0.25·Dense + 0.25·Graph + 0.25·Coverage
  → Top-10
```

| 版本 | 核心改动 | R@10 | Expand-gold → Top10 |
|------|----------|-----:|--------------------:|
| Expansion only | 不重排 | pool 91.52% | — |
| v2 Greedy | 结构选择 | ~82.61% | 很少 |
| v3 Allocation | 硬 7+3 槽位 | ~81–82% | 有，但挤掉 BM25 gold |
| v4 Risk-aware | −Risk + inherit | 82.83% | 仍很少 |
| v5 Semantic | BM25 rescore, k=20 | **85.00%** | **0 / 20** |
| Dense ranking (A) | Dense 作主 relevance | 80.43% | **6 / 20 (30%)** |
| Hybrid (H) | BM25+Dense+Graph | **86.09%** | （融合保底+校准） |

默认权重（与 v5 对齐，未为刷分调参）：α=0.5，β=0.25，γ=0.25，λ=0.2。

---

## Standard Evidence Graph

由 Evidence Units 构建轻量异构结构图 `G=(V,E)`，**不使用 LLM 抽实体**，只利用标准文档显式结构。

| 节点 | 数量 | 边 | 权重 | 数量 |
|------|-----:|----|-----:|-----:|
| document / chapter / clause / evidence | 13 / 81 / 500 / 565 | `parent_of` | 1.0 | 1147 |
| | | `next_to` | 0.3 | 420 |
| | | `refers_to` | 0.5 | 245 |

```bash
python src/sage_rag/build_graph.py \
  --input data/evidence_units/evidence_units.jsonl \
  --output data/sage_graph
```

---

## 环境准备

```bash
git clone <your-repo-url>
cd sage_rag
pip install -r requirements.txt
```

- FAISS 冲突时：`pip uninstall faiss faiss-cpu -y` 后安装 `faiss-cpu==1.9.0.post1`
- HuggingFace 超时：`$env:HF_HUB_OFFLINE = "1"`（PowerShell）
- PowerShell 请用 `;` 分隔命令，不要用 `&&`
- 评测推荐环境：含 `jieba` / `faiss` / `sentence-transformers` 的 conda env

---

## 快速复现

```bash
# A. Clean Benchmark 基线
python scripts/build_qa_clean.py
python scripts/evaluate_clean_benchmark.py
python scripts/run_root_cause_analysis_clean.py

# B. Standard Evidence Graph
python src/sage_rag/build_graph.py \
  --input data/evidence_units/evidence_units.jsonl \
  --output data/sage_graph

# C. SAGE v5
python src/evaluation/evaluate_sage_v5.py

# D. Dense+Graph 实验（含 ablation A/B/C/D + Hybrid）
python src/evaluation/evaluate_sage_dense_graph.py
```

### 从 PDF 完整重建

| Step | 命令 | 输出 |
|------|------|------|
| 1 解析 | `python src/parsing/pdf_to_structure.py --input data/raw_pdf --output data/parsed_json` | `parsed_json/` |
| 2 切分 | `python src/chunking/build_evidence_units.py --input data/parsed_json --output data/evidence_units` | `evidence_units.jsonl` |
| 3 结构图 | `python src/sage_rag/build_graph.py --input data/evidence_units/evidence_units.jsonl --output data/sage_graph` | `sage_graph/` |
| 4 Dense 索引 | `python src/embedding/build_index.py --input data/evidence_units/evidence_units.jsonl --output data/vector_store` | `vector_store/` |
| 5 BM25 索引 | `python scripts/build_bm25_index.py` | `bm25_index/` |
| 6 Clean + 评测 | 见上方「快速复现 A / C / D」 | `clean_benchmark/` 等 |

---

## 数据说明

仓库默认不提交大数据（见 `.gitignore`）。本地至少需要：

```
data/
├── evidence_units/evidence_units.jsonl   # 565
├── sage_graph/                           # Standard Evidence Graph
├── vector_store/                         # FAISS + embeddings.npy
├── bm25_index/
└── qa_dataset/
    ├── qa_pairs_v2.jsonl                 # 492
    └── qa_pairs_clean.jsonl              # 460 ★
```

| 资源 | 规模 |
|------|------|
| PDF | 13 |
| Evidence Units | 565 |
| Graph nodes / edges | 1159 / 1812 |
| QA V2 / Clean | 492 / 460 |

---

## 论文材料索引

| 用途 | 路径 |
|------|------|
| Clean 主表指标 | `results/retrieval/clean_benchmark/` |
| SAGE v5 | `results/retrieval/sage_v5/` |
| Dense+Graph 实验 | `results/retrieval/sage_dense_graph/` |
| Expanded 瓶颈分析 | `results/retrieval/` 下 `expanded_candidate_analysis.md`（若已生成） |
| Clean 失败分布 / 案例 | `results/root_cause_analysis_clean/` |
| Design Motivation | `results/root_cause_analysis_clean/design_motivation.md` |
| 结构图统计 | `results/sage_graph_statistics.md` |
| QA 清洗 / 对齐 | `results/qa_quality/`、`results/retrieval/clean_benchmark/cleaning_log.md` |

---

## 下一步

把 Dense 作为 **expanded 证据校准项** 接入正式 SAGE 流水线（保留 BM25 主 relevance），而不是用 Dense 单独终排；可选：对 `next_to` 边降权或阈值门控，以及论文表格 / ablation 定稿。

暂不为刷分调参；当前默认权重与 v5 对齐，用于验证假设。

---

## License

Research use. See repository for details.
