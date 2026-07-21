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
| 5. Design Motivation | ✅ | 结构模块动机初稿（无算法） |
| 6. SAGE-RAG 图构建 | ✅ | Evidence Unit → Standard Evidence Graph（层级 / 相邻 / 引用） |
| 7. SAGE-RAG 检索 | 🔜 | 结构感知检索器 + Clean Benchmark 对比 |

**论文主评测集：** `data/qa_dataset/qa_pairs_clean.jsonl`（460 queries）  
**主实验结果：** `results/retrieval/clean_benchmark/`  
**主失败分析：** `results/root_cause_analysis_clean/`  
**结构图：** `data/sage_graph/`（统计见 `results/sage_graph_statistics.md`）

---

## 项目目录

```
sage_rag/
├── README.md
├── requirements.txt
├── build_qa_dataset.py          # 模板 QA（V1）CLI
├── evaluate_dense.py            # Dense 评测入口
├── evaluate_bm25.py             # BM25 评测入口
├── test_dense_retriever.py      # Dense 冒烟测试
│
├── src/                         # 核心代码
│   ├── parsing/                 # PDF → 结构化 JSON
│   ├── chunking/                # Evidence Unit 构建
│   ├── embedding/               # BGE-M3 + FAISS
│   ├── generation/              # QA 生成与自然语言质量检查
│   ├── retrieval/               # Dense / BM25 / Hybrid / RRF
│   ├── evaluation/              # 检索指标与报告
│   ├── analysis/                # 对齐检测、失败分类、Root Cause
│   └── sage_rag/                # ★ SAGE-RAG（结构证据图）
│       ├── build_graph.py       # CLI：EU → Graph
│       └── graph/               # schema / builder / store
│
├── scripts/                     # 实验脚本（见下方）
├── docs/                        # 设计说明
├── configs/ baselines/ tests/   # 预留
│
├── data/                        # 本地数据（默认不入 Git）
│   ├── raw_pdf/                 # 原始标准 PDF（13）
│   ├── parsed_json/             # 结构化解析结果
│   ├── evidence_units/          # 565 条 Evidence Units
│   ├── sage_graph/              # ★ Standard Evidence Graph
│   │   ├── nodes.jsonl
│   │   ├── edges.jsonl
│   │   └── graph_statistics.md
│   ├── vector_store/            # Dense FAISS 索引
│   ├── bm25_index/              # BM25 索引
│   └── qa_dataset/
│       ├── qa_pairs_v1.jsonl    # 模板化 QA（对照）
│       ├── qa_pairs_v2.jsonl    # 自然语言 QA（492，已改写）
│       └── qa_pairs_clean.jsonl # ★ 论文 Clean Benchmark（460）
│
└── results/                     # 实验产出（论文材料）
    ├── retrieval/
    │   ├── dense/               # V2（及 V1 对照）Dense 结果
    │   ├── bm25/                # V2 BM25 结果
    │   ├── hybrid/              # V2 Hybrid 结果
    │   ├── clean_benchmark/     # ★ Clean 三基线指标与对比
    │   ├── bm25_vs_dense.md
    │   └── hybrid_vs_baselines.md
    ├── sage_graph_statistics.md # ★ 结构图节点/边统计
    ├── root_cause_analysis/     # V2 全量 RCA（含 Dataset Issue）
    ├── root_cause_analysis_clean/  # ★ Clean RCA + Design Motivation
    ├── qa_quality/              # QA 对齐审计与改写日志
    ├── ablation/                # V1 vs V2 Dense 对照
    ├── benchmark/               # 数据画像、索引报告
    ├── figures/ tables/         # 统计图 / 表
    └── logs/
```

### 关键脚本一览

| 脚本 | 用途 |
|------|------|
| `src/sage_rag/build_graph.py` | ★ Evidence Unit → Standard Evidence Graph |
| `scripts/build_bm25_index.py` | 构建 BM25 索引 |
| `scripts/evaluate_hybrid.py` | Hybrid 评测 |
| `scripts/build_qa_clean.py` | 剔除 Dataset Issue → Clean QA |
| `scripts/evaluate_clean_benchmark.py` | Clean 集上跑 Dense/BM25/Hybrid |
| `scripts/run_root_cause_analysis.py` | V2 Root Cause（含 Dataset Issue） |
| `scripts/run_root_cause_analysis_clean.py` | ★ Clean Root Cause + Motivation |
| `scripts/audit_qa_alignment.py` | QA–Gold 对齐审计 |
| `scripts/rewrite_misaligned_qa.py` | 批量改写错位问题 |
| `scripts/compare_*` | 基线对比报告 |

---

## 主实验结果（Clean Benchmark）

460 queries，top_k=10。相对排序：**BM25 > Hybrid > Dense**。

| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR |
|-----------|---------:|---------:|----------:|----:|
| BM25 | **0.5435** | **0.7652** | **82.61%** | **0.6439** |
| Hybrid (RRF) | 0.4783 | 0.7152 | 78.26% | 0.5777 |
| Dense (BGE-M3) | 0.3913 | 0.5674 | 65.43% | 0.4698 |

与原始 V2（492）对比：清洗后三条基线 Recall@10 约 **+3.4 ~ +4.1 pp**（部分失败来自标注噪声）。详见 [results/retrieval/clean_benchmark/comparison.md](results/retrieval/clean_benchmark/comparison.md)。

### Clean Hybrid 失败归因（100 misses，已排除 Dataset Issue）

| Failure Type | Count | % |
|--------------|------:|--:|
| Document Identity / Version Disambiguation | 37 | 37% |
| Appendix Retrieval Failure | 25 | 25% |
| Semantic Misunderstanding | 25 | 25% |
| Cross-reference Failure | 7 | 7% |
| Hierarchy Structure Failure | 3 | 3% |
| Table / Structured Content Failure | 3 | 3% |

→ 动机：Version Relation / Appendix Links / Cross-reference Graph / Hierarchy Graph / Table Nodes。详见 [results/root_cause_analysis_clean/design_motivation.md](results/root_cause_analysis_clean/design_motivation.md)。

---

## Standard Evidence Graph（已完成）

由 Evidence Units 构建轻量异构结构图 `G=(V,E)`，**不使用 LLM 抽实体**，只利用标准文档显式结构。

| 节点 | id 约定 | 数量 |
|------|---------|-----:|
| document | `{document_id}` | 13 |
| chapter | `{doc}::chapter::{chapter_id}` | 81 |
| clause | `{doc}::clause::{parent_clause}` | 500 |
| evidence | `{unit_id}`（不变） | 565 |

| 边 | 含义 | 权重 | 数量 |
|----|------|-----:|-----:|
| `parent_of` | document → chapter → clause → evidence | 1.0 | 1147 |
| `next_to` | 同章相邻条款 | 0.3 | 420 |
| `refers_to` | 规则匹配交叉引用（见 / 参见 / 附录 / GB/T / ISO） | 0.5 | 245 |

```bash
python src/sage_rag/build_graph.py \
  --input data/evidence_units/evidence_units.jsonl \
  --output data/sage_graph
```

产出：`data/sage_graph/{nodes,edges}.jsonl`、`graph_statistics.md`，并同步 `results/sage_graph_statistics.md`。

---

## 环境准备

```bash
git clone <your-repo-url>
cd sage_rag
# 推荐 conda 环境（需 faiss-cpu）；或 venv + pip
pip install -r requirements.txt
```

- FAISS 冲突时：`pip uninstall faiss faiss-cpu -y` 后安装 `faiss-cpu==1.9.0.post1`
- HuggingFace 超时：`$env:HF_HUB_OFFLINE = "1"`（PowerShell）
- PowerShell 请用 `;` 分隔命令，不要用 `&&`

---

## 快速复现（索引已存在时）

```bash
# A. Clean Benchmark 评测（论文主表）
python scripts/build_qa_clean.py
python scripts/evaluate_clean_benchmark.py
python scripts/run_root_cause_analysis_clean.py

# B. 原始 V2 三基线（对照）
python evaluate_dense.py --qa data/qa_dataset/qa_pairs_v2.jsonl --top-k 10 \
  --output results/retrieval/dense/retrieval_results_v2.jsonl \
  --report results/retrieval/dense/retrieval_dense_report_v2.md
python evaluate_bm25.py --qa data/qa_dataset/qa_pairs_v2.jsonl --top-k 10
python scripts/evaluate_hybrid.py --qa data/qa_dataset/qa_pairs_v2.jsonl --top-k 10
python scripts/compare_hybrid_baselines.py
python scripts/run_root_cause_analysis.py

# C. Standard Evidence Graph（SAGE-RAG 图构建）
python src/sage_rag/build_graph.py \
  --input data/evidence_units/evidence_units.jsonl \
  --output data/sage_graph
```

### 从 PDF 完整重建

| Step | 命令 | 输出 |
|------|------|------|
| 1 解析 | `python src/parsing/pdf_to_structure.py --input data/raw_pdf --output data/parsed_json` | `parsed_json/` |
| 2 切分 | `python src/chunking/build_evidence_units.py --input data/parsed_json --output data/evidence_units` | `evidence_units.jsonl` |
| 3 结构图 | `python src/sage_rag/build_graph.py --input data/evidence_units/evidence_units.jsonl --output data/sage_graph` | `sage_graph/` |
| 4 Dense 索引 | `python src/embedding/build_index.py --input data/evidence_units/evidence_units.jsonl --output data/vector_store` | `vector_store/` |
| 5 BM25 索引 | `python scripts/build_bm25_index.py` | `bm25_index/` |
| 6 QA V2 | `python scripts/regenerate_qa.py --full -o data/qa_dataset/qa_pairs_v2.jsonl` | `qa_pairs_v2.jsonl` |
| 7 Clean + 评测 | 见上方「快速复现 A」 | `clean_benchmark/` 等 |

---

## 数据说明

仓库默认不提交大数据（见 `.gitignore`）。本地至少需要：

```
data/
├── evidence_units/evidence_units.jsonl   # 565
├── sage_graph/                           # Standard Evidence Graph
├── vector_store/                         # FAISS
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
| Clean 主表指标 | `results/retrieval/clean_benchmark/*_metrics.json` |
| Clean vs V2 对比 | `results/retrieval/clean_benchmark/comparison.md` |
| Clean 失败分布 | `results/root_cause_analysis_clean/root_cause_clean_statistics.md` |
| Clean 失败案例 | `results/root_cause_analysis_clean/root_cause_clean_examples.md` |
| Design Motivation | `results/root_cause_analysis_clean/design_motivation.md` |
| 结构图统计 | `results/sage_graph_statistics.md` |
| V2 三基线对比 | `results/retrieval/hybrid_vs_baselines.md` |
| QA 清洗日志 | `results/retrieval/clean_benchmark/cleaning_log.md` |
| QA 对齐/改写 | `results/qa_quality/` |

---

## 下一步

在 Standard Evidence Graph 之上实现 **SAGE-RAG 结构感知检索**，并在 Clean Benchmark 上与 BM25 / Dense / Hybrid 对比：

1. Hierarchy-aware retrieval（沿 `parent_of` / `next_to` 扩展）  
2. Cross-reference expansion（沿 `refers_to`）  
3. Appendix / Table 定向召回  
4. Document / Version 消歧  

---

## License

Research use. See repository for details.
