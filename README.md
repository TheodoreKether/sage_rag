# Sage RAG

面向**结构化技术文档**（国标 GB、ISO、IEC 等）的研究型 RAG 基准。与将 PDF 当纯文本切块的通用方案不同，本项目在解析、切分、检索全链路保留**条款级结构与编号体系**。

**当前能力：**

- PDF 解析 → Evidence Unit（565 条）
- Dense（BGE-M3 + FAISS）/ BM25 / Hybrid（RRF）三条检索基线
- 自然语言 QA 评测集 V2（492 条，已做对齐审计与高可疑问题改写）
- Failure Analysis（失败分类 → 统计 / 案例 / 设计动机）

---

## 项目结构

```
sage_rag/
├── configs/                 # 配置（预留）
├── data/                    # 数据集与中间产物（不纳入 Git）
│   ├── raw_pdf/
│   ├── parsed_json/
│   ├── evidence_units/
│   ├── vector_store/        # Dense / FAISS
│   ├── bm25_index/          # BM25
│   └── qa_dataset/
├── src/
│   ├── parsing/             # PDF → 结构化 JSON
│   ├── chunking/            # Evidence Unit
│   ├── embedding/           # BGE-M3 + FAISS
│   ├── generation/          # QA 生成与质量检查
│   ├── retrieval/           # Dense / BM25 / Hybrid / RRF
│   ├── evaluation/          # 检索指标与报告
│   └── analysis/            # 数据集画像、QA 对齐审计、失败分类
├── scripts/                 # 批处理与实验脚本
├── results/
│   ├── retrieval/           # dense / bm25 / hybrid 及对比报告
│   ├── failure_analysis/    # 失败统计、案例、分布图、设计动机
│   ├── qa_quality/          # QA 对齐审计与改写日志
│   ├── benchmark/           # 流水线阶段报告
│   └── ablation/            # V1 vs V2 等对照
├── docs/
├── build_qa_dataset.py
├── evaluate_dense.py
├── evaluate_bm25.py
└── test_dense_retriever.py
```

详细目录说明见 **[docs/project_structure.md](docs/project_structure.md)**。

---

## 环境准备

```bash
git clone <your-repo-url>
cd sage_rag
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows:     .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

使用真实 LLM 生成 QA 时额外安装：`pip install openai`，并配置 `OPENAI_API_KEY`。

PowerShell 下不要用 `&&`，请逐条执行或用 `;` 分隔。

---

## 数据说明

**仓库不包含数据集**（见 `.gitignore`）。两种方式：

**A. 从 PDF 完整复现** — 将 13 份标准 PDF 放入 `data/raw_pdf/`，按下方流水线执行。

**B. 使用已有中间数据** — 至少具备：

```
data/
├── parsed_json/*.json
├── evidence_units/evidence_units.jsonl    # 565 条
├── vector_store/                          # faiss.index + metadata.json
├── bm25_index/                            # BM25 索引（可重建）
└── qa_dataset/
    ├── qa_pairs_v1.jsonl                  # 模板化 QA（878）
    └── qa_pairs_v2.jsonl                  # 自然语言 QA（492，已对齐改写）
```

可从任意步骤续跑。

| 阶段 | 规模 |
|------|------|
| 原始 PDF | 13 份 |
| Evidence Units / 向量 | 565（一一对应） |
| QA V2 | 492 |

---

## 流水线总览

```
PDF → parsed_json → evidence_units
  → vector_store (Dense)
  → bm25_index (BM25)
  → qa_dataset (V2)
  → Dense / BM25 / Hybrid 评测
  → Failure Analysis
```

---

## Step 1–4：数据构建

| Step | 命令 | 输出 |
|------|------|------|
| 1 解析 | `python src/parsing/pdf_to_structure.py --input data/raw_pdf --output data/parsed_json` | `data/parsed_json/` |
| 2 分析（可选） | `python src/analysis/dataset_profiler.py --input data/parsed_json` | `results/benchmark/` |
| 3 切分 | `python src/chunking/build_evidence_units.py --input data/parsed_json --output data/evidence_units` | `evidence_units.jsonl` |
| 4 Dense 索引 | `python src/embedding/build_index.py --input data/evidence_units/evidence_units.jsonl --output data/vector_store` | `data/vector_store/` |

**BM25 索引：**

```bash
python scripts/build_bm25_index.py
# 输出: data/bm25_index/
```

---

## Step 5：QA 数据集

推荐使用 **V2 自然语言 QA**（`qa_pairs_v2.jsonl`）。

```bash
# 抽样验证
python scripts/regenerate_qa.py --sample 50 \
    -o data/qa_dataset/qa_pairs_natural_sample50.jsonl

# 全量 V2（正式评测建议 --llm-backend openai）
python scripts/regenerate_qa.py --full -o data/qa_dataset/qa_pairs_v2.jsonl

# 旧版模板 QA（对照用）
python build_qa_dataset.py --input data/evidence_units --output data/qa_dataset
```

### QA 质量：对齐审计与改写

部分 V2 问题曾出现「问整本标准标题 / 发布日期，Gold 却是具体条款」的错位。可用下列脚本审计并批量改写（**保留 `qa_id` 与 evidence**）：

```bash
# 审计问题–Gold 对齐
python scripts/audit_qa_alignment.py
# 输出: results/qa_quality/qa_alignment_audit.md

# 批量改写高可疑问题
python scripts/rewrite_misaligned_qa.py
# 输出: results/qa_quality/qa_rewrite_log.md
# 会备份 qa_pairs_v2.jsonl.bak.*
```

改写后请**重新跑全部基线评测与 Failure Analysis**，旧检索结果会与新问题集不一致。

---

## Step 6：检索基线评测

默认 QA：`data/qa_dataset/qa_pairs_v2.jsonl`，`top_k=10`。

### 6.1 Dense

```bash
python test_dense_retriever.py --index data/vector_store --query "..."

python evaluate_dense.py --qa data/qa_dataset/qa_pairs_v2.jsonl --index data/vector_store --top-k 10 --output results/retrieval/dense/retrieval_results_v2.jsonl --report results/retrieval/dense/retrieval_dense_report_v2.md
```

常用参数：`--sample 100`（调试）、`--device cuda` / `cpu`。模型已缓存时可用：`$env:HF_HUB_OFFLINE = "1"`。

### 6.2 BM25

```bash
python scripts/test_bm25.py
python evaluate_bm25.py --qa data/qa_dataset/qa_pairs_v2.jsonl --top-k 10
# 输出: results/retrieval/bm25/
```

### 6.3 Hybrid（Dense Top-100 + BM25 Top-100 → RRF）

```bash
python scripts/test_hybrid.py
python scripts/evaluate_hybrid.py --qa data/qa_dataset/qa_pairs_v2.jsonl --top-k 10
# 输出: results/retrieval/hybrid/
```

### 6.4 对比报告

```bash
python scripts/compare_bm25_dense.py
# → results/retrieval/bm25_vs_dense.md

python scripts/compare_hybrid_baselines.py
# → results/retrieval/hybrid_vs_baselines.md

python scripts/compare_qa_benchmarks.py
# → results/ablation/qa_v2_dense_comparison.md
```

---

## Step 7：Failure Analysis

在三条基线的 `retrieval_results*.jsonl` 就绪后运行（**不修改检索代码、不重跑检索**）：

```bash
python scripts/run_failure_analysis.py
# 输出: results/failure_analysis/
#   failure_statistics.json / .md
#   failure_examples.md
#   failure_distribution.png
#   design_motivation.md
```

失败类别包括：Hierarchical / Cross-document / Version / Table / Appendix / Cross-reference / Semantic / Lexical（Dense 命中且 BM25 未命中）。

---

## 当前基准结果

基于 **QA V2（对齐改写后）** 全量评测：492 queries，top_k=10。完整对比见 [results/retrieval/hybrid_vs_baselines.md](results/retrieval/hybrid_vs_baselines.md)。

| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|-----------|---------:|---------:|----------:|----:|--------:|
| BM25 | **0.5081** | **0.7276** | **78.66%** | **0.6071** | **0.6509** |
| Hybrid (RRF) | 0.4492 | 0.6789 | 74.19% | 0.5447 | 0.5924 |
| Dense (BGE-M3) | 0.3659 | 0.5366 | 61.99% | 0.4416 | 0.4839 |

**结论：** BM25 仍为最强基线；Hybrid 明显优于 Dense，但尚未超过 BM25。QA 对齐改写后三条基线 Recall@10 均较改写前提升约 8–10 个百分点（改写前约 BM25 70.5% / Hybrid 65.9% / Dense 52.4%），说明此前部分「假失败」来自问题–Gold 错位。

V1 模板化 QA（878）Dense Recall@10 约 4.4%，含「根据条款…」等结构导航用语，不适合作为检索主评测集。

### Failure Analysis（改写后）

| Retriever | Failures (Recall@10 miss) | Failure Rate |
|-----------|--------------------------:|-------------:|
| BM25 | 105 | 21.34% |
| Hybrid | 127 | 25.81% |
| Dense | 187 | 38.01% |

Hybrid 主要失败类别仍集中在 Semantic / Appendix / Cross-document / Version / Cross-reference 等，用于动机化 SAGE-RAG 结构模块。详见 [results/failure_analysis/](results/failure_analysis/)。

---

## 复现基线评测

索引已存在时：

```bash
# 1) Dense（请写入 v2 路径，供对比与 Failure Analysis 使用）
python evaluate_dense.py --qa data/qa_dataset/qa_pairs_v2.jsonl --top-k 10 \
  --output results/retrieval/dense/retrieval_results_v2.jsonl \
  --report results/retrieval/dense/retrieval_dense_report_v2.md

# 2) BM25
python evaluate_bm25.py --qa data/qa_dataset/qa_pairs_v2.jsonl --top-k 10

# 3) Hybrid
python scripts/evaluate_hybrid.py --qa data/qa_dataset/qa_pairs_v2.jsonl --top-k 10

# 4) 对比
python scripts/compare_bm25_dense.py
python scripts/compare_hybrid_baselines.py

# 5) Failure Analysis
python scripts/run_failure_analysis.py
```

若尚无 BM25 索引：先执行 `python scripts/build_bm25_index.py`。

---

## 常见问题

| 问题 | 处理 |
|------|------|
| `Input path does not exist` | 按「数据说明」放置对应数据 |
| 只跑检索评测 | 需索引 + `qa_pairs_v2.jsonl`，直接 Step 6 |
| Dense 依赖缺失 / FAISS 冲突 | `pip uninstall faiss faiss-cpu -y` 后安装 `faiss-cpu==1.9.0.post1` |
| HuggingFace 超时 | `$env:HF_HUB_OFFLINE = "1"` |
| Windows `&&` 报错 | PowerShell 逐条执行或用 `;` |
| Failure Analysis 导入失败 | 在项目根目录运行脚本；脚本会将根目录加入 `sys.path` |
| QA 改写后指标异常 | 必须重跑评测；旧 `retrieval_results` 对应旧问题文本 |

**待实现：** SAGE-RAG 结构图检索（Hierarchical / Cross-reference / Table / Appendix）、LLM 生成与端到端 RAG 评测。

---

## License

Research use. See repository for details.
