# Sage RAG

面向**结构化技术文档**（国标 GB、ISO、IEC 等）的研究型 RAG 基准。与将 PDF 当纯文本切块的通用方案不同，本项目在解析、切分、检索全链路保留**条款级结构与编号体系**。

**核心能力：** PDF 解析 → Evidence Unit（565 条）→ BGE-M3 + FAISS → 自然语言 QA 评测集 → Dense 检索评测（Recall@K / MRR / nDCG）。

---

## 项目结构

```
sage_rag/
├── configs/              # 配置（预留）
├── data/                 # 数据集与中间产物（不纳入 Git）
├── src/                  # 核心流水线
├── scripts/              # 批处理与实验脚本
├── baselines/            # 对比基线 bm25 / hybrid / sage 等（预留）
├── results/              # 实验输出
│   ├── retrieval/dense/  # Dense 检索结果与报告
│   ├── benchmark/        # 流水线阶段报告
│   ├── ablation/         # 对照实验（V1 vs V2）
│   ├── case_study/       # 个案分析
│   ├── figures/ tables/ logs/
├── docs/                 # 项目文档
├── tests/                # 单元测试（预留）
├── build_qa_dataset.py   # QA 构建（根目录 CLI）
├── evaluate_dense.py     # Dense 评测
└── test_dense_retriever.py
```

| 目录 | 职责 |
|------|------|
| `src/parsing/` | PDF → 结构化 JSON |
| `src/chunking/` | Evidence Unit 构建 |
| `src/embedding/` | BGE-M3 编码 + FAISS |
| `src/generation/` | 自然语言 QA 生成 |
| `src/retrieval/` | Dense Retriever（`RetrieverBase`） |
| `src/evaluation/` | 检索指标与报告 |

详细目录说明、产物路径与设计原则见 **[docs/project_structure.md](docs/project_structure.md)**。

---

## 环境准备

```bash
git clone <your-repo-url> && cd sage_rag
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows:     .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

使用真实 LLM 生成 QA 时需额外：`pip install openai`，并配置 `OPENAI_API_KEY` 等环境变量。

---

## 数据说明

**仓库不包含数据集**（见 `.gitignore`）。两种方式：

**A. 从 PDF 完整复现** — 将 13 份标准 PDF 放入 `data/raw_pdf/`，按下方流水线 Step 1–7 执行。

**B. 使用已有中间数据** — 解压后确保至少具备：

```
data/
├── parsed_json/*.json
├── evidence_units/evidence_units.jsonl    # 565 条
├── vector_store/                          # faiss.index + metadata.json
└── qa_dataset/
    ├── qa_pairs_v1.jsonl                  # 模板化 QA（878）
    └── qa_pairs_v2.jsonl                  # 自然语言 QA（492）
```

可从任意步骤续跑，无需从头执行。

---

## 流水线

```
PDF → parsed_json → evidence_units → vector_store → qa_dataset → Dense 评测 → results/
```

| Step | 命令 | 主要输出 |
|------|------|----------|
| 1 解析 | `python src/parsing/pdf_to_structure.py --input data/raw_pdf --output data/parsed_json` | `data/parsed_json/` |
| 2 分析（可选） | `python src/analysis/dataset_profiler.py --input data/parsed_json` | `results/benchmark/`、`results/figures/` |
| 3 切分 | `python src/chunking/build_evidence_units.py --input data/parsed_json --output data/evidence_units` | `evidence_units.jsonl` |
| 4 索引 | `python src/embedding/build_index.py --input data/evidence_units/evidence_units.jsonl --output data/vector_store` | `data/vector_store/` |
| 5 QA | 见下方 | `data/qa_dataset/` |
| 6 测试 | `python test_dense_retriever.py --index data/vector_store --query "..."` | 终端输出 |
| 7 评测 | `python evaluate_dense.py --qa data/qa_dataset/qa_pairs_v2.jsonl` | `results/retrieval/dense/` |

**Step 5 — QA 生成（推荐 V2 自然语言）：**

```bash
# 抽样验证（50 unit）
python scripts/regenerate_qa.py --sample 50 \
    -o data/qa_dataset/qa_pairs_natural_sample50.jsonl

# 全量 V2（建议 --llm-backend openai）
python scripts/regenerate_qa.py --full -o data/qa_dataset/qa_pairs_v2.jsonl

# 旧版模板 QA（对比用）
python build_qa_dataset.py --input data/evidence_units --output data/qa_dataset
```

**Step 7 — 常用评测参数：** `--qa`、`--index data/vector_store`、`--top-k 10`、`--sample 100`（调试）、`--device cuda`。

**V1 vs V2 对比报告：**

```bash
python scripts/compare_qa_benchmarks.py
# 输出: results/ablation/qa_v2_dense_comparison.md
```

---

## 当前基准结果

Dense 检索：BGE-M3 + FAISS IndexFlatIP，top_k=10。Retriever 实现未改动，差异来自 QA 问题风格。

| 数据集 | 文件 | QA 数 | Recall@10 | MRR |
|--------|------|------:|----------:|----:|
| V1 模板化 | `qa_pairs_v1.jsonl` | 878 | 4.44% | 0.0164 |
| V2 自然语言 | `qa_pairs_v2.jsonl` | 492 | **52.24%** | **0.3501** |

V1 问题含「根据条款…」「第 X 章…」等结构导航用语，不适合评测 Dense Retriever；V2 模拟真实用户提问。完整对比见 [results/ablation/qa_v2_dense_comparison.md](results/ablation/qa_v2_dense_comparison.md)。

| 阶段 | 规模 |
|------|------|
| 原始 PDF | 13 份 |
| Evidence Units / 向量 | 565（一一对应） |
| Clauses | ~501 |

---

## 一键复现

```bash
python src/parsing/pdf_to_structure.py --input data/raw_pdf --output data/parsed_json
python src/chunking/build_evidence_units.py --input data/parsed_json --output data/evidence_units
python src/embedding/build_index.py --input data/evidence_units/evidence_units.jsonl --output data/vector_store
python scripts/regenerate_qa.py --sample 50
python evaluate_dense.py --qa data/qa_dataset/qa_pairs_natural_sample50.jsonl --top-k 10
```

---

## 常见问题

| 问题 | 处理 |
|------|------|
| `Input path does not exist` | 按「数据说明」放置对应目录下的数据文件 |
| 只想跑检索评测 | 需 `data/vector_store/` + QA 文件，直接 Step 7 |
| 只想验证 QA 模块 | 需 `evidence_units.jsonl`，运行 `scripts/regenerate_qa.py --sample 50` |
| Windows `&&` 报错 | PowerShell 请逐条执行，或用 `;` 分隔 |
| 模型已缓存仍联网 | `$env:HF_HUB_OFFLINE = "1"`（PowerShell） |
| placeholder vs 真实 LLM | placeholder 离线占位，正式 benchmark 应用 `--llm-backend openai` 等 |

**待实现：** BM25 / Hybrid / Graph-RAG 基线（`baselines/`）、LLM 生成与端到端 RAG 评测。

---

## License

Research use. See repository for details.
