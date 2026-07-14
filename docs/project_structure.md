# Sage RAG — Project Structure

面向结构化技术标准（GB / ISO / IEC）的研究型 RAG 基准项目。本文档描述目录布局、模块职责与执行流水线。

## 顶层目录

```
sage_rag/
├── configs/          # 配置文件（预留）
├── data/             # 数据集与中间产物（不纳入 Git 内容）
├── src/              # 核心 Python 模块
├── scripts/          # 批处理与实验脚本
├── baselines/        # 对比基线实现（预留）
├── results/          # 实验输出与报告
├── docs/             # 项目文档
├── tests/            # 单元测试（预留）
├── build_qa_dataset.py      # 根目录 CLI 入口（QA 构建）
├── evaluate_dense.py        # 根目录 CLI 入口（Dense 评测）
├── test_dense_retriever.py  # 根目录 CLI 入口（检索 smoke test）
└── requirements.txt
```

## `data/` — 数据资产

| 路径 | 说明 |
|------|------|
| `data/raw_pdf/` | 原始 PDF（13 份标准） |
| `data/parsed_json/` | 条款级结构化 JSON |
| `data/evidence_units/` | Evidence Unit（检索粒度，565 条） |
| `data/vector_store/` | FAISS 索引、`embeddings.npy`、`metadata.json` |
| `data/qa_dataset/` | QA 评测集（含 `qa_pairs_v1.jsonl`、`qa_pairs_v2.jsonl`） |

**原则：** 向量库、Evidence Units、QA 数据集均为实验资产，迁移时只移动路径引用，不重建。

## `src/` — 核心模块

| 模块 | 职责 |
|------|------|
| `src/parsing/` | PDF → 结构化 JSON |
| `src/analysis/` | 数据集统计分析 |
| `src/chunking/` | Evidence Unit 构建 |
| `src/embedding/` | BGE-M3 编码 + FAISS 索引 |
| `src/generation/` | 自然语言 QA 生成与质量过滤 |
| `src/retrieval/` | Dense Retriever（`RetrieverBase` 接口） |
| `src/evaluation/` | Recall@K / MRR / nDCG 评测 |
| `src/ingestion/` | 数据摄取（预留） |
| `src/routing/` | 查询路由（预留） |
| `src/selection/` | 证据选择（预留） |

## `scripts/` — 实验脚本

| 脚本 | 用途 |
|------|------|
| `scripts/regenerate_qa.py` | V2 自然语言 QA 分步/全量重生成 |
| `scripts/compare_qa_benchmarks.py` | V1 vs V2 检索对比与失败分析 |

## `baselines/` — 对比基线（预留）

```
baselines/
├── bm25/
├── hybrid/
├── lightrag/
├── hipporag/
└── sage/
```

当前为空目录占位，后续实现统一 `RetrieverBase` 接口以便公平对比。

## `results/` — 实验输出

```
results/
├── retrieval/
│   └── dense/           # Dense 检索逐题结果与汇总报告
├── benchmark/           # 流水线阶段报告（解析/切分/索引/QA 质量）
├── ablation/            # 对照实验（如 V1 vs V2 QA）
├── case_study/          # 个案分析素材
├── figures/             # 统计图表
├── tables/              # 结构化统计表（JSON）
└── logs/                # 运行日志
```

### 主要产物示例

| 类别 | 示例文件 |
|------|----------|
| Dense 检索 | `results/retrieval/dense/retrieval_results_v2.jsonl` |
| 检索报告 | `results/retrieval/dense/retrieval_dense_report_v2.md` |
| QA 质量 | `results/benchmark/qa_quality_report_v2.md` |
| Ablation | `results/ablation/qa_v2_dense_comparison.md` |
| 统计表 | `results/tables/dataset_statistics.json` |

## 执行流水线

```
PDF (raw_pdf)
    → parsing → parsed_json
    → chunking → evidence_units.jsonl
    → embedding → vector_store/
    → generation → qa_dataset/
    → retrieval (Dense) + evaluation → results/retrieval/dense/
```

### 常用命令

```bash
# Evidence Unit
python src/chunking/build_evidence_units.py --input data/parsed_json --output data/evidence_units

# 向量索引
python src/embedding/build_index.py --input data/evidence_units/evidence_units.jsonl --output data/vector_store

# QA 重生成（V2）
python scripts/regenerate_qa.py --full -o data/qa_dataset/qa_pairs_v2.jsonl

# Dense 评测
python evaluate_dense.py --qa data/qa_dataset/qa_pairs_v2.jsonl

# V1 vs V2 对比报告
python scripts/compare_qa_benchmarks.py
```

## 遗留目录

| 路径 | 说明 |
|------|------|
| `index/` | 早期索引占位（bm25 / graph / vector_store），保留未删 |
| `logs/` | 已迁移至 `results/logs/`，根目录保留 `.gitkeep` |

## 设计原则

1. **数据与代码分离** — `data/` 存资产，`src/` 存逻辑，`results/` 存输出。
2. **可复现实验** — 同一 `vector_store/` 与 QA 文件可复跑评测。
3. **基线可扩展** — 新检索器实现 `RetrieverBase.retrieve()` 即可接入 `src/evaluation/`。
4. **零功能迁移** — 目录调整仅更新路径默认值，不改变算法与评测逻辑。
