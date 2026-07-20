# Sage RAG — Project Structure

面向结构化技术标准（GB / ISO / IEC）的研究型 RAG 基准。本文档与根目录 `README.md` 互补，侧重目录职责。

## 当前论文阶段

已完成：数据构建 → QA V2 / Clean → Dense/BM25/Hybrid 基线 → Root Cause → Design Motivation。  
下一阶段：SAGE-RAG 结构感知检索。

**主评测集：** `data/qa_dataset/qa_pairs_clean.jsonl`  
**主结果：** `results/retrieval/clean_benchmark/`、`results/root_cause_analysis_clean/`

## 顶层目录

```
sage_rag/
├── configs/ baselines/ tests/   # 预留
├── data/                        # 本地数据与索引（默认不入 Git）
├── src/                         # 核心模块
├── scripts/                     # 实验脚本
├── results/                     # 论文实验产出
├── docs/                        # 文档
├── build_qa_dataset.py
├── evaluate_dense.py
├── evaluate_bm25.py
├── test_dense_retriever.py
├── requirements.txt
└── README.md
```

## `data/`

| 路径 | 说明 |
|------|------|
| `raw_pdf/` | 原始 PDF（13） |
| `parsed_json/` | 条款级结构化 JSON |
| `evidence_units/` | Evidence Unit（565） |
| `vector_store/` | Dense FAISS 索引 |
| `bm25_index/` | BM25 索引 |
| `qa_dataset/qa_pairs_v1.jsonl` | 模板化 QA（对照） |
| `qa_dataset/qa_pairs_v2.jsonl` | 自然语言 QA（492） |
| `qa_dataset/qa_pairs_clean.jsonl` | Clean Benchmark（460）★ |

## `src/`

| 模块 | 职责 |
|------|------|
| `parsing/` | PDF → 结构化 JSON |
| `chunking/` | Evidence Unit |
| `embedding/` | BGE-M3 + FAISS |
| `generation/` | QA 生成与质量检查 |
| `retrieval/` | Dense / BM25 / Hybrid / RRF |
| `evaluation/` | 检索指标与报告 |
| `analysis/` | 对齐检测、失败分类、Root Cause |

## `results/`（论文材料）

| 路径 | 说明 |
|------|------|
| `retrieval/clean_benchmark/` | ★ Clean 三基线指标与对比 |
| `retrieval/{dense,bm25,hybrid}/` | V2 全量基线 |
| `root_cause_analysis_clean/` | ★ Clean RCA + Design Motivation |
| `root_cause_analysis/` | V2 RCA（含 Dataset Issue） |
| `qa_quality/` | QA 对齐审计与改写日志 |
| `ablation/` | V1 vs V2 Dense |
| `benchmark/` `figures/` `tables/` | 数据画像与统计图 |

## 设计原则

1. **不改已有检索器**做分析：RCA / Clean 只读评测结果。  
2. **新实验写入新目录**，避免覆盖论文主表。  
3. 大数据默认 `.gitignore`；迁移时移动路径引用，非必要不重建索引。
