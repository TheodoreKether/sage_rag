# LightRAG ↔ Sage RAG Mapping

> 官方代码路径：`baselines/lightrag/LightRAG/`  
> 本仓库主评测集：`data/qa_dataset/qa_pairs_clean.jsonl`（460）  
> 证据语料：`data/evidence_units/evidence_units.jsonl`（565）

本文建立概念与模块对应关系，并标明 **不对齐点**（评测时必须用 Adapter 桥接）。

---

## 1. 总览对照

| Sage RAG | LightRAG | 对齐程度 | 说明 |
|----------|----------|----------|------|
| Evidence Units | text chunks（`text_chunks` + `chunks_vdb`） | ⚠️ 需主动对齐 | 默认 LightRAG 会自切分；应用 **EU 作为 custom chunks** |
| QA Dataset（Clean） | 无内置同构格式；evaluation 用 RAGAS JSON | ❌ | 外挂评测脚本读我们的 jsonl |
| Vector Store（FAISS/BGE） | `chunks_vdb` / `entities_vdb` / `relationships_vdb` | ❌ 独立重建 | 不可复用现有 FAISS；LightRAG 自建 working_dir |
| BM25 Index | 无直接对应 | — | LightRAG 主路径非 BM25 |
| Evaluation（Recall@k / MRR） | RAGAS / LLM judge | ❌ | 必须用本仓库 `src/evaluation` |
| `RetrieverBase.retrieve()` | `aquery_data` / `query_data` | ✅ 可适配 | Adapter 包装即可 |
| SAGE-RAG（未来结构图） | `chunk_entity_relation_graph`（LLM 抽实体图） | ❌ 不同图 | 勿混为一谈 |

---

## 2. Evidence Units ↔ LightRAG Chunks

### 2.1 我们的 Evidence Unit

字段（节选）：`unit_id`、`document_id`、`parent_clause`、`text`、结构元数据（chapter / table / appendix 等）。

评测金标：`supporting_evidence[].unit_id`。

### 2.2 LightRAG 侧

| LightRAG 概念 | 存储 | 字段 |
|---------------|------|------|
| Document | `full_docs` | `doc_id`（可指定，否则 md5） |
| Chunk | `text_chunks` | `content`, `full_doc_id`, `chunk_order_index`, … |
| Chunk 向量 | `chunks_vdb` | 与 chunk key 对齐 |
| Chunk ID 生成 | `make_custom_chunk_id(doc, text)` 或 md5(`doc:content`) | 通常为 `chunk-<hash>`，**≠ unit_id** |

### 2.3 推荐映射策略（不改官方代码）

```text
按 document_id 分组 Evidence Units
    → 每个 document_id 作为 LightRAG doc_id
    → 每个 EU.text 作为一条 custom chunk
    → Adapter 落盘映射表：
         lightrag_chunk_id  →  unit_id
         （可选）content_hash / file_path 辅助校验
```

可选辅助：插入时把 `file_path` 设为 `unit_id`（若该路径在所用 insert API 中按 chunk 可设），查询结果用 `file_path` 反查；**仍建议显式映射表**，避免路径规范化副作用。

### 2.4 明确不推荐

| 做法 | 原因 |
|------|------|
| 把整本标准 PDF/全文丢给 `ainsert` 让官方重切 | chunk 边界 ≠ EU → Recall 与 BM25/Dense **不可比** |
| 假定 `chunk_id == unit_id` | `ainsert_custom_chunks` 强制内容哈希 ID |
| 复用本仓库 FAISS 给 LightRAG | API/schema 不兼容；且会绕过其 entities/relations 向量 |

---

## 3. QA Dataset ↔ LightRAG Evaluation

### 3.1 我们的 Clean QA

路径：`data/qa_dataset/qa_pairs_clean.jsonl`

| 字段 | 用途 |
|------|------|
| `qa_id` | 样本 ID |
| `question` | 查询文本 → 送入 `aquery_data` |
| `supporting_evidence[].unit_id` | Gold |
| `answer` | 生成评测可选；**检索主实验不用** |

### 3.2 LightRAG 官方评测

| 资源 | 用途 |
|------|------|
| `lightrag/evaluation/sample_dataset.json` | 少量问答 + RAGAS |
| `eval_rag_quality.py` | 调 API + RAGAS 指标 |
| `reproduce/batch_eval.py` | 回答两两比较 |

### 3.3 Mapping

| 我们的 | LightRAG | 适配方式 |
|--------|----------|----------|
| `question` | `aquery_data(query=...)` | 直接传 |
| `unit_id` gold | `data.chunks[].chunk_id` | **映射表** 转回 unit_id 再算 Recall |
| `answer` | `aquery` 输出 | 仅附录/生成实验；主表不做 |
| RAGAS metrics | — | 不作为 Clean Benchmark 主指标 |

**结论：** QA 文件本身 **不要改、不要替换**；评测脚本在仓库侧新建。

---

## 4. Vector Store ↔ LightRAG Storage

### 4.1 我们现有

| 资产 | 路径 | 用途 |
|------|------|------|
| Dense | `data/vector_store/`（BGE-M3 + FAISS） | Dense / Hybrid 基线 |
| BM25 | `data/bm25_index/` | BM25 / Hybrid 基线 |

### 4.2 LightRAG

独立 `working_dir`（建议新建，例如 `data/lightrag_index/`）：

```text
working_dir/
  kv_store_*.json          # full_docs / text_chunks / llm_cache ...
  vdb_*.json 或 faiss 文件 # chunks / entities / relationships
  graph_*.graphml 等       # NetworkX 默认
  doc_status_*.json
```

| 我们的 | LightRAG | 关系 |
|--------|----------|------|
| EU 文本语料 | `text_chunks` 内容来源 | 同源（若 custom 注入） |
| FAISS EU 向量 | `chunks_vdb` | **平行重建**（可选用同一 embedding 模型以公平，但仍是新索引） |
| — | `entities_vdb` / `relationships_vdb` | LightRAG 独有 |
| BM25 | — | 无映射 |

**不要**把 LightRAG 的 working_dir 指到现有 `data/vector_store/`。

---

## 5. Evaluation ↔ LightRAG

### 5.1 我们的评测栈

| 模块 | 角色 |
|------|------|
| `RetrieverBase.retrieve(query, top_k) → list[EvidenceUnit]` | 统一检索接口 |
| `src/evaluation/*` | Recall@k、MRR、nDCG、报告 |
| `scripts/evaluate_clean_benchmark.py` | Clean 主表入口 |
| `results/retrieval/clean_benchmark/` | 已有 BM25/Dense/Hybrid 数字 |

### 5.2 Mapping

| 我们的 | LightRAG | Adapter 职责 |
|--------|----------|--------------|
| `retrieve()` | `aquery_data` | 调官方 API，解析 `chunks` |
| `EvidenceUnit.unit_id` | `chunk_id` | 查映射表 |
| `EvidenceUnit.score/rank` | 列表顺序 / 若有 score 字段 | 尽量保留顺序作 rank |
| `top_k` | `QueryParam.chunk_top_k`（及 mode 相关 `top_k`） | 对齐实验超参 |
| metrics 计算 | — | **继续用本仓库 evaluation，不改官方** |

### 5.3 模式选择建议（论文表格）

| 报告名 | QueryParam.mode | 备注 |
|--------|-----------------|------|
| LightRAG-mix（主） | `mix` | 官方默认，完整方法 |
| LightRAG-hybrid | `hybrid` | 仅 KG local+global |
| LightRAG-naive | `naive` | 接近其内部 Dense；勿与我们的 Dense 混淆命名 |
| LightRAG-local / global | 可选消融 | 附录 |

---

## 6. 接口级 Mapping（开发时查阅）

```text
Sage                              LightRAG
────────────────────────────────────────────────────────
EvidenceUnit.text          ←→     chunk content
EvidenceUnit.unit_id       ←→     (Adapter map) ← chunk_id
EvidenceUnit.document_id   ←→     full_docs / full_doc_id
QA.question                ←→     aquery_data(query)
gold unit_ids              ←→     mapped retrieved unit_ids
DenseRetriever             ≈      naive_query 路径（不同模型/索引）
HybridRetriever (RRF)      ≠      QueryParam(mode="hybrid"|"mix")
BM25Retriever              ≠      （无）
FAISS index                ≠      NanoVectorDB / FaissVectorDBStorage（自建）
```

---

## 7. 成本与依赖 Mapping

| 我们基线 | LLM？ | Embedding？ |
|----------|-------|-------------|
| BM25 / Dense / Hybrid | 否 | Dense/Hybrid 需 embedding（已有） |
| LightRAG 索引 | **是**（每 chunk 抽实体） | 是（chunk + entity + relation） |
| LightRAG `aquery_data` | **通常是**（关键词抽取；可预填 `hl_keywords`/`ll_keywords` 做消融） | 是 |

适配时需单独准备 API Key / 本地 LLM，并评估 565 chunks × 抽取成本。

---

## 8. 目录级建议 Mapping（未来落地，当前不创建）

| 用途 | 路径（已落地） |
|------|----------------|
| 官方只读克隆 | `baselines/lightrag/LightRAG/` |
| Adapter 包 | `baselines/lightrag/adapter/` |
| 索引 working_dir | `baselines/lightrag/rag_storage/` |
| chunk↔unit 映射 | `baselines/lightrag/maps/` |
| 评测脚本 | `baselines/lightrag/scripts/` |
| 结果 | `baselines/lightrag/results/`（可选 mirror 到 `results/retrieval/lightrag/`） |
| 独立环境 | conda env `lightrag`（**非** `ailearn`） |

官方仓库内部文件：**只读，不写入实验产物。**
