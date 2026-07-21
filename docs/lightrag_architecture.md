# LightRAG 架构调研（Technical Investigation）

> 调研对象：`baselines/lightrag/LightRAG/`（官方仓库，版本 **1.5.5**）  
> 约束：只读分析，**不修改**任何官方代码。  
> 目的：理解整体架构，为接入 Sage RAG Clean Benchmark 做最小代价适配设计。

---

## 1. 项目整体架构

LightRAG 是一个以 **知识图谱（KG）+ 向量检索** 为核心的 GraphRAG 框架。相对 Microsoft GraphRAG，它强调轻量存储、增量更新与多后端可插拔。

### 1.1 顶层目录

| 路径 | 角色 |
|------|------|
| `lightrag/` | **核心 Python 包**（SDK） |
| `lightrag/lightrag.py` | 门面类 `LightRAG`：insert / query / 存储生命周期 |
| `lightrag/operate.py` | KG 抽取、merge、`kg_query` / `naive_query` |
| `lightrag/pipeline.py` | 文档入队 → 解析 → chunk → 抽取的状态机 |
| `lightrag/base.py` | `QueryParam`、存储抽象、`TextChunkSchema` |
| `lightrag/chunker/` | F/R/V/P 四种切分策略 |
| `lightrag/kg/` | KV / Vector / Graph / DocStatus 后端实现 |
| `lightrag/llm/` | OpenAI / Ollama / HF / Gemini 等 LLM & embedding 绑定 |
| `lightrag/parser/` | PDF/DOCX 等解析与路由 |
| `lightrag/evaluation/` | RAGAS 评测示例（端到端生成质量，非 Recall@k） |
| `lightrag/api/` | REST Server（生产入口，研究可不用） |
| `examples/` | SDK 用法示例 |
| `reproduce/` | 论文复现脚本（context 抽取、batch LLM judge） |
| `docs/ProgramingWithCore.md` | 嵌入式 / 研究用 Core API 说明 |

### 1.2 运行时组件（`LightRAG` 实例内）

初始化后（必须 `await initialize_storages()`）持有：

| 组件 | 默认实现 | 内容 |
|------|----------|------|
| `full_docs` | JsonKV | 原文文档 |
| `text_chunks` | JsonKV | chunk 文本与元数据 |
| `chunks_vdb` | NanoVectorDB | chunk 向量 |
| `entities_vdb` | NanoVectorDB | 实体向量 |
| `relationships_vdb` | NanoVectorDB | 关系向量 |
| `chunk_entity_relation_graph` | NetworkX | 实体-关系图 |
| `doc_status` | JsonDocStatus | 文档处理状态 / 自定义 chunk journal |
| `llm_response_cache` | JsonKV | LLM 缓存（抽取 / 关键词 / 回答） |

可替换后端：`Faiss` / `Milvus` / `Qdrant` / `Neo4j` / `Postgres` 等（见 `lightrag/kg/`）。

### 1.3 架构示意

```text
                    ┌─────────────────────────────────────┐
                    │           LightRAG (SDK)            │
                    │  insert / ainsert / aquery_*        │
                    └──────────────┬──────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
  Indexing Pipeline         Graph + Vectors           Query Pipeline
  (pipeline.py)             (operate + kg/)           (operate.kg_query)
         │                         │                         │
  enqueue → parse →         entities / edges /        keywords → VDB
  chunk → embed →           chunk graph               + graph walk →
  extract_entities →                                  context → (LLM)
  merge_nodes_edges
```

---

## 2. 数据流（端到端）

### 2.1 索引侧（Indexing）

```text
Raw text / files
    → apipeline_enqueue_documents   # 入队 + doc_status
    → apipeline_process_enqueue_documents
        → parse（可选）
        → process_single_document
            → Chunker (F/R/V/P 或 legacy chunking_func)
            → chunks → text_chunks KV + chunks_vdb.upsert（embedding）
            → extract_entities（LLM，每 chunk）
            → merge_nodes_and_edges → graph + entities_vdb + relationships_vdb
    → _insert_done / index_done_callback（落盘）
```

SDK 捷径：`ainsert(text)` ≡ enqueue（固定 **F** chunker）+ process。

自定义切分捷径：`ainsert_custom_chunks(full_text, text_chunks, doc_id)`  
→ **跳过官方 chunker**，仍走 embedding + entity extraction + KG merge。  
（标记为 deprecated，但仍是“保留 EU 边界”的关键官方入口。）

### 2.2 查询侧（Query）

```text
query + QueryParam(mode=...)
    → aquery_llm / aquery / aquery_data
        → mode ∈ {local, global, hybrid, mix} → kg_query
        → mode == naive → naive_query（仅 chunks_vdb）
        → mode == bypass → 空检索，直送 LLM
    → （可选）LLM 生成回答
```

**检索评测关键 API：`aquery_data` / `query_data`**  
内部强制 `only_need_context=True`，返回结构化 `entities` / `relationships` / `chunks`，**不调用生成 LLM**（但 `kg_query` 路径仍可能调用 LLM 做 **关键词抽取**）。

---

## 3. Retrieval Pipeline

入口：`LightRAG.aquery_data` → `operate.kg_query` 或 `operate.naive_query`。

### 3.1 Query 模式

| mode | 行为（官方语义） |
|------|------------------|
| `local` | 低层关键词 → 实体检索 → 关联 chunk |
| `global` | 高层关键词 → 关系检索 → 关联实体/chunk |
| `hybrid` | local + global 结果合并 |
| `mix`（默认） | KG（local/global）+ naive 向量 chunk |
| `naive` | 仅 `chunks_vdb` 向量检索 |
| `bypass` | 无检索 |

参数集中在 `QueryParam`（`base.py`）：`top_k`、`chunk_top_k`、`enable_rerank`、`hl_keywords` / `ll_keywords` 等。

### 3.2 `kg_query` 主路径（`operate.py`）

1. `get_keywords_from_query` / `extract_keywords_only`（LLM，可缓存）  
2. `_build_query_context`  
   - `_perform_kg_search`：entities_vdb / relationships_vdb + graph 邻居  
   - `_get_vector_context`：chunks_vdb（mix 等）  
   - token 截断、可选 rerank、合并 chunk  
3. 若 `only_need_context`：返回 context 字符串 + `raw_data`  
4. 否则拼 system prompt → LLM 生成

### 3.3 `naive_query`

仅对 `chunks_vdb` 做向量 Top-K，再从 `text_chunks` 取正文。适合作为“LightRAG 内置 Dense 对照”，**不是**我们仓库里已有的 BGE-M3+FAISS Dense。

### 3.4 与 “Retriever” 概念的对应

LightRAG **没有**名为 `retrieve()` 的公开方法。等价接口：

| 需求 | API |
|------|-----|
| 结构化检索结果（无生成） | `query_data` / `aquery_data` |
| 仅 context 字符串 | `aquery(..., QueryParam(only_need_context=True))` |
| 完整 RAG 回答 | `aquery` / `aquery_llm` |

返回 chunk 字段示例：`chunk_id`、`content`、`file_path`、`reference_id`。

---

## 4. Graph Construction Pipeline

位置：`operate.extract_entities` → `merge_nodes_and_edges`；由 `pipeline.process_single_document` 或 `ainsert_custom_chunks` 触发。

### 4.1 步骤

1. 对每个 text chunk 调用抽取 LLM（prompt 在 `prompt.py` / `addon_params` 实体类型配置）  
2. 解析实体 / 关系（含 gleaning 多轮）  
3. 描述过长则 summary（LLM）  
4. 合并同名实体与边，写：  
   - `chunk_entity_relation_graph`（节点/边）  
   - `entities_vdb` / `relationships_vdb`（带 embedding）  
5. chunk ↔ entity 通过 `source_id` / chunk 列表关联

### 4.2 可选旁路

- `insert_custom_kg` / `ainsert_custom_kg`：直接注入自定义 entities / relationships / chunks（见 `examples/insert_custom_kg.py`）。  
  **不适合**我们做“官方 LightRAG 算法 baseline”，除非做消融。

---

## 5. Query Pipeline（生成）

`aquery` → `aquery_llm`：

1. 与 `aquery_data` 相同的检索/上下文构建  
2. 填入 `PROMPTS["rag_response"]`  
3. 调用 role=`query` 的 LLM  
4. 可选 stream；结果可走 LLM cache

研究评测若只关心 **Evidence Unit Recall**，应停在 `aquery_data`，不要用生成答案做主指标。

---

## 6. Embedding Pipeline

| 环节 | 位置 |
|------|------|
| 函数注入 | `LightRAG(embedding_func=...)`（`llm/` 下 openai / hf / ollama…） |
| Chunk 向量写入 | `process_single_document` 内 `chunks_vdb.upsert(chunks)` |
| 实体/关系向量 | `merge_nodes_and_edges` 路径 upsert |
| 查询向量 | `BaseVectorStorage.query` 内部对 query 调同一 `embedding_func` |
| 批大小 / 并发 | `embedding_batch_num`、`embedding_func_max_async` |

默认研究配置常用 OpenAI embedding；也可换本地 HF（`lightrag/llm/hf.py`）。  
**与本仓库 Dense 基线的 BGE-M3 + FAISS 索引相互独立**——LightRAG 会建自己的 `working_dir` 向量库。

---

## 7. Index Construction

### 7.1 文档读取入口

| 入口 | 说明 |
|------|------|
| `LightRAG.ainsert` / `insert` | 字符串或字符串列表；SDK 固定 F-chunker |
| `apipeline_enqueue_documents` + `apipeline_process_enqueue_documents` | Server/完整管线；可选 F/R/V/P |
| `ainsert_custom_chunks` | 调用方已切好的 chunks |
| `lightrag/parser/` + Server 上传 | 文件级解析（研究可跳过） |

### 7.2 Chunk 构建位置

| 策略 | 模块 |
|------|------|
| F fixed-token | `chunker/token_size.py`（`ainsert` 默认） |
| R recursive | `chunker/recursive_character.py` |
| V semantic-vector | `chunker/semantic_vector.py` |
| P paragraph-semantic | `chunker/paragraph_semantic.py` |
| 组装 chunk dict / ID | `utils_pipeline.build_chunks_dict_from_chunking_result` |
| 自定义 chunk ID | `utils_pipeline.make_custom_chunk_id(doc_key, text)` → `chunk-<md5>` |

`build_chunks_dict_from_chunking_result` 若输入 dict 带显式 `chunk_id`，会优先使用（并可能加 `doc_id-` 前缀）。  
`ainsert_custom_chunks` **不接受**外部 `chunk_id`，一律内容哈希。

### 7.3 持久化

默认 `working_dir` 下 JSON + NanoVectorDB 文件。评测应固定独立目录，例如：  
`data/lightrag_store/` 或 `baselines/lightrag/rag_storage/`（勿覆盖官方示例目录）。

---

## 8. Retrieval Interface（接口清单）

### 8.1 关键公开 API

```text
LightRAG.initialize_storages() / finalize_storages()
LightRAG.ainsert(...) / ainsert_custom_chunks(...)
LightRAG.aquery_data(query, QueryParam)   # ← 检索评测首选
LightRAG.aquery / aquery_llm              # 端到端 QA
QueryParam(mode, top_k, chunk_top_k, only_need_context, ...)
```

### 8.2 内部“检索器”

无独立 Retriever 类；检索逻辑内嵌于 `operate.kg_query` / `naive_query` 与各 `*VectorStorage.query`。

### 8.3 Evaluation 示例

| 路径 | 类型 | 与我们的差距 |
|------|------|--------------|
| `lightrag/evaluation/eval_rag_quality.py` | RAGAS（Faithfulness 等），常打 REST API | **不是** unit_id Recall@k |
| `lightrag/evaluation/offline_retrieval_check.py` | 词法 oracle 冒烟 | 仅样例文档 |
| `reproduce/batch_eval.py` | LLM pairwise judge | 生成质量对比 |
| `reproduce/Step_*.py` | 论文数据管线复现 | 非我们的 QA 格式 |

**结论：** LightRAG 官方评测不能直接复用为我们的 Clean Benchmark；需外挂 Adapter + 本仓库 `src/evaluation/`。

---

## 9. 关键模块速查表

| 关注点 | 位置 |
|--------|------|
| 文档读取入口 | `lightrag.py`：`ainsert` / `apipeline_enqueue_documents` |
| Chunk 构建 | `pipeline.process_single_document` + `chunker/*` |
| Graph 构建 | `operate.extract_entities` + `merge_nodes_and_edges` |
| Embedding 构建 | `chunks_vdb` / `entities_vdb` / `relationships_vdb` upsert |
| Retriever 实现 | `operate.kg_query` / `naive_query`（无独立类） |
| Query 入口 | `aquery` / `aquery_llm` / `aquery_data` |
| retrieve() 等价 | **`aquery_data` / `query_data`** |
| Evaluation 示例 | `lightrag/evaluation/*`、`reproduce/*` |

---

## 10. 对基准实验的直接含义

1. LightRAG 默认会 **重新切分文档** 并用 LLM **抽图**——成本高，且 chunk ID ≠ 我们的 `unit_id`。  
2. 要做与 BM25/Dense/Hybrid **同语料粒度** 的公平对比，应 **跳过官方 chunker**，把 Evidence Units 当作 LightRAG 的 text chunks 注入。  
3. 主指标应走 **`aquery_data` → 映射回 `unit_id` → 本仓库 Recall/MRR`**，而不是 RAGAS。  
4. 官方代码保持不变；所有 glue 放在仓库 Adapter / scripts 层。
