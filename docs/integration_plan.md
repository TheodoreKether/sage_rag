# LightRAG 接入计划（Integration Plan）

> 状态：**仅方案设计，不开发、不改官方代码、不替换数据。**  
> 官方路径：`baselines/lightrag/LightRAG/`（v1.5.5）  
> 主评测集：`data/qa_dataset/qa_pairs_clean.jsonl`（保持不变）  
> 详设参见：`docs/lightrag_architecture.md`、`docs/lightrag_mapping.md`

---

## 目标

在 **尽量不改动 LightRAG 官方实现** 的前提下，将其作为额外 retrieval baseline，在 Clean Benchmark 上报告与 BM25 / Dense / Hybrid 可比的 **Recall@k / MRR / nDCG**。

---

## 最小修改原则

### 必须改 / 必须新建（仅在我们仓库侧）

| 项 | 原因 |
|----|------|
| 新建 Adapter（实现 `RetrieverBase`） | 官方无 `retrieve()`，需 `aquery_data` + unit_id 映射 |
| 新建索引构建脚本 | 用 Evidence Units 注入，跳过官方重切分 |
| 新建评测脚本 / 结果目录 | 官方 RAGAS ≠ 我们的指标 |
| 新建 LightRAG `working_dir` | 与现有 FAISS/BM25 隔离 |
| 配置 embedding + LLM | 官方运行时依赖；可用 env，不改源码 |

### 建议用 Adapter（不要改官方）

| 职责 | Adapter 做法 |
|------|----------------|
| EU → LightRAG documents/chunks | 调 `ainsert_custom_chunks`（或按 doc 批量） |
| `chunk_id` ↔ `unit_id` | 外部 JSON 映射表 |
| `aquery_data` → `list[EvidenceUnit]` | 解析 `data.chunks`，查映射，填 rank |
| 异步 API → 同步 `retrieve()` | `asyncio.run` / 持久 event loop |
| 模式与 top_k | 构造 `QueryParam` |
| 失败/空结果 | 返回 []，与现有评测兼容 |

### 绝对不要改

| 范围 | 原因 |
|------|------|
| `baselines/lightrag/LightRAG/lightrag/**` 源码 | 保持官方可复现、可升级 |
| `operate.kg_query` / `extract_entities` 算法逻辑 | 否则不再是 “LightRAG baseline” |
| `data/qa_dataset/qa_pairs_clean.jsonl` | 主表固定 |
| `data/evidence_units/evidence_units.jsonl` | 语料固定；只读喂入 |
| 现有 `data/vector_store/`、`data/bm25_index/` | 避免污染已发基线 |
| 已有 Dense/BM25/Hybrid 实现与 clean_benchmark 主结果 | 新结果写新目录或新文件名 |

### 可选、但不要默认做

- 用 `ainsert` 全文重切分（不公平）  
- 用 RAGAS 替代 Recall（指标不一致）  
- 把 SAGE 结构图塞进 `insert_custom_kg`（变成另一方法）  
- Fork 官方只为改 `chunk_id` 生成（映射表更便宜）

---

## 推荐最小方案（一句话）

**只读官方 SDK → Adapter 把 565 个 Evidence Units 当 custom chunks 建库 → `aquery_data(mode="mix")` → 映射 unit_id → 本仓库 evaluation 跑 Clean 460。**

---

## 分步计划

### 第一步：环境与冒烟（只读官方 + 仓库外配置）

1. 单独 venv / 依赖：按官方 `pyproject.toml` 安装（勿强行并进主环境除非必要）。  
2. 配置 LLM + Embedding（OpenAI 兼容或本地 HF/Ollama）。  
3. 跑通官方 `examples/lightrag_openai_compatible_demo.py` 量级冒烟（可用短文本）。  
4. 确认 `initialize_storages` / `ainsert` / `aquery_data` 可用。  

**产出：** 环境说明短文档或 `.env.example`（写在我们侧，不进官方树）。  
**不产出：** 对官方的 patch。

### 第二步：语料注入 Adapter（索引构建）

1. 只读加载 `evidence_units.jsonl`。  
2. 按 `document_id` 分组。  
3. 对每个文档调用 `ainsert_custom_chunks`：  
   - `full_text` = 文档内 EU 文本简单拼接（满足 API）  
   - `text_chunks` = 各 EU 的 `text`（**保持 EU 边界**）  
   - `doc_id` = `document_id`  
4. 构建并保存 `chunk_id → unit_id`（及反向）映射。  
   - 可在插入前用官方同款 `make_custom_chunk_id(doc_id, text)` **在 Adapter 内预计算**（import 官方函数，不改其实现）。  
5. `working_dir` 指向 `baselines/lightrag/rag_storage/`（新建）。  

**产出：** `baselines/lightrag/scripts/build_index.py` + `maps/` + `rag_storage/`。  
**风险点：** 565 × LLM 实体抽取，耗时与费用；需缓存（官方 `enable_llm_cache` 默认开）。

### 第三步：检索 Adapter + Clean 评测

1. 实现 `LightRAGRetriever(RetrieverBase)`：  
   - `retrieve(query, top_k)` → `aquery_data` + 映射 → `EvidenceUnit` 列表  
   - 主配置：`QueryParam(mode="mix", chunk_top_k=top_k, enable_rerank=与实验设定一致)`  
2. 脚本 `baselines/lightrag/scripts/evaluate.py`：读 Clean QA，复用 `src/evaluation` 指标。  
3. 结果写入 `baselines/lightrag/results/`；可选 `--mirror-paper` 同步到 `results/retrieval/lightrag/`。  
4. 可选消融：`naive` / `hybrid` / `local` 同脚本不同 mode。  

**产出：** metrics.json、retrieval_results.jsonl、evaluation_report.md。

### 第四步：报告与可比性核查

1. 与 `results/retrieval/clean_benchmark/` 中 BM25/Dense/Hybrid 并表。  
2. 文档中写清：  
   - 相同：EU 语料、Clean QA、Recall 定义  
   - 不同：LightRAG 额外 KG 索引、查询期关键词 LLM、embedding 模型若不一致需声明  
3. 若 embedding 与 BGE-M3 不同，在论文/报告中标注（公平性说明）；可选第二轮用同一 embedding 函数注入 LightRAG 以加强可比性。  

**产出：** `results/retrieval/lightrag_vs_baselines.md`（可选）。

---

## 预计工作量

| 阶段 | 人天（估计） | 说明 |
|------|--------------|------|
| 第一步 环境冒烟 | 0.5–1 | 依赖与 API 连通性 |
| 第二步 索引构建 | 1–2 | 脚本简单；**墙钟时间**取决于 LLM（可能数小时起） |
| 第三步 Adapter + 评测 | 1–1.5 | 接口清晰，主耗在调参与空结果排查 |
| 第四步 报告与公平性 | 0.5 | |
| **合计** | **约 3–5 人天** | 不含大规模抽图排队等待 |

并行：索引跑的时候可写 Adapter 与评测脚本。

---

## 可能风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| LLM 抽图成本高 / 限流 | 索引建不完 | 官方 cache；降并发；本地小模型；先子集冒烟 |
| `chunk_id` 映射失败 | Recall 全 0 | 预计算 `make_custom_chunk_id`；插入后抽样校验 |
| `sanitize_text_for_encoding` 改写文本 | 哈希与预计算不一致 | 插入前后用同一 sanitize；或插入后从 `text_chunks` 反查 |
| 查询仍依赖关键词 LLM | 费用、不稳、空 keywords | cache；失败重试；必要时预填 keywords 做对照实验 |
| Rerank 默认开启 | 与基线设定不一致 | 实验固定 `enable_rerank=False` 或统一开启并声明 |
| `ainsert_custom_chunks` 标 deprecated | 未来升级 API 变 | 锁定 git commit；升级时再测；备选：自定义 chunker 函数注入（仍不改官方文件） |
| Windows / faiss / 异步循环 | 运行失败 | 沿用已验证 conda env；注意 event loop 策略 |
| 与 Hybrid 名称混淆 | 论文误读 | 报告中写 `LightRAG-hybrid` vs `RRF-Hybrid` |

---

## 需要修改 / 新建的文件列表

### 新建（我们仓库，推荐）

```text
baselines/lightrag/adapter/          # 已落地
baselines/lightrag/scripts/          # build_index / evaluate / smoke_test
baselines/lightrag/rag_storage/      # working_dir（gitignore）
baselines/lightrag/maps/             # id 映射（gitignore）
baselines/lightrag/results/          # 本方法评测产出
baselines/lightrag/README.md
baselines/lightrag/.env.example
docs/lightrag_architecture.md
docs/lightrag_mapping.md
docs/integration_plan.md
```

### 可能微调（非必须）

```text
.gitignore                                       # 已放行 adapter/scripts
README.md                                        # 增加 baseline 一行说明
results/retrieval/lightrag/                      # --mirror-paper 时同步
```

### 明确不修改

```text
baselines/lightrag/LightRAG/**                   # 全部官方代码
data/qa_dataset/qa_pairs_clean.jsonl
data/qa_dataset/qa_pairs_v2.jsonl
data/evidence_units/evidence_units.jsonl
data/vector_store/**
data/bm25_index/**
src/retrieval/bm25.py | dense_retriever.py | hybrid.py
results/retrieval/clean_benchmark/**             # 不覆盖；只追加对照文档
```

---

## 官方代码保持不变的部分（清单）

- 全部检索算法：`operate.kg_query` / `naive_query` / Rerank  
- 全部图构建：`extract_entities` / `merge_nodes_and_edges`  
- 全部默认 chunker：F/R/V/P  
- 全部存储后端实现：`lightrag/kg/*`  
- 全部 LLM 绑定实现：`lightrag/llm/*`  
- API Server / WebUI / Docker  
- 官方 evaluation / reproduce 脚本  

我们只通过 **公开构造函数参数 + 公开方法调用**（`LightRAG(...)`、`ainsert_custom_chunks`、`aquery_data`、`QueryParam`）使用它们。

---

## 决策记录（建议默认）

| 决策 | 默认选择 |
|------|----------|
| 语料粒度 | Evidence Units（custom chunks） |
| QA | `qa_pairs_clean.jsonl` |
| 主 query mode | `mix` |
| 主指标 | Recall@1/5/10、MRR、nDCG（本仓库） |
| 官方源码 | 只读 |
| 结果目录 | 新建，不覆盖 clean_benchmark 三基线文件 |

---

## 下一步（需你确认后再开发）

当前调研结束。若进入实现，建议严格按 **第一步 → 第二步 → 第三步 → 第四步** 执行，且第一批 PR/提交只加 Adapter 与脚本，**零官方 diff**。
