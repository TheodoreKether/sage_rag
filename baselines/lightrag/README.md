# LightRAG Baseline (Sage RAG)

官方仓库只读克隆：`LightRAG/`（v1.5.5）。  
适配与实验产物全部收在本目录；**不修改**官方源码。

## 布局

```text
baselines/lightrag/
├── LightRAG/          # 官方代码（只读）
├── adapter/           # EU 注入 / id 映射 / RetrieverBase
├── scripts/           # build_index / evaluate / smoke_test
├── rag_storage/       # LightRAG working_dir（大文件，本地）
├── maps/              # chunk_id ↔ unit_id
├── results/           # 本方法评测产出
├── .env.example
└── README.md
```

共用只读数据（不复制）：

- `data/evidence_units/evidence_units.jsonl`
- `data/qa_dataset/qa_pairs_clean.jsonl`

## 环境（独立，勿用 ailearn）

```powershell
# 已创建则跳过
& "$env:USERPROFILE\anaconda3\Scripts\conda.exe" create -y -n lightrag python=3.11
& "$env:USERPROFILE\anaconda3\Scripts\conda.exe" run -n lightrag pip install -e .\LightRAG
& "$env:USERPROFILE\anaconda3\Scripts\conda.exe" run -n lightrag pip install python-dotenv tqdm

# 激活
conda activate lightrag
```

配置密钥：

```powershell
copy .env.example .env
# 编辑 .env 填入 LLM / Embedding API
```

## 流程

```powershell
cd baselines/lightrag
conda activate lightrag

# 1) 冒烟（不需 API）
python scripts/smoke_test.py

# 2) 可选：小规模 live（需 .env，会调 LLM 抽图）
python scripts/smoke_test.py --live

# 3) 全量建库（565 EU，耗时/费用取决于 LLM）
python scripts/build_index.py -v

# 4) Clean QA 评测
python scripts/evaluate.py --mode mix --top-k 10
# 可选：同步一份到论文目录 results/retrieval/lightrag/
python scripts/evaluate.py --mode mix --mirror-paper
```

## 注意

- DashScope 模型名须用官方 ID（如 `qwen-plus` / `qwen3.7-plus`），不要写 `Qwen3.7-Plus`。
- 首次运行会使用本地 `.tiktoken_cache/`（避免从 Azure 拉 tiktoken 失败）。
- 全量建库对每个 EU 做 LLM 实体抽取，墙钟时间可能数小时；建议 `MAX_ASYNC_LLM=1` 降低超时。
- 主 query mode 建议 `mix`；消融可用 `naive` / `hybrid` / `local` / `global`。
- 评测默认 `enable_rerank=False`；需要时加 `--enable-rerank`。
- 官方代码与 `ailearn` 环境保持隔离。

