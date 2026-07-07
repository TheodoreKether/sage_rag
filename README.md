# Sage RAG

面向**结构化技术文档**（国标 GB、ISO、IEC、企业标准等）的研究型检索增强生成（RAG）系统。

与将 PDF 当作纯文本处理的通用 RAG 不同，本项目强调：

- **条款级检索** — 以章节、条款、子条款为粒度进行索引与召回
- **结构感知 RAG** — 在解析、切分、检索全链路中保留文档层级、编号体系与语义关系

## 项目结构

```
sage_rag/
├── data/
│   ├── raw_pdf/          # 原始 PDF（未纳入 Git）
│   ├── parsed_json/      # 解析后的结构化 JSON
│   ├── evidence_units/   # Evidence Unit（检索候选单元）
│   ├── vector_store/     # FAISS 向量索引与 embedding 矩阵
│   └── qa_dataset/       # QA 评测数据集
├── index/                # 图索引、BM25 索引（待实现）
├── src/                  # 核心流水线模块
├── configs/              # 配置文件
├── logs/                 # 运行日志
└── results/              # 分析报告与实验结果
```

## 已实现模块

| 阶段 | 模块 | 状态 |
|------|------|------|
| PDF 解析 | `src/parsing/` | ✅ 已完成 |
| 数据集分析 | `src/analysis/` | ✅ 已完成 |
| Evidence Unit 构建 | `src/chunking/` | ✅ 已完成 |
| 向量编码与 FAISS 索引 | `src/embedding/` | ✅ 已完成 |
| QA 数据集构建 | `src/generation/` | ✅ 已完成 |
| 检索 / 生成 / 评测 | `src/retrieval/` 等 | ⏳ 待实现 |

---

## 设计原理（截至目前）

本项目面向**异构技术标准**（GB / ISO / IEC 等）构建研究型 RAG 基准。与"把 PDF 当纯文本切块"的通用方案不同，流水线在每个阶段都尽量保留文档的结构信息。

### 整体思路

```
PDF  →  结构化 JSON  →  Evidence Unit  →  向量索引
                              ↓
                         QA 评测数据集
```

各阶段职责清晰分离，便于替换组件（如换 embedding 模型、换 LLM）而不影响其他环节。

### 阶段 1：PDF 解析（`src/parsing/`）

**输入：** 原始 PDF  
**输出：** 条款级 JSON（`data/parsed_json/`）

解析器用 PyMuPDF / pdfplumber 提取文本，再按标准文档的编号规则（如 `5.1.2`、`Table 3`）识别章节与条款，构建层级树：

```
Document → Chapter → Clause → Sub-clause
```

每条 clause 保留 `clause_id`、正文、页码、所属章节等字段。这一步解决的核心问题是：**让非结构化的 PDF 变成可程序处理的结构化数据**。

### 阶段 2：Evidence Unit 构建（`src/chunking/`）

**输入：** 解析 JSON  
**输出：** `data/evidence_units/evidence_units.jsonl`（565 条）

Evidence Unit 是**检索的最小语义单元**，设计原则：

- **一条 clause 至少产生 1 条 unit** — 短条款整体作为一个单元
- **长条款（>512 tokens）语义切分** — 按段落 → 换行 → 句子 → 定长依次尝试，避免超出 embedding 模型上下文
- **不跨 clause 合并** — 保持条款边界，便于引用时定位到具体 `parent_clause`
- **保留结构元数据** — `unit_id`、`document_id`、`chapter_id`、`parent_clause`、`split_index/total` 等

`unit_id` 编码了文档与条款路径（如 `GB_T_39401-2020::5::5::1`），保证全局唯一。

> 565 条 unit 来自 501 条 clause：部分长条款被切分为多条 unit（`split_total > 1`），因此 unit 数 ≥ clause 数。

### 阶段 3：向量编码与索引（`src/embedding/`）

**输入：** `evidence_units.jsonl`  
**输出：** `data/vector_store/`（565 个向量）

对每条 Evidence Unit 的 `text` 字段调用 embedding 模型（`BAAI/bge-m3`），生成 1024 维稠密向量，经 L2 归一化后写入 FAISS `IndexFlatIP` 索引。

**为什么向量数量与 `evidence_units.jsonl` 行数一致？**

这是**刻意的一对一映射**设计：

| 规则 | 说明 |
|------|------|
| 一条 unit → 一个向量 | 每条有效 Evidence Unit 编码为一个向量，索引行号与 metadata 中的 `index` 字段对齐 |
| 仅跳过无效行 | 空文本、损坏 JSON、缺少 `unit_id` 的记录会被跳过；当前数据集 565 条全部有效，故 0 跳过 |
| 不做合并或拆分 | 向量化不改变 unit 粒度，embedding 阶段不引入额外的 chunk 逻辑 |
| metadata 完整保留 | `metadata.json` 存下原文与全部结构字段，检索命中后可通过 `index` 还原 Evidence Unit |

因此：**565 条 JSONL = 565 个向量 = 565 条 metadata**，三者一一对应。QA 数据集（878 条）是评测用的问答对，一条 unit 可对应 1–3 个 QA，数量不必与向量数相同。

向量索引与检索逻辑解耦，后续 Dense RAG、Hybrid RAG、Graph-RAG 等基线可共用同一份 `vector_store/`，保证实验可比性。

### 阶段 4：QA 数据集构建（`src/generation/`）

**输入：** Evidence Unit  
**输出：** `data/qa_dataset/qa_pairs.jsonl`（878 条）

基于每条 Evidence Unit，用 LLM 生成 1–3 个问答对（7 种题型可配置），并记录 `supporting_evidence` 指向来源 unit。用于后续 RAG 检索与生成效果的离线评测。

Prompt 模板、LLM 后端、数据集构建、质量校验四层分离，支持替换不同模型而不改 pipeline 代码。

---

## 环境准备

### 1. 克隆代码

```bash
git clone <your-repo-url>
cd sage_rag
```

### 2. 创建虚拟环境

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

> 若使用真实 LLM 生成 QA（非 placeholder 后端），还需额外安装：`pip install openai`

---

## 数据说明（重要）

**GitHub 仓库不包含数据集**（见 `.gitignore`）。你需要单独获取数据文件，并按以下目录放置。

### 方式 A：从原始 PDF 完整跑通（推荐首次复现）

将 13 份标准 PDF 放入：

```
data/raw_pdf/
├── GB_T 39401-2020_7783.pdf
├── GB_T 39412-2020_3263.pdf
├── ...
└── ISO 11694-6-2014_en.pdf
```

然后依次执行下方 **Step 1 → Step 5** 的全部命令。

### 方式 B：直接使用已生成的中间数据

若已收到打包好的数据，解压到项目根目录，确保目录结构如下：

```
data/
├── raw_pdf/                        # 可选，仅当需要重新解析时
├── parsed_json/*.json              # 13 个解析结果
├── evidence_units/evidence_units.jsonl
├── vector_store/                   # 可选，仅当需要重新构建索引时
│   ├── faiss.index
│   ├── embeddings.npy
│   └── metadata.json
└── qa_dataset/qa_pairs.jsonl       # 可选，仅当需要重新生成 QA 时

results/                            # 可选，分析报告
├── dataset_profile.md
├── dataset_statistics.json
├── evidence_statistics.md
├── embedding_report.md
├── qa_quality_report.md
└── figures/
```

收到中间数据后，可从对应步骤继续，无需从头运行。

---

## 流水线：从原始 PDF 到向量索引

以下命令均在**项目根目录**下执行（已激活虚拟环境）。

### Step 1 — PDF 解析 → 结构化 JSON

将原始 PDF 解析为条款级 JSON，输出到 `data/parsed_json/`。

```bash
python src/parsing/pdf_to_structure.py \
    --input data/raw_pdf \
    --output data/parsed_json
```

**Windows (PowerShell)**

```powershell
python src/parsing/pdf_to_structure.py --input data/raw_pdf --output data/parsed_json
```

**输入：** `data/raw_pdf/*.pdf`  
**输出：** `data/parsed_json/*.json`（每个 PDF 对应一个 JSON）

每条 JSON 包含 `standard_id`、`doc_type`、`title`、`toc`、`chapters`（含嵌套 `clauses`）及 `quality` 质检信息。

---

### Step 2 — 数据集分析（可选）

对解析结果做统计分析，生成报告与图表。

```bash
python src/analysis/dataset_profiler.py \
    --input data/parsed_json \
    --markdown results/dataset_profile.md \
    --json results/dataset_statistics.json \
    --figures results/figures
```

**输出：**

| 文件 | 说明 |
|------|------|
| `results/dataset_profile.md` | 可读性分析报告 |
| `results/dataset_statistics.json` | 结构化统计数据 |
| `results/figures/*.png` | 分布图表 |

---

### Step 3 — Evidence Unit 构建

将解析 JSON 切分为检索候选单元（Evidence Unit），每条 clause 对应 1 条或多条 unit（长条款会语义切分）。

```bash
python src/chunking/build_evidence_units.py \
    --input data/parsed_json \
    --output data/evidence_units \
    --stats results/evidence_statistics.md
```

**输入：** `data/parsed_json/*.json`  
**输出：** `data/evidence_units/evidence_units.jsonl`（约 565 条）

---

### Step 4 — 向量编码与 FAISS 索引

将每条 Evidence Unit 编码为稠密向量，构建 FAISS 检索索引。首次运行会从 HuggingFace 下载 `BAAI/bge-m3` 模型（约 2.3 GB）。

```bash
python src/embedding/build_index.py \
    --input data/evidence_units/evidence_units.jsonl \
    --output data/vector_store \
    --report results/embedding_report.md
```

**Windows (PowerShell)**

```powershell
python src/embedding/build_index.py --input data/evidence_units/evidence_units.jsonl --output data/vector_store
```

**输入：** `data/evidence_units/evidence_units.jsonl`  
**输出：**

| 文件 | 说明 |
|------|------|
| `data/vector_store/faiss.index` | FAISS 向量索引（IndexFlatIP） |
| `data/vector_store/embeddings.npy` | 原始 embedding 矩阵 (565 × 1024) |
| `data/vector_store/metadata.json` | 与向量对齐的 Evidence Unit 元数据 |
| `results/embedding_report.md` | 构建报告（向量数、维度、耗时、设备等） |

#### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `BAAI/bge-m3` | HuggingFace 模型名或本地模型目录 |
| `--batch-size` | `16` | 推理批大小 |
| `--device` | `auto` | `auto` / `cpu` / `cuda`（有 GPU 时建议 `cuda`） |
| `--max-length` | `8192` | 最大序列长度 |

**国内镜像（首次下载模型）：**

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python src/embedding/build_index.py --input data/evidence_units/evidence_units.jsonl --output data/vector_store
```

**离线模型：** 若已下载模型到本地，指定路径即可：

```bash
python src/embedding/build_index.py --model /path/to/bge-m3 --output data/vector_store
```

---

### Step 5 — QA 数据集构建

基于 Evidence Unit，通过 LLM 生成问答对，用于后续 RAG 评测。

**默认使用 placeholder 后端（离线、无需 API Key，适合验证流程）：**

```bash
python build_qa_dataset.py \
    --input data/evidence_units \
    --output data/qa_dataset \
    --quality-report results/qa_quality_report.md
```

**Windows (PowerShell)**

```powershell
python build_qa_dataset.py --input data/evidence_units --output data/qa_dataset
```

**输入：** `data/evidence_units/evidence_units.jsonl`  
**输出：**

| 文件 | 说明 |
|------|------|
| `data/qa_dataset/qa_pairs.jsonl` | QA 评测数据集（约 878 条） |
| `results/qa_quality_report.md` | 质量过滤报告 |

#### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--llm-backend` | `placeholder` | LLM 后端：`placeholder` / `openai` / `deepseek` / `qwen` / `glm` / `ollama` |
| `--question-types` | 全部 7 种 | 题型：`definition,requirement,procedure,comparison,enumeration,constraint,exception` |
| `--pairs-min` / `--pairs-max` | `1` / `3` | 每个 Evidence Unit 生成的 QA 数量范围 |
| `--limit` | 无 | 仅处理前 N 条（调试用） |
| `--seed` | `42` | 随机种子 |

#### 使用真实 LLM（OpenAI 兼容 API）

```bash
pip install openai

# Linux / macOS
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"   # 以 DeepSeek 为例
export OPENAI_MODEL="deepseek-chat"

python build_qa_dataset.py --llm-backend deepseek
```

**Windows (PowerShell)**

```powershell
$env:OPENAI_API_KEY = "your-api-key"
$env:OPENAI_BASE_URL = "https://api.deepseek.com/v1"
$env:OPENAI_MODEL = "deepseek-chat"

python build_qa_dataset.py --llm-backend deepseek
```

---

## 一键复现清单

从原始 PDF 完整跑到当前阶段，按顺序执行：

```bash
# 0. 环境
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. 放置 PDF 到 data/raw_pdf/

# 2. 解析
python src/parsing/pdf_to_structure.py --input data/raw_pdf --output data/parsed_json

# 3. 分析（可选）
python src/analysis/dataset_profiler.py --input data/parsed_json

# 4. Evidence Unit
python src/chunking/build_evidence_units.py --input data/parsed_json --output data/evidence_units

# 5. 向量索引
python src/embedding/build_index.py --input data/evidence_units/evidence_units.jsonl --output data/vector_store

# 6. QA 数据集
python build_qa_dataset.py --input data/evidence_units --output data/qa_dataset
```

---

## 数据流概览

```
data/raw_pdf/*.pdf
        │
        ▼  Step 1  pdf_to_structure.py
data/parsed_json/*.json
        │
        ├──▶ Step 2  dataset_profiler.py  ──▶ results/dataset_profile.md
        │
        ▼  Step 3  build_evidence_units.py
data/evidence_units/evidence_units.jsonl
        │
        ├──▶ Step 4  build_index.py  ──▶ data/vector_store/
        │                              (faiss.index + embeddings.npy + metadata.json)
        │                                    └──▶ results/embedding_report.md
        │
        ▼  Step 5  build_qa_dataset.py
data/qa_dataset/qa_pairs.jsonl
        │
        └──▶ results/qa_quality_report.md
```

---

## 当前数据集规模（参考）

| 阶段 | 数量 |
|------|------|
| 原始 PDF | 13 份 |
| 解析 JSON | 13 个 |
| Clauses | ~501 |
| Evidence Units | 565 |
| 向量（FAISS index） | 565（与 Evidence Unit 一一对应） |
| QA Pairs（placeholder 后端） | 878 |

---

## 常见问题

**Q: 运行报错 `Input path does not exist`？**  
A: 检查对应目录下是否有数据文件。若从 GitHub 克隆，需先按上文「数据说明」放置数据集。

**Q: Windows 下 `&&` 报错？**  
A: PowerShell 旧版本不支持 `&&`，请逐条执行命令，或用 `;` 分隔。

**Q: 只想验证 QA 模块，不想重新解析？**  
A: 只需 `data/evidence_units/evidence_units.jsonl`，直接运行 Step 5 即可。

**Q: 只想重建向量索引？**  
A: 只需 `data/evidence_units/evidence_units.jsonl`，直接运行 Step 4 即可。向量数应与 JSONL 有效行数一致。

**Q: 向量数量和 Evidence Unit 数量为什么相同？**  
A: embedding 阶段对每条有效 Evidence Unit 生成 exactly 一个向量，不做合并或二次切分；空文本或损坏行会被跳过（当前数据集无跳过）。

**Q: placeholder 和真实 LLM 的区别？**  
A: `placeholder` 是离线占位后端，用于开发测试，不调用外部 API；真实评测应使用 `--llm-backend openai`（或 deepseek 等兼容端点）。

---

## License

Research use. See repository for details.
