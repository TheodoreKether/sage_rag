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
│   └── qa_dataset/       # QA 评测数据集
├── index/                # 向量索引、图索引、BM25 索引（待实现）
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
| QA 数据集构建 | `src/generation/` | ✅ 已完成 |
| 向量化 / 检索 / 生成 / 评测 | `src/embedding/` 等 | ⏳ 待实现 |

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

然后依次执行下方 **Step 1 → Step 4** 的全部命令。

### 方式 B：直接使用已生成的中间数据

若已收到打包好的数据，解压到项目根目录，确保目录结构如下：

```
data/
├── raw_pdf/                        # 可选，仅当需要重新解析时
├── parsed_json/*.json              # 13 个解析结果
├── evidence_units/evidence_units.jsonl
└── qa_dataset/qa_pairs.jsonl       # 可选，仅当需要重新生成 QA 时

results/                            # 可选，分析报告
├── dataset_profile.md
├── dataset_statistics.json
├── evidence_statistics.md
├── qa_quality_report.md
└── figures/
```

收到中间数据后，可从对应步骤继续，无需从头运行。

---

## 流水线：从原始 PDF 到 QA 数据集

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

### Step 4 — QA 数据集构建

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

# 5. QA 数据集
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
        ▼  Step 4  build_qa_dataset.py
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
| QA Pairs（placeholder 后端） | 878 |

---

## 常见问题

**Q: 运行报错 `Input path does not exist`？**  
A: 检查对应目录下是否有数据文件。若从 GitHub 克隆，需先按上文「数据说明」放置数据集。

**Q: Windows 下 `&&` 报错？**  
A: PowerShell 旧版本不支持 `&&`，请逐条执行命令，或用 `;` 分隔。

**Q: 只想验证 QA 模块，不想重新解析？**  
A: 只需 `data/evidence_units/evidence_units.jsonl`，直接运行 Step 4 即可。

**Q: placeholder 和真实 LLM 的区别？**  
A: `placeholder` 是离线占位后端，用于开发测试，不调用外部 API；真实评测应使用 `--llm-backend openai`（或 deepseek 等兼容端点）。

---

## License

Research use. See repository for details.
