# Sage RAG 项目上下文（GPT 交接文档）

> 将本文档粘贴给 GPT，可快速恢复项目上下文并继续协作。最后更新：2026-07-14。

---

## 1. 项目定位

**Sage RAG** 是面向结构化技术文档（GB / ISO / IEC 等）的研究型 RAG 基准系统。

核心差异：不以「PDF 纯文本切块」为主，而是保留**章节、条款、编号体系**，以 **Evidence Unit** 为检索粒度。

**仓库路径：** `d:\WorkCode\sage_rag`  
**环境：** Windows + PowerShell，Python venv 在 `.venv/`

---

## 2. 流水线现状（已完成）

| 阶段 | 模块 | 输出 | 规模 |
|------|------|------|------|
| PDF 解析 | `src/parsing/` | `data/parsed_json/` | 13 份标准 |
| Evidence Unit | `src/chunking/` | `data/evidence_units/evidence_units.jsonl` | **565 条** |
| Embedding + FAISS | `src/embedding/` | `data/vector_store/` | 565 向量，BGE-M3，dim=1024 |
| QA 生成（旧） | `src/generation/` | `data/qa_dataset/qa_pairs.jsonl` | **878 条（模板化）** |
| QA 生成（新） | `scripts/regenerate_qa.py` | `qa_pairs_natural_sample50.jsonl` | **30 条（验证集）** |
| Dense Retriever | `src/retrieval/` | `DenseRetriever` | FAISS IndexFlatIP |
| 检索评测 | `src/evaluation/` | `evaluate_dense.py` | Recall@K / MRR / nDCG |

**待实现：** BM25、Hybrid、Graph-RAG、端到端 RAG 生成评测。

---

## 3. 当前最重要的问题：QA 数据集重设计

### 3.1 问题背景

旧版 QA（878 条）由 placeholder LLM 按**模板**生成，问题大量包含：

- 根据条款……
- 根据第 X 章……
- 标准规定……
- 条款 X 要求……

这类问题**泄露文档结构**，不代表真实用户检索意图，导致 Dense Retriever 评测 Recall 极低。

### 3.2 解决方案（已实现）

重设计 QA 生成模块，要求问题模拟**未读过文档**但**懂领域**的用户（工程师、审计员、标准使用者）：

**禁止出现在问题中：**
- 根据条款 / 依据条款 / 按照 X.X / 本标准规定 / 第 X 章 / 第 X 节 / 条款 X
- clause/section/chapter/table 编号
- 复制 evidence 标题或首句

**问题应自然，例如：**
- 一次完整的代码安全审计包括哪些主要阶段？
- 「遗留调试代码」在相关规范中指的是什么？
- 执行「关键状态数据外部可控」相关检查通常包含哪些步骤？

**答案：** 自然语言概括 evidence，非逐字复制，保留技术正确性。

**JSON schema 不变：** `qa_id`, `question`, `answer`, `supporting_evidence`（含 `unit_id`）, `difficulty`, `question_type`, `document_type`, `document_id`

### 3.3 新增/修改的核心文件

| 文件 | 作用 |
|------|------|
| `scripts/regenerate_qa.py` | 分步重生成 CLI（`--sample 50` 验证，`--full` 全量） |
| `src/generation/prompts/qa_generation_natural.yaml` | 自然语言 Prompt（默认） |
| `src/generation/qa_quality.py` | 禁止短语、条款号、复制检测 |
| `src/generation/qa_validator.py` | 去重 + 严格自然语言校验 |
| `src/generation/qa_builder.py` | 抽样、重试、质量统计 |
| `src/generation/llm_interface.py` | Placeholder 改为自然语言模板（离线测试用） |

### 3.4 11 种题型

`definition, requirement, procedure, purpose, comparison, constraint, enumeration, exception, explanation, cause, application`

`requirement` 在采样时降权（0.6），避免过度使用。

---

## 4. 实验结果（关键验证）

### 4.1 验证集生成

```bash
python scripts/regenerate_qa.py \
    --input data/evidence_units/evidence_units.jsonl \
    --sample 50 \
    --output data/qa_dataset/qa_pairs_natural_sample50.jsonl
```

- 抽样 50 个 Evidence Unit
- 1 个过短跳过
- **30 条 QA 成功**（placeholder 后端；19 个 unit 在 5 次重试后仍失败，多为 ISO/IEC 法文/英文 boilerplate）
- 质量报告：`results/qa_quality_report_natural_sample50.md`

### 4.2 Dense Retriever 对比（BGE-M3 + FAISS，top_k=10，同一评测框架）

| 数据集 | n | Recall@1 | Recall@5 | Recall@10 | MRR |
|--------|--:|---------:|---------:|----------:|----:|
| 旧版模板化 QA（全量） | 878 | 0.80% | 2.39% | **4.44%** | 0.0164 |
| 旧版模板化 QA（同 30 gold unit） | 30 | 0.00% | 3.33% | 10.00% | 0.0154 |
| **自然语言 QA（验证集）** | **30** | **56.67%** | **66.67%** | **70.00%** | **0.6069** |

**结论：** 自然语言 QA 方向正确；Recall@10 从 ~4–10% 提升到 **70%**（同 unit 对照）。

评测报告：
- `results/retrieval_dense_report_natural_sample50.md`
- `results/retrieval_results_natural_sample50.jsonl`

### 4.3 自然语言问题样例（来自验证集）

```json
{"question": "「遗留调试代码」在相关规范中指的是什么？", "question_type": "definition"}
{"question": "执行「关键状态数据外部可控」相关检查通常包含哪些步骤？", "question_type": "procedure"}
{"question": "如何理解「早期放大攻击」的含义？", "question_type": "definition"}
{"question": "What is the purpose of `single level reference designation`?", "question_type": "purpose"}
```

### 4.4 旧版问题样例（同 gold unit，对比）

```
条款 9.1 设定了哪些限制、阈值或约束条件？
标准 GB_T_39412-2020 中 9.3 规定了哪些要求或条件？
```

---

## 5. 推荐下一步（论文「逐步验证」路线）

1. **人工抽查** `data/qa_dataset/qa_pairs_natural_sample50.jsonl`（30 条）确认提问风格
2. **用真实 LLM 重跑 50 条抽样**（提高 ISO/IEC 覆盖率）：
   ```bash
   python scripts/regenerate_qa.py --sample 50 --llm-backend openai
   # 或 deepseek / qwen 等 OpenAI 兼容端点
   ```
3. **再次跑 Dense 评测**，确认 Recall 提升可复现
4. **全量生成**（验证通过后）：
   ```bash
   python scripts/regenerate_qa.py --full \
       --output data/qa_dataset/qa_pairs_natural.jsonl \
       --llm-backend openai
   ```
5. **保留旧集** `qa_pairs.jsonl` 作为对比基线，论文中报告「模板 QA vs 自然语言 QA」的 ablation

---

## 6. 常用命令速查

```powershell
# 激活环境
.venv\Scripts\Activate.ps1

# 自然语言 QA 验证集（50 unit 抽样）
python scripts/regenerate_qa.py --sample 50

# Dense 评测（自然语言验证集）
python evaluate_dense.py `
    --qa data/qa_dataset/qa_pairs_natural_sample50.jsonl `
    --index data/vector_store `
    --top-k 10 `
    --output results/retrieval_results_natural_sample50.jsonl `
    --report results/retrieval_dense_report_natural_sample50.md `
    --device cpu

# 离线模式（模型已缓存）
$env:HF_HUB_OFFLINE = "1"

# 真实 LLM
$env:OPENAI_API_KEY = "your-key"
$env:OPENAI_BASE_URL = "https://api.deepseek.com/v1"
$env:OPENAI_MODEL = "deepseek-chat"
python scripts/regenerate_qa.py --sample 50 --llm-backend deepseek
```

---

## 7. 数据与目录结构

```
data/
├── evidence_units/evidence_units.jsonl     # 565 条，gold 来源
├── vector_store/                           # FAISS + metadata
└── qa_dataset/
    ├── qa_pairs.jsonl                      # 旧版 878 条（勿覆盖）
    ├── qa_pairs_natural_sample50.jsonl       # 新版验证 30 条
    └── qa_pairs_legacy_matched30.jsonl       # 同 unit 旧版对照 30 条

results/
├── qa_quality_report_natural_sample50.md
├── retrieval_dense_report_natural_sample50.md
├── retrieval_dense_report_legacy_matched30.md
└── retrieval_results_*.jsonl
```

---

## 8. 设计约束（给 GPT 的指令）

继续开发时请遵守：

1. **不要覆盖** `qa_pairs.jsonl`，新 QA 写入独立文件（如 `qa_pairs_natural.jsonl`）
2. **问题禁止**结构导航用语（条款号、章节号、表号）
3. **先小批量验证**（`--sample 50`），再 `--full` 扩展
4. **placeholder 仅用于离线管线测试**；正式 benchmark 用真实 LLM
5. **检索评测 gold** = `supporting_evidence[].unit_id`
6. **最小改动原则**：不重构无关模块；匹配现有代码风格
7. **Windows 环境**：PowerShell，路径用反斜杠或引号包裹

---

## 9. 给 GPT 的示例续作请求

你可以直接对 GPT 说：

> 我在做 Sage RAG 项目（见交接文档）。自然语言 QA 验证集 30 条上 Dense Recall@10=70%，旧模板 QA 仅 4–10%。请帮我：
> 1. 用 DeepSeek API 对 50 个 unit 重新生成自然语言 QA；
> 2. 对比新旧 Recall；
> 3. 若质量 OK，生成全量 `qa_pairs_natural.jsonl`；
> 4. 更新论文实验表格草稿。

或：

> 请实现 BM25 retriever，接入现有 `RetrieverBase` 接口，在同一 QA 验证集上评测并与 Dense 对比。

---

## 10. 关键代码入口

| 任务 | 入口 |
|------|------|
| 重生成 QA | `scripts/regenerate_qa.py` |
| 全量构建 QA | `build_qa_dataset.py` / `src/generation/build_qa_dataset.py` |
| Prompt 模板 | `src/generation/prompts/qa_generation_natural.yaml` |
| 质量规则 | `src/generation/qa_quality.py` |
| Dense 检索 | `src/retrieval/dense_retriever.py` |
| 检索评测 | `evaluate_dense.py` / `src/evaluation/evaluate_dense.py` |
| Evidence Unit | `src/chunking/evidence_units.py` |

---

*本文档与 `README.md` 同步维护；详细命令与参数见 README Step 5 / Step 7。*
