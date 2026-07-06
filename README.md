# Sage RAG

A research-oriented Retrieval-Augmented Generation (RAG) system for **structured technical documents**, with a primary focus on standards and regulations (e.g., GB, ISO, industry norms).

## Overview

Sage RAG is designed for documents that have explicit hierarchical structure—chapters, sections, clauses, tables, and cross-references. Unlike generic document RAG pipelines that treat PDFs as flat text, this project emphasizes:

- **Clause-level retrieval** — indexing and retrieving at the granularity of individual clauses and sub-clauses
- **Structure-aware RAG** — preserving document hierarchy, numbering schemes, and semantic relationships during parsing, chunking, and retrieval

## Project Layout

```
sage_rag/
├── data/           # Raw inputs, parsed artifacts, chunks, and evaluation datasets
├── index/          # Vector store, graph index, and BM25 index
├── src/            # Core pipeline modules (ingestion → evaluation)
├── scripts/        # Runnable entry points and experiments
├── configs/        # YAML/JSON configuration files
├── logs/           # Runtime logs
└── results/        # Experiment outputs and metrics
```

## Pipeline Modules (`src/`)

| Module       | Responsibility                                      |
|--------------|-----------------------------------------------------|
| `ingestion`  | Load raw PDFs and metadata                          |
| `parsing`    | Extract structured content from standards documents |
| `chunking`   | Split parsed content into clause-level chunks       |
| `embedding`  | Encode chunks into dense vectors                    |
| `retrieval`  | Hybrid retrieval (dense + sparse + graph)           |
| `routing`    | Route queries to appropriate retrieval strategies   |
| `selection`  | Rerank and filter retrieved candidates              |
| `generation` | LLM-based answer synthesis with citations           |
| `evaluation`   | Benchmark against gold datasets                   |

## Getting Started

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## Status

This repository currently contains the **project scaffold only**. Pipeline logic will be implemented incrementally as part of ongoing research.
