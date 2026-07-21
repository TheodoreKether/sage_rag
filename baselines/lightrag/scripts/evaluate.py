#!/usr/bin/env python
"""Evaluate LightRAG on Clean QA (Recall@k / MRR via sage_rag metrics).

Does not import src.retrieval package (avoids faiss/bm25 in lightrag env).

Usage:
  conda activate lightrag
  cd baselines/lightrag
  python scripts/evaluate.py --mode mix --top-k 10
  python scripts/evaluate.py --sample 5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import random
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

BASELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BASELINE_ROOT.parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapter.paths import (  # noqa: E402
    PAPER_RESULTS_DIR,
    QA_CLEAN,
    RAG_STORAGE_DIR,
    RESULTS_DIR,
)
from adapter.retriever import LightRAGRetriever  # noqa: E402

logger = logging.getLogger(__name__)


def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_metrics = _load_module(
    "sage_rag_retrieval_metrics",
    REPO_ROOT / "src" / "evaluation" / "retrieval_metrics.py",
)
_report = _load_module(
    "sage_rag_evaluation_report",
    REPO_ROOT / "src" / "evaluation" / "report.py",
)


@dataclass
class QASample:
    qa_id: str
    question: str
    gold_unit_ids: list[str]
    question_type: str = ""
    document_id: str = ""


def load_qa_samples(qa_path: Path) -> tuple[list[QASample], int]:
    samples: list[QASample] = []
    skipped = 0
    with qa_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            question = (record.get("question") or "").strip()
            gold: list[str] = []
            for item in record.get("supporting_evidence") or []:
                if isinstance(item, dict):
                    uid = (item.get("unit_id") or "").strip()
                    if uid:
                        gold.append(uid)
            if not question or not gold:
                skipped += 1
                continue
            samples.append(
                QASample(
                    qa_id=str(record.get("qa_id", f"line_{line_no}")),
                    question=question,
                    gold_unit_ids=gold,
                    question_type=str(record.get("question_type", "")),
                    document_id=str(record.get("document_id", "")),
                )
            )
    return samples, skipped


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate LightRAG on Clean Benchmark")
    p.add_argument("--qa", type=Path, default=QA_CLEAN)
    p.add_argument("--working-dir", type=Path, default=RAG_STORAGE_DIR)
    p.add_argument(
        "--mode",
        default="mix",
        choices=["mix", "hybrid", "local", "global", "naive"],
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--enable-rerank", action="store_true")
    p.add_argument(
        "--mirror-paper",
        action="store_true",
        help="Copy metrics to results/retrieval/lightrag/",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.mode
    results_path = RESULTS_DIR / f"retrieval_results_{tag}.jsonl"
    report_path = RESULTS_DIR / f"evaluation_report_{tag}.md"
    metrics_path = RESULTS_DIR / f"metrics_{tag}.json"

    all_samples, load_skipped = load_qa_samples(args.qa)
    if args.sample is not None and args.sample < len(all_samples):
        rng = random.Random(args.seed)
        idx = sorted(rng.sample(range(len(all_samples)), args.sample))
        eval_samples = [all_samples[i] for i in idx]
    else:
        eval_samples = all_samples

    retriever = LightRAGRetriever(
        working_dir=args.working_dir,
        mode=args.mode,
        enable_rerank=args.enable_rerank,
    )

    records: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    failed = 0
    start = time.perf_counter()
    try:
        for sample in tqdm(eval_samples, desc=f"LightRAG-{tag}", unit="qa"):
            try:
                hits = retriever.retrieve(sample.question, top_k=args.top_k)
                retrieved_ids = [h.unit_id for h in hits]
                metrics = _metrics.compute_retrieval_metrics(
                    retrieved_ids,
                    sample.gold_unit_ids,
                    recall_ks=(1, 5, 10),
                    ndcg_ks=(5, 10),
                )
                record = {
                    "qa_id": sample.qa_id,
                    "question": sample.question,
                    "gold_unit_ids": sample.gold_unit_ids,
                    "retrieved_unit_ids": retrieved_ids,
                    "retrieved": [
                        {
                            "rank": h.rank,
                            "unit_id": h.unit_id,
                            "score": h.score,
                        }
                        for h in hits
                    ],
                    "metrics": metrics,
                }
                records.append(record)
                metric_rows.append(metrics)
            except Exception as exc:
                failed += 1
                logger.warning("Failed qa_id=%s: %s", sample.qa_id, exc)
    finally:
        retriever.close()

    elapsed = time.perf_counter() - start
    avg = _metrics.average_metrics(metric_rows)
    type_counts = Counter(s.question_type for s in eval_samples if s.question_type)
    summary = _report.EvaluationSummary(
        retriever_name=f"LightRAG-{tag}",
        model_name=args.mode,
        index_dir=str(args.working_dir),
        qa_file=str(args.qa),
        top_k=args.top_k,
        total_qa_pairs=len(all_samples) + load_skipped,
        evaluated_pairs=len(records),
        skipped_pairs=load_skipped,
        failed_pairs=failed,
        elapsed_seconds=elapsed,
        average_metrics=avg,
        dataset_stats={
            "question_types": dict(type_counts),
            "multi-gold QA pairs": sum(1 for s in eval_samples if len(s.gold_unit_ids) > 1),
        },
        results_jsonl=str(results_path),
        report_title=f"LightRAG ({tag}) Retrieval Evaluation Report",
    )
    _report.write_report(summary, report_path)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics_payload = {
        "retriever": f"LightRAG-{tag}",
        "mode": args.mode,
        "qa": str(args.qa),
        "top_k": args.top_k,
        "evaluated_pairs": summary.evaluated_pairs,
        "failed_pairs": summary.failed_pairs,
        "elapsed_seconds": summary.elapsed_seconds,
        "metrics": {k: float(v) for k, v in avg.items()},
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2))

    if args.mirror_paper:
        PAPER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for src in (results_path, report_path, metrics_path):
            shutil.copy2(src, PAPER_RESULTS_DIR / src.name)
        logger.info("Mirrored artifacts to %s", PAPER_RESULTS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
