"""Build clean QA benchmark by removing Dataset / Annotation Issues."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.failure_classifier import UnitInfo
from src.analysis.root_cause_classifier import detect_dataset_issue

QA_V2 = ROOT / "data/qa_dataset/qa_pairs_v2.jsonl"
QA_CLEAN = ROOT / "data/qa_dataset/qa_pairs_clean.jsonl"
UNITS = ROOT / "data/evidence_units/evidence_units.jsonl"
OUT_DIR = ROOT / "results/retrieval/clean_benchmark"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean QA benchmark")
    parser.add_argument("--qa-in", type=Path, default=QA_V2)
    parser.add_argument("--qa-out", type=Path, default=QA_CLEAN)
    parser.add_argument("--units", type=Path, default=UNITS)
    parser.add_argument("--log-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    units = {}
    for r in load_jsonl(args.units):
        u = UnitInfo.from_record(r)
        units[u.unit_id] = u

    qas = load_jsonl(args.qa_in)
    removed: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []

    for qa in qas:
        golds = qa.get("supporting_evidence") or []
        if not golds:
            removed.append(
                {
                    "qa_id": qa.get("qa_id"),
                    "reason": "Empty supporting_evidence",
                    "question": qa.get("question"),
                }
            )
            continue
        uid = golds[0]["unit_id"]
        gold = units.get(uid)
        if gold is None:
            removed.append(
                {
                    "qa_id": qa.get("qa_id"),
                    "reason": f"Missing evidence unit: {uid}",
                    "question": qa.get("question"),
                }
            )
            continue

        is_issue, reason = detect_dataset_issue(
            str(qa.get("question") or ""),
            str(qa.get("question_type") or ""),
            gold,
            doc_title=gold.title or "",
        )
        if is_issue:
            removed.append(
                {
                    "qa_id": qa["qa_id"],
                    "reason": reason,
                    "question": qa.get("question"),
                    "gold_unit_id": uid,
                    "question_type": qa.get("question_type"),
                }
            )
        else:
            kept.append(qa)

    write_jsonl(args.qa_out, kept)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log = {
        "source_qa": str(args.qa_in),
        "clean_qa": str(args.qa_out),
        "original_count": len(qas),
        "removed_count": len(removed),
        "clean_count": len(kept),
        "removed_ids": [r["qa_id"] for r in removed],
        "removed_details": removed,
    }
    (args.log_dir / "cleaning_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# QA Clean Benchmark Log",
        "",
        f"- Source: `{args.qa_in}` ({len(qas)})",
        f"- Clean: `{args.qa_out}` ({len(kept)})",
        f"- Removed Dataset Issues: **{len(removed)}**",
        "",
        "## Removed examples",
        "",
    ]
    for item in removed[:20]:
        lines += [
            f"### `{item['qa_id']}`",
            "",
            f"- **Reason:** {item['reason']}",
            f"- **Question:** {item.get('question')}",
            "",
        ]
    (args.log_dir / "cleaning_log.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Original: {len(qas)}")
    print(f"Removed:  {len(removed)}")
    print(f"Clean:    {len(kept)}")
    print(f"Wrote:    {args.qa_out}")
    print(f"Log:      {args.log_dir / 'cleaning_log.json'}")


if __name__ == "__main__":
    main()
