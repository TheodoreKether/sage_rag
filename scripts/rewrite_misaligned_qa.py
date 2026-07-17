"""Batch-rewrite high-severity misaligned QA questions."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_qa_alignment import main as audit_main  # noqa: E402
from src.analysis.qa_question_rewriter import rewrite_question  # noqa: E402

QA_PATH = ROOT / "data/qa_dataset/qa_pairs_v2.jsonl"
UNITS_PATH = ROOT / "data/evidence_units/evidence_units.jsonl"
AUDIT_PATH = ROOT / "results/qa_quality/qa_alignment_audit.json"
OUT_DIR = ROOT / "results/qa_quality"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite misaligned QA questions")
    parser.add_argument(
        "--qa-file",
        type=Path,
        default=QA_PATH,
        help="QA dataset JSONL to update",
    )
    parser.add_argument(
        "--audit-file",
        type=Path,
        default=AUDIT_PATH,
        help="Alignment audit JSON with high_ids",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview rewrites without writing QA file",
    )
    args = parser.parse_args()

    if not args.audit_file.is_file():
        raise FileNotFoundError(f"Audit file not found: {args.audit_file}")

    audit = json.loads(args.audit_file.read_text(encoding="utf-8"))
    high_ids = set(audit["summary"]["high_ids"])
    units = {r["unit_id"]: r for r in load_jsonl(UNITS_PATH)}
    qas = load_jsonl(args.qa_file)

    changes: list[dict] = []
    updated = 0
    for qa in qas:
        qa_id = qa["qa_id"]
        if qa_id not in high_ids:
            continue
        golds = qa.get("supporting_evidence") or []
        if not golds:
            continue
        unit = units.get(golds[0]["unit_id"])
        if not unit:
            continue

        old_q = qa.get("question", "")
        new_q = rewrite_question(qa, unit)
        if new_q != old_q:
            changes.append(
                {
                    "qa_id": qa_id,
                    "question_type": qa.get("question_type"),
                    "gold_unit_id": golds[0]["unit_id"],
                    "old_question": old_q,
                    "new_question": new_q,
                    "gold_preview": (unit.get("text") or "")[:160],
                }
            )
            qa["question"] = new_q
            updated += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "qa_rewrite_log.json"
    log_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "target_count": len(high_ids),
                "rewritten_count": updated,
                "changes": changes,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    md_lines = [
        "# QA Question Rewrite Log",
        "",
        f"- Target high-severity items: **{len(high_ids)}**",
        f"- Rewritten: **{updated}**",
        "",
        "## Samples",
        "",
    ]
    for item in changes[:20]:
        md_lines += [
            f"### `{item['qa_id']}`",
            "",
            f"- **Old:** {item['old_question']}",
            f"- **New:** {item['new_question']}",
            f"- **Gold:** {item['gold_preview']}",
            "",
        ]
    (OUT_DIR / "qa_rewrite_log.md").write_text("\n".join(md_lines), encoding="utf-8")

    if args.dry_run:
        print(f"Dry run: would rewrite {updated}/{len(high_ids)} questions")
        print(f"Log preview: {log_path}")
        return

    backup = args.qa_file.with_suffix(
        args.qa_file.suffix + f".bak.{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    shutil.copy2(args.qa_file, backup)
    write_jsonl(args.qa_file, qas)
    print(f"Rewrote {updated}/{len(high_ids)} questions")
    print(f"Backup: {backup}")
    print(f"Updated: {args.qa_file}")
    print(f"Log: {log_path}")

    # Re-run alignment audit on updated dataset
    audit_main()
    post = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    print(
        "Post-rewrite audit: "
        f"suspect={post['summary']['suspect_count']}, "
        f"high={post['summary']['high_severity']}, "
        f"medium={post['summary']['medium_severity']}"
    )


if __name__ == "__main__":
    main()
