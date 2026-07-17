"""Audit QA pairs for question–gold evidence misalignment."""

from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QA_PATH = ROOT / "data/qa_dataset/qa_pairs_v2.jsonl"
UNITS_PATH = ROOT / "data/evidence_units/evidence_units.jsonl"
OUT_DIR = ROOT / "results/qa_quality"

TEMPLATE_PREFIXES = [
    "为什么要关注",
    "的设计目标或主要用途是什么",
    "通常适用于哪些系统或业务场景",
    "在实际项目中",
    "一般如何落地应用",
    "的核心机制应如何理解",
    "各类别或方案如何区分",
    "相关的关键字段或条目有哪些",
]

DATE_PAT = re.compile(
    r"\d{4}[-年/]\d{1,2}[-月/]\d{1,2}|20\d{2}-\d{2}-\d{2}|发布|实施日"
)


def normalize(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def tokens(s: str) -> set[str]:
    s = (s or "").lower()
    out: set[str] = set()
    for m in re.finditer(r"[a-z0-9_\-]{2,}", s):
        out.add(m.group())
    chars = re.sub(r"[^\u4e00-\u9fff]", "", s)
    for i in range(len(chars) - 1):
        out.add(chars[i : i + 2])
    return out


def extract_quoted(q: str) -> list[str]:
    return re.findall(r"「([^」]{2,80})」", q)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    units = {r["unit_id"]: r for r in load_jsonl(UNITS_PATH)}
    doc_titles: dict[str, str] = {}
    for u in units.values():
        doc_titles[u["document_id"]] = (u.get("title") or "").strip()

    qas = load_jsonl(QA_PATH)
    issues: list[dict] = []

    for qa in qas:
        q = (qa.get("question") or "").strip()
        golds = qa.get("supporting_evidence") or []
        if not golds:
            issues.append(
                {
                    "qa_id": qa["qa_id"],
                    "flags": ["no_gold"],
                    "question": q,
                    "severity": "high",
                }
            )
            continue

        uid = golds[0]["unit_id"]
        u = units.get(uid)
        if not u:
            issues.append(
                {
                    "qa_id": qa["qa_id"],
                    "flags": ["missing_unit"],
                    "question": q,
                    "severity": "high",
                }
            )
            continue

        title = doc_titles.get(qa.get("document_id") or u["document_id"], "") or (
            u.get("title") or ""
        )
        evidence = u.get("text") or ""
        chapter = u.get("chapter_title") or ""
        flags: list[str] = []
        details: dict = {}
        quoted = extract_quoted(q)
        nq, nt, ne, nc = map(normalize, [q, title, evidence, chapter])

        title_in_q = False
        if title and len(title) >= 4:
            if title in q or (nt and nt in nq):
                title_in_q = True
            else:
                for chunk in re.split(r"[\s　]+", title):
                    if len(chunk) >= 4 and chunk in q:
                        title_in_q = True
                        break
            for qt in quoted:
                if SequenceMatcher(None, normalize(qt), nt).ratio() >= 0.75:
                    title_in_q = True

        if title_in_q:
            q_topic = quoted[0] if quoted else title
            topic_bits = [p for p in re.split(r"[\s　]+", q_topic) if len(p) >= 4]
            topic_in_ev = normalize(q_topic) in ne or any(p in evidence for p in topic_bits)
            title_like = SequenceMatcher(None, normalize(q_topic), nt).ratio() >= 0.75
            if title_like and not topic_in_ev:
                flags.append("question_uses_doc_title")
                details["title"] = title
                details["quoted"] = quoted

        if DATE_PAT.search(q) and any(k in q for k in ("发布", "实施", "日期")):
            if not re.search(r"20\d{2}|发布|实施", evidence[:120]):
                flags.append("question_uses_publish_date")
                details["date_match"] = DATE_PAT.findall(q)

        for qt in quoted:
            nqt = normalize(qt)
            if len(qt) < 4:
                continue
            if title and SequenceMatcher(None, nqt, nt).ratio() >= 0.75:
                continue
            if nqt in ne or nqt in nc:
                continue
            if len(nqt) >= 4 and nqt[: min(6, len(nqt))] in ne:
                continue
            qt_tok = tokens(qt)
            ev_tok = tokens(evidence[:600])
            if qt_tok and len(qt_tok & ev_tok) / max(len(qt_tok), 1) < 0.25:
                flags.append("quoted_topic_not_in_evidence")
                details.setdefault("bad_quoted", []).append(qt)

        q_clean = q
        for p in TEMPLATE_PREFIXES:
            q_clean = q_clean.replace(p, " ")
        for qt in quoted:
            if title and SequenceMatcher(None, normalize(qt), nt).ratio() >= 0.7:
                q_clean = q_clean.replace(f"「{qt}」", " ")
        q_tok = tokens(q_clean) | tokens(re.sub(r"「[^」]+」", " ", q))
        if title:
            q_tok -= tokens(title)
        ev_tok = tokens(evidence)
        if len(q_tok) >= 3:
            overlap = len(q_tok & ev_tok) / len(q_tok)
            details["overlap"] = round(overlap, 3)
            if overlap < 0.12:
                flags.append("low_question_evidence_overlap")
        elif title_in_q:
            flags.append("template_only_around_title")

        ans = (qa.get("answer") or "").strip()
        if ans.startswith("主要目的在于：") or ans.startswith("主要目的在于:"):
            body = ans.split("：", 1)[-1].split(":", 1)[-1].strip()
            if (
                body
                and SequenceMatcher(
                    None, normalize(body[:120]), normalize(evidence[:120])
                ).ratio()
                >= 0.85
            ):
                flags.append("answer_is_evidence_dump")

        if qa.get("question_type") == "purpose" and "question_uses_doc_title" in flags:
            flags.append("purpose_on_doc_title")

        if flags:
            hard = {
                "question_uses_doc_title",
                "question_uses_publish_date",
                "quoted_topic_not_in_evidence",
                "purpose_on_doc_title",
                "template_only_around_title",
                "no_gold",
                "missing_unit",
            }
            severity = "high" if hard & set(flags) else "medium"
            issues.append(
                {
                    "qa_id": qa["qa_id"],
                    "question": q,
                    "question_type": qa.get("question_type"),
                    "document_id": qa.get("document_id"),
                    "gold_unit_id": uid,
                    "chapter_title": chapter,
                    "gold_preview": evidence[:180].replace("\n", " "),
                    "flags": sorted(set(flags)),
                    "severity": severity,
                    "details": details,
                }
            )

    by_flag: Counter[str] = Counter()
    by_sev: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    for it in issues:
        by_sev[it["severity"]] += 1
        by_type[it.get("question_type") or "?"] += 1
        for f in it["flags"]:
            by_flag[f] += 1

    high = [i for i in issues if i["severity"] == "high"]
    medium = [i for i in issues if i["severity"] == "medium"]

    summary = {
        "total_qa": len(qas),
        "suspect_count": len(issues),
        "high_severity": len(high),
        "medium_severity": len(medium),
        "by_flag": dict(by_flag.most_common()),
        "by_question_type": dict(by_type.most_common()),
        "high_ids": [i["qa_id"] for i in high],
    }

    (OUT_DIR / "qa_alignment_audit.json").write_text(
        json.dumps({"summary": summary, "issues": issues}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# QA–Gold Alignment Audit",
        "",
        f"Total QA pairs: **{len(qas)}**",
        f"Suspect pairs: **{len(issues)}** (high={len(high)}, medium={len(medium)})",
        "",
        "## Flag counts",
        "",
    ]
    for f, c in by_flag.most_common():
        lines.append(f"- `{f}`: {c}")
    lines += ["", "## By question type (suspects)", ""]
    for t, c in by_type.most_common():
        lines.append(f"- {t}: {c}")

    lines += ["", "## High-severity examples", ""]
    for i, it in enumerate(high[:30], 1):
        lines += [
            f"### {i}. `{it['qa_id']}`",
            "",
            f"- **Flags:** {', '.join(it['flags'])}",
            f"- **Type:** {it['question_type']}",
            f"- **Question:** {it['question']}",
            f"- **Chapter:** {it['chapter_title']}",
            f"- **Gold preview:** {it['gold_preview']}",
            "",
        ]

    lines += ["", "## All high-severity IDs", "", "```"]
    lines.extend(summary["high_ids"])
    lines.append("```")

    (OUT_DIR / "qa_alignment_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT_DIR / 'qa_alignment_audit.md'}")


if __name__ == "__main__":
    main()
