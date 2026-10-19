"""Audit generated question maps and atomic-requirement JSON files.

This utility only reads question-paper-derived JSON. It does not load student
submissions, execute code, classify answers, or assign marks.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


def shorten(value: Any, limit: int = 260) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def audit_file(path: Path, sample_count: int) -> tuple[list[str], dict[str, int]]:
    data = load_json(path)
    requirements = data.get("requirements", [])
    questions = data.get("questions", [])
    if not isinstance(requirements, list):
        raise ValueError("'requirements' must be a list")
    if not isinstance(questions, list):
        raise ValueError("'questions' must be a list")

    lines: list[str] = []
    kinds = Counter(str(item.get("requirement_type", "missing")) for item in requirements if isinstance(item, dict))
    review_items = [item for item in requirements if isinstance(item, dict) and item.get("needs_review")]
    ids = [item.get("requirement_id") for item in requirements if isinstance(item, dict)]
    duplicate_ids = sorted({rid for rid in ids if rid is not None and ids.count(rid) > 1})

    bad_spans: list[tuple[str, str, Any]] = []
    missing_fields: list[str] = []
    for item in requirements:
        if not isinstance(item, dict):
            missing_fields.append("non-object requirement record")
            continue
        rid = str(item.get("requirement_id", "<missing-id>"))
        source_text = item.get("source_text")
        atomic_text = item.get("atomic_text")
        span = item.get("source_span")
        if not isinstance(source_text, str) or not isinstance(atomic_text, str):
            missing_fields.append(f"{rid}: source_text/atomic_text is missing or not text")
            continue
        if not isinstance(span, dict) or not isinstance(span.get("start"), int) or not isinstance(span.get("end"), int):
            bad_spans.append((rid, atomic_text, span))
            continue
        start, end = span["start"], span["end"]
        if start < 0 or end < start or end > len(source_text) or source_text[start:end] != atomic_text:
            bad_spans.append((rid, atomic_text, span))

    lines.append(f"PAPER FILE: {path.name}")
    lines.append(f"Questions in extracted output: {len(questions)}")
    lines.append(f"Requirement records: {len(requirements)}")
    lines.append(f"Needs review: {len(review_items)} ({(100 * len(review_items) / len(requirements)) if requirements else 0:.1f}%)")
    lines.append("Requirement type counts: " + json.dumps(dict(sorted(kinds.items())), ensure_ascii=False))
    lines.append(f"Duplicate requirement IDs: {len(duplicate_ids)}")
    if duplicate_ids:
        lines.append("  IDs: " + ", ".join(duplicate_ids[:30]))
    lines.append(f"Source-span mismatches: {len(bad_spans)}")
    for rid, atomic_text, span in bad_spans[:20]:
        lines.append(f"  {rid}: span={span!r}; atomic_text={atomic_text!r}")
    lines.append(f"Missing/invalid text fields: {len(missing_fields)}")
    for problem in missing_fields[:20]:
        lines.append("  " + problem)

    diagnostics = data.get("diagnostics", [])
    lines.append(f"Parser/extraction diagnostics carried into output: {len(diagnostics) if isinstance(diagnostics, list) else 'invalid format'}")
    if isinstance(diagnostics, list):
        for diagnostic in diagnostics:
            lines.append("  " + json.dumps(diagnostic, ensure_ascii=False, sort_keys=True))

    lines.append(f"Sample needs-review requirements (up to {sample_count}):")
    for item in review_items[:sample_count]:
        lines.append(
            "  - " + str(item.get("requirement_id", "<missing-id>"))
            + f" | question={item.get('question_id')!r}"
            + f" | category={item.get('source_category')!r}"
            + f" | atomic_text={shorten(item.get('atomic_text'))!r}"
        )
    lines.append("")

    summary = {
        "questions": len(questions),
        "requirements": len(requirements),
        "needs_review": len(review_items),
        "bad_spans": len(bad_spans),
        "duplicate_ids": len(duplicate_ids),
        "diagnostics": len(diagnostics) if isinstance(diagnostics, list) else 0,
    }
    return lines, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/processed/requirements")
    parser.add_argument("--pattern", default="*.requirements.json")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--output", default="data/processed/requirements/requirement_extraction_review.txt")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")
    if args.sample_count < 0:
        parser.error("--sample-count must be zero or greater")

    candidates = input_dir.rglob(args.pattern) if args.recursive else input_dir.glob(args.pattern)
    files = sorted(path for path in candidates if path.is_file() and path.name != Path(args.output).name)
    if not files:
        parser.error(f"No files matching {args.pattern!r} found in {input_dir}")

    report: list[str] = ["REQUIREMENT EXTRACTION QUALITY REVIEW", f"Input directory: {input_dir}", f"Files reviewed: {len(files)}", ""]
    totals = Counter()
    failures: list[str] = []
    for path in files:
        try:
            file_lines, stats = audit_file(path, args.sample_count)
            report.extend(file_lines)
            totals.update(stats)
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            failures.append(f"{path}: {exc}")

    report.extend(["=" * 88, "AGGREGATE SUMMARY"])
    for key in ("questions", "requirements", "needs_review", "bad_spans", "duplicate_ids", "diagnostics"):
        report.append(f"{key}: {totals[key]}")
    if totals["requirements"]:
        report.append(f"needs_review_rate: {100 * totals['needs_review'] / totals['requirements']:.1f}%")
    report.append(f"Files failed to audit: {len(failures)}")
    report.extend("  " + failure for failure in failures)
    report.append("Student code executed: no")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report[-12:]))
    print(f"\nFull review saved to: {output}")
    return 1 if failures or totals["bad_spans"] or totals["duplicate_ids"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
