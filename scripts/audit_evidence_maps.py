from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evaluation.evidence_mapper import extract_submission_segments


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _is_code_type(segment_type: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(segment_type or "").casefold()).strip("_")
    return "code" in normalized.split("_") or normalized in {"python", "ipython", "program"}


def _looks_like_q_assignment(text: str) -> bool:
    first_line = next((line for line in str(text or "").splitlines() if line.strip()), "")
    return bool(re.match(r"^\s*(?:#\s*)?q\s*\d{1,3}\s*=", first_line, re.IGNORECASE))


def _inventory(qmap: dict[str, Any]) -> list[str]:
    ids = []
    for entry in qmap.get("questions", []) or []:
        if isinstance(entry, dict) and (entry.get("question_id") or entry.get("id")):
            ids.append(str(entry.get("question_id") or entry.get("id")))
    for entry in qmap.get("standalone_tasks", []) or []:
        if isinstance(entry, dict) and (entry.get("task_id") or entry.get("question_id")):
            ids.append(str(entry.get("task_id") or entry.get("question_id")))
    return ids


def audit_one(qmap: dict[str, Any], reqs: dict[str, Any], submission: dict[str, Any], source_name: str, output: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    question_ids = _inventory(qmap)
    if len(question_ids) != len(set(question_ids)):
        errors.append("question map has duplicate question/task IDs")
    out_question_ids = [str(q.get("question_id")) for q in output.get("questions", []) if isinstance(q, dict)]
    if out_question_ids != question_ids:
        errors.append("output question/task IDs or order differ from the question map")
    if output.get("student_code_executed") is not False:
        errors.append("student_code_executed is not false")
    if output.get("source_submission_filename") != source_name:
        errors.append("source submission filename mismatch")

    # Compare all requirement objects, not just counts or IDs.
    input_records = [r for r in reqs.get("requirements", []) or [] if isinstance(r, dict) and r.get("requirement_id") is not None]
    input_by_id = {str(r["requirement_id"]): r for r in input_records}
    if len(input_by_id) != len(input_records):
        errors.append("input flat requirements contain duplicate IDs")
    output_records = []
    for question in output.get("questions", []) or []:
        if isinstance(question, dict):
            output_records.extend(r for r in question.get("requirements", []) or [] if isinstance(r, dict) and r.get("requirement_id") is not None)
    output_records.extend(r for r in output.get("global_requirements", []) or [] if isinstance(r, dict) and r.get("requirement_id") is not None)
    out_ids = [str(r["requirement_id"]) for r in output_records]
    if len(out_ids) != len(set(out_ids)):
        errors.append("output requirement IDs are duplicated across question/global lists")
    if set(out_ids) != set(input_by_id):
        errors.append(f"requirement ID set differs (input={len(input_by_id)}, output={len(set(out_ids))}, missing={len(set(input_by_id)-set(out_ids))}, extra={len(set(out_ids)-set(input_by_id))})")
    out_by_id = {str(r["requirement_id"]): r for r in output_records}
    changed = [rid for rid in set(input_by_id) & set(out_by_id) if _canonical(input_by_id[rid]) != _canonical(out_by_id[rid])]
    if changed:
        errors.append(f"{len(changed)} full requirement objects changed; examples: {changed[:3]}")

    # Re-check evidence against the raw submission's segments and exact source offsets.
    segments = extract_submission_segments(submission, source_name)
    segment_by_id = {s.segment_id: s for s in segments}
    if len(segment_by_id) != len(segments):
        errors.append("submission has duplicate segment IDs")
    blocks = []
    for q in output.get("questions", []) or []:
        if isinstance(q, dict):
            blocks.extend(b for b in q.get("evidence_blocks", []) or [] if isinstance(b, dict))
    blocks.extend(b for b in output.get("unassigned_evidence_blocks", []) or [] if isinstance(b, dict))
    block_ids = [str(b.get("block_id")) for b in blocks]
    if len(block_ids) != len(set(block_ids)):
        errors.append("evidence block IDs are duplicated")
    intervals: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for block in blocks:
        sid = str(block.get("segment_id", ""))
        source = segment_by_id.get(sid)
        if source is None:
            errors.append(f"evidence block references missing segment {sid!r}")
            continue
        try:
            start = int(block["segment_start_offset"])
            end = int(block["segment_end_offset"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"block {block.get('block_id')} has invalid source offsets")
            continue
        if start < 0 or end < start or end > len(source.source_text):
            errors.append(f"block {block.get('block_id')} offsets out of range")
            continue
        if block.get("source_text") != source.source_text[start:end]:
            errors.append(f"block {block.get('block_id')} source text does not match its raw offset slice")
        if block.get("output_text", "") != source.output_text:
            errors.append(f"block {block.get('block_id')} output text differs from raw parser output")
        if block.get("mapping_method") == "explicit_question_heading":
            if _is_code_type(source.segment_type):
                errors.append(f"code segment {sid} was incorrectly treated as an explicit question heading")
            elif _looks_like_q_assignment(block.get("source_text", "")):
                errors.append(f"block {block.get('block_id')} looks like a q-number variable assignment, not a question heading")
        intervals[sid].append((start, end, str(block.get("block_id"))))
        if block.get("mapping_confidence") not in (0, 0.0, 0.75, 1, 1.0):
            errors.append(f"block {block.get('block_id')} has unexpected structural confidence")

    expected_segments = {s.segment_id for s in segments if s.source_text.strip() or s.output_text.strip()}
    present_segments = set(intervals)
    if expected_segments != present_segments:
        errors.append(f"segment coverage differs (expected={len(expected_segments)}, mapped={len(present_segments)}, missing={sorted(expected_segments-present_segments)[:4]})")
    for sid in expected_segments & present_segments:
        source = segment_by_id[sid].source_text
        ranges = sorted(intervals[sid])
        cursor = 0
        for start, end, block_id in ranges:
            if start != cursor:
                errors.append(f"segment {sid} has a gap/overlap before block {block_id}: expected offset {cursor}, got {start}")
                break
            cursor = end
        if cursor != len(source):
            errors.append(f"segment {sid} coverage ends at {cursor}, raw length is {len(source)}")

    expected_evidence_blocks = len(blocks)
    summary = output.get("summary", {})
    summary_checks = {
        "question_or_task_count": len(question_ids),
        "submission_segments": len(segments),
        "evidence_blocks": expected_evidence_blocks,
        "mapped_blocks": sum(len(q.get("evidence_blocks", []) or []) for q in output.get("questions", []) or [] if isinstance(q, dict)),
        "unassigned_blocks": len(output.get("unassigned_evidence_blocks", []) or []),
    }
    for key, expected in summary_checks.items():
        if summary.get(key) != expected:
            errors.append(f"summary.{key}={summary.get(key)!r}, actual={expected}")
    return {
        "submission": source_name,
        "status": "PASS" if not errors else "FAIL",
        "question_ids": len(question_ids),
        "input_requirements": len(input_records),
        "output_requirements": len(out_ids),
        "source_segments": len(segments),
        "evidence_blocks": expected_evidence_blocks,
        "mapped_blocks": summary_checks["mapped_blocks"],
        "unassigned_blocks": summary_checks["unassigned_blocks"],
        "student_code_executed": output.get("student_code_executed"),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Independently audit evidence-map JSONs against raw submission/question/requirements files.")
    parser.add_argument("--question-map", required=True)
    parser.add_argument("--requirements", required=True)
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    qmap_path, req_path, source_dir, evidence_dir = map(Path, (args.question_map, args.requirements, args.submission_dir, args.evidence_dir))
    for path in (qmap_path, req_path):
        if not path.is_file():
            parser.error(f"Input file not found: {path}")
    if not source_dir.is_dir() or not evidence_dir.is_dir():
        parser.error("--submission-dir and --evidence-dir must be existing directories")
    qmap = json.loads(qmap_path.read_text(encoding="utf-8"))
    reqs = json.loads(req_path.read_text(encoding="utf-8"))
    files = sorted(source_dir.rglob(args.pattern) if args.recursive else source_dir.glob(args.pattern))
    files = [p for p in files if p.is_file() and not p.name.endswith(".evidence-map.json")]
    results = []
    for source_path in files:
        relative = source_path.relative_to(source_dir)
        evidence_path = evidence_dir / relative.parent / f"{relative.stem}.evidence-map.json"
        if not evidence_path.is_file():
            results.append({"submission": source_path.name, "status": "FAIL", "errors": [f"evidence output missing: {evidence_path}"]})
            continue
        try:
            source = json.loads(source_path.read_text(encoding="utf-8"))
            output = json.loads(evidence_path.read_text(encoding="utf-8"))
            results.append(audit_one(qmap, reqs, source, source_path.name, output))
        except Exception as exc:
            results.append({"submission": source_path.name, "status": "FAIL", "errors": [f"{type(exc).__name__}: {exc}"]})
    if not files:
        parser.error("No submission JSON files matched the pattern")
    passed = sum(r["status"] == "PASS" for r in results)
    report = {
        "audit_version": "1.1",
        "question_map": str(qmap_path),
        "requirements": str(req_path),
        "source_submissions": len(files),
        "passed": passed,
        "failed": len(results) - passed,
        "student_code_executed": False,
        "results": results,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Raw evidence-map audit: {passed}/{len(results)} passed; failed={len(results)-passed}; student code executed=no")
    print(f"Report: {report_path}")
    for record in results:
        if record["status"] != "PASS":
            print(f"FAIL {record['submission']}: {'; '.join(record.get('errors', []))}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
