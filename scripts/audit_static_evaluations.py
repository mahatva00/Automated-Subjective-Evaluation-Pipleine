from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALID = {"complete", "partial", "absent", "needs_review"}
FORBIDDEN_GRADE_KEYS = {"marks_awarded", "score", "grade", "final_marks"}

def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def collect_source_requirements(evidence_map):
    expected = []
    for question in evidence_map.get("questions", []) or []:
        for req in question.get("requirements", []) or []:
            expected.append(str(req.get("requirement_id") or ""))
    for req in evidence_map.get("global_requirements", []) or []:
        expected.append(str(req.get("requirement_id") or ""))
    return expected

def collect_results(evaluation):
    results = list(evaluation.get("global_evaluations", []) or [])
    for question in evaluation.get("questions", []) or []:
        results.extend(question.get("evaluations", []) or [])
    return results

def audit(evidence_map, evaluation):
    errors = []
    if evidence_map.get("student_code_executed") is True or evaluation.get("student_code_executed") is not False:
        errors.append("student_code_executed must be false in both source and evaluation")
    if evaluation.get("grading_policy_applied") is not False:
        errors.append("grading_policy_applied must be false")
    source_ids = collect_source_requirements(evidence_map)
    result_rows = collect_results(evaluation)
    result_ids = [str(x.get("requirement_id") or "") for x in result_rows]
    if len(source_ids) != len(set(source_ids)):
        errors.append("input evidence map contains duplicate requirement IDs")
    if Counter(source_ids) != Counter(result_ids):
        missing = sorted(set(source_ids) - set(result_ids))
        extra = sorted(set(result_ids) - set(source_ids))
        errors.append(f"requirement ID mismatch; missing={missing[:10]}, extra={extra[:10]}")
    blocks = {}
    for q in evidence_map.get("questions", []) or []:
        for b in q.get("evidence_blocks", []) or []:
            blocks[str(b.get("block_id") or "")] = b
    for result in result_rows:
        rid = result.get("requirement_id", "<missing>")
        if result.get("status") not in VALID:
            errors.append(f"{rid}: invalid status")
        try:
            conf = float(result.get("confidence"))
            if not (0 <= conf <= 1):
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"{rid}: invalid confidence")
        for ev in result.get("evidence", []) or []:
            bid, quote = str(ev.get("block_id") or ""), str(ev.get("quote") or "")
            if bid not in blocks:
                errors.append(f"{rid}: evidence cites unknown block {bid}")
                continue
            b = blocks[bid]
            if not quote or (quote not in str(b.get("source_text") or "") and quote not in str(b.get("output_text") or "")):
                errors.append(f"{rid}: quotation not found verbatim in block {bid}")
        if result.get("status") in {"complete", "partial"} and not result.get("evidence"):
            errors.append(f"{rid}: {result.get('status')} requires evidence citations")
        if result.get("status") == "absent" and (result.get("context_complete") is not True or result.get("evidence_sufficient") is not True):
            errors.append(f"{rid}: absent cannot be asserted without sufficient complete context")
        def walk(obj, path="$"):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key.lower() in FORBIDDEN_GRADE_KEYS:
                        errors.append(f"{rid}: forbidden grading field {path}.{key}")
                    walk(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for index, value in enumerate(obj):
                    walk(value, f"{path}[{index}]")
        walk(result)
    counts = Counter(str(x.get("status")) for x in result_rows)
    return {
        "audit_version": "1.0",
        "passed": not errors,
        "student_code_executed": False,
        "input_requirement_count": len(source_ids),
        "output_assessment_count": len(result_ids),
        "unique_input_requirement_ids": len(set(source_ids)),
        "unique_output_requirement_ids": len(set(result_ids)),
        "status_counts": {key: counts.get(key, 0) for key in sorted(VALID)},
        "errors": errors,
    }

def main():
    p = argparse.ArgumentParser(description="Independently audit static answer-evaluation JSON against its evidence map.")
    p.add_argument("--evidence-map", required=True)
    p.add_argument("--evaluation", required=True)
    p.add_argument("--report", required=True)
    a = p.parse_args()
    source_path, output_path, report_path = Path(a.evidence_map), Path(a.evaluation), Path(a.report)
    result = audit(load(source_path), load(output_path))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Static evaluation raw audit: {'PASS' if result['passed'] else 'FAIL'}")
    print(f"Input requirements: {result['input_requirement_count']}; output assessments: {result['output_assessment_count']}")
    print("Statuses: " + json.dumps(result["status_counts"]))
    print(f"Report: {report_path}")
    if result["errors"]:
        for error in result["errors"]:
            print(f"ERROR: {error}")
        return 1
    print("Student code executed: no")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
