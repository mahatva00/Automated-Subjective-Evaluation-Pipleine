#!/usr/bin/env python3
"""Audit the approved pilot JSONs against the approval CSV and source offsets."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.annotation.validation import validate_document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/annotations/approved_pilot")
    parser.add_argument("--report", default="data/annotations/approved_pilot/raw_approval_audit.json")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"ERROR: directory not found: {input_dir}")
        return 2
    paths = sorted(input_dir.glob("*.annotation.json"))
    if not paths:
        print("ERROR: no annotation JSON files found")
        return 2
    errors = []
    docs = {}
    annotations_by_key = {}
    annotation_count = 0
    semantic_count = 0
    relation_count = 0
    for path in paths:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{path.name}: invalid JSON: {exc}")
            continue
        doc_id = doc.get("document_id")
        if not doc_id or doc_id in docs:
            errors.append(f"{path.name}: missing/duplicate document_id {doc_id!r}")
            continue
        docs[doc_id] = doc
        validation_errors = validate_document(doc)
        errors.extend(f"{path.name}: {e}" for e in validation_errors)
        if doc.get("annotation_metadata", {}).get("status") != "draft":
            errors.append(f"{path.name}: partial-coverage document must remain draft")
        if any(a.get("review_status") != "reviewed" for a in doc.get("annotations", [])):
            errors.append(f"{path.name}: expected every selected pilot span to be reviewed")
        if any(r.get("review_status", "unreviewed") != "unreviewed" for r in doc.get("relations", [])):
            errors.append(f"{path.name}: pilot relation status changed unexpectedly")
        for ann in doc.get("annotations", []):
            key = (doc_id, ann.get("annotation_id"))
            if key in annotations_by_key:
                errors.append(f"duplicate annotation identity: {key}")
            annotations_by_key[key] = ann
            annotation_count += 1
            semantic_count += int(ann.get("layer") == "semantic")
            start, end = ann.get("start_char"), ann.get("end_char")
            if not isinstance(start, int) or not isinstance(end, int) or doc["text"][start:end] != ann.get("text"):
                errors.append(f"{path.name}: raw source offset mismatch for {ann.get('annotation_id')}")
        relation_count += len(doc.get("relations", []))

    csv_path = input_dir / "pilot_review.csv"
    csv_rows = []
    if not csv_path.is_file():
        errors.append("pilot_review.csv is missing")
    else:
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                csv_rows = list(csv.DictReader(f))
        except Exception as exc:
            errors.append(f"pilot_review.csv cannot be read: {exc}")
    csv_keys = {(r.get("document_id"), r.get("annotation_id")) for r in csv_rows}
    if len(csv_rows) != 89:
        errors.append(f"expected 89 approval rows, got {len(csv_rows)}")
    if any(r.get("review_decision") != "approve" for r in csv_rows):
        errors.append("one or more CSV rows do not say approve")
    if csv_keys != set(annotations_by_key):
        errors.append("CSV annotation identities do not exactly match annotation JSON records")
    if len(csv_keys) != len(csv_rows):
        errors.append("duplicate approval-row identity found in CSV")

    label_counts = Counter()
    split_counts = Counter()
    for doc in docs.values():
        split_counts[doc.get("source", {}).get("filename", "unknown")] += 1
        for ann in doc.get("annotations", []):
            if ann.get("layer") == "semantic":
                label_counts[ann.get("label", "unknown")] += 1
    result = {
        "audit_version": "1.0.0",
        "documents": len(docs),
        "annotation_rows": annotation_count,
        "semantic_span_rows_for_mvp": semantic_count,
        "relation_proposals_not_used": relation_count,
        "approval_csv_rows": len(csv_rows),
        "all_csv_decisions_approve": all(r.get("review_decision") == "approve" for r in csv_rows),
        "annotation_ids_match_csv": csv_keys == set(annotations_by_key),
        "source_offsets_match": not any("raw source offset mismatch" in e for e in errors),
        "all_documents_stay_draft": all(d.get("annotation_metadata", {}).get("status") == "draft" for d in docs.values()),
        "semantic_source_label_counts": dict(sorted(label_counts.items())),
        "passed": not errors,
        "errors": errors,
        "limitations": [
            "Approves the selected 89 pilot spans only; documents are not exhaustively annotated.",
            "Relationship proposals remain unreviewed and are not used for MVP model training.",
            "Unannotated text is unknown, not a negative class."
        ]
    }
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Documents: {result['documents']}; approved spans: {annotation_count}; semantic spans: {semantic_count}; relations not used: {relation_count}")
    print(f"Approval/raw offset audit: {'PASS' if result['passed'] else 'FAIL'}")
    for error in errors:
        print(f"ERROR: {error}")
    print(f"Report: {report}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
