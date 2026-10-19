#!/usr/bin/env python3
"""Build a partial-label span-classification dataset from approved pilot annotations.

This exports only explicitly reviewed semantic spans. Unannotated document text is
never converted into negative examples. The output is for an MVP role classifier,
not a complete document-level token-classification dataset.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT
# When merged into the repository, validation lives at src/annotation/validation.py.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.scope import validate_scope_ledger, classify_span_scope

SEMANTIC_ROLE_MAP = {
    "requirement": "requirement",
    "submission_instruction": "instruction_or_assessment_constraint",
    "grading_instruction": "instruction_or_assessment_constraint",
    "administrative_instruction": "instruction_or_assessment_constraint",
    "option": "option_or_alternative",
    "alternative_marker": "option_or_alternative",
    "context": "context_or_example",
    "definition": "context_or_example",
    "example": "context_or_example",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_rows(input_dir: Path, manifest_path: Path, scope_dir: Path, pattern: str = "*.annotation.json") -> tuple[list[dict], dict, dict]:
    manifest = load_json(manifest_path)
    if manifest.get("manifest_version") != "1.0.0" or not isinstance(manifest.get("documents"), list):
        raise ValueError("manifest must use version 1.0.0 and contain a documents array")
    assignments = {}
    groups_by_split = defaultdict(set)
    for entry in manifest["documents"]:
        doc_id = entry.get("document_id")
        split = entry.get("split")
        if not doc_id or split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid manifest entry: {entry!r}")
        if doc_id in assignments:
            raise ValueError(f"duplicate document_id in manifest: {doc_id}")
        group = entry.get("group_id")
        if not group:
            raise ValueError(f"manifest group_id is required for {doc_id}")
        previous = [s for s, groups in groups_by_split.items() if group in groups and s != split]
        if previous:
            raise ValueError(f"group leakage for {group}: {previous[0]} and {split}")
        groups_by_split[split].add(group)
        assignments[doc_id] = entry

    paths = sorted(input_dir.glob(pattern))
    if not paths:
        raise ValueError(f"no annotation files matched {pattern!r} in {input_dir}")
    loaded_ids = set()
    rows = []
    validation_errors = []
    scope_counts = Counter()
    try:
        from src.annotation.validation import validate_document
    except ImportError:
        validate_document = None

    for path in paths:
        doc = load_json(path)
        doc_id = doc.get("document_id")
        if not doc_id or doc_id in loaded_ids:
            raise ValueError(f"missing or duplicate document_id in {path}")
        loaded_ids.add(doc_id)
        entry = assignments.get(doc_id)
        if entry is None:
            raise ValueError(f"document {doc_id} is missing from manifest")
        if validate_document is not None:
            errs = validate_document(doc)
            if errs:
                validation_errors.append((path.name, errs))
                continue
        scope_path = scope_dir / (path.name.removesuffix(".annotation.json") + ".scope.json")
        if not scope_path.is_file():
            raise ValueError(f"scope sidecar missing for {path.name}: {scope_path}; generate and review question-only scope first")
        scope = load_json(scope_path)
        scope_errors = validate_scope_ledger(doc, scope)
        if scope_errors:
            raise ValueError(f"invalid scope sidecar for {path.name}: " + "; ".join(scope_errors))
        # Relation proposals are intentionally not exported or learned in this MVP.
        for ann in doc.get("annotations", []):
            if ann.get("layer") != "semantic":
                continue
            start, end = ann.get("start_char"), ann.get("end_char")
            text = ann.get("text")
            if not isinstance(start, int) or not isinstance(end, int) or doc["text"][start:end] != text:
                raise ValueError(f"source span mismatch for {ann.get('annotation_id')} in {path.name}")
            # Scope first: labels on out-of-scope or unresolved text have no meaning in this task.
            # They are skipped rather than reclassified as background or another semantic class.
            scope_status, question_id = classify_span_scope(start, end, scope, doc["text"])
            if scope_status != "in_scope":
                scope_counts[f"excluded_{scope_status}"] += 1
                continue
            if ann.get("review_status") not in {"reviewed", "adjudicated"}:
                raise ValueError(f"unreviewed in-scope semantic annotation {ann.get('annotation_id')} in {path.name}")
            raw_label = ann.get("label")
            role = SEMANTIC_ROLE_MAP.get(raw_label)
            if role is None:
                raise ValueError(f"no subject-independent MVP role mapping for in-scope semantic label {raw_label!r}")
            scope_counts["included_in_scope"] += 1
            rows.append({
                "document_id": doc_id,
                "source_filename": doc.get("source", {}).get("filename", path.name),
                "annotation_id": ann["annotation_id"],
                "split": entry["split"],
                "group_id": entry["group_id"],
                "layer": "semantic",
                "source_label": raw_label,
                "label": role,
                "text": text,
                "start_char": start,
                "end_char": end,
                "question_id": question_id,
                "scope_status": "in_scope",
                "scope_policy_version": "question_scope_policy_v1.2",
                "coverage_note": "reviewed_selected_span_inside_reviewed_question_scope; surrounding unannotated text is unknown, not a negative example",
            })
    if validation_errors:
        msg = "\n".join(f"{name}: {'; '.join(errs)}" for name, errs in validation_errors)
        raise ValueError("annotation validation failed:\n" + msg)
    missing = set(assignments) - loaded_ids
    if missing:
        raise ValueError("manifest documents not found in input directory: " + ", ".join(sorted(missing)))
    if not any(r["split"] == "train" for r in rows):
        raise ValueError("no semantic training examples assigned to train split")
    if not any(r["split"] == "validation" for r in rows):
        raise ValueError("no semantic validation examples assigned to validation split")
    return rows, manifest, dict(scope_counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/annotations/approved_pilot")
    parser.add_argument("--manifest", default="data/annotations/approved_pilot/span_classifier_manifest.json")
    parser.add_argument("--pattern", default="*.annotation.json")
    parser.add_argument("--scope-dir", default="data/annotations/scope_reviews", help="Reviewed question-only scope sidecars; each is <stem>.scope.json")
    parser.add_argument("--output", default="data/annotations/distilbert_mvp/semantic_span_dataset_scope_v1.jsonl")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing the selected output and its summary")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    manifest_path = Path(args.manifest)
    if not input_dir.is_dir() or not manifest_path.is_file():
        print("ERROR: annotation input directory or manifest does not exist")
        return 2
    scope_dir = Path(args.scope_dir)
    if not scope_dir.is_dir():
        print(f"ERROR: scope directory does not exist: {scope_dir}; prepare and review scope sidecars first")
        return 2
    try:
        rows, manifest, scope_counts = build_rows(input_dir, manifest_path, scope_dir, args.pattern)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2
    out = Path(args.output)
    summary_path = out.with_name(out.stem + "_summary.json")
    if out.exists() and not args.overwrite:
        print(f"ERROR: output already exists: {out}; choose a new path or pass --overwrite")
        return 2
    if summary_path.exists() and not args.overwrite:
        print(f"ERROR: summary already exists: {summary_path}; choose a new output path or pass --overwrite")
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    counts = Counter(row["label"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    summary = {
        "dataset_version": "mvp_semantic_span_v1",
        "task": "coarse_semantic_role_classification",
        "num_examples": len(rows),
        "by_label": dict(sorted(counts.items())),
        "by_split": dict(sorted(split_counts.items())),
        "document_count": len(manifest["documents"]),
        "uses_unannotated_as_negative": False,
        "scope_policy_version": "question_scope_policy_v1.2",
        "scope_counts": scope_counts,
        "excludes_unresolved_and_out_of_scope_spans": True,
        "relations_used": False,
        "limitations": [
            "Partial-coverage pilot; only selected reviewed spans are examples.",
            "Three Machine Learning papers only; this does not demonstrate cross-domain generalization.",
            "Classification applies to candidate spans; it does not discover document boundaries on its own."
        ]
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(rows)} reviewed semantic span examples -> {out}")
    print("Label counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("Split counts: " + ", ".join(f"{k}={v}" for k, v in sorted(split_counts.items())))
    print(f"Summary: {summary_path}")
    print("Out-of-scope or unresolved annotations exported: no")
    print("Unannotated text converted to negatives: no")
    print("Relation training: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
