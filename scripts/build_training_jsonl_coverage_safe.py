#!/usr/bin/env python3
"""Export only reviewed, fully coverage-audited documents to training JSONL."""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.annotation.validation import validate_document
from src.annotation.coverage import validate_coverage
from src.annotation.scope import validate_scope_ledger, coverage_chunk_requires_review, filter_document_annotations_to_scope

APPROVED = {"reviewed", "adjudicated"}
SPLITS = {"train", "validation", "test"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Root containing canonical annotation JSON files")
    parser.add_argument("--coverage-dir", required=True, help="Root containing matching coverage sidecars")
    parser.add_argument("--scope-dir", required=True, help="Root containing reviewed question-only scope sidecars")
    parser.add_argument("--manifest", required=True, help="Dataset split/group manifest JSON")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--pattern", default="*.annotation.json")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output JSONL")
    args = parser.parse_args()
    root, cov_root, scope_root = Path(args.input_dir), Path(args.coverage_dir), Path(args.scope_dir)
    if not root.is_dir() or not cov_root.is_dir() or not scope_root.is_dir():
        print("ERROR: input-dir, coverage-dir, and scope-dir must all exist")
        return 2
    try:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"ERROR: cannot read manifest: {exc}")
        return 2
    if manifest.get("manifest_version") != "1.0.0" or not isinstance(manifest.get("documents"), list):
        print("ERROR: manifest must have manifest_version='1.0.0' and a documents array")
        return 2

    entries: dict[str, dict] = {}
    for i, entry in enumerate(manifest["documents"]):
        if not isinstance(entry, dict):
            print(f"ERROR: manifest documents[{i}] must be an object")
            return 2
        doc_id = entry.get("document_id")
        if not isinstance(doc_id, str) or not doc_id:
            print(f"ERROR: manifest documents[{i}] has no document_id")
            return 2
        if doc_id in entries:
            print(f"ERROR: duplicate manifest document_id: {doc_id}")
            return 2
        if entry.get("split") not in SPLITS | {"unassigned"}:
            print(f"ERROR: invalid split for {doc_id}: {entry.get('split')!r}")
            return 2
        if not isinstance(entry.get("group_id"), str) or not entry["group_id"].strip():
            print(f"ERROR: group_id is required for {doc_id}")
            return 2
        entries[doc_id] = entry

    files = sorted(root.rglob(args.pattern) if args.recursive else root.glob(args.pattern))
    docs: dict[str, tuple[Path, dict]] = {}
    for path in files:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"ERROR: cannot parse {path}: {exc}")
            return 2
        doc_id = doc.get("document_id") if isinstance(doc, dict) else None
        if not doc_id:
            print(f"ERROR: {path} has no document_id")
            return 2
        if doc_id in docs:
            print(f"ERROR: duplicate document_id in annotation files: {doc_id}")
            return 2
        docs[doc_id] = (path, doc)

    exported: list[dict] = []
    group_splits: dict[str, str] = {}
    for doc_id, entry in entries.items():
        split = entry["split"]
        if split == "unassigned":
            continue
        pair = docs.get(doc_id)
        if pair is None:
            print(f"ERROR: assigned document {doc_id!r} has no annotation file in {root}")
            return 2
        actual_path, doc = pair
        annotation_path = entry.get("annotation_path")
        if annotation_path and (root / annotation_path).resolve() != actual_path.resolve():
            print(f"ERROR: manifest path for {doc_id} does not match found file {actual_path}")
            return 2
        errors = validate_document(doc)
        if errors:
            print(f"ERROR: invalid annotation document {actual_path}:")
            for error in errors:
                print(f"  - {error}")
            return 2
        status = doc["annotation_metadata"]["status"]
        if status not in APPROVED:
            print(f"ERROR: refusing {doc_id}: status={status!r}; expected reviewed/adjudicated")
            return 2

        relative = actual_path.relative_to(root)
        scope_path = scope_root / relative.parent / (relative.name.removesuffix(".annotation.json") + ".scope.json")
        if not scope_path.is_file():
            print(f"ERROR: refusing {doc_id}: question-scope sidecar not found: {scope_path}")
            return 2
        try:
            scope = json.loads(scope_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"ERROR: cannot parse scope sidecar {scope_path}: {exc}")
            return 2
        scope_errors = validate_scope_ledger(doc, scope)
        pending = [s.get("segment_id") for s in scope.get("segments", []) if s.get("scope_status") == "pending"]
        if pending:
            scope_errors.append(f"scope review is incomplete: {len(pending)} pending segment(s); unresolved text cannot contribute training labels")
        if scope_errors:
            print(f"ERROR: refusing {doc_id}: invalid, incomplete, or unsafe question scope:")
            for error in scope_errors:
                print(f"  - {error}")
            return 2

        cov_name = relative.name.removesuffix(".annotation.json") + ".coverage.json"
        cov_rel = (relative.parent / cov_name) if args.recursive else Path(cov_name)
        cov_path = cov_root / cov_rel
        if not cov_path.is_file():
            print(f"ERROR: refusing {doc_id}: coverage sidecar not found: {cov_path}")
            return 2
        try:
            coverage = json.loads(cov_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"ERROR: cannot parse {cov_path}: {exc}")
            return 2
        errors = validate_coverage(doc, coverage, scope=scope)
        if errors:
            print(f"ERROR: refusing {doc_id}: invalid or incomplete coverage:")
            for error in errors:
                print(f"  - {error}")
            return 2
        required_chunks = [c for c in coverage.get("chunks", []) if coverage_chunk_requires_review(c, scope, doc["text"])]
        if not required_chunks or any(
            c.get("structure_status") != "reviewed" or c.get("semantic_status") != "reviewed"
            for c in required_chunks
        ):
            print(f"ERROR: refusing {doc_id}: coverage of in-scope question text is incomplete")
            return 2

        group_id = entry["group_id"]
        previous_split = group_splits.get(group_id)
        if previous_split is not None and previous_split != split:
            print(f"ERROR: group leakage: group_id {group_id!r} occurs in both {previous_split} and {split}")
            return 2
        group_splits[group_id] = split
        row, filter_audit = filter_document_annotations_to_scope(doc, scope)
        # The source text stays byte-for-byte intact for offsets/auditability; only the
        # annotation/relation lists in this downstream copy are scope-filtered.
        unreviewed_annotations = [a.get("annotation_id") for a in row.get("annotations", [])
                                  if a.get("review_status") not in {"reviewed", "adjudicated"}]
        if unreviewed_annotations:
            print(f"ERROR: refusing {doc_id}: in-scope annotations are not reviewed/adjudicated: {unreviewed_annotations}")
            return 2
        unreviewed_relations = [r.get("relation_id") for r in row.get("relations", [])
                                if r.get("review_status") not in {"reviewed", "adjudicated"}]
        if unreviewed_relations:
            print(f"ERROR: refusing {doc_id}: in-scope relations are not reviewed/adjudicated: {unreviewed_relations}")
            return 2
        errors = validate_document(row)
        if errors:
            print(f"ERROR: scope-filtered document {doc_id} is invalid:")
            for error in errors:
                print(f"  - {error}")
            return 2
        row["dataset_assignment"] = {"split": split, "group_id": group_id,
                                      "subject": entry.get("subject"), "course": entry.get("course")}
        row["scope_review"] = scope
        row["scope_filter_audit"] = filter_audit
        row["scope_loss_contract"] = "out_of_scope_pending_and_separator_tokens_must_use_ignore_index"
        exported.append(row)
    if not exported:
        print("ERROR: no reviewed/adjudicated documents with complete coverage were assigned to train/validation/test")
        return 2
    split_counts = defaultdict(int)
    for row in exported:
        split_counts[row["dataset_assignment"]["split"]] += 1
    out = Path(args.output)
    if out.exists() and not args.overwrite:
        print(f"ERROR: output already exists: {out}; choose a new path or pass --overwrite")
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in exported), encoding="utf-8")
    print(f"Exported {len(exported)} fully reviewed documents to {out}")
    print("Split counts: " + ", ".join(f"{s}={split_counts[s]}" for s in ("train", "validation", "test")))
    print("Only reviewed in-scope text can contribute background/O labels; out-of-scope/pending tokens must be ignored; student code execution: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
