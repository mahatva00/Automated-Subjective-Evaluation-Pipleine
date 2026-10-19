#!/usr/bin/env python3
"""Audit source-anchored annotation coverage sidecars and report layer-level progress."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.annotation.coverage import coverage_summary, validate_coverage
from src.annotation.scope import validate_scope_ledger, validate_annotations_against_scope
from src.annotation.validation import validate_document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--coverage-dir", required=True)
    parser.add_argument("--scope-dir", help="Optional question-only scope sidecars; review is required only for in-scope question content")
    parser.add_argument("--pattern", default="*.annotation.json")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--report", help="Optional JSON report path")
    parser.add_argument("--require-complete", action="store_true", help="Exit non-zero unless both layers are complete for every document")
    args = parser.parse_args()
    annotation_root, coverage_root = Path(args.annotation_dir), Path(args.coverage_dir)
    scope_root = Path(args.scope_dir) if args.scope_dir else None
    if scope_root is not None and not scope_root.is_dir():
        print(f"ERROR: scope directory does not exist: {scope_root}")
        return 2
    if not annotation_root.is_dir():
        print(f"ERROR: annotation directory does not exist: {annotation_root}")
        return 2
    if not coverage_root.is_dir():
        print(f"ERROR: coverage directory does not exist: {coverage_root}")
        return 2
    files = sorted(annotation_root.rglob(args.pattern) if args.recursive else annotation_root.glob(args.pattern))
    if not files:
        print(f"ERROR: no annotation files matched {args.pattern!r}")
        return 2
    results = []
    invalid = 0
    incomplete = 0
    for path in files:
        rel = path.relative_to(annotation_root)
        cov_rel_name = rel.name.removesuffix(".annotation.json") + ".coverage.json"
        cov_rel = (rel.parent / cov_rel_name) if args.recursive else Path(cov_rel_name)
        cov_path = coverage_root / cov_rel
        errors: list[str] = []
        summary = {"document_id": None, "total_chunks": 0, "chunks_requiring_review": 0,
                   "scope_aware_coverage": scope_root is not None, "structure_reviewed_chunks": 0,
                   "structure_coverage_percent": 0.0, "semantic_reviewed_chunks": 0,
                   "semantic_coverage_percent": 0.0, "complete_both_layers": False}
        scope_complete = None
        pending_scope_segments = None
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            errors.extend(validate_document(doc))
            scope = None
            if scope_root is not None:
                scope_rel = (rel.parent / (rel.name.removesuffix(".annotation.json") + ".scope.json")) if args.recursive else Path(rel.name.removesuffix(".annotation.json") + ".scope.json")
                scope_path = scope_root / scope_rel
                if not scope_path.is_file():
                    errors.append(f"question-scope sidecar missing: {scope_path}")
                else:
                    scope = json.loads(scope_path.read_text(encoding="utf-8"))
                    errors.extend(validate_scope_ledger(doc, scope))
                    errors.extend(validate_annotations_against_scope(doc, scope))
                    pending_scope_segments = sum(1 for seg in scope.get("segments", []) if seg.get("scope_status") == "pending")
                    scope_complete = pending_scope_segments == 0
                    if args.require_complete and not scope_complete:
                        errors.append(f"question scope remains incomplete: {pending_scope_segments} pending segments")
            if not cov_path.is_file():
                errors.append(f"coverage sidecar missing: {cov_path}")
            else:
                coverage = json.loads(cov_path.read_text(encoding="utf-8"))
                errors.extend(validate_coverage(doc, coverage, scope=scope))
                summary = coverage_summary(coverage, scope=scope, source_text=doc.get("text", ""))
            status = doc.get("annotation_metadata", {}).get("status")
            if status in {"reviewed", "adjudicated"} and not summary["complete_both_layers"]:
                errors.append(f"document status is {status!r} but scope-aware coverage summary is incomplete")
        except Exception as exc:
            errors.append(f"could not read/validate files: {exc}")
            status = None
        is_invalid = bool(errors)
        is_incomplete = not summary.get("complete_both_layers", False) or (scope_root is not None and not bool(scope_complete))
        invalid += int(is_invalid)
        incomplete += int(is_incomplete)
        state = "FAIL" if is_invalid else ("INCOMPLETE" if is_incomplete else "READY")
        print(f"{state} {path.name}: structure={summary['structure_coverage_percent']:.1f}% "
              f"semantic={summary['semantic_coverage_percent']:.1f}% chunks={summary['total_chunks']} status={status}")
        for error in errors:
            print(f"  - {error}")
        results.append({"annotation_file": str(path), "coverage_file": str(cov_path), "document_status": status,
                        "scope_complete": scope_complete, "pending_scope_segments": pending_scope_segments,
                        **summary, "errors": errors})
    report = {"report_version": "1.1.0", "scope_aware": scope_root is not None, "documents": results,
              "document_count": len(files), "invalid_document_count": invalid,
              "incomplete_document_count": incomplete,
              "all_documents_complete": bool(files) and invalid == 0 and incomplete == 0,
              "training_readiness_note": "Coverage is a human attestation. With --scope-dir, only chunks containing in-scope question text require structure/semantic review; out-of-scope and pending tokens must not be trained as background/O. This audit cannot prove semantic correctness of labels."}
    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Report saved: {out}")
    print(f"\nDocuments={len(files)}; invalid={invalid}; incomplete={incomplete}; fully covered={len(files)-incomplete}")
    if invalid or (args.require_complete and incomplete):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
