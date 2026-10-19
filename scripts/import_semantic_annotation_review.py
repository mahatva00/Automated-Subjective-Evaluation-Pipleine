from __future__ import annotations
import argparse
import json
import sys
from datetime import date
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.annotation.review_workflow import ReviewError, collect_json, load_index, import_one

def main() -> int:
    p = argparse.ArgumentParser(description="Import explicit human semantic review decisions into a separate canonical annotation copy.")
    p.add_argument("--annotation-dir", required=True, type=Path)
    p.add_argument("--proposal-dir", required=True, type=Path)
    p.add_argument("--scope-dir", required=True, type=Path)
    p.add_argument("--review-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--pattern", default="*.semantic-review.tsv")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--reviewer", required=True)
    p.add_argument("--review-date", required=True)
    p.add_argument("--confirm-reviewed-decisions", action="store_true", help="Required human attestation that non-pending TSV decisions were made by a reviewer")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if not a.confirm_reviewed_decisions:
        print("ERROR: pass --confirm-reviewed-decisions after reviewing the TSV; no files written", file=sys.stderr)
        return 2
    try:
        date.fromisoformat(a.review_date)
        pattern = a.pattern
        review_paths = sorted(a.review_dir.rglob(pattern) if a.recursive else a.review_dir.glob(pattern))
        if not review_paths:
            raise ReviewError(f"No review TSVs matched {pattern!r} under {a.review_dir}")
        ann_paths = collect_json(a.annotation_dir, "*.annotation.json") if a.recursive else sorted(a.annotation_dir.glob("*.annotation.json"))
        prop_paths = collect_json(a.proposal_dir, "*.semantic-proposal.json") if a.recursive else sorted(a.proposal_dir.glob("*.semantic-proposal.json"))
        scope_paths = collect_json(a.scope_dir, "*.scope.json") if a.recursive else sorted(a.scope_dir.glob("*.scope.json"))
        ann_by_doc = load_index(ann_paths, "document_id")
        prop_by_doc = load_index(prop_paths, "document_id")
        scope_by_doc = load_index(scope_paths, "document_id")
        count = 0
        for review_path in review_paths:
            # The review file carries document_id; read the first data row without interpreting decisions.
            import csv
            with review_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                if not reader.fieldnames or "document_id" not in reader.fieldnames:
                    raise ReviewError(f"Review TSV lacks document_id: {review_path}")
                first = next(reader, None)
            if first is None:
                raise ReviewError(f"Review TSV contains no candidate rows: {review_path}")
            doc_id = first.get("document_id", "")
            if doc_id not in ann_by_doc or doc_id not in prop_by_doc or doc_id not in scope_by_doc:
                raise ReviewError(f"Canonical annotation, proposal, or scope ledger missing for {doc_id} ({review_path})")
            ann_path, prop_path, scope_path = ann_by_doc[doc_id], prop_by_doc[doc_id], scope_by_doc[doc_id]
            out_path = a.output_dir / ann_path.name
            result = import_one(ann_path, prop_path, scope_path, review_path, out_path, a.reviewer, a.review_date, a.overwrite)
            print(f"OK {result['document_id']}: annotations_created={result['annotations_created']}; decisions={result['decisions']}; document_status=draft")
            print(f"   canonical={result['output']}")
            print(f"   audit={result['audit']}")
            count += 1
        print(f"Review sheets imported: {count}; no document auto-finalized")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
if __name__ == "__main__":
    raise SystemExit(main())
