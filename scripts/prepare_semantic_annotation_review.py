from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.annotation.review_workflow import ReviewError, collect_json, load_index, prepare_one

def main() -> int:
    p = argparse.ArgumentParser(description="Create review TSVs from semantic proposal JSON without approving labels.")
    p.add_argument("--annotation-dir", required=True, type=Path)
    p.add_argument("--proposal-dir", required=True, type=Path)
    p.add_argument("--scope-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--pattern", default="*.semantic-proposal.json")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    try:
        proposal_paths = collect_json(a.proposal_dir, a.pattern) if a.recursive else sorted(a.proposal_dir.glob(a.pattern))
        if not proposal_paths:
            raise ReviewError(f"No proposal files matched {a.pattern!r} under {a.proposal_dir}")
        ann_paths = collect_json(a.annotation_dir, "*.annotation.json") if a.recursive else sorted(a.annotation_dir.glob("*.annotation.json"))
        scope_paths = collect_json(a.scope_dir, "*.scope.json") if a.recursive else sorted(a.scope_dir.glob("*.scope.json"))
        ann_by_doc = load_index(ann_paths, "document_id")
        scope_by_doc = load_index(scope_paths, "document_id")
        count = 0
        for prop_path in proposal_paths:
            import json
            proposal = json.loads(prop_path.read_text(encoding="utf-8"))
            doc_id = proposal.get("document_id")
            if doc_id not in ann_by_doc or doc_id not in scope_by_doc:
                print(f"SKIP {prop_path.name}: canonical annotation or scope ledger not found for {doc_id}")
                continue
            ann_path = ann_by_doc[doc_id]
            out_name = prop_path.name.replace(".semantic-proposal.json", ".semantic-review.tsv")
            result = prepare_one(ann_path, prop_path, scope_by_doc[doc_id], a.output_dir / out_name, a.overwrite)
            print(f"OK {result['document_id']}: proposals={result['proposal_count']}; output={result['output']}")
            count += 1
        print(f"Review sheets prepared: {count}; labels approved: 0")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
if __name__ == "__main__":
    raise SystemExit(main())
