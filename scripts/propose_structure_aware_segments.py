from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.annotation.structure_segmentation import propose_structure, sha256_text


def main() -> int:
    ap = argparse.ArgumentParser(description="Create pending, scope-aware structure-segmentation proposals for a question paper.")
    ap.add_argument("--annotation-dir", required=True, help="Directory containing canonical *.annotation.json drafts")
    ap.add_argument("--scope-dir", required=True, help="Directory containing reviewed *.scope.json sidecars")
    ap.add_argument("--output-dir", required=True, help="New output directory; canonical drafts are never modified")
    ap.add_argument("--pattern", default="*.annotation.json")
    ap.add_argument("--overwrite", action="store_true", help="Allow replacing only this proposal output")
    args = ap.parse_args()
    ann_dir, scope_dir, out_dir = Path(args.annotation_dir), Path(args.scope_dir), Path(args.output_dir)
    if not ann_dir.is_dir(): ap.error(f"Annotation directory does not exist: {ann_dir}")
    if not scope_dir.is_dir(): ap.error(f"Scope directory does not exist: {scope_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(ann_dir.glob(args.pattern))
    if not files:
        ap.error(f"No annotation drafts matched {args.pattern!r} in {ann_dir}")
    failed = 0
    for ann_path in files:
        stem = ann_path.name.removesuffix(".annotation.json")
        scope_path = scope_dir / f"{stem}.scope.json"
        if not scope_path.exists():
            print(f"SKIP {ann_path.name}: no scope ledger at {scope_path}")
            failed += 1
            continue
        json_path = out_dir / f"{stem}.structure-proposal.json"
        tsv_path = out_dir / f"{stem}.structure-proposal.tsv"
        if not args.overwrite and (json_path.exists() or tsv_path.exists()):
            print(f"SKIP {ann_path.name}: proposal output already exists; pass --overwrite to replace proposal output only")
            failed += 1
            continue
        try:
            annotation = json.loads(ann_path.read_text(encoding="utf-8"))
            scope = json.loads(scope_path.read_text(encoding="utf-8"))
            proposal = propose_structure(annotation, scope)
            json_path.write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            with tsv_path.open("w", encoding="utf-8-sig", newline="") as f:
                fields = ["document_id", "candidate_id", "candidate_type", "span_role", "parent_candidate_id", "paired_candidate_id", "question_id", "scope_segment_id", "start_char", "end_char", "body_start_char", "body_end_char", "header_start_char", "header_end_char", "continuation_of_candidate_id", "continuation_group_id", "continuation_sequence", "linked_to_previous_scope_fragment", "text_sha256", "proposal_confidence", "review_status", "annotation_label", "training_label", "text", "proposal_rationale"]
                w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
                w.writeheader()
                for c in proposal["candidates"]:
                    row = {k: c.get(k, "") for k in fields}
                    row["document_id"] = proposal["document_id"]
                    row["annotation_label"] = ""
                    row["training_label"] = ""
                    w.writerow(row)
            print(f"PASS {ann_path.name}: {len(proposal['candidates'])} pending candidates; questions={len(proposal['question_nodes'])}; candidate_types={len(proposal['candidate_type_counts'])}; excluded out-of-scope={proposal['scope_summary']['out_of_scope_segments_excluded']}; output={json_path}")
        except Exception as e:
            print(f"ERROR {ann_path.name}: {type(e).__name__}: {e}")
            failed += 1
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())
