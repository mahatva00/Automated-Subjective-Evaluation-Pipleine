from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.annotation.semantic_proposals import propose_semantic_annotations, write_tsv


def main() -> int:
    ap = argparse.ArgumentParser(description="Create pending semantic span proposals from reviewed question scope and a structure proposal.")
    ap.add_argument("--annotation-dir", required=True, help="Directory containing canonical *.annotation.json drafts")
    ap.add_argument("--scope-dir", required=True, help="Directory containing reviewed *.scope.json sidecars")
    ap.add_argument("--structure-dir", required=True, help="Directory containing matching *.structure-proposal.json v2 outputs")
    ap.add_argument("--output-dir", required=True, help="New output directory; canonical inputs are never modified")
    ap.add_argument("--pattern", default="*.annotation.json")
    ap.add_argument("--overwrite", action="store_true", help="Allow replacing only matching semantic proposal outputs")
    args = ap.parse_args()
    ann_dir, scope_dir, struct_dir, out_dir = map(Path, (args.annotation_dir, args.scope_dir, args.structure_dir, args.output_dir))
    for label, path in [("annotation", ann_dir), ("scope", scope_dir), ("structure", struct_dir)]:
        if not path.is_dir(): ap.error(f"{label} directory does not exist: {path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    drafts = sorted(ann_dir.glob(args.pattern))
    if not drafts: ap.error(f"No annotation drafts matched {args.pattern!r} in {ann_dir}")
    failures = 0
    for draft_path in drafts:
        stem = draft_path.name.removesuffix(".annotation.json")
        scope_path = scope_dir / f"{stem}.scope.json"
        structure_path = struct_dir / f"{stem}.structure-proposal.json"
        out_json = out_dir / f"{stem}.semantic-proposal.json"
        out_tsv = out_dir / f"{stem}.semantic-proposal.tsv"
        missing = [p for p in (scope_path, structure_path) if not p.is_file()]
        if missing:
            print(f"SKIP {draft_path.name}: missing input(s): {', '.join(str(p) for p in missing)}")
            failures += 1
            continue
        if not args.overwrite and (out_json.exists() or out_tsv.exists()):
            print(f"SKIP {draft_path.name}: output exists; use --overwrite to replace these proposal outputs only")
            failures += 1
            continue
        try:
            annotation = json.loads(draft_path.read_text(encoding="utf-8"))
            scope = json.loads(scope_path.read_text(encoding="utf-8"))
            structure = json.loads(structure_path.read_text(encoding="utf-8"))
            proposal = propose_semantic_annotations(annotation, scope, structure)
            out_json.write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            write_tsv(proposal, str(out_tsv))
            counts = proposal["label_proposal_counts"]
            print(f"PASS {draft_path.name}: semantic_candidates={proposal['candidate_count']}; labels={counts}; "
                  f"ignored_unassigned={proposal['scope_summary']['unassigned_in_scope_regions_omitted']}; "
                  f"questions={len(proposal['question_reconstructions'])}; output={out_json}")
        except Exception as exc:
            print(f"ERROR {draft_path.name}: {type(exc).__name__}: {exc}")
            failures += 1
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
