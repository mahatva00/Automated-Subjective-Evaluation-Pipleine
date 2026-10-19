from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
for import_root in (PROJECT_ROOT, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from src.evaluation.evidence_mapper import build_from_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Map statically parsed student segments to questions in a question map.")
    parser.add_argument("--question-map", required=True, help="Path to a *.question-map.json file")
    parser.add_argument("--requirements", required=True, help="Path to a *.requirements.json file")
    parser.add_argument("--submission-dir", required=True, help="Directory containing parsed submission JSONs; student code is never executed")
    parser.add_argument("--output-dir", required=True, help="Separate output directory for evidence maps")
    parser.add_argument("--pattern", default="*.json", help="Submission JSON filename pattern")
    parser.add_argument("--recursive", action="store_true", help="Search submission directory recursively")
    args = parser.parse_args()

    qmap = Path(args.question_map)
    reqs = Path(args.requirements)
    source_dir = Path(args.submission_dir)
    out_dir = Path(args.output_dir)
    if not qmap.is_file() or not reqs.is_file():
        parser.error("--question-map and --requirements must point to existing files")
    if not source_dir.is_dir():
        parser.error("--submission-dir must be an existing directory")
    files = sorted(source_dir.rglob(args.pattern) if args.recursive else source_dir.glob(args.pattern))
    files = [f for f in files if f.is_file() and not f.name.endswith(".evidence-map.json")]
    if not files:
        print("No submission JSON files matched the pattern.", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = failed = 0
    for path in files:
        try:
            result = build_from_paths(qmap, reqs, path)
            relative = path.relative_to(source_dir)
            dest = out_dir / relative.parent / f"{relative.stem}.evidence-map.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            s = result["summary"]
            print(f"OK {path.name}: {s['mapped_blocks']}/{s['evidence_blocks']} blocks mapped; {s['unassigned_blocks']} unassigned; code executed=no -> {dest}")
            ok += 1
        except Exception as exc:
            print(f"FAILED {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed += 1
    print(f"\nProcessed submissions: {ok}; failed: {failed}; student code executed: no")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
