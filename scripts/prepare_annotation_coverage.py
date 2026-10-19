#!/usr/bin/env python3
"""Create source-hashed coverage sidecars for annotation drafts without labeling them."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.annotation.coverage import build_coverage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory of canonical *.annotation.json files")
    parser.add_argument("--output-dir", required=True, help="Directory for *.coverage.json sidecars")
    parser.add_argument("--pattern", default="*.annotation.json")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-chars", type=int, default=900, help="Approximate maximum source chars per review chunk")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing coverage sidecars")
    args = parser.parse_args()
    source_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    if not source_root.is_dir():
        print(f"ERROR: input directory does not exist: {source_root}")
        return 2
    files = sorted(source_root.rglob(args.pattern) if args.recursive else source_root.glob(args.pattern))
    if not files:
        print(f"ERROR: no files matched {args.pattern!r} in {source_root}")
        return 2
    created = skipped = failed = total_chunks = 0
    for path in files:
        relative = path.relative_to(source_root)
        relative_output = relative.name.removesuffix(".annotation.json") + ".coverage.json"
        if args.recursive and len(relative.parts) > 1:
            relative_output = str(relative.parent / relative_output)
        out = output_root / relative_output
        if out.exists() and not args.overwrite:
            print(f"SKIP {path}: {out} already exists (use --overwrite only to regenerate an unreviewed ledger)")
            skipped += 1
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            coverage = build_coverage(doc, path.name, args.max_chars)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            created += 1
            total_chunks += len(coverage["chunks"])
            print(f"CREATED {out}: {len(coverage['chunks'])} chunks, {len(doc['text'])} source characters; all statuses pending")
        except Exception as exc:
            failed += 1
            print(f"FAIL {path}: {exc}")
    print(f"\nFiles: {len(files)}; created={created}; skipped={skipped}; failed={failed}; chunks created={total_chunks}")
    print("No labels were generated and no annotation source JSON was modified.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
