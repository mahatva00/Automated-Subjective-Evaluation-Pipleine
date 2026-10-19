#!/usr/bin/env python3
"""Validate document-level annotation JSON files."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.annotation.validation import validate_document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--pattern", default="*.annotation.json")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()
    root = Path(args.input_dir)
    if not root.is_dir():
        print(f"ERROR: input directory does not exist: {root}")
        return 2
    files = sorted(root.rglob(args.pattern) if args.recursive else root.glob(args.pattern))
    if not files:
        print(f"ERROR: no files matched {args.pattern!r} in {root}")
        return 2
    failed = 0
    for path in files:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            errors = validate_document(doc)
        except Exception as exc:
            errors = [f"could not read or parse JSON: {exc}"]
        if errors:
            failed += 1
            print(f"FAIL {path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"OK   {path}: {len(doc.get('annotations', []))} annotations, {len(doc.get('relations', []))} relations, status={doc['annotation_metadata']['status']}")
    print(f"\nFiles reviewed: {len(files)}; failed: {failed}; passed: {len(files) - failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
