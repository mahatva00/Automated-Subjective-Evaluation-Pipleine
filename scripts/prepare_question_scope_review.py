#!/usr/bin/env python3
"""Create line/range-based scope review JSON+TSV sidecars for question-only extraction."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.annotation.scope import build_scope_review

FIELDS = ["document_id", "segment_id", "start_char", "end_char", "scope_status", "question_id", "reason", "reviewer", "review_date", "text_sha256", "text_preview_json"]

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--pattern", default="*.annotation.json")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--max-chars", type=int, default=900)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(); root, outroot = Path(a.input_dir), Path(a.output_dir)
    if not root.is_dir(): print(f"ERROR: input directory does not exist: {root}"); return 2
    paths = sorted(root.rglob(a.pattern) if a.recursive else root.glob(a.pattern))
    if not paths: print(f"ERROR: no files matched {a.pattern!r}"); return 2
    made = 0
    for path in paths:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            ledger, rows = build_scope_review(doc, path.name, max_chars=a.max_chars)
        except Exception as exc:
            print(f"ERROR: {path}: {exc}"); return 2
        rel = path.relative_to(root); parent = outroot / rel.parent
        stem = path.name.removesuffix(".annotation.json")
        json_out, tsv_out = parent / f"{stem}.scope.json", parent / f"{stem}.scope-review.tsv"
        if not a.overwrite and (json_out.exists() or tsv_out.exists()):
            print(f"ERROR: scope output already exists for {path.name}; use --overwrite or a new output directory"); return 2
        parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with tsv_out.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
            w.writeheader(); w.writerows(rows)
        nonblank = sum(1 for s in ledger["segments"] if s["scope_status"] == "pending")
        print(f"Prepared {path.name}: {len(rows)} segments; {nonblank} nonblank pending; JSON={json_out}; TSV={tsv_out}")
        made += 1
    print(f"Prepared scope reviews for {made} documents. No scope decisions or annotation labels were inferred.")
    return 0
if __name__ == "__main__": raise SystemExit(main())
