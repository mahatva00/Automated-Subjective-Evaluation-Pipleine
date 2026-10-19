#!/usr/bin/env python3
"""Create training-coverage review templates from canonical annotation drafts and optional scope ledgers."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.annotation.token_dataset import DatasetError, coverage_template, index_documents, read_json

def text_preview(text: str, limit: int = 150) -> str:
    return text.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")[:limit]

def write_tsv(path: Path, coverage: dict, canonical_text: str) -> None:
    columns = [
        "coverage_id", "start_char", "end_char", "scope_status", "status",
        "reviewed_layers", "source_segment_id", "question_id", "page_numbers",
        "text_preview", "reviewer", "review_date", "notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for seg in coverage["segments"]:
            writer.writerow({
                "coverage_id": seg["coverage_id"],
                "start_char": seg["start_char"],
                "end_char": seg["end_char"],
                "scope_status": seg.get("scope_status") or "",
                "status": seg["status"],
                "reviewed_layers": ";".join(seg.get("reviewed_layers", [])),
                "source_segment_id": seg.get("source_segment_id") or "",
                "question_id": seg.get("question_id") or "",
                "page_numbers": ";".join(str(p) for p in seg.get("page_numbers", [])),
                "text_preview": text_preview(canonical_text[seg["start_char"]:seg["end_char"]]),
                "reviewer": seg.get("reviewer", ""),
                "review_date": seg.get("review_date", ""),
                "notes": seg.get("notes", ""),
            })

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", required=True, help="Canonical *.annotation.json directory")
    parser.add_argument("--scope-dir", help="Optional directory containing reviewed *.scope.json ledgers")
    parser.add_argument("--output-dir", required=True, help="New directory for training-coverage JSON and TSV templates")
    parser.add_argument("--pattern", default="*.annotation.json")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        docs = index_documents(Path(args.annotation_dir), args.pattern, args.recursive)
        scopes = {}
        if args.scope_dir:
            scope_root = Path(args.scope_dir)
            if not scope_root.is_dir():
                raise DatasetError(f"scope directory does not exist: {scope_root}")
            for path in sorted(scope_root.rglob("*.scope.json") if args.recursive else scope_root.glob("*.scope.json")):
                scope = read_json(path)
                doc_id = scope.get("document_id")
                if doc_id:
                    if doc_id in scopes:
                        raise DatasetError(f"duplicate scope ledger for document_id {doc_id}")
                    scopes[doc_id] = (path, scope)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        prepared = 0
        for doc_id, (source_path, doc) in sorted(docs.items()):
            scope_pair = scopes.get(doc_id)
            scope_doc = scope_pair[1] if scope_pair else None
            cov = coverage_template(doc, scope_doc)
            stem = source_path.stem
            json_path = out / f"{stem}.training-coverage.json"
            tsv_path = out / f"{stem}.training-coverage.tsv"
            existing = [p for p in (json_path, tsv_path) if p.exists()]
            if existing and not args.overwrite:
                print(f"SKIP {doc_id}: output exists; use --overwrite only after archiving")
                continue
            json_path.write_text(json.dumps(cov, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            write_tsv(tsv_path, cov, doc["text"])
            print(f"OK {doc_id}: coverage segments={len(cov['segments'])}; scope_source={'existing ledger' if scope_doc else 'none (all page text pending)'}")
            prepared += 1
        print(f"Prepared templates: {prepared}; labels approved: 0; training dataset exported: no")
        return 0 if prepared else 1
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
