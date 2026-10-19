#!/usr/bin/env python3
"""Import a manually reviewed training-coverage TSV into its JSON ledger."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.annotation.token_dataset import DatasetError, read_json, sha256_text

VALID_STATUS = {"reviewed", "out_of_scope", "pending", "separator"}
VALID_LAYERS = {"structure", "semantic"}
FIXED_COLUMNS = ("coverage_id", "start_char", "end_char", "scope_status")

def parse_layers(value: str) -> list[str]:
    layers = [x.strip() for x in value.split(";") if x.strip()]
    if len(set(layers)) != len(layers) or any(x not in VALID_LAYERS for x in layers):
        raise DatasetError(f"invalid reviewed_layers value {value!r}; use structure;semantic")
    return layers

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-dir", required=True, help="Directory containing template JSON and reviewed TSV files")
    parser.add_argument("--output-dir", required=True, help="Destination directory for imported coverage JSON")
    parser.add_argument("--pattern", default="*.training-coverage.tsv")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--finalize", action="store_true", help="Mark ledger reviewed after validating every row; this is an explicit human action")
    parser.add_argument("--reviewer", help="Reviewer identifier (required with --finalize)")
    parser.add_argument("--review-date", help="ISO date YYYY-MM-DD (defaults to today only when finalizing)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        if args.finalize and not (args.reviewer or "").strip():
            raise DatasetError("--reviewer is required with --finalize")
        root = Path(args.coverage_dir)
        if not root.is_dir():
            raise DatasetError(f"coverage directory does not exist: {root}")
        files = sorted(root.rglob(args.pattern) if args.recursive else root.glob(args.pattern))
        if not files:
            raise DatasetError(f"no TSV files matched {args.pattern!r}")
        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        completed = 0
        for tsv_path in files:
            json_path = tsv_path.with_suffix(".json")
            # Input base e.g. FAT-QP.annotation.training-coverage.tsv -> matching .json
            if not json_path.exists():
                raise DatasetError(f"template JSON missing for {tsv_path}: expected {json_path}")
            template = read_json(json_path)
            target = outdir / json_path.name
            if target.exists() and not args.overwrite:
                raise DatasetError(f"destination exists: {target}; use --overwrite after archiving")
            with tsv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            by_id = {r.get("coverage_id"): r for r in rows}
            if len(by_id) != len(rows) or set(by_id) != {s["coverage_id"] for s in template["segments"]}:
                raise DatasetError(f"{tsv_path}: coverage_id set differs from template")
            for row in rows:
                for field in FIXED_COLUMNS:
                    if row.get(field, "") != str(next(s[field] if field != "scope_status" else (s.get(field) or "") for s in template["segments"] if s["coverage_id"] == row["coverage_id"])):
                        raise DatasetError(f"{tsv_path}: immutable field {field} changed for {row['coverage_id']}")
            updated = json.loads(json.dumps(template))
            for seg in updated["segments"]:
                row = by_id[seg["coverage_id"]]
                status = (row.get("status") or "").strip()
                layers = parse_layers(row.get("reviewed_layers") or "")
                if status not in VALID_STATUS:
                    raise DatasetError(f"{tsv_path}: invalid status for {seg['coverage_id']}: {status!r}")
                source_scope = seg.get("scope_status")
                if source_scope == "out_of_scope" and status != "out_of_scope":
                    raise DatasetError(f"{tsv_path}: cannot override source out-of-scope segment {seg['coverage_id']}")
                if source_scope == "separator" and status != "separator":
                    raise DatasetError(f"{tsv_path}: cannot relabel source separator {seg['coverage_id']}")
                if source_scope == "pending" and status != "pending":
                    raise DatasetError(f"{tsv_path}: cannot promote scope-pending segment {seg['coverage_id']}")
                if source_scope == "in_scope" and status not in {"reviewed", "pending"}:
                    raise DatasetError(f"{tsv_path}: in-scope region must remain reviewed or pending: {seg['coverage_id']}")
                if status == "reviewed" and not layers:
                    raise DatasetError(f"{tsv_path}: reviewed segment {seg['coverage_id']} needs reviewed_layers")
                if status != "reviewed" and layers:
                    raise DatasetError(f"{tsv_path}: non-reviewed segment {seg['coverage_id']} cannot list reviewed_layers")
                seg["status"] = status
                seg["reviewed_layers"] = layers
                seg["reviewer"] = (row.get("reviewer") or args.reviewer or "").strip()
                seg["review_date"] = (row.get("review_date") or args.review_date or "").strip()
                seg["notes"] = row.get("notes") or ""
            updated["ledger_status"] = "reviewed" if args.finalize else "draft"
            if args.finalize:
                review_date = args.review_date or date.today().isoformat()
                # Keep reviewer metadata at the top level; do not modify source/annotation data.
                updated["coverage_review_metadata"] = {
                    "reviewer": args.reviewer.strip(),
                    "review_date": review_date,
                    "status": "reviewed",
                    "note": "Coverage review finalized explicitly by CLI flag; pending regions remain masked."
                }
            target.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"OK {template.get('document_id')}: imported {len(updated['segments'])} coverage rows; ledger_status={updated['ledger_status']}")
            completed += 1
        print(f"Imported coverage ledgers: {completed}; annotation labels created: 0")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
