#!/usr/bin/env python3
"""Compile reviewed canonical QP annotations plus reviewed coverage into masked BIO windows."""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.annotation.token_dataset import (
    DatasetError, compile_document, find_coverage_files, index_documents,
    label_maps, read_manifest, validate_approved_document,
)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", required=True, help="Directory containing canonical *.annotation.json")
    parser.add_argument("--manifest", required=True, help="Explicit paper-level train/validation/test split manifest")
    parser.add_argument("--coverage-dir", required=True, help="Directory containing *.training-coverage.json files")
    parser.add_argument("--output-dir", required=True, help="New output directory for JSONL windows and label maps")
    parser.add_argument("--tokenizer", default="distilbert-base-uncased", help="Fast Hugging Face tokenizer name or local path")
    parser.add_argument("--pattern", default="*.annotation.json")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=64, help="Token overlap between adjacent windows")
    parser.add_argument("--local-files-only", action="store_true", help="Do not try to download a tokenizer")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        if args.max_length < 8:
            raise DatasetError("--max-length must be at least 8")
        if args.stride < 0:
            raise DatasetError("--stride must be non-negative")
        annotation_root = Path(args.annotation_dir)
        manifest = read_manifest(Path(args.manifest))
        docs = index_documents(annotation_root, args.pattern, args.recursive)
        coverages = find_coverage_files(Path(args.coverage_dir), args.recursive)
        assigned = [entry for entry in manifest.values() if entry["split"] != "unassigned"]
        if not assigned:
            raise DatasetError("manifest contains no documents assigned to train/validation/test")
        for doc_id, entry in manifest.items():
            if entry["split"] == "unassigned":
                continue
            if doc_id not in docs:
                raise DatasetError(f"manifest assigns {doc_id} but no annotation file was found")
            actual_path, doc = docs[doc_id]
            declared = (annotation_root / entry["annotation_path"]).resolve()
            if declared != actual_path.resolve():
                raise DatasetError(f"manifest path for {doc_id} resolves to {declared}, found {actual_path}")
            validate_approved_document(doc, actual_path)
            if doc_id not in coverages:
                raise DatasetError(f"training coverage ledger missing for {doc_id}")
        try:
            from transformers import AutoTokenizer
        except Exception as exc:
            raise DatasetError(f"transformers is required to compile token windows: {exc}") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, use_fast=True, local_files_only=args.local_files_only
        )
        if not getattr(tokenizer, "is_fast", False):
            raise DatasetError("a fast tokenizer with offset_mapping support is required")
        compiled_by_split = {"train": [], "validation": [], "test": []}
        per_document = []
        total_annotations = 0
        for doc_id, entry in manifest.items():
            if entry["split"] == "unassigned":
                print(f"SKIP {doc_id}: manifest split is unassigned")
                continue
            _, doc = docs[doc_id]
            coverage_path, coverage = coverages[doc_id]
            windows, stats = compile_document(
                doc, coverage, entry, tokenizer,
                max_length=args.max_length, stride=args.stride
            )
            compiled_by_split[entry["split"]].extend(windows)
            total_annotations += stats["approved_annotations"]
            per_document.append(stats)
            print(
                f"OK {doc_id}: approved_annotations={stats['approved_annotations']}; "
                f"tokens={stats['token_count_before_specials']}; windows={stats['window_count']}; "
                f"alignment_warnings={stats['alignment_warning_count']}"
            )
        if total_annotations == 0:
            raise DatasetError("no reviewed annotations found in assigned documents; refusing to create a labels-free training dataset")
        outdir = Path(args.output_dir)
        out_files = [outdir / f"{split}.bio.jsonl" for split in compiled_by_split]
        out_files += [outdir / "label_maps.json", outdir / "dataset_summary.json", outdir / "alignment_warnings.json"]
        existing = [p for p in out_files if p.exists()]
        if existing and not args.overwrite:
            raise DatasetError("output exists (use --overwrite only after archiving): " + ", ".join(map(str, existing)))
        outdir.mkdir(parents=True, exist_ok=True)
        for split, rows in compiled_by_split.items():
            (outdir / f"{split}.bio.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
                encoding="utf-8",
            )
        (outdir / "label_maps.json").write_text(json.dumps(label_maps(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        warnings = [w for s in per_document for w in s["alignment_warnings"]]
        summary = {
            "dataset_schema_version": "distilbert-bio-dataset-v1.0",
            "tokenizer": args.tokenizer,
            "tokenizer_is_fast": True,
            "max_length": args.max_length,
            "stride": args.stride,
            "ignore_index": -100,
            "documents": len(per_document),
            "windows_by_split": {k: len(v) for k, v in compiled_by_split.items()},
            "approved_annotations_total": total_annotations,
            "documents_detail": per_document,
            "alignment_warning_count": len(warnings),
            "review_contract": (
                "Only reviewed/adjudicated canonical annotations and reviewed coverage can create labels. "
                "O is assigned only to tokens covered by reviewed coverage for that layer. "
                "Pending, out-of-scope, separator, special, unreviewed and ambiguous boundary tokens use -100. "
                "Student code is never executed; no labels are inferred from proposals."
            ),
        }
        (outdir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (outdir / "alignment_warnings.json").write_text(json.dumps(warnings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"COMPILED documents={len(per_document)}; annotations={total_annotations}; windows={sum(map(len, compiled_by_split.values()))}")
        print("Windows: " + ", ".join(f"{s}={len(rows)}" for s, rows in compiled_by_split.items()))
        print(f"Alignment warnings: {len(warnings)}")
        print(f"Output: {outdir}")
        print("Student code executed: no; proposal labels consumed: no")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
