#!/usr/bin/env python3
"""Create blank human-annotation templates from processed PDF parser JSONs."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEPARATOR = "\n\f\n"


def safe_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._:-]+", "-", value.strip().lower()).strip("-._:")
    return value or "document"


def build_template(parser_record: dict, input_path: Path) -> dict:
    submission = parser_record.get("submission") or {}
    segments = parser_record.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("source parser JSON must contain a non-empty 'segments' array")
    usable = []
    for original_index, segment in enumerate(segments):
        if not isinstance(segment, dict) or not isinstance(segment.get("content"), str):
            continue
        usable.append((original_index, segment))
    usable.sort(key=lambda item: (item[1].get("position", item[0]), item[0]))
    if not usable:
        raise ValueError("source parser JSON contains no segments with string content")

    page_texts = [segment["content"] for _, segment in usable]
    chunks: list[str] = []
    page_ranges: list[dict] = []
    cursor = 0
    for i, ((original_index, segment), page_text) in enumerate(zip(usable, page_texts)):
        if i:
            chunks.append(SEPARATOR)
            cursor += len(SEPARATOR)
        start = cursor
        chunks.append(page_text)
        cursor += len(page_text)
        source_location = segment.get("source_location") or {}
        page_number = source_location.get("page_number")
        range_item = {
            "source_segment_id": str(segment.get("segment_id") or f"segment_{original_index + 1:04d}"),
            "page_number": page_number if isinstance(page_number, int) and page_number >= 1 else None,
            "sequence_index": i,
            "start_char": start,
            "end_char": cursor,
            "extraction_method": segment.get("extraction_method"),
            "original_type": segment.get("original_type"),
        }
        page_ranges.append(range_item)
    text = "".join(chunks)
    if not text.strip():
        raise ValueError("combined source text is empty; review the upstream parser output")

    filename = str(submission.get("source_filename") or input_path.name)
    source_hash = submission.get("source_hash")
    fingerprint = str(source_hash or hashlib.sha256(text.encode("utf-8")).hexdigest())[:12]
    document_id = f"{safe_id(Path(filename).stem)}_{fingerprint}"
    source = {
        "filename": filename,
        "format": str(submission.get("source_format") or Path(filename).suffix.lstrip(".") or "unknown"),
        "document_type": "question_paper",
        "relative_path": str(submission.get("source_relative_path") or ""),
        "language": "und",
        "source_parser_schema_version": str(parser_record.get("schema_version") or "unknown"),
    }
    if source_hash:
        source["source_hash"] = str(source_hash)
    course_name = submission.get("course_name")
    if course_name is not None:
        source["course"] = str(course_name)
    return {
        "schema_version": "1.0.0",
        "document_id": document_id,
        "source": source,
        "text": text,
        "page_ranges": page_ranges,
        "annotations": [],
        "relations": [],
        "annotation_metadata": {
            "status": "draft",
            "guidelines_version": "1.0.0",
            "annotator_ids": [],
            "notes": "Blank annotation template. No labels were inferred automatically.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing processed PDF-parser JSON files")
    parser.add_argument("--pattern", default="*-QP.json", help="Input filename pattern (default: *-QP.json)")
    parser.add_argument("--output-dir", required=True, help="Directory for blank annotation drafts")
    parser.add_argument("--recursive", action="store_true", help="Search input directory recursively")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing draft JSON files and manifest template")
    args = parser.parse_args()
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    if not input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {input_dir}")
        return 2
    files = sorted(input_dir.rglob(args.pattern) if args.recursive else input_dir.glob(args.pattern))
    if not files:
        print(f"ERROR: no files matched {args.pattern!r} in {input_dir}")
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_docs = []
    failures = 0
    for path in files:
        try:
            parser_record = json.loads(path.read_text(encoding="utf-8"))
            draft = build_template(parser_record, path)
            destination = output_dir / f"{Path(draft['source']['filename']).stem}.annotation.json"
            if destination.exists() and not args.overwrite:
                raise FileExistsError(f"output exists (use --overwrite after archiving): {destination}")
            destination.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            manifest_docs.append({
                "document_id": draft["document_id"],
                "annotation_path": destination.name,
                "split": "unassigned",
                "group_id": draft["document_id"],
                "subject": None,
            })
            print(f"OK {path.name}: {len(draft['page_ranges'])} source segments, {len(draft['text'])} characters -> {destination}")
        except Exception as exc:
            failures += 1
            print(f"ERROR {path}: {exc}")
    manifest_path = output_dir / "dataset_manifest.template.json"
    if manifest_path.exists() and not args.overwrite:
        print(f"WARNING: left existing manifest unchanged: {manifest_path} (use --overwrite to regenerate)")
    elif manifest_docs:
        manifest = {"manifest_version": "1.0.0", "documents": manifest_docs}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Manifest template: {manifest_path}")
    print(f"\nPrepared: {len(manifest_docs)}; failed: {failures}; annotations generated: 0")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
