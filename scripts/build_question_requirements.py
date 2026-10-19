"""Batch-run question parsing and baseline requirement extraction on parsed-PDF JSON.

This reads PDF-parser JSON files containing page segments. It does not open or
execute student submissions. For each question-paper JSON it writes a question
map and a requirements JSON to separate output directories.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.question_parser import parse_question_paper
from src.requirements.requirement_extractor import extract_atomic_requirements


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    segments = data.get("segments")
    if not isinstance(segments, list):
        raise ValueError("input is not PDF-parser JSON: expected a 'segments' list")
    return data


def process_file(
    source: Path,
    question_map_dir: Path,
    requirements_dir: Path,
) -> dict[str, Any]:
    """Parse one PDF-parser JSON file and save both intermediate outputs."""
    document = _read_json(source)
    question_map = parse_question_paper(document)
    requirements = extract_atomic_requirements(question_map)

    question_map_dir.mkdir(parents=True, exist_ok=True)
    requirements_dir.mkdir(parents=True, exist_ok=True)
    map_path = question_map_dir / f"{source.stem}.question-map.json"
    requirements_path = requirements_dir / f"{source.stem}.requirements.json"
    map_path.write_text(
        json.dumps(question_map, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    requirements_path.write_text(
        json.dumps(requirements, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    all_requirements = requirements.get("requirements", [])
    return {
        "source": str(source),
        "question_map": str(map_path),
        "requirements_file": str(requirements_path),
        "questions": len(question_map.get("questions", [])),
        "requirements": len(all_requirements),
        "needs_review": sum(bool(item.get("needs_review")) for item in all_requirements),
        "parser_diagnostics": len(question_map.get("diagnostics", [])),
        "extraction_diagnostics": len(requirements.get("diagnostics", [])),
        "student_code_executed": requirements["extraction"]["student_code_executed"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", default="data/processed/pdfs",
        help="Directory containing PDF-parser JSON files (default: data/processed/pdfs)",
    )
    parser.add_argument(
        "--pattern", default="*-QP.json",
        help="Glob pattern for question papers (default: *-QP.json)",
    )
    parser.add_argument(
        "--question-map-dir", default="data/processed/questions",
        help="Directory for parsed question-map JSON files",
    )
    parser.add_argument(
        "--requirements-dir", default="data/processed/requirements",
        help="Directory for extracted requirement JSON files",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Search input directory recursively",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    question_map_dir = Path(args.question_map_dir)
    requirements_dir = Path(args.requirements_dir)
    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")

    candidates = input_dir.rglob(args.pattern) if args.recursive else input_dir.glob(args.pattern)
    sources = sorted(path for path in candidates if path.is_file())
    if not sources:
        parser.error(f"No files matching {args.pattern!r} found in {input_dir}")

    failures = []
    summaries = []
    for source in sources:
        try:
            summary = process_file(source, question_map_dir, requirements_dir)
            summaries.append(summary)
            print(
                f"OK {source.name}: {summary['questions']} questions, "
                f"{summary['requirements']} requirements, "
                f"{summary['needs_review']} needs review, "
                f"{summary['parser_diagnostics']} parser diagnostics"
            )
            print(f"   question map: {summary['question_map']}")
            print(f"   requirements: {summary['requirements_file']}")
        except (ValueError, OSError, TypeError, KeyError) as exc:
            failures.append((source, str(exc)))
            print(f"FAILED {source}: {exc}", file=sys.stderr)

    print(f"\nProcessed successfully: {len(summaries)}/{len(sources)}")
    print(f"Student code executed: no")
    if failures:
        print("Failures:")
        for source, reason in failures:
            print(f"- {source}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
