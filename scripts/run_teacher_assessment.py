#!/usr/bin/env python3
"""Run a static assessment from one teacher input folder; student code is never executed."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.orchestration.teacher_assessment import run_teacher_assessment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Folder containing a question-paper PDF, student IPYNB/PDFs, and an assessment policy")
    parser.add_argument("--output-dir", default=None, help="Output base directory (default: sibling of input folder); each run gets a timestamped child")
    parser.add_argument("--question-paper", default=None, help="Optional explicit QP PDF path when autodiscovery is ambiguous")
    parser.add_argument("--policy-file", default=None, help="Optional explicit policy JSON/TXT/MD/PDF/DOCX path")
    parser.add_argument("--provider", choices=("ollama", "groq", "gemini"), default=os.environ.get("INE_EVALUATOR_PROVIDER", "ollama"))
    parser.add_argument("--model", default=os.environ.get("INE_EVALUATOR_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-evidence-blocks", type=int, default=4)
    parser.add_argument("--max-evidence-chars", type=int, default=2800)
    parser.add_argument("--model-dir", default=os.environ.get("INE_DISTILBERT_MODEL_DIR"), help="Optional pre-trained DistilBERT MVP checkpoint; model predictions are advisory only")
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--resume-run", default=None, help="Resume an existing run directory after confirming all teacher input hashes match")
    args = parser.parse_args()
    try:
        report = run_teacher_assessment(args.input_dir, args.output_dir,
            question_paper=args.question_paper, policy_file=args.policy_file,
            provider=args.provider, model=args.model, base_url=args.base_url,
            timeout=args.timeout, batch_size=args.batch_size,
            max_evidence_blocks=args.max_evidence_blocks, max_evidence_chars=args.max_evidence_chars,
            model_dir=args.model_dir, min_confidence=args.min_confidence,
            resume_run_dir=args.resume_run)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("No student code was executed.", file=sys.stderr)
        return 2
    print(f"Assessment run: {report['run_id']}")
    print(f"Students: {report['summary']['students_assessed']}/{report['summary']['students_discovered']} evaluated")
    print(f"Run report: {Path(report['run_dir']) / 'RUN_REPORT.json'}")
    print(f"HTML summary: {Path(report['run_dir']) / 'ASSESSMENT_SUMMARY.html'}")
    print("Student code executed: no")
    return 0 if report["summary"]["students_failed_or_partial"] == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
