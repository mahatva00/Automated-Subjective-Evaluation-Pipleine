"""Repeat evaluation of one frozen evidence map without reparsing or remapping."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.static_answer_evaluator import OllamaClient, evaluate_evidence_map


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark evaluator repeatability on a frozen evidence map."
    )
    parser.add_argument("--evidence-map", required=True)
    parser.add_argument("--model", default="qwen2.5:3b-instruct")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.batch_size < 1 or args.timeout < 1:
        parser.error("--repeats, --batch-size, and --timeout must be positive")

    source = Path(args.evidence_map)
    if not source.is_file():
        parser.error(f"Evidence map does not exist: {source}")
    evidence_map = json.loads(source.read_text(encoding="utf-8"))
    if evidence_map.get("student_code_executed") is True:
        parser.error("Evidence map says student code was executed; refusing to benchmark")

    client = OllamaClient(args.base_url, args.model, args.timeout)
    repetitions = []
    for repeat in range(1, args.repeats + 1):
        started = time.perf_counter()
        evaluation = evaluate_evidence_map(evidence_map, client, batch_size=args.batch_size)
        elapsed = time.perf_counter() - started
        rows = [
            row
            for question in evaluation.get("questions", [])
            for row in question.get("evaluations", [])
        ]
        repetitions.append({
            "repeat": repeat,
            "elapsed_seconds": round(elapsed, 2),
            "status_counts": dict(Counter(row.get("status") for row in rows)),
            "diagnostic_counts": dict(Counter(
                (row.get("evidence_diagnostic") or {}).get("category")
                for row in rows
            )),
            "validated_rows": sum(not row.get("validation_errors") for row in rows),
            "total_rows": len(rows),
        })
        print(
            f"Repeat {repeat}: {len(rows)} requirements, "
            f"{repetitions[-1]['validated_rows']} validated, {elapsed:.1f}s",
            flush=True,
        )

    report = {
        "artifact_type": "frozen_evidence_evaluation_benchmark",
        "evidence_map": str(source),
        "model": args.model,
        "student_code_executed": False,
        "repeats": repetitions,
        "repeatability": {
            "status_counts": [
                item["status_counts"] for item in repetitions
            ],
            "diagnostic_counts": [
                item["diagnostic_counts"] for item in repetitions
            ],
        },
        "note": "Every repetition reused the same evidence map; no parsing or mapping stage ran.",
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen benchmark report saved: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
