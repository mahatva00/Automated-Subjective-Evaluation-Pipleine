from __future__ import annotations
import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.static_answer_evaluator import (
    OllamaClient,
    get_requirement_text,
    select_evidence_blocks,
    validate_assessment,
)


def make_batch(question: dict, batch_size: int, max_blocks: int, max_chars: int) -> tuple[dict, list[dict], dict[str, tuple[list[dict], bool, set[str]]]]:
    q_blocks = question.get("evidence_blocks", []) or []
    reqs = (question.get("requirements", []) or [])[:batch_size]
    payloads = []
    contexts = {}
    for req in reqs:
        selected, complete = select_evidence_blocks(
            req, q_blocks, max_blocks=max_blocks, max_chars=max_chars, max_block_chars=760
        )
        rid = str(req.get("requirement_id") or "")
        supplied = {str(b.get("block_id") or "") for b in selected}
        contexts[rid] = (selected, complete, supplied)
        payloads.append({
            "requirement_id": rid,
            "question_number": req.get("question_number", question.get("question_number")),
            "question_title": question.get("question_title", ""),
            "requirement_type": req.get("requirement_type") or req.get("source_category"),
            "requirement_text": get_requirement_text(req),
            "original_source_text": req.get("source_text") or "",
            "context_complete": complete,
            "evidence_blocks": selected,
        })
    return {"requirements_to_assess": payloads}, reqs, contexts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark local Ollama static assessment on two real evidence batches. Student code is never executed."
    )
    parser.add_argument("--evidence-map", required=True)
    parser.add_argument("--model", default="qwen2.5:3b-instruct")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-evidence-blocks", type=int, default=2)
    parser.add_argument("--max-evidence-chars", type=int, default=1400)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    if args.batch_size < 1 or args.timeout < 1:
        parser.error("--batch-size and --timeout must be positive")

    source = Path(args.evidence_map)
    if not source.is_file():
        parser.error(f"Evidence map does not exist: {source}")
    evidence_map = json.loads(source.read_text(encoding="utf-8"))
    if evidence_map.get("student_code_executed") is True:
        parser.error("Evidence map says student code was executed; refusing to benchmark")
    questions = [q for q in evidence_map.get("questions", []) or [] if q.get("requirements")]
    if not questions:
        parser.error("No question-scoped requirements are available to benchmark")

    first = questions[0]
    largest = max(questions, key=lambda q: len(q.get("requirements", []) or []))
    sample_questions = [first]
    if largest is not first:
        sample_questions.append(largest)
    elif len(questions) > 1:
        sample_questions.append(sorted(questions, key=lambda q: len(q.get("requirements", []) or []), reverse=True)[1])

    client = OllamaClient(args.base_url, args.model, args.timeout)
    samples = []
    elapsed_values = []
    print(f"Ollama benchmark model: {args.model}")
    print(f"Sample: at most {args.batch_size} requirements per question, {len(sample_questions)} requests")
    print("Uses existing submission text as evidence; never executes student code.", flush=True)

    for index, question in enumerate(sample_questions, start=1):
        payload, reqs, contexts = make_batch(question, args.batch_size, args.max_evidence_blocks, args.max_evidence_chars)
        if not reqs:
            continue
        started = time.perf_counter()
        raw_response = client.chat_json(payload)
        elapsed = time.perf_counter() - started
        elapsed_values.append(elapsed)
        raw_list = raw_response.get("assessments", []) if isinstance(raw_response, dict) else []
        raw_by_id = {str(x.get("requirement_id")): x for x in raw_list if isinstance(x, dict) and x.get("requirement_id") is not None}
        checked = []
        q_blocks = question.get("evidence_blocks", []) or []
        for req in reqs:
            rid = str(req.get("requirement_id") or "")
            selected, complete, supplied = contexts[rid]
            checked.append(validate_assessment(raw_by_id.get(rid, {}), req, q_blocks, supplied, complete, args.model))
        valid = [item for item in checked if not item.get("validation_errors")]
        statuses = Counter(item.get("status", "missing") for item in checked)
        row = {
            "question_id": question.get("question_id"),
            "question_number": question.get("question_number"),
            "requested_requirements": len(reqs),
            "returned_assessments": len(raw_list),
            "validated_assessments": len(valid),
            "status_counts": dict(statuses),
            "elapsed_seconds": round(elapsed, 2),
            "requirement_ids": [str(r.get("requirement_id") or "") for r in reqs],
        }
        samples.append(row)
        print(
            f"Sample {index}: question={row['question_number'] or row['question_id']}; "
            f"{len(valid)}/{len(reqs)} validated; statuses={dict(statuses)}; {elapsed:.1f}s",
            flush=True,
        )

    total_reqs = sum(len(q.get("requirements", []) or []) for q in evidence_map.get("questions", []) or []) + len(evidence_map.get("global_requirements", []) or [])
    planned_calls = sum(math.ceil(len(q.get("requirements", []) or []) / args.batch_size) for q in evidence_map.get("questions", []) or [])
    planned_calls += math.ceil(len(evidence_map.get("global_requirements", []) or []) / args.batch_size)
    # The first call can include model-loading cost, so project from warmed calls where possible.
    warm_times = elapsed_values[1:] or elapsed_values
    warm_average = (sum(warm_times) / len(warm_times)) if warm_times else None
    estimated_minutes = (warm_average * planned_calls / 60) if warm_average is not None else None
    report = {
        "artifact_type": "ollama_performance_benchmark",
        "evidence_map": str(source),
        "model": args.model,
        "student_code_executed": False,
        "total_requirements_in_full_run": total_reqs,
        "batch_size": args.batch_size,
        "estimated_provider_calls": planned_calls,
        "samples": samples,
        "warmed_sample_average_seconds": round(warm_average, 2) if warm_average is not None else None,
        "rough_projected_minutes": round(estimated_minutes, 1) if estimated_minutes is not None else None,
        "projection_warning": "Projection is based on at most two question batches and is indicative, not a runtime guarantee.",
    }
    print(f"Estimated provider calls for full run: {planned_calls} across {total_reqs} requirements")
    if estimated_minutes is not None:
        print(f"Rough warmed-call projection: {estimated_minutes:.1f} minutes (indicative, not guaranteed)")
    report_path = Path(args.report) if args.report else Path("data/processed/evaluations") / source.parent.name / f"ollama-benchmark-{args.model.replace(':', '-').replace('/', '-')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Benchmark report saved: {report_path}")
    print("No evaluation checkpoint or final evaluation file was modified.")
    return 0 if samples and all(row["validated_assessments"] == row["requested_requirements"] for row in samples) else 2

if __name__ == "__main__":
    raise SystemExit(main())
