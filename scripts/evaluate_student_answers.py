from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.static_answer_evaluator import GeminiClient, GroqClient, OllamaClient, evaluate_evidence_map


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate student answers statically using Ollama, Gemini API, or Groq API. Student code is never executed.")
    parser.add_argument("--evidence-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", choices=("ollama", "gemini", "groq"), default="ollama")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=int, default=180, help="Maximum inactivity/request timeout per provider call, in seconds")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-evidence-blocks", type=int, default=4)
    parser.add_argument("--max-evidence-chars", type=int, default=2800)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (defaults to OUTPUT.checkpoint.json)")
    parser.add_argument("--resume", action="store_true", help="Resume completed requirements from an existing compatible checkpoint")
    args = parser.parse_args()

    source_path = Path(args.evidence_map)
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_path.with_suffix(output_path.suffix + ".checkpoint.json")
    if not source_path.is_file():
        parser.error(f"Evidence map does not exist: {source_path}")
    if args.timeout < 1 or args.batch_size < 1:
        parser.error("--timeout and --batch-size must be positive integers")
    source_bytes = source_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    evidence_map = json.loads(source_bytes.decode("utf-8"))
    model = args.model or {"gemini": "gemini-3.8-flash", "groq": "openai/gpt-oss-20b", "ollama": "qwen2.5-coder:7b"}[args.provider]
    if args.provider == "gemini":
        try:
            client = GeminiClient(model=model, timeout=args.timeout)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.provider == "groq":
        try:
            client = GroqClient(model=model, timeout=args.timeout)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        client = OllamaClient(args.base_url, model, args.timeout)

    completed = []
    if args.resume and checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("source_sha256") != source_hash:
            parser.error("Checkpoint source hash does not match this evidence map; refusing to resume against different input.")
        if checkpoint.get("provider") != args.provider or checkpoint.get("model") != model:
            parser.error("Checkpoint provider/model differs from this run; use the same provider/model or choose a new checkpoint path.")
        completed = checkpoint.get("completed_results", []) or []
        print(f"Resuming from checkpoint: {len(completed)} saved requirement results")
    elif checkpoint_path.exists() and not args.resume:
        print(f"Existing checkpoint ignored (pass --resume to use it): {checkpoint_path}")

    request_counter = 0
    saved_count = [len(completed)]

    def report_request(batch_count):
        nonlocal request_counter
        request_counter += 1
        total = sum(len(q.get("requirements", []) or []) for q in evidence_map.get("questions", []) or []) + len(evidence_map.get("global_requirements", []) or [])
        print(f"Starting provider request {request_counter}: batch size {batch_count}; {saved_count[0]}/{total} already checkpointed", flush=True)

    def save_checkpoint(rows, total):
        saved_count[0] = len(rows)
        atomic_write_json(checkpoint_path, {
            "checkpoint_version": 1,
            "source_evidence_map_path": str(source_path),
            "source_sha256": source_hash,
            "provider": args.provider,
            "model": model,
            "completed_requirements": len(rows),
            "total_requirements": total,
            "completed_results": rows,
        })
        print(f"Progress: {len(rows)}/{total} requirements | checkpoint saved", flush=True)

    try:
        print(f"Input: {source_path}")
        print(f"Provider/model: {args.provider}/{model}")
        print(f"Request timeout: {args.timeout}s per provider call; batch size: {args.batch_size}")
        print("Static inspection only; student code execution is disabled.", flush=True)
        output = evaluate_evidence_map(
            evidence_map, client, batch_size=args.batch_size,
            max_evidence_blocks=args.max_evidence_blocks,
            max_evidence_chars=args.max_evidence_chars,
            completed_results=completed,
            progress_callback=save_checkpoint,
            before_request_callback=report_request,
        )
        output["source_evidence_map_path"] = str(source_path)
        output["source_evidence_map_sha256"] = source_hash
        output["provider"] = args.provider
        output["checkpoint_path"] = str(checkpoint_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output_path, output)
        print(f"Saved evaluation JSON: {output_path}")
        print("Summary: " + json.dumps(output["summary"], ensure_ascii=False))
        print("Student code executed: no")
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Completed results are saved in the checkpoint. Rerun with --resume to continue.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: evaluation failed: {exc}", file=sys.stderr)
        print(f"Checkpoint (if any completed batches): {checkpoint_path}", file=sys.stderr)
        print("Resume with the same provider/model and add --resume.", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
