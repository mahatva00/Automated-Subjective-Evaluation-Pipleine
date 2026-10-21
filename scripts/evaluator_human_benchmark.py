"""Prepare blinded human labels and compare them with frozen evaluator outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.static_answer_evaluator import VALID_STATUSES, select_evidence_blocks

STATUS_ORDER = ("needs_review", "absent", "partial", "complete")
LABEL_FIELDS = (
    "case_id",
    "submission_alias",
    "question_number",
    "question_title",
    "requirement_text",
    "evidence_json",
    "human_status",
    "human_rationale",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_rows(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in evaluation.get("global_evaluations", []) or [] if isinstance(row, dict)]
    for question in evaluation.get("questions", []) or []:
        if isinstance(question, dict):
            rows.extend(
                row for row in question.get("evaluations", []) or []
                if isinstance(row, dict)
            )
    return rows


def _load_candidates(
    run_dir: Path,
    excluded_tokens: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evidence_root = run_dir / "intermediate" / "evidence_maps"
    evaluation_root = run_dir / "intermediate" / "evaluations"
    evidence_paths = sorted(evidence_root.rglob("*.evidence-map.json"))
    if not evidence_paths:
        raise ValueError(f"No evidence maps found under {evidence_root}")

    candidates: list[dict[str, Any]] = []
    source_artifacts = []
    submission_filenames = []
    for evidence_path in evidence_paths:
        if any(token.casefold() in evidence_path.name.casefold() for token in excluded_tokens):
            continue
        relative = evidence_path.relative_to(evidence_root)
        evaluation_path = evaluation_root / relative.parent / relative.name.replace(
            ".evidence-map.json", ".evaluation.json"
        )
        if not evaluation_path.is_file():
            raise ValueError(f"Evaluation file is missing for evidence map {evidence_path}")
        evidence_map = json.loads(evidence_path.read_text(encoding="utf-8"))
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evidence_map.get("student_code_executed") is not False or evaluation.get("student_code_executed") is not False:
            raise ValueError(f"Static-only guardrail is missing or failed for {evidence_path.name}")
        evidence_hash = _sha256(evidence_path)
        if evaluation.get("source_evidence_map_sha256") != evidence_hash:
            raise ValueError(f"Evaluation/evidence-map hash mismatch for {evidence_path.name}")

        requirement_context: dict[str, tuple[dict[str, Any], list[dict[str, Any]], str | None]] = {}
        all_blocks = [
            block
            for question in evidence_map.get("questions", []) or []
            for block in question.get("evidence_blocks", []) or []
            if isinstance(block, dict)
        ]
        for question in evidence_map.get("questions", []) or []:
            if not isinstance(question, dict):
                continue
            blocks = [block for block in question.get("evidence_blocks", []) or [] if isinstance(block, dict)]
            for requirement in question.get("requirements", []) or []:
                if not isinstance(requirement, dict):
                    continue
                requirement_id = str(requirement.get("requirement_id") or "")
                if not requirement_id or requirement_id in requirement_context:
                    raise ValueError(f"Missing or duplicate requirement ID in {evidence_path.name}: {requirement_id!r}")
                requirement_context[requirement_id] = (requirement, blocks, str(question.get("question_title") or ""))
        for requirement in evidence_map.get("global_requirements", []) or []:
            if not isinstance(requirement, dict):
                continue
            requirement_id = str(requirement.get("requirement_id") or "")
            if not requirement_id or requirement_id in requirement_context:
                raise ValueError(f"Missing or duplicate requirement ID in {evidence_path.name}: {requirement_id!r}")
            requirement_context[requirement_id] = (requirement, all_blocks, "")

        evaluation_rows = _evaluation_rows(evaluation)
        evaluated_ids = [str(row.get("requirement_id") or "") for row in evaluation_rows]
        if len(set(evaluated_ids)) != len(evaluated_ids) or set(evaluated_ids) != set(requirement_context):
            raise ValueError(f"Evaluation does not cover the exact requirement set in {evidence_path.name}")

        source_filename = str(
            (evidence_map.get("source_submission_filename") or evidence_path.name)
        )
        submission_filenames.append(source_filename)
        source_artifacts.append({
            "submission_filename": source_filename,
            "evidence_map": relative.as_posix(),
            "evidence_map_sha256": evidence_hash,
            "evaluation": (evaluation_root / relative.parent / relative.name.replace(
                ".evidence-map.json", ".evaluation.json"
            )).relative_to(evaluation_root).as_posix(),
            "evaluation_sha256": _sha256(evaluation_path),
        })
        for row in evaluation_rows:
            requirement_id = str(row.get("requirement_id") or "")
            requirement, blocks, question_title = requirement_context[requirement_id]
            status = str(row.get("status") or "")
            if status not in VALID_STATUSES:
                raise ValueError(f"Unknown model status {status!r} for {requirement_id}")
            selected, _ = select_evidence_blocks(requirement, blocks)
            selected_ids = [str(block.get("block_id") or "") for block in selected]
            diagnostic = row.get("evidence_diagnostic")
            recorded_ids = diagnostic.get("selected_block_ids") if isinstance(diagnostic, dict) else None
            if not isinstance(recorded_ids, list) or recorded_ids != selected_ids:
                raise ValueError(
                    f"Cannot reconstruct the exact evaluator context for {requirement_id}; "
                    "the review set would not match the frozen model input."
                )
            evidence_preview = [
                {
                    key: block[key]
                    for key in ("block_id", "component", "source_text", "output_text", "visual_evidence")
                    if key in block
                }
                for block in selected
            ]
            candidates.append({
                "submission_filename": source_filename,
                "requirement_id": requirement_id,
                "question_number": row.get("question_number") or requirement.get("question_number"),
                "question_title": question_title,
                "requirement_text": str(
                    row.get("requirement_text")
                    or requirement.get("atomic_text")
                    or requirement.get("requirement_text")
                    or requirement.get("source_text")
                    or ""
                ),
                "evidence_json": json.dumps(evidence_preview, ensure_ascii=False, separators=(",", ":")),
                "model_status": status,
                "model_confidence": row.get("confidence"),
                "model_rationale": row.get("rationale"),
                "model_evidence": row.get("evidence") or [],
                "model_validation_errors": row.get("validation_errors") or [],
                "evidence_map_sha256": evidence_hash,
            })
    if not candidates:
        raise ValueError("No eligible evaluation cases remain after applying submission exclusions.")
    aliases = {
        filename: f"SUB-{index:02d}"
        for index, filename in enumerate(sorted(set(submission_filenames)), start=1)
    }
    for candidate in candidates:
        candidate["submission_alias"] = aliases[candidate["submission_filename"]]
    return candidates, {
        "source_artifacts": source_artifacts,
        "submission_aliases": {alias: filename for filename, alias in aliases.items()},
    }


def _stratified_sample(
    candidates: list[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    randomizer = random.Random(seed)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    aliases = sorted({str(row["submission_alias"]) for row in candidates})
    for candidate in candidates:
        grouped[str(candidate["model_status"])][str(candidate["submission_alias"])].append(candidate)
    for by_submission in grouped.values():
        for rows in by_submission.values():
            randomizer.shuffle(rows)
    randomizer.shuffle(aliases)

    sampled = []
    while len(sampled) < min(sample_size, len(candidates)):
        made_progress = False
        for status in STATUS_ORDER:
            for alias in aliases:
                rows = grouped.get(status, {}).get(alias, [])
                if rows:
                    sampled.append(rows.pop())
                    made_progress = True
                    if len(sampled) == sample_size:
                        break
            if len(sampled) == sample_size:
                break
        if not made_progress:
            break
    return sampled


def prepare_review_files(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    sample_size: int = 30,
    seed: int = 42,
    exclude_submission_tokens: list[str] | None = None,
) -> tuple[Path, Path]:
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    source = Path(run_dir).resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Run directory does not exist: {source}")
    excluded = list(exclude_submission_tokens or [])
    candidates, source_metadata = _load_candidates(source, excluded)
    sample = _stratified_sample(candidates, sample_size, seed)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    labels_path = destination / "evaluator-human-review.tsv"
    key_path = destination / "evaluator-human-review-key.json"
    key_cases = {}
    with labels_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LABEL_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for index, candidate in enumerate(sample, start=1):
            case_id = f"CASE-{index:04d}"
            writer.writerow({
                "case_id": case_id,
                "submission_alias": candidate["submission_alias"],
                "question_number": candidate["question_number"] or "",
                "question_title": candidate["question_title"],
                "requirement_text": candidate["requirement_text"],
                "evidence_json": candidate["evidence_json"],
                "human_status": "",
                "human_rationale": "",
            })
            key_cases[case_id] = {
                key: candidate[key]
                for key in (
                    "submission_filename", "submission_alias", "question_number",
                    "requirement_id", "model_status", "model_confidence",
                    "model_rationale", "model_evidence", "model_validation_errors",
                    "evidence_map_sha256",
                )
            }
    key_data = {
        "schema_version": "1.0",
        "artifact_type": "blinded_evaluator_human_benchmark_key",
        "source_run_dir": str(source),
        "sample_size_requested": sample_size,
        "sample_size_written": len(sample),
        "sampling_seed": seed,
        "excluded_submission_tokens": excluded,
        "source_artifacts": source_metadata["source_artifacts"],
        "submission_aliases": source_metadata["submission_aliases"],
        "cases": key_cases,
        "label_instructions": (
            "Complete human_status with complete, partial, absent, or needs_review and add a concise "
            "human_rationale. Label against the requirement text and frozen evidence only; do not copy "
            "the evaluator's answer. Keep this key private until labels are complete to preserve blinding."
        ),
    }
    key_path.write_text(json.dumps(key_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return labels_path, key_path


def compare_labels(
    labels_path: str | Path,
    key_path: str | Path,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    labels_file = Path(labels_path)
    key_file = Path(key_path)
    key_data = json.loads(key_file.read_text(encoding="utf-8"))
    if key_data.get("artifact_type") != "blinded_evaluator_human_benchmark_key":
        raise ValueError("The key file is not a supported evaluator benchmark key.")
    cases = key_data.get("cases")
    if not isinstance(cases, dict):
        raise ValueError("Benchmark key has no cases object.")
    rows = []
    seen_ids = set()
    with labels_file.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None or not set(LABEL_FIELDS).issubset(reader.fieldnames):
            raise ValueError("Review file is missing required label columns.")
        for row in reader:
            case_id = str(row.get("case_id") or "")
            if not case_id or case_id in seen_ids:
                raise ValueError(f"Review file contains a missing or duplicate case_id: {case_id!r}")
            seen_ids.add(case_id)
            if case_id not in cases:
                raise ValueError(f"Review file case {case_id} is not present in the benchmark key.")
            human_status = str(row.get("human_status") or "").strip()
            if not human_status:
                continue
            if human_status not in VALID_STATUSES:
                raise ValueError(f"Unknown human status {human_status!r} for {case_id}")
            rows.append((case_id, human_status, cases[case_id], str(row.get("human_rationale") or "")))
    if not rows:
        raise ValueError("No human_status labels have been completed.")

    confusion = {human: {model: 0 for model in sorted(VALID_STATUSES)} for human in sorted(VALID_STATUSES)}
    support = Counter()
    correct = 0
    disagreements = []
    for case_id, human_status, machine, human_rationale in rows:
        model_status = str(machine.get("model_status") or "")
        if model_status not in VALID_STATUSES:
            raise ValueError(f"Unknown model status {model_status!r} for {case_id}")
        confusion[human_status][model_status] += 1
        support[human_status] += 1
        if model_status == human_status:
            correct += 1
        else:
            disagreements.append({
                "case_id": case_id,
                "submission_alias": machine.get("submission_alias"),
                "question_number": machine.get("question_number"),
                "requirement_id": machine.get("requirement_id"),
                "model_status": model_status,
                "human_status": human_status,
                "model_rationale": machine.get("model_rationale"),
                "human_rationale": human_rationale,
            })
    per_status = {}
    f1_values = []
    for status in sorted(VALID_STATUSES):
        true_positive = confusion[status][status]
        false_positive = sum(confusion[actual][status] for actual in confusion if actual != status)
        false_negative = sum(confusion[status][predicted] for predicted in confusion if predicted != status)
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else None
        recall = true_positive / recall_denominator if recall_denominator else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else (0.0 if support[status] else None)
        )
        if f1 is not None:
            f1_values.append(f1)
        per_status[status] = {
            "support": support[status],
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
        }
    report = {
        "schema_version": "1.0",
        "artifact_type": "evaluator_human_agreement_report",
        "source_run_dir": key_data.get("source_run_dir"),
        "labeled_cases": len(rows),
        "exact_agreement": round(correct / len(rows), 4),
        "macro_f1_over_human_observed_statuses": round(sum(f1_values) / len(f1_values), 4) if f1_values else None,
        "per_status": per_status,
        "confusion_matrix": confusion,
        "disagreements": disagreements,
        "note": "This small human-reviewed sample is a diagnostic, not a certification of evaluator accuracy.",
    }
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create a blinded TSV label sheet and a separate evaluator key.")
    prepare.add_argument("--run-dir", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--sample-size", type=int, default=30)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--exclude-submission-token", action="append", default=[])
    compare = commands.add_parser("compare", help="Compare completed human labels with the evaluator key.")
    compare.add_argument("--labels-tsv", required=True)
    compare.add_argument("--key-json", required=True)
    compare.add_argument("--report-json", default=None)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            labels, key = prepare_review_files(
                args.run_dir,
                args.output_dir,
                sample_size=args.sample_size,
                seed=args.seed,
                exclude_submission_tokens=args.exclude_submission_token,
            )
            print(f"Blinded label sheet: {labels}")
            print(f"Separate evaluator key (do not open until labels are complete): {key}")
        else:
            report_path = args.report_json or str(Path(args.labels_tsv).with_suffix(".agreement.json"))
            report = compare_labels(args.labels_tsv, args.key_json, report_path)
            print(f"Labeled cases: {report['labeled_cases']}")
            print(f"Exact agreement: {report['exact_agreement']:.1%}")
            print(f"Agreement report: {report_path}")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
