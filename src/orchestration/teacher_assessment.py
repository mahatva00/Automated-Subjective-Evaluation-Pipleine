"""One-folder, static-only teacher assessment runner built on the verified modules."""
from __future__ import annotations
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.ingestion import pdf_parser, notebook_parser, question_parser
from src.requirements.requirement_extractor import extract_atomic_requirements
from src.evaluation.evidence_mapper import build_evidence_map
from src.evaluation.static_answer_evaluator import (
    GeminiClient,
    GroqClient,
    OllamaClient,
    ProviderRateLimitError,
    evaluate_evidence_map,
)
from src.policy.policy_parser import parse_assessment_policy
from src.evaluation.grading_engine import score_evaluation, score_question_statuses
from src.reporting.student_report import build_student_report, evaluation_rows, write_student_report
from scripts.audit_static_evaluations import audit as audit_static_evaluation

PDF_SUFFIX = ".pdf"
STUDENT_SUFFIXES = {".ipynb", ".pdf"}
POLICY_SUFFIXES = {".json", ".txt", ".md", ".markdown", ".pdf", ".docx"}
QP_NAME_RE = re.compile(r"\b(?:qp|question[\s_-]*paper|exam[\s_-]*paper|problem[\s_-]*statement)\b", re.I)
POLICY_NAME_RE = re.compile(r"\b(?:policy|rubric|marking(?:[\s_-]*(?:scheme|criteria|policy))?|grading(?:[\s_-]*(?:scheme|criteria|policy))?|assessment[\s_-]*policy)\b", re.I)
STUDENT_PATH_RE = re.compile(r"\b(?:student|submission|submissions|learner|answers?)\b", re.I)
POLICY_TEXT_RE = re.compile(r"\b(?:assessment policy|marking scheme|grading criteria|partial credit|strict marking|lenient marking|rubric|credit factors?)\b", re.I)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _path_text(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return " ".join([str(rel), *[str(part) for part in rel.parts]])


def _read_text_candidate(path: Path, parsed_pdf: dict[str, Any] | None = None) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            data = parsed_pdf or pdf_parser.parse_pdf(path.resolve(), path.resolve().parent)
            return "\n".join(str(s.get("content") or "") for s in data.get("segments", []) if isinstance(s, dict))
        if suffix in {".txt", ".md", ".markdown"}:
            return path.read_text(encoding="utf-8-sig")
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            return json.dumps(data, ensure_ascii=False)
        if suffix == ".docx":
            from docx import Document
            doc = Document(str(path))
            parts = [p.text for p in doc.paragraphs]
            for table in doc.tables:
                parts.extend(" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows)
            return "\n".join(parts)
    except Exception:
        return ""
    return ""


def _qp_content_score(parsed_pdf: dict[str, Any]) -> int:
    try:
        qm = question_parser.parse_question_paper(parsed_pdf)
    except Exception:
        return 0
    questions = len(qm.get("questions", []) or [])
    standalone = len(qm.get("standalone_tasks", []) or [])
    text = "\n".join(str(s.get("content") or "") for s in parsed_pdf.get("segments", []) if isinstance(s, dict))
    marks = len(re.findall(r"\[\s*\d+\s*marks?\s*\]", text, re.I))
    return questions * 8 + standalone * 6 + marks * 2


def _policy_content_score(text: str) -> int:
    return sum(2 for _ in POLICY_TEXT_RE.finditer(text))


def _select_candidate(candidates: list[tuple[int, Path]], role: str) -> Path | None:
    eligible = [(score, path) for score, path in candidates if score > 0]
    if not eligible:
        return None
    top_score = max(score for score, _ in eligible)
    winners = [path for score, path in eligible if score == top_score]
    if len(winners) > 1:
        names = ", ".join(str(p) for p in winners[:8])
        raise ValueError(f"Multiple equally plausible {role} files were found: {names}. Put the file in a clearly named folder or use the explicit override option.")
    return winners[0]


def discover_input_files(input_dir: Path, question_paper: Path | None = None, policy_file: Path | None = None,
                         pdf_cache: dict[Path, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Discover QP/policy/submissions; ambiguous roles produce an actionable error."""
    root = input_dir.resolve()
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in STUDENT_SUFFIXES | POLICY_SUFFIXES)
    if pdf_cache is None:
        pdf_cache = {}
    for path in files:
        if path.suffix.lower() == ".pdf" and path.resolve() not in pdf_cache:
            try:
                pdf_cache[path.resolve()] = pdf_parser.parse_pdf(path.resolve(), root)
            except Exception as exc:
                pdf_cache[path.resolve()] = {"submission": {"source_filename": path.name, "extraction_status": "failed"}, "segments": [], "diagnostics": [f"PDF parser failed: {type(exc).__name__}: {exc}"]}

    q_pdf_override = question_paper.resolve() if question_paper else None
    policy_override = policy_file.resolve() if policy_file else None
    if q_pdf_override and (not q_pdf_override.is_file() or q_pdf_override.suffix.lower() != ".pdf"):
        raise ValueError(f"--question-paper must be an existing PDF: {q_pdf_override}")
    if q_pdf_override and q_pdf_override != root and root not in q_pdf_override.parents:
        raise ValueError("--question-paper must be inside --input-dir so the run remains self-contained.")
    if policy_override and (not policy_override.is_file() or policy_override.suffix.lower() not in POLICY_SUFFIXES):
        raise ValueError(f"--policy-file must be an existing JSON/TXT/MD/PDF/DOCX file: {policy_override}")
    if policy_override and policy_override != root and root not in policy_override.parents:
        raise ValueError("--policy-file must be inside --input-dir so the run remains self-contained.")

    q_candidates = []
    p_candidates = []
    for path in files:
        role_text = _path_text(path, root)
        if path.suffix.lower() == ".pdf":
            parsed = pdf_cache[path.resolve()]
            qp_score = _qp_content_score(parsed)
        else:
            qp_score = 0
        has_policy_name = bool(POLICY_NAME_RE.search(role_text))
        has_qp_name = bool(QP_NAME_RE.search(role_text))
        is_student_path = bool(STUDENT_PATH_RE.search(role_text))
        text = _read_text_candidate(path, pdf_cache.get(path.resolve()))
        p_content_score = _policy_content_score(text)
        if path.suffix.lower() == ".pdf" and not has_policy_name and not is_student_path:
            if has_qp_name:
                qp_score += 100
            if p_content_score and not has_qp_name:
                qp_score -= p_content_score * 2
            q_candidates.append((max(qp_score, 0), path.resolve()))
        if path.suffix.lower() in POLICY_SUFFIXES and (has_policy_name or p_content_score > 0):
            score = (100 if has_policy_name else 0) + p_content_score
            p_candidates.append((score, path.resolve()))

    qp = q_pdf_override or _select_candidate(q_candidates, "question paper")
    policy = policy_override or _select_candidate(p_candidates, "assessment policy")
    if qp is None:
        raise ValueError("Could not identify a question-paper PDF. Name it with 'QP'/'question paper' or place it in a 'Question Paper' folder.")
    if policy == qp:
        raise ValueError("The selected question paper is also being identified as the assessment policy; use --policy-file or rename/relocate the policy.")
    if policy is not None and policy.suffix.lower() == ".pdf" and bool(STUDENT_PATH_RE.search(_path_text(policy, root))):
        raise ValueError("A student-submission PDF was selected as the assessment policy; move the policy into a clearly named policy file/folder.")

    submissions = []
    for path in files:
        resolved = path.resolve()
        if resolved == qp or (policy is not None and resolved == policy):
            continue
        if path.suffix.lower() not in STUDENT_SUFFIXES:
            continue
        if path.suffix.lower() == ".pdf":
            text = _read_text_candidate(path, pdf_cache.get(resolved))
            # Exclude a likely policy PDF even when it had no filename cue, rather than grading it as a student.
            if _policy_content_score(text) >= 2 and not STUDENT_PATH_RE.search(_path_text(path, root)):
                continue
        submissions.append(resolved)
    if not submissions:
        raise ValueError("No student submissions (.ipynb or PDF) were found after excluding the question paper and policy.")
    return {"question_paper": qp, "policy_file": policy, "submissions": submissions, "all_files": files,
            "pdf_cache": pdf_cache, "policy_missing": policy is None}


def _default_client(provider: str, model: str | None, base_url: str, timeout: int):
    defaults = {"ollama": "qwen2.5:3b-instruct", "groq": "openai/gpt-oss-20b", "gemini": "gemini-3.8-flash"}
    model_name = model or defaults[provider]
    if provider == "ollama":
        return OllamaClient(base_url, model_name, timeout)
    if provider == "groq":
        return GroqClient(model=model_name, timeout=timeout)
    if provider == "gemini":
        return GeminiClient(model=model_name, timeout=timeout)
    raise ValueError(f"Unsupported provider: {provider}")


def _relative_output_name(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


def _load_checkpoint(path: Path, source_hash: str, provider: str, model: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Evaluation checkpoint is unreadable: {path}: {exc}") from exc
    if data.get("source_sha256") != source_hash:
        raise ValueError(f"Evaluation checkpoint source hash differs from the current evidence map: {path}. Use a new run instead of resuming.")
    if data.get("provider") != provider or data.get("model") != model:
        raise ValueError(f"Evaluation checkpoint provider/model differs from this run: {path}. Use the same provider/model or start a new run.")
    rows = data.get("completed_results", [])
    if not isinstance(rows, list):
        raise ValueError(f"Evaluation checkpoint results are malformed: {path}")
    return rows


def run_teacher_assessment(input_dir: str | Path, output_dir: str | Path | None = None, *,
                           question_paper: str | Path | None = None, policy_file: str | Path | None = None,
                           provider: str = "ollama", model: str | None = None,
                           base_url: str = "http://localhost:11434", timeout: int = 180,
                           batch_size: int = 4, max_evidence_blocks: int = 4,
                           max_evidence_chars: int = 2800, model_dir: str | Path | None = None,
                           min_confidence: float = 0.55, resume_run_dir: str | Path | None = None,
                           client_factory: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Run a fresh one-folder assessment. Model roles remain advisory and never remove requirements."""
    root = Path(input_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {root}")
    base_out = Path(output_dir).resolve() if output_dir else root.parent / f"{root.name}_assessment_output"
    if base_out == root or root in base_out.parents:
        raise ValueError("Output directory must be outside the teacher input directory.")
    if provider not in {"ollama", "groq", "gemini"}:
        raise ValueError("provider must be one of ollama, groq, gemini")
    if batch_size < 1 or timeout < 1:
        raise ValueError("batch_size and timeout must be positive")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")
    resume_manifest = None
    if resume_run_dir is not None:
        run_dir = Path(resume_run_dir).resolve()
        manifest_path = run_dir / "RUN_MANIFEST.json"
        if not run_dir.is_dir() or not manifest_path.is_file():
            raise ValueError(f"--resume-run must point to an existing run directory with RUN_MANIFEST.json: {run_dir}")
        try:
            resume_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Cannot read resume manifest {manifest_path}: {exc}") from exc
        if Path(str(resume_manifest.get("input_dir", ""))).resolve() != root:
            raise ValueError("Resume run was created for a different input directory; use a new run.")
        run_id = run_dir.name
    else:
        run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_out / run_id
        suffix = 2
        while run_dir.exists():
            run_dir = base_out / f"{run_id}_{suffix:02d}"
            suffix += 1
        run_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "intermediate").mkdir(exist_ok=True)
    stages: list[dict[str, Any]] = []
    warnings: list[str] = []

    try:
        discovered = discover_input_files(root,
            Path(question_paper).resolve() if question_paper else None,
            Path(policy_file).resolve() if policy_file else None)
    except Exception as exc:
        discovery_report = {"schema_version": "1.0", "artifact_type": "teacher_assessment_run", "run_id": run_id,
                            "input_dir": str(root), "pipeline_status": "discovery_failed", "error": f"{type(exc).__name__}: {exc}",
                            "student_code_executed": False}
        write_json(run_dir / "RUN_REPORT.json", discovery_report)
        raise
    pdf_cache = discovered["pdf_cache"]
    input_files_snapshot = []
    for input_path in discovered["all_files"]:
        input_files_snapshot.append({"path": str(input_path.relative_to(root)), "sha256": sha256_file(input_path), "suffix": input_path.suffix.lower()})
    if resume_manifest is not None:
        prior_snapshot = resume_manifest.get("input_files")
        if not isinstance(prior_snapshot, list) or sorted((x.get("path"), x.get("sha256")) for x in prior_snapshot) != sorted((x.get("path"), x.get("sha256")) for x in input_files_snapshot):
            raise ValueError("Teacher input files changed since this run was created; resume refused to prevent stale assessment reuse.")
    write_json(run_dir / "RUN_MANIFEST.json", {"run_id": run_id, "input_dir": str(root), "run_dir": str(run_dir),
        "input_files": input_files_snapshot, "provider": provider, "model": model,
        "distilbert_model_dir": str(Path(model_dir).resolve()) if model_dir else os.environ.get("INE_DISTILBERT_MODEL_DIR"),
        "student_code_executed": False, "stage_outputs": stages})
    qp_path: Path = discovered["question_paper"]
    policy_path: Path | None = discovered["policy_file"]
    qp_payload = pdf_cache.get(qp_path.resolve()) or pdf_parser.parse_pdf(qp_path, root)
    parsed_qp_out = run_dir / "intermediate" / "parsed_question_paper.json"
    write_json(parsed_qp_out, qp_payload)
    stages.append({"stage": "question_paper_ingestion", "status": "completed", "output": str(parsed_qp_out)})
    question_map = question_parser.parse_question_paper(qp_payload)
    if not (question_map.get("questions") or question_map.get("standalone_tasks")):
        failure = {"schema_version": "1.0", "artifact_type": "teacher_assessment_run", "run_id": run_id,
                   "input_dir": str(root), "run_dir": str(run_dir), "pipeline_status": "question_paper_parse_failed",
                   "discovered": {"question_paper": str(qp_path), "assessment_policy": str(policy_path) if policy_path else None,
                                  "student_submissions": len(discovered["submissions"])},
                   "parser_diagnostics": question_map.get("diagnostics", []),
                   "warnings": ["No questions or standalone tasks were recognized. Inspect the source PDF or parser diagnostics."],
                   "student_code_executed": False}
        write_json(run_dir / "RUN_REPORT.json", failure)
        raise ValueError(f"Question parser recognized no questions or standalone tasks. Diagnostic report: {run_dir / 'RUN_REPORT.json'}")
    qmap_out = run_dir / "intermediate" / "question_map.json"
    write_json(qmap_out, question_map)
    # Do not turn question-paper preamble instructions into assessable requirements.
    requirements = extract_atomic_requirements({**question_map, "global_instructions": []})
    req_out = run_dir / "intermediate" / "requirements_base" / f"{qp_path.stem}.requirements.json"
    write_json(req_out, requirements)
    stages.append({"stage": "question_mapping_and_requirement_extraction", "status": "completed",
                   "question_count": len(question_map.get("questions", []) or []) + len(question_map.get("standalone_tasks", []) or []),
                   "requirement_count": len(requirements.get("requirements", []) or []), "output": str(req_out)})

    if policy_path:
        policy_payload = pdf_cache.get(policy_path.resolve()) if policy_path.suffix.lower() == ".pdf" else None
        try:
            policy = parse_assessment_policy(policy_path, root, parsed_pdf=policy_payload)
            stages.append({"stage": "assessment_policy_parsing", "status": "completed", "output_format": policy.get("source_format"),
                           "numeric_scoring_enabled": policy.get("numeric_scoring_enabled")})
        except Exception as exc:
            warnings.append(f"Assessment policy could not be normalized; numeric scoring is disabled: {type(exc).__name__}: {exc}")
            policy = parse_assessment_policy(None)
            policy.update({"source_filename": policy_path.name, "source_path": str(policy_path),
                           "source_sha256": sha256_file(policy_path), "source_format": policy_path.suffix.lower().lstrip("."),
                           "warnings": [f"Policy normalization failed: {type(exc).__name__}: {exc}"]})
            stages.append({"stage": "assessment_policy_parsing", "status": "warning", "reason": "normalization_failed"})
    else:
        policy = parse_assessment_policy(None)
        warnings.append("Assessment policy was not detected; requirement evaluation will continue, but numeric grading is disabled.")
        stages.append({"stage": "assessment_policy_parsing", "status": "warning", "reason": "policy_not_found"})
    policy_out = run_dir / "intermediate" / "normalized_assessment_policy.json"
    write_json(policy_out, policy)

    mapping_requirements = requirements
    used_model_dir = Path(model_dir).resolve() if model_dir else (Path(os.environ["INE_DISTILBERT_MODEL_DIR"]).resolve() if os.environ.get("INE_DISTILBERT_MODEL_DIR") else None)
    model_result: dict[str, Any] = {"attempted": False, "completed": False, "mode": "advisory_only"}
    if used_model_dir is not None:
        classifier_script = Path(__file__).resolve().parents[2] / "scripts" / "classify_requirement_candidates.py"
        if not classifier_script.is_file():
            warnings.append("DistilBERT model path was supplied but the classifier script is absent; baseline requirements are used.")
            model_result["reason"] = "classifier_script_missing"
        elif not (used_model_dir / "config.json").is_file():
            warnings.append(f"DistilBERT checkpoint does not exist at {used_model_dir}; baseline requirements are used.")
            model_result["reason"] = "checkpoint_missing"
        else:
            classify_dir = run_dir / "intermediate" / "requirements_classified"
            cmd = [sys.executable, str(classifier_script), "--model-dir", str(used_model_dir),
                   "--input-dir", str(req_out.parent), "--output-dir", str(classify_dir),
                   "--pattern", "*.requirements.json", "--min-confidence", str(min_confidence)]
            proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[2]), text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            classify_dir.mkdir(parents=True, exist_ok=True)
            inference_log = classify_dir / "inference.log"
            inference_log.write_text(proc.stdout, encoding="utf-8")
            model_result.update({"attempted": True, "return_code": proc.returncode, "checkpoint_dir": str(used_model_dir),
                                 "command": cmd, "log": str(inference_log), "stdout_tail": proc.stdout[-5000:]})
            classified_path = classify_dir / req_out.name
            if proc.returncode == 0 and classified_path.is_file():
                mapping_requirements = json.loads(classified_path.read_text(encoding="utf-8"))
                model_result.update({"completed": True, "report": str(classify_dir / "mvp_classification_report.json"),
                                     "output": str(classified_path),
                                     "use": "advisory metadata only; candidate set and atomic boundaries are unchanged"})
                stages.append({"stage": "distilbert_candidate_classification", "status": "completed", "output": str(classified_path)})
            else:
                warnings.append("DistilBERT candidate inference failed; the baseline requirements were retained.")
                model_result["reason"] = "inference_failed"
                stages.append({"stage": "distilbert_candidate_classification", "status": "warning", "return_code": proc.returncode})
    else:
        model_result["reason"] = "no_checkpoint_configured"
        model_result["note"] = "Set INE_DISTILBERT_MODEL_DIR or pass --model-dir to add advisory candidate roles; assessment does not train the model per assignment."
        stages.append({"stage": "distilbert_candidate_classification", "status": "skipped", "reason": "no_checkpoint_configured"})

    output_reqs_dir = run_dir / "intermediate" / "requirements_for_mapping"
    output_reqs_dir.mkdir(parents=True, exist_ok=True)
    req_for_mapping_path = output_reqs_dir / req_out.name
    write_json(req_for_mapping_path, mapping_requirements)

    try:
        client = client_factory(provider=provider, model=model, base_url=base_url, timeout=timeout) if client_factory else _default_client(provider, model, base_url, timeout)
        client_model = getattr(client, "model", model or provider)
    except Exception as exc:
        client = None
        client_model = model or provider
        warnings.append(f"Could not initialize evaluation provider: {type(exc).__name__}: {exc}")

    submission_results = []
    provider_rate_limit_pause: str | None = None
    for index, submission_path in enumerate(discovered["submissions"], start=1):
        relative = _relative_output_name(submission_path, root)
        rel_stem = relative.with_suffix("").as_posix()
        report_id = re.sub(r"[^A-Za-z0-9._-]+", "_", rel_stem).strip("._") or f"student_{index:04d}"
        try:
            if submission_path.suffix.lower() == ".ipynb":
                parsed = notebook_parser.parse_notebook(submission_path, root)
            else:
                parsed = pdf_cache.get(submission_path.resolve()) or pdf_parser.parse_pdf(submission_path, root)
        except Exception as exc:
            error_warning = f"Submission parser failed: {type(exc).__name__}: {exc}"
            report_meta = {"report_id": report_id, "display_name": submission_path.stem,
                           "source_filename": submission_path.name, "source_relative_path": relative.as_posix(),
                           "source_sha256": sha256_file(submission_path)}
            report = build_student_report(report_meta, None, None, policy, None, [error_warning])
            jpath, hpath = write_student_report(run_dir / "reports", report)
            submission_results.append({"student": report_meta, "status": "extraction_failed", "report_json": str(jpath), "report_html": str(hpath), "error": error_warning})
            stages.append({"stage": "student_ingestion", "submission": submission_path.name, "status": "failed", "error": error_warning})
            continue
        parsed_path = run_dir / "intermediate" / "parsed_submissions" / relative.parent / (relative.stem + ".parsed.json")
        write_json(parsed_path, parsed)
        sub_meta = dict(parsed.get("submission", {}) or {})
        sub_meta.update({"report_id": report_id, "display_name": sub_meta.get("student_roll_no") or submission_path.stem,
                         "source_relative_path": relative.as_posix(), "source_filename": submission_path.name,
                         "source_sha256": sha256_file(submission_path)})
        per_student_warnings = list(parsed.get("diagnostics", []) or [])
        extraction_status = sub_meta.get("extraction_status", "unknown")
        if extraction_status == "failed" or not parsed.get("segments"):
            per_student_warnings.append("Submission extraction failed or produced no segments; evaluation was not attempted.")
            report = build_student_report(sub_meta, None, None, policy, None, per_student_warnings)
            jpath, hpath = write_student_report(run_dir / "reports", report)
            submission_results.append({"student": sub_meta, "status": "extraction_failed", "report_json": str(jpath), "report_html": str(hpath)})
            continue
        try:
            evidence_map = build_evidence_map(question_map, mapping_requirements, parsed, submission_path.name)
        except Exception as exc:
            error_warning = f"Evidence mapping failed: {type(exc).__name__}: {exc}"
            per_student_warnings.append(error_warning)
            report = build_student_report(sub_meta, None, None, policy, None, per_student_warnings)
            jpath, hpath = write_student_report(run_dir / "reports", report)
            submission_results.append({"student": sub_meta, "status": "mapping_failed", "report_json": str(jpath), "report_html": str(hpath), "error": error_warning})
            stages.append({"stage": "evidence_mapping", "submission": submission_path.name, "status": "failed", "error": error_warning})
            continue
        evidence_path = run_dir / "intermediate" / "evidence_maps" / relative.parent / (relative.stem + ".evidence-map.json")
        write_json(evidence_path, evidence_map)
        stages.append({"stage": "evidence_mapping", "submission": submission_path.name, "status": "completed",
                       "mapped_blocks": evidence_map.get("summary", {}).get("mapped_blocks"),
                       "unassigned_blocks": evidence_map.get("summary", {}).get("unassigned_blocks"), "output": str(evidence_path)})
        eval_path = run_dir / "intermediate" / "evaluations" / relative.parent / (relative.stem + ".evaluation.json")
        checkpoint_path = eval_path.with_suffix(eval_path.suffix + ".checkpoint.json")
        log_path = run_dir / "logs" / relative.parent / (relative.stem + ".evaluation.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation = None
        evaluation_error = None
        if client is None:
            evaluation_error = "Evaluation provider client could not be initialized."
        elif provider_rate_limit_pause is not None:
            evaluation_error = (
                "Provider requests were paused after an earlier rate-limit failure: "
                + provider_rate_limit_pause
            )
            per_student_warnings.append(evaluation_error)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"SKIPPED: {evaluation_error}\n")
            stages.append({
                "stage": "static_answer_evaluation",
                "submission": submission_path.name,
                "status": "skipped_rate_limited",
                "error": evaluation_error,
                "log": str(log_path),
            })
        else:
            source_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
            checkpoint_model = str(client_model)
            completed = _load_checkpoint(checkpoint_path, source_hash, provider, checkpoint_model)
            def save_progress(rows: list[dict[str, Any]], total: int) -> None:
                write_json(checkpoint_path, {"checkpoint_version": 1, "source_evidence_map_path": str(evidence_path),
                    "source_sha256": source_hash, "provider": provider, "model": checkpoint_model,
                    "completed_requirements": len(rows), "total_requirements": total, "completed_results": rows})
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"checkpoint: {len(rows)}/{total} requirement assessments saved\n")
            def before_request(batch_count: int) -> None:
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"provider request: batch_size={batch_count}; model={checkpoint_model}; provider={provider}\n")
            try:
                evaluation = evaluate_evidence_map(evidence_map, client, batch_size=batch_size,
                    max_evidence_blocks=max_evidence_blocks, max_evidence_chars=max_evidence_chars,
                    completed_results=completed, progress_callback=save_progress,
                    before_request_callback=before_request)
                evaluation.update({"source_evidence_map_path": str(evidence_path), "source_evidence_map_sha256": source_hash,
                                   "provider": provider, "checkpoint_path": str(checkpoint_path)})
                write_json(eval_path, evaluation)
                stages.append({"stage": "static_answer_evaluation", "submission": submission_path.name, "status": "completed",
                               "summary": evaluation.get("summary"), "output": str(eval_path), "checkpoint": str(checkpoint_path), "log": str(log_path)})
            except ProviderRateLimitError as exc:
                provider_rate_limit_pause = str(exc)
                evaluation_error = f"{type(exc).__name__}: {exc}"
                rate_limit_warning = (
                    "Provider evaluation paused after rate-limit retries were exhausted. "
                    "Completed requirements remain checkpointed; resume the run later."
                )
                warnings.append(rate_limit_warning)
                per_student_warnings.append(rate_limit_warning)
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"RATE_LIMIT_PAUSE: {evaluation_error}\n")
                stages.append({
                    "stage": "static_answer_evaluation",
                    "submission": submission_path.name,
                    "status": "paused_rate_limited",
                    "error": evaluation_error,
                    "checkpoint": str(checkpoint_path) if checkpoint_path.exists() else None,
                    "log": str(log_path),
                })
            except Exception as exc:
                evaluation_error = f"{type(exc).__name__}: {exc}"
                per_student_warnings.append(f"Static answer evaluation failed: {evaluation_error}")
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"ERROR: {evaluation_error}\n")
                stages.append({"stage": "static_answer_evaluation", "submission": submission_path.name, "status": "failed", "error": evaluation_error,
                               "checkpoint": str(checkpoint_path) if checkpoint_path.exists() else None, "log": str(log_path)})
        scoring = None
        if evaluation is not None:
            audit_result = audit_static_evaluation(evidence_map, evaluation)
            audit_path = run_dir / "intermediate" / "evaluation_audits" / relative.parent / (relative.stem + ".evaluation-audit.json")
            write_json(audit_path, audit_result)
            stages.append({"stage": "static_evaluation_audit", "submission": submission_path.name,
                           "status": "passed" if audit_result.get("passed") else "failed",
                           "error_count": len(audit_result.get("errors", [])), "output": str(audit_path)})
            if audit_result.get("passed"):
                if policy.get("component_scoring_enabled"):
                    scoring = score_question_statuses(evaluation_rows(evaluation), policy)
                else:
                    scoring = score_evaluation(evaluation, mapping_requirements, policy)
            else:
                audit_message = f"Independent evaluation audit failed for {submission_path.name}; numeric scoring was skipped."
                warnings.append(audit_message)
                per_student_warnings.append(audit_message)
                scoring = {"artifact_type": "policy_scoring_result", "numeric_scoring_enabled": False,
                           "score_status": "blocked_by_evaluation_audit", "score": None, "max_marks": None,
                           "percentage": None, "final_grade": None, "teacher_approval_required": True,
                           "notes": list(audit_result.get("errors", []))}
        if scoring is not None:
            score_path = run_dir / "intermediate" / "scores" / relative.parent / (relative.stem + ".score.json")
            write_json(score_path, scoring)
            stages.append({"stage": "policy_scoring", "submission": submission_path.name,
                           "status": "completed" if scoring.get("numeric_scoring_enabled") else "warning",
                           "score_status": scoring.get("score_status"), "output": str(score_path)})
        report = build_student_report(sub_meta, evaluation, scoring, policy, evidence_map, per_student_warnings)
        jpath, hpath = write_student_report(run_dir / "reports", report)
        stages.append({"stage": "student_report_generation", "submission": submission_path.name, "status": "completed",
                       "json_report": str(jpath), "html_report": str(hpath)})
        submission_results.append({"student": sub_meta, "status": "completed" if evaluation is not None else "evaluation_failed",
                                   "evidence_map": str(evidence_path), "evaluation": str(eval_path) if evaluation is not None else None,
                                   "score_status": (scoring or {}).get("score_status", "not_scored"),
                                   "score": (scoring or {}).get("score"),
                                   "max_marks": (scoring or {}).get("max_marks"),
                                   "percentage": (scoring or {}).get("percentage"),
                                   "report_json": str(jpath), "report_html": str(hpath), "error": evaluation_error})

    input_files = input_files_snapshot
    completed_count = sum(x["status"] == "completed" for x in submission_results)
    failed_count = len(submission_results) - completed_count
    all_warnings = warnings + list(policy.get("warnings", []) or [])
    if any(s.get("status") in {"failed", "warning"} for s in stages):
        all_warnings.append("One or more processing stages reported warnings or failures; inspect the stage list.")
    batch_report = {
        "schema_version": "1.0", "artifact_type": "teacher_assessment_run", "run_id": run_id,
        "input_dir": str(root), "run_dir": str(run_dir), "pipeline_status": "completed_with_warnings" if all_warnings or failed_count else "completed",
        "discovered": {"question_paper": str(qp_path), "assessment_policy": str(policy_path) if policy_path else None,
                       "student_submissions": len(discovered["submissions"])},
        "stages": stages, "distilbert": model_result, "student_results": submission_results,
        "summary": {"students_discovered": len(submission_results), "students_assessed": completed_count,
                    "students_failed_or_partial": failed_count, "student_code_executed": False,
                    "numeric_scoring_enabled_by_policy": bool(policy.get("numeric_scoring_enabled"))},
        "warnings": all_warnings, "input_files": input_files,
        "guardrails": {"student_code_executed": False, "original_input_files_modified": False,
                       "distilbert_predictions_affect_requirement_inclusion": False, "official_grades_issued": False},
    }
    write_json(run_dir / "RUN_REPORT.json", batch_report)
    write_json(run_dir / "RUN_MANIFEST.json", {"run_id": run_id, "input_dir": str(root), "run_dir": str(run_dir),
        "input_files": input_files, "provider": provider, "model": client_model,
        "distilbert_model_dir": str(used_model_dir) if used_model_dir else None,
        "student_code_executed": False, "stage_outputs": stages})
    _write_batch_html(run_dir / "ASSESSMENT_SUMMARY.html", batch_report)
    return batch_report


def _write_batch_html(path: Path, report: dict[str, Any]) -> None:
    import html
    entries = []
    run_dir = Path(report.get("run_dir") or ".")
    for row in report.get("student_results", []) or []:
        student = row.get("student", {}) or {}
        title = html.escape(str(student.get("student_roll_no") or student.get("source_filename") or "Student"))
        status = html.escape(str(row.get("status")))
        report_path = Path(row.get("report_html") or "")
        try:
            report_relative = report_path.relative_to(run_dir)
        except ValueError:
            report_relative = Path("reports") / report_path.name
        report_rel = html.escape(report_relative.as_posix())
        score = row.get("score")
        max_marks = row.get("max_marks")
        points = f"{score} / {max_marks}" if score is not None and max_marks is not None else "Not scored"
        percentage = f"{row['percentage']}%" if row.get("percentage") is not None else "—"
        entries.append(
            f"<tr><td>{title}</td><td>{status}</td><td>{html.escape(str(row.get('score_status')))}</td>"
            f"<td>{html.escape(points)}</td><td>{html.escape(percentage)}</td>"
            f"<td><a href=\"{report_rel}\">Review report</a></td></tr>"
        )
    warns = "".join(f"<li>{html.escape(str(w))}</li>" for w in report.get("warnings", []) or []) or "<li>None recorded.</li>"
    summary = report.get("summary", {})
    content = f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>Assessment batch summary</title><style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:.6rem;text-align:left}}th{{background:#f2f2f2}}</style></head><body><h1>Assessment batch summary</h1><p><b>Run:</b> {html.escape(str(report.get('run_id')))}</p><p><b>Status:</b> {html.escape(str(report.get('pipeline_status')))}</p><p><b>Question paper:</b> {html.escape(str((report.get('discovered') or {}).get('question_paper')))}</p><p><b>Students discovered:</b> {summary.get('students_discovered',0)}; assessed: {summary.get('students_assessed',0)}; failed/partial: {summary.get('students_failed_or_partial',0)}</p><div style=\"padding:.8rem;border-left:4px solid #a67c00;background:#fff8df\">Static inspection only: student code was not executed. Numeric scores may be provisional or unavailable. No official grades were issued.</div><h2>Student reports</h2><table><thead><tr><th>Student</th><th>Status</th><th>Scoring</th><th>Provisional points</th><th>Percentage</th><th>Review report</th></tr></thead><tbody>{''.join(entries)}</tbody></table><h2>Warnings</h2><ul>{warns}</ul></body></html>"""
    write_text = content
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(write_text, encoding="utf-8")
