#!/usr/bin/env python3
"""Raw-audit enriched model outputs against original V5 requirement JSONs."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def canonical_without_predictions(doc: dict) -> dict:
    clone = json.loads(json.dumps(doc))
    for req in clone.get("requirements", []):
        if isinstance(req, dict):
            req.pop("mvp_model_assessment", None)
    return clone


def audit(input_dir: Path, output_dir: Path, pattern: str, label_mapping_path: Path | None = None) -> dict:
    sources = sorted(input_dir.rglob(pattern))
    if not sources:
        raise ValueError(f"no input files matched {pattern!r} in {input_dir}")
    allowed_labels = None
    if label_mapping_path and label_mapping_path.is_file():
        map_json = json.loads(label_mapping_path.read_text(encoding="utf-8"))
        allowed_labels = set(map_json.get("id2label", {}).values())
    file_results = []
    all_errors = []
    total_candidates = 0
    for src in sources:
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        errors = []
        if not dst.is_file():
            errors.append(f"missing output file: {dst}")
            file_results.append({"file": str(rel), "passed": False, "errors": errors})
            all_errors.extend(f"{rel}: {e}" for e in errors)
            continue
        try:
            original = json.loads(src.read_text(encoding="utf-8"))
            output = json.loads(dst.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"JSON read/parse failed: {exc}")
            file_results.append({"file": str(rel), "passed": False, "errors": errors})
            all_errors.extend(f"{rel}: {e}" for e in errors)
            continue
        orig_reqs, out_reqs = original.get("requirements"), output.get("requirements")
        if not isinstance(orig_reqs, list) or not isinstance(out_reqs, list):
            errors.append("input or output lacks top-level requirements list")
            orig_reqs = orig_reqs if isinstance(orig_reqs, list) else []
            out_reqs = out_reqs if isinstance(out_reqs, list) else []
        if len(orig_reqs) != len(out_reqs):
            errors.append(f"requirement count changed: input={len(orig_reqs)} output={len(out_reqs)}")
        orig_ids = [x.get("requirement_id") for x in orig_reqs]
        out_ids = [x.get("requirement_id") for x in out_reqs]
        if orig_ids != out_ids:
            errors.append("requirement IDs/order changed")
        # Verify all original data is byte-semantically equivalent as parsed JSON after removing model fields.
        if canonical_without_predictions(output) != original:
            errors.append("one or more original JSON fields changed beyond added mvp_model_assessment fields")
        for idx, req in enumerate(out_reqs):
            assessment = req.get("mvp_model_assessment") if isinstance(req, dict) else None
            if not isinstance(assessment, dict):
                errors.append(f"requirement[{idx}] lacks mvp_model_assessment")
                continue
            role, conf, probs = assessment.get("predicted_role"), assessment.get("confidence"), assessment.get("probabilities")
            if role is not None and allowed_labels is not None and role not in allowed_labels:
                errors.append(f"requirement[{idx}] has unknown predicted role {role!r}")
            if conf is not None and (not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0):
                errors.append(f"requirement[{idx}] has confidence outside [0,1]")
            if not isinstance(probs, dict):
                errors.append(f"requirement[{idx}] probabilities must be an object")
            elif probs:
                if any(not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in probs.values()):
                    errors.append(f"requirement[{idx}] has an invalid probability")
                if not math.isclose(sum(probs.values()), 1.0, abs_tol=0.015):
                    errors.append(f"requirement[{idx}] probabilities do not sum to approximately 1")
            if assessment.get("confidence_is_calibrated") is not False:
                errors.append(f"requirement[{idx}] must explicitly flag confidence as uncalibrated")
        total_candidates += len(out_reqs)
        file_results.append({"file": str(rel), "passed": not errors, "requirement_count": len(out_reqs), "errors": errors})
        all_errors.extend(f"{rel}: {e}" for e in errors)
    return {
        "audit_version": "1.0.0",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files_checked": len(sources),
        "candidates_checked": total_candidates,
        "passed": not all_errors and all(x["passed"] for x in file_results),
        "errors": all_errors,
        "files": file_results,
        "scope_note": "Checks output integrity and prediction metadata, not whether model predictions are semantically correct.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/processed/requirements_v5")
    parser.add_argument("--output-dir", default="data/processed/requirements_distilbert_mvp")
    parser.add_argument("--pattern", default="*.requirements.json")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--model-dir", default="models/distilbert_span_role_mvp_v1")
    parser.add_argument("--report", default="data/processed/requirements_distilbert_mvp/raw_output_audit.json")
    args = parser.parse_args()
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    if not input_dir.is_dir() or not output_dir.is_dir():
        print("ERROR: input or output directory does not exist")
        return 2
    try:
        result = audit(input_dir, output_dir, args.pattern, Path(args.model_dir) / "mvp_label_mapping.json")
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Files checked: {result['files_checked']}; candidates checked: {result['candidates_checked']}")
    print(f"Raw output integrity: {'PASS' if result['passed'] else 'FAIL'}")
    if result["errors"]:
        for e in result["errors"][:40]:
            print(f"ERROR: {e}")
    print(f"Report: {report_path}")
    print("Note: integrity passing does not establish semantic prediction quality.")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
