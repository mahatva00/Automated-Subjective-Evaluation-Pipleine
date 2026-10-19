#!/usr/bin/env python3
"""Add DistilBERT semantic-role predictions to V5 requirement candidate JSONs.

Original V5 files are read-only; enriched copies are written to a separate folder.
This is a candidate classifier, not a boundary extractor or final grading engine.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/distilbert_span_role_mvp_v1")
    parser.add_argument("--input-dir", default="data/processed/requirements_v5")
    parser.add_argument("--output-dir", default="data/processed/requirements_distilbert_mvp")
    parser.add_argument("--pattern", default="*.requirements.json")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    args = parser.parse_args()
    model_dir, input_dir, output_dir = Path(args.model_dir), Path(args.input_dir), Path(args.output_dir)
    if not (model_dir / "config.json").is_file():
        print(f"ERROR: trained model not found: {model_dir}. Run train_distilbert_mvp.py first.")
        return 2
    if not input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {input_dir}")
        return 2
    if args.batch_size < 1 or args.max_length < 16 or not 0 <= args.min_confidence <= 1:
        print("ERROR: invalid batch-size, max-length, or confidence threshold")
        return 2
    files = sorted(input_dir.rglob(args.pattern) if args.recursive else input_dir.glob(args.pattern))
    if not files:
        print(f"ERROR: no files match {args.pattern!r} in {input_dir}")
        return 2
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        print("ERROR: DistilBERT dependencies are missing. Install requirements-distilbert-mvp.txt")
        print(f"Details: {exc}")
        return 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    raw_mapping = json.loads((model_dir / "mvp_label_mapping.json").read_text(encoding="utf-8"))
    id2label = {int(k): v for k, v in raw_mapping["id2label"].items()}

    total = 0
    role_counts = Counter()
    candidate_counts = Counter()
    for input_path in files:
        try:
            doc = json.loads(input_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"ERROR: could not read {input_path}: {exc}")
            return 2
        requirements = doc.get("requirements")
        if not isinstance(requirements, list):
            print(f"ERROR: {input_path} has no top-level requirements array")
            return 2
        texts = [str(row.get("atomic_text") or row.get("source_text") or "") for row in requirements]
        assessments = [None] * len(texts)
        nonempty_indexes = [i for i, t in enumerate(texts) if t.strip()]
        with torch.no_grad():
            for start in range(0, len(nonempty_indexes), args.batch_size):
                batch_indexes = nonempty_indexes[start:start + args.batch_size]
                batch_texts = [texts[i] for i in batch_indexes]
                enc = tokenizer(batch_texts, truncation=True, max_length=args.max_length, padding=True, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                probs = torch.softmax(model(**enc).logits, dim=1).detach().cpu()
                for local_idx, doc_idx in enumerate(batch_indexes):
                    confidence, label_idx = torch.max(probs[local_idx], dim=0)
                    role = id2label[int(label_idx)]
                    conf = float(confidence)
                    if role == "requirement" and conf >= args.min_confidence:
                        disposition = "likely_assessable_candidate"
                    elif conf < args.min_confidence:
                        disposition = "manual_review_low_confidence"
                    else:
                        disposition = "check_candidate_role"
                    assessments[doc_idx] = {
                        "model": "DistilBERT span-role MVP v1",
                        "predicted_role": role,
                        "confidence": round(conf, 6),
                        "confidence_is_calibrated": False,
                        "disposition": disposition,
                        "probabilities": {id2label[j]: round(float(probs[local_idx][j]), 6) for j in range(len(id2label))},
                    }
        for idx, row in enumerate(requirements):
            row["mvp_model_assessment"] = assessments[idx] or {
                "model": "DistilBERT span-role MVP v1",
                "predicted_role": None,
                "confidence": None,
                "confidence_is_calibrated": False,
                "disposition": "manual_review_empty_text",
                "probabilities": {},
            }
            candidate_counts[row["mvp_model_assessment"]["disposition"]] += 1
            if row["mvp_model_assessment"]["predicted_role"]:
                role_counts[row["mvp_model_assessment"]["predicted_role"]] += 1
            total += 1
        try:
            relative = input_path.relative_to(input_dir)
        except ValueError:
            relative = Path(input_path.name)
        out_path = output_dir / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"OK {input_path.name}: {len(requirements)} candidates -> {out_path}")

    report = {
        "model": "DistilBERT span-role MVP v1",
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "files_processed": len(files),
        "candidate_records_classified": total,
        "predicted_roles": dict(sorted(role_counts.items())),
        "dispositions": dict(sorted(candidate_counts.items())),
        "min_confidence_threshold": args.min_confidence,
        "limitations": [
            "Predictions classify existing V5 candidates; no new question or requirement boundaries are discovered.",
            "The model was trained on only three Machine Learning papers, with one used for a tiny validation set.",
            "Confidence values are not calibrated and should not be interpreted as correctness probabilities.",
            "Use this output for triage/inspection, not automatic grading or silent deletion of requirements."
        ],
        "student_code_executed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "mvp_classification_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Processed files: {len(files)}; candidates classified: {total}")
    print(f"Report: {report_path}")
    print("Original V5 files modified: no")
    print("Student code executed: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
