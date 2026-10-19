#!/usr/bin/env python3
"""Train a small DistilBERT semantic-role classifier from approved pilot spans.

This is an MVP span classifier, not a full-document boundary detector. It uses
only explicitly approved annotation spans, trains on FAT + PAT2, and validates
on LPS5 (all three are ML papers). Results must not be described as cross-domain
performance.
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"line {line_no} has no text")
            if not isinstance(row.get("label"), str) or not row["label"]:
                raise ValueError(f"line {line_no} has no label")
            if row.get("split") not in {"train", "validation", "test"}:
                raise ValueError(f"line {line_no} has invalid split")
            rows.append(row)
    return rows


def compute_metrics(y_true: list[int], y_pred: list[int], num_labels: int) -> dict:
    correct = sum(a == b for a, b in zip(y_true, y_pred))
    per_class = []
    supports = []
    for label in range(num_labels):
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        support = sum(a == label for a in y_true)
        supports.append(support)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            per_class.append(f1)
    return {
        "accuracy": correct / len(y_true) if y_true else 0.0,
        "macro_f1_over_classes_present_in_validation": sum(per_class) / len(per_class) if per_class else 0.0,
        "validation_support_by_label_id": {str(i): supports[i] for i in range(num_labels) if supports[i]},
        "num_validation_examples": len(y_true),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/annotations/distilbert_mvp/semantic_span_dataset.jsonl")
    parser.add_argument("--output-dir", default="models/distilbert_span_role_mvp_v1")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 16:
        print("ERROR: epochs/batch-size must be positive and max-length must be >= 16")
        return 2
    dataset_path = Path(args.dataset)
    if not dataset_path.is_file():
        print(f"ERROR: dataset not found: {dataset_path}. Run prepare_distilbert_span_dataset.py first.")
        return 2
    try:
        rows = load_jsonl(dataset_path)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]
    test_rows = [r for r in rows if r["split"] == "test"]
    if not train_rows or not val_rows:
        print("ERROR: both train and validation splits must contain examples")
        return 2
    train_docs = sorted({r["document_id"] for r in train_rows})
    val_docs = sorted({r["document_id"] for r in val_rows})
    overlap = set(train_docs) & set(val_docs)
    if overlap:
        print(f"ERROR: document leakage between train and validation: {sorted(overlap)}")
        return 2
    label_names = sorted({r["label"] for r in train_rows})
    label2id = {label: i for i, label in enumerate(label_names)}
    unknown_val = sorted({r["label"] for r in val_rows} - set(label2id))
    if unknown_val:
        print(f"ERROR: validation contains label(s) absent from training: {unknown_val}")
        return 2
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".distilbert_write_probe.tmp"
        with probe.open("w", encoding="utf-8") as f:
            f.write("write-test")
        probe.unlink()
    except OSError as exc:
        print(f"ERROR: model output directory is not writable: {out_dir}")
        print(f"Details: {exc}")
        print("Choose a new output directory under models/ and close programs that may be locking its files.")
        return 2

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        print("ERROR: DistilBERT dependencies are missing. Install requirements-distilbert-mvp.txt")
        print(f"Details: {exc}")
        return 2

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if not tokenizer.is_fast:
        print("ERROR: a fast tokenizer is required")
        return 2
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_names),
        id2label={i: label for label, i in label2id.items()},
        label2id=label2id,
    ).to(device)

    def collate(batch):
        tokens = tokenizer(
            [x["text"] for x in batch],
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors="pt",
        )
        labels = torch.tensor([label2id[x["label"]] for x in batch], dtype=torch.long)
        return {k: v.to(device) for k, v in tokens.items()}, labels

    train_loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_rows, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    train_counts = Counter(r["label"] for r in train_rows)
    weights = torch.tensor(
        [len(train_rows) / (len(label_names) * train_counts[label]) for label in label_names],
        dtype=torch.float,
        device=device,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_macro_f1 = -1.0
    best_state = None
    best_metrics = None
    epoch_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batch_count += 1
        model.eval()
        val_true, val_pred = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                logits = model(**inputs).logits
                preds = torch.argmax(logits, dim=1)
                val_true.extend(labels.detach().cpu().tolist())
                val_pred.extend(preds.detach().cpu().tolist())
        metrics = compute_metrics(val_true, val_pred, len(label_names))
        metrics["epoch"] = epoch
        metrics["train_loss"] = total_loss / max(batch_count, 1)
        epoch_history.append(metrics)
        print(
            f"epoch {epoch}/{args.epochs} train_loss={metrics['train_loss']:.4f} "
            f"val_accuracy={metrics['accuracy']:.3f} "
            f"val_macro_f1={metrics['macro_f1_over_classes_present_in_validation']:.3f}"
        )
        if metrics["macro_f1_over_classes_present_in_validation"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1_over_classes_present_in_validation"]
            # Keep the best state in memory and write it only once after all epochs.
            # Repeatedly overwriting the same checkpoint can trigger Windows file-lock issues.
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = metrics

    if best_state is None or best_metrics is None:
        print("ERROR: no best checkpoint state was recorded")
        return 2
    try:
        model.load_state_dict(best_state)
        # Use PyTorch .bin format; avoid the safetensors serializer that failed in the prior run.
        model.save_pretrained(out_dir, safe_serialization=False)
        tokenizer.save_pretrained(out_dir)
        (out_dir / "mvp_label_mapping.json").write_text(
            json.dumps({"label2id": label2id, "id2label": {str(i): label for label, i in label2id.items()}}, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"ERROR: final checkpoint save failed for {out_dir}")
        print(f"Details: {type(exc).__name__}: {exc}")
        print("The model was held in memory until training completed, but could not be written. Use a fresh writable output directory and close programs that may be locking model files.")
        return 2

    summary = {
        "model_name": args.model_name,
        "task": "coarse semantic role classification for candidate spans",
        "model_output_dir": str(out_dir),
        "checkpoint_format": "pytorch_model.bin",
        "device": str(device),
        "labels": label2id,
        "train_documents": train_docs,
        "validation_documents": val_docs,
        "train_examples": len(train_rows),
        "validation_examples": len(val_rows),
        "test_examples_not_used": len(test_rows),
        "best_validation_metrics": best_metrics,
        "epoch_history": epoch_history,
        "limitations": [
            "Only three Machine Learning papers; no cross-domain evaluation.",
            "Only selected semantic spans were annotated; unannotated text was not used as negative examples.",
            "This model classifies existing candidate spans. It does not discover boundaries or rebuild hierarchy by itself.",
            "The validation sample is one paper and is too small for a reliable performance claim."
        ],
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved MVP classifier: {out_dir}")
    print("Checkpoint format: pytorch_model.bin (saved once after training)")
    print(f"Best LPS5 validation metrics: {json.dumps(best_metrics)}")
    print("Warning: this validation is a small same-subject smoke check, not a cross-domain score.")
    print("Student code executed: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
