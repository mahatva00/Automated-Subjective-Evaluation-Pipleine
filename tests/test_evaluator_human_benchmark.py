from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluator_human_benchmark import compare_labels, prepare_review_files
from src.evaluation.static_answer_evaluator import select_evidence_blocks


def _write_submission(run_dir: Path, filename: str, status: str) -> None:
    evidence_path = (
        run_dir / "intermediate" / "evidence_maps" / "submissions"
        / f"{filename}.evidence-map.json"
    )
    evaluation_path = (
        run_dir / "intermediate" / "evaluations" / "submissions"
        / f"{filename}.evaluation.json"
    )
    requirement = {
        "requirement_id": "r1",
        "question_id": "q1",
        "question_number": "1",
        "atomic_text": "Count the missing values in each column.",
        "requirement_type": "action",
        "linked_evidence_block_ids": ["code1"],
        "linked_code_block_ids": ["code1"],
    }
    blocks = [{
        "block_id": "code1",
        "segment_id": "cell1",
        "segment_position": 1,
        "component": "code",
        "source_text": "df.isnull().sum()",
        "output_text": "",
    }]
    evidence_map = {
        "student_code_executed": False,
        "source_submission_filename": f"{filename}.ipynb",
        "global_requirements": [],
        "questions": [{
            "question_id": "q1",
            "question_number": "1",
            "question_title": "Data quality",
            "evidence_blocks": blocks,
            "requirements": [requirement],
        }],
    }
    selected, context_complete = select_evidence_blocks(requirement, blocks)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence_map, ensure_ascii=False), encoding="utf-8")
    evaluation = {
        "student_code_executed": False,
        "source_evidence_map_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "global_evaluations": [],
        "questions": [{
            "question_id": "q1",
            "question_number": "1",
            "evaluations": [{
                "requirement_id": "r1",
                "question_number": "1",
                "requirement_text": requirement["atomic_text"],
                "status": status,
                "rationale": f"Model rationale for {filename}",
                "confidence": 0.8,
                "evidence": [],
                "validation_errors": [],
                "evidence_diagnostic": {
                    "selected_block_ids": [block["block_id"] for block in selected],
                    "context_complete": context_complete,
                },
            }],
        }],
    }
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(json.dumps(evaluation, ensure_ascii=False), encoding="utf-8")


class EvaluatorHumanBenchmarkTests(unittest.TestCase):
    def test_prepare_creates_blinded_deterministic_sample_and_excludes_anomaly_token(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            _write_submission(run_dir, "student_1001", "complete")
            _write_submission(run_dir, "student_1002", "needs_review")
            _write_submission(run_dir, "student_1063", "absent")
            labels_path, key_path = prepare_review_files(
                run_dir,
                root / "review",
                sample_size=5,
                seed=17,
                exclude_submission_tokens=["1063"],
            )
            with labels_path.open(encoding="utf-8-sig", newline="") as stream:
                labels = list(csv.DictReader(stream, delimiter="\t"))
            key = json.loads(key_path.read_text(encoding="utf-8"))

            self.assertEqual(len(labels), 2)
            self.assertEqual(len(key["cases"]), 2)
            self.assertTrue(all(row["human_status"] == "" for row in labels))
            self.assertNotIn("model_status", labels[0])
            self.assertNotIn("student_1063.ipynb", key["submission_aliases"].values())
            self.assertIn("df.isnull().sum()", labels[0]["evidence_json"])
            self.assertEqual(
                {case["model_status"] for case in key["cases"].values()},
                {"complete", "needs_review"},
            )

    def test_compare_reports_agreement_confusion_and_disagreements(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            _write_submission(run_dir, "student_1001", "complete")
            _write_submission(run_dir, "student_1002", "needs_review")
            labels_path, key_path = prepare_review_files(
                run_dir, root / "review", sample_size=2, seed=3
            )
            with labels_path.open("r", encoding="utf-8-sig", newline="") as stream:
                labels = list(csv.DictReader(stream, delimiter="\t"))
            for row in labels:
                row["human_status"] = (
                    "complete" if row["submission_alias"] == "SUB-01" else "partial"
                )
                row["human_rationale"] = "Manual reference decision."
            with labels_path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=labels[0].keys(),
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(labels)

            report = compare_labels(labels_path, key_path, root / "agreement.json")

            self.assertEqual(report["labeled_cases"], 2)
            self.assertEqual(report["exact_agreement"], 0.5)
            self.assertEqual(len(report["disagreements"]), 1)
            self.assertEqual(report["confusion_matrix"]["partial"]["needs_review"], 1)
            self.assertTrue((root / "agreement.json").is_file())


if __name__ == "__main__":
    unittest.main()
