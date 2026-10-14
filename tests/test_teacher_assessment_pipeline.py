from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

from src.policy.policy_parser import parse_assessment_policy
from src.evaluation.grading_engine import score_evaluation
from src.reporting.student_report import build_student_report, render_student_report
from src.orchestration.teacher_assessment import discover_input_files, run_teacher_assessment


class FakeClient:
    model = "fake-static-test-model"
    def __init__(self):
        self.call_count = 0
    def chat_json(self, payload):
        self.call_count += 1
        assessments = []
        for req in payload.get("requirements_to_assess", []):
            blocks = req.get("evidence_blocks", [])
            ev = []
            if blocks:
                block = blocks[0]
                text = block.get("source_text", "") or block.get("output_text", "")
                if text:
                    ev = [{"block_id": block["block_id"], "quote": text[:min(40, len(text))], "purpose": "synthetic test evidence"}]
            assessments.append({"requirement_id": req["requirement_id"], "status": "complete" if ev else "needs_review",
                                "rationale": "Synthetic pipeline test; this is not a real student assessment.",
                                "confidence": 0.9 if ev else 0.0, "evidence_sufficient": bool(ev), "evidence": ev})
        return {"assessments": assessments}


def make_pdf(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    doc.save(path)
    doc.close()


class TeacherPipelineTests(unittest.TestCase):
    def test_policy_does_not_invent_credit_factors(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "policy.txt"
            p.write_text("Use lenient marking. Award partial credit where appropriate.", encoding="utf-8")
            policy = parse_assessment_policy(p, td)
            self.assertIsNone(policy["credit_map"])
            self.assertFalse(policy["numeric_scoring_enabled"])
            self.assertEqual(policy["strictness"], "lenient")

    def test_text_policy_percent_factors_are_recognized_only_when_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "policy.txt"
            p.write_text("complete: 100%; partial: 50%; absent: 0%", encoding="utf-8")
            policy = parse_assessment_policy(p, td)
            self.assertEqual(policy["credit_map"], {"complete": 1.0, "partial": 0.5, "absent": 0.0})

    def test_json_policy_explicit_credit_map(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "policy.json"
            p.write_text(json.dumps({"credit_map": {"complete": 1, "partial": 0.5, "absent": 0}, "strictness": "balanced"}), encoding="utf-8")
            policy = parse_assessment_policy(p, td)
            self.assertTrue(policy["numeric_scoring_enabled"])
            self.assertEqual(policy["credit_map"]["partial"], 0.5)

    def test_marks_not_duplicated_within_source_node(self):
        requirements = {"requirements": [
            {"requirement_id": "r1", "source_node_id": "part_a", "source_node_marks": 2, "question_id": "q1"},
            {"requirement_id": "r2", "source_node_id": "part_a", "source_node_marks": 2, "question_id": "q1"},
        ], "questions": [{"question_id": "q1", "marks": 2}]}
        evaluation = {"student_code_executed": False, "questions": [{"evaluations": [
            {"requirement_id": "r1", "status": "complete"}, {"requirement_id": "r2", "status": "partial"}]}]}
        policy = {"credit_map": {"complete": 1, "partial": 0.5, "absent": 0},
                  "mark_allocation": {"mode": "per_source_node_equal_split"}, "teacher_approval_required": True}
        result = score_evaluation(evaluation, requirements, policy)
        self.assertEqual(result["max_marks"], 2.0)
        self.assertEqual(result["score"], 1.5)
        self.assertAlmostEqual(result["percentage"], 75.0)

    def test_one_folder_end_to_end_with_fake_evaluator(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "teacher_input"
            qp_dir = root / "Question Paper"
            student_dir = root / "Student Submissions"
            policy_dir = root / "Assessment Policy"
            root.mkdir()
            qp = qp_dir / "Assignment-QP.pdf"
            make_pdf(qp, "Lab Assessment\n1. Load the dataset and display the first records. [2 marks]")
            student_dir.mkdir(parents=True)
            notebook = student_dir / "23MCA1006_Assignment.ipynb"
            notebook.write_text(json.dumps({"cells": [
                {"cell_type": "markdown", "metadata": {}, "source": ["1. Load the dataset and display the first records."]},
                {"cell_type": "code", "metadata": {}, "execution_count": 1, "outputs": [], "source": ["df.head()"]}
            ], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}), encoding="utf-8")
            policy_dir.mkdir(parents=True)
            policy_file = policy_dir / "marking_policy.json"
            policy_file.write_text(json.dumps({"strictness": "balanced", "credit_map": {"complete": 1, "partial": 0.5, "absent": 0}}), encoding="utf-8")
            output_base = Path(td) / "results"
            clients = []
            def factory(**_):
                client = FakeClient()
                clients.append(client)
                return client
            report = run_teacher_assessment(root, output_base, provider="ollama", client_factory=factory)
            self.assertEqual(report["summary"]["students_discovered"], 1)
            self.assertEqual(report["summary"]["students_assessed"], 1)
            self.assertFalse(report["summary"]["student_code_executed"])
            audit_stages = [s for s in report["stages"] if s["stage"] == "static_evaluation_audit"]
            self.assertEqual(len(audit_stages), 1)
            self.assertEqual(audit_stages[0]["status"], "passed")
            run_dir = Path(report["run_dir"])
            self.assertTrue((run_dir / "RUN_REPORT.json").is_file())
            self.assertTrue((run_dir / "ASSESSMENT_SUMMARY.html").is_file())
            student_jsons = list((run_dir / "reports").glob("*.report.json"))
            student_htmls = list((run_dir / "reports").glob("*.report.html"))
            self.assertEqual(len(student_jsons), 1)
            self.assertEqual(len(student_htmls), 1)
            saved = json.loads(student_jsons[0].read_text(encoding="utf-8"))
            self.assertFalse(saved["student_code_executed"])
            self.assertEqual(saved["assessment_status"], "completed")
            self.assertGreater(clients[0].call_count, 0)
            resumed = run_teacher_assessment(root, output_base, provider="ollama", resume_run_dir=run_dir, client_factory=factory)
            self.assertEqual(resumed["run_dir"], str(run_dir))
            self.assertEqual(clients[1].call_count, 0, "resume should reuse compatible checkpointed requirement assessments")

    def test_report_is_explicitly_not_an_official_grade(self):
        report = build_student_report({"report_id": "test", "source_filename": "sample.ipynb"},
            {"student_code_executed": False, "summary": {"complete": 1}, "questions": []},
            {"score_status": "provisional", "numeric_scoring_enabled": False, "final_grade": None, "notes": []},
            {"source_filename": "policy.txt", "numeric_scoring_enabled": False}, {}, [])
        rendered = render_student_report(report)
        self.assertIn("No official grade has been issued", rendered)
        self.assertIn("student-submitted code was not executed", rendered)


if __name__ == "__main__":
    unittest.main()
