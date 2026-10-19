from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

from src.policy.policy_parser import parse_assessment_policy
from src.evaluation.grading_engine import score_component_evaluation, score_evaluation
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
    def test_discovery_accepts_pdf_and_notebook_submissions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "teacher_input"
            qp_dir = root / "Question Paper"
            student_dir = root / "Student Submissions"
            qp_dir.mkdir(parents=True)
            student_dir.mkdir()
            make_pdf(qp_dir / "Assignment-QP.pdf", "Lab Assessment\n1. Answer the question.")
            make_pdf(student_dir / "23MCA1006_Assignment.pdf", "1. Answer the question.\nStudent response.")
            (student_dir / "23MCA1007_Assignment.ipynb").write_text(
                json.dumps({
                    "cells": [{
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": ["1. Answer the question.\nNotebook response."],
                    }],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }),
                encoding="utf-8",
            )

            discovered = discover_input_files(root)
            self.assertEqual(
                {path.suffix.lower() for path in discovered["submissions"]},
                {".pdf", ".ipynb"},
            )

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

    def test_markdown_policy_extracts_question_component_weights(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "assessment_policy.md"
            p.write_text(
                "Q1. DATA PREPROCESSING [10 marks]\n"
                "Split: Code 50%, Explanation 25%, Output 25%\n\n"
                "Q2. TRAINING [30 marks]\n"
                "Split: Code 40%, Explanation 30%, Output 30%\n",
                encoding="utf-8",
            )
            policy = parse_assessment_policy(p, td)
            self.assertTrue(policy["component_scoring_enabled"])
            self.assertEqual(policy["question_component_policy"]["1"]["marks"], 10.0)
            self.assertEqual(
                policy["question_component_policy"]["1"]["component_weights"],
                {"code": 0.5, "explanation": 0.25, "output": 0.25},
            )
            self.assertEqual(
                policy["question_component_policy"]["1"]["subquestion_marks"],
                {},
            )
            self.assertEqual(policy["question_component_policy"]["2"]["marks"], 30.0)

    def test_markdown_policy_rejects_incomplete_component_split(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "assessment_policy.md"
            p.write_text(
                "Q1. DATA PREPROCESSING [10 marks]\n"
                "Split: Code 50%, Explanation 25%, Output 15%\n",
                encoding="utf-8",
            )
            policy = parse_assessment_policy(p, td)
            self.assertFalse(policy["component_scoring_enabled"])
            self.assertEqual(policy["question_component_policy"], {})
            self.assertTrue(any("sum to 100%" in warning for warning in policy["warnings"]))

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

    def test_component_scoring_with_missing_explanation_awards_no_explanation_marks(self):
        result = score_component_evaluation(
            {
                "student_code_executed": False,
                "questions": [{
                    "question_number": "1",
                    "component_evaluations": {
                        "code": "complete",
                        "output": "partial",
                    },
                }],
            },
            {
                "question_component_policy": {
                    "1": {
                        "marks": 10,
                        "component_weights": {
                            "code": 0.5,
                            "explanation": 0.25,
                            "output": 0.25,
                        },
                    },
                },
                "teacher_approval_required": True,
            },
        )
        self.assertEqual(result["max_marks"], 10.0)
        self.assertEqual(result["score"], 6.25)
        self.assertEqual(
            next(row for row in result["component_rows"] if row["component"] == "explanation")["awarded_marks"],
            0.0,
        )

    def test_component_scoring_keeps_needs_review_provisional(self):
        result = score_component_evaluation(
            {
                "student_code_executed": False,
                "questions": [{
                    "question_number": "3",
                    "component_evaluations": {
                        "explanation": "needs_review",
                        "output": "complete",
                    },
                }],
            },
            {
                "question_component_policy": {
                    "3": {
                        "marks": 10,
                        "component_weights": {
                            "code": 0.0,
                            "explanation": 0.7,
                            "output": 0.3,
                        },
                    },
                },
            },
        )
        self.assertEqual(result["score_status"], "provisional")
        self.assertEqual(result["score"], 3.0)
        self.assertEqual(result["pending_review_components"], 1)

    def test_one_folder_end_to_end_with_fake_evaluator(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "teacher_input"
            qp_dir = root / "Question Paper"
            student_dir = root / "Student Submissions"
            policy_dir = root / "Assessment Policy"
            root.mkdir()
            qp = qp_dir / "Assignment-QP.pdf"
            make_pdf(
                qp,
                "Lab Assessment\nInstructions:\nUse a dataset with at least 500 instances.\n"
                "1. Load the dataset and display the first records. [2 marks]",
            )
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
            requirement_stage = next(
                stage for stage in report["stages"]
                if stage["stage"] == "question_mapping_and_requirement_extraction"
            )
            extracted_requirements = json.loads(
                Path(requirement_stage["output"]).read_text(encoding="utf-8")
            )
            self.assertEqual(extracted_requirements["global_requirements"], [])
            self.assertFalse(
                any(
                    item.get("source_category") == "global_instruction"
                    for item in extracted_requirements["requirements"]
                )
            )
            evaluation_stage = next(
                stage for stage in report["stages"]
                if stage["stage"] == "static_answer_evaluation"
            )
            evaluation = json.loads(
                Path(evaluation_stage["output"]).read_text(encoding="utf-8")
            )
            self.assertEqual(evaluation["global_evaluations"], [])
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

    def test_report_propagates_question_component_findings(self):
        evidence_map = {
            "questions": [{
                "question_id": "q1",
                "question_number": "1",
                "question_title": "Build the model",
                "mapping_status": "evidence_found",
                "component_blocks": {
                    "question": [{"block_id": "q1"}],
                    "question_copy": [{"block_id": "q1_copy"}],
                    "code": [{"block_id": "c1"}],
                    "output": [],
                    "explanation": [],
                    "ai_log": [],
                },
                "missing_components": ["output", "explanation"],
                "ai_log_present": False,
                "requirements": [{
                    "requirement_id": "r1",
                    "linked_evidence_block_ids": ["c1"],
                    "evidence_link_status": "subquestion_anchor",
                    "evidence_link_confidence": 1.0,
                }],
            }],
            "summary": {"mapped_blocks": 3},
        }
        report = build_student_report(
            {"report_id": "test", "source_filename": "sample.ipynb"},
            None,
            None,
            {"source_filename": "policy.txt", "numeric_scoring_enabled": False},
            evidence_map,
            [],
        )
        component = report["question_components"][0]
        self.assertEqual(component["present_components"], ["question", "question_copy", "code"])
        self.assertEqual(component["missing_components"], ["output", "explanation"])
        self.assertEqual(component["requirements"][0]["linked_evidence_block_ids"], ["c1"])
        self.assertIn("Answer component extraction", render_student_report(report))


if __name__ == "__main__":
    unittest.main()
