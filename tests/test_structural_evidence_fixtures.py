import unittest

from src.evaluation.evidence_mapper import build_evidence_map
from src.evaluation.static_answer_evaluator import select_evidence_blocks


class StructuralEvidenceFixtureTests(unittest.TestCase):
    def setUp(self):
        self.question_map = {
            "schema_version": "1.1",
            "source_filename": "Structural-QP.pdf",
            "questions": [
                {
                    "question_id": "q1_occurrence_1",
                    "number": "1",
                    "occurrence": 1,
                    "title": "Structured response",
                    "text": "1. Structured response\n(a) Provide an implementation and result.\n(b) Explain the result.",
                    "source_pages": [1],
                },
                {
                    "question_id": "q2_occurrence_1",
                    "number": "2",
                    "occurrence": 1,
                    "title": "Independent response",
                    "text": "2. Independent response\n(a) Provide an implementation.",
                    "source_pages": [1],
                },
            ],
            "standalone_tasks": [],
        }
        self.requirements = {
            "schema_version": "1.1",
            "source_filename": "Structural-QP.pdf",
            "requirements": [
                {
                    "requirement_id": "q1_occurrence_1_part_a_r1",
                    "question_id": "q1_occurrence_1",
                    "question_number": "1",
                    "source_node_id": "q1_occurrence_1_part_a",
                    "atomic_text": "Provide an implementation and result.",
                    "requirement_type": "action",
                },
                {
                    "requirement_id": "q1_occurrence_1_part_b_r1",
                    "question_id": "q1_occurrence_1",
                    "question_number": "1",
                    "source_node_id": "q1_occurrence_1_part_b",
                    "atomic_text": "Explain the result.",
                    "requirement_type": "reasoning",
                },
                {
                    "requirement_id": "q2_occurrence_1_part_a_r1",
                    "question_id": "q2_occurrence_1",
                    "question_number": "2",
                    "source_node_id": "q2_occurrence_1_part_a",
                    "atomic_text": "Provide an implementation.",
                    "requirement_type": "action",
                },
            ],
            "global_requirements": [],
        }

    def test_fixture_preserves_component_roles_and_subquestion_boundaries(self):
        submission = {
            "question_blocks": {
                "1": [
                    {"block_id": "q1", "segment_id": "q1", "component": "question",
                     "source_text": "1. Structured response", "output_text": ""},
                    {"block_id": "q1a_copy", "segment_id": "q1a_copy",
                     "component": "question_copy", "subquestion_anchor": "a",
                     "source_text": "(a) Provide an implementation and result.", "output_text": ""},
                    {"block_id": "q1a_code", "segment_id": "q1a",
                     "component": "code", "subquestion_anchor": "a",
                     "source_text": "implementation()", "output_text": ""},
                    {"block_id": "q1a_output", "segment_id": "q1a",
                     "component": "output", "subquestion_anchor": "a",
                     "source_text": "", "output_text": "result"},
                    {"block_id": "q1a_explanation", "segment_id": "q1a_exp",
                     "component": "explanation", "subquestion_anchor": "a",
                     "source_text": "The implementation produced the result.", "output_text": ""},
                    {"block_id": "q1a_ai", "segment_id": "q1a_ai",
                     "component": "ai_log", "subquestion_anchor": "a",
                     "source_text": "No AI used.", "output_text": ""},
                    {"block_id": "q1b_copy", "segment_id": "q1b_copy",
                     "component": "question_copy", "subquestion_anchor": "b",
                     "source_text": "(b) Explain the result.", "output_text": ""},
                    {"block_id": "q1b_explanation", "segment_id": "q1b",
                     "component": "explanation", "subquestion_anchor": "b",
                     "source_text": "The result is consistent.", "output_text": ""},
                ],
                "2": [
                    {"block_id": "q2", "segment_id": "q2", "component": "question",
                     "source_text": "2. Independent response", "output_text": ""},
                    {"block_id": "q2a_code", "segment_id": "q2a",
                     "component": "code", "subquestion_anchor": "a",
                     "source_text": "independent_implementation()", "output_text": ""},
                ],
            },
            "segments": [],
        }
        result = build_evidence_map(
            self.question_map, self.requirements, submission, "student.ipynb"
        )
        questions = {q["question_number"]: q for q in result["questions"]}
        q1 = {r["requirement_id"]: r for r in questions["1"]["requirements"]}
        q2 = questions["2"]["requirements"][0]

        self.assertEqual(
            q1["q1_occurrence_1_part_a_r1"]["mapping_diagnostics"]["category"],
            "mapped_code",
        )
        self.assertEqual(
            q1["q1_occurrence_1_part_a_r1"]["evidence_bundle"]["block_ids"],
            ["q1a_code", "q1a_output", "q1a_explanation"],
        )
        self.assertEqual(
            q1["q1_occurrence_1_part_b_r1"]["mapping_diagnostics"]["category"],
            "explanation_only",
        )
        self.assertEqual(
            q1["q1_occurrence_1_part_b_r1"]["evidence_bundle"]["block_ids"],
            ["q1b_explanation"],
        )
        self.assertNotIn(
            "q1a_ai",
            q1["q1_occurrence_1_part_a_r1"]["evidence_bundle"]["block_ids"],
        )
        self.assertEqual(q2["linked_code_block_ids"], ["q2a_code"])

    def test_fixture_keeps_visual_metadata_auxiliary_and_non_authoritative(self):
        requirement = {
            "requirement_id": "visual",
            "atomic_text": "Draw a chart.",
            "requirement_type": "deliverable",
            "linked_evidence_block_ids": [],
        }
        blocks = [{
            "block_id": "visual_code",
            "segment_id": "cell1",
            "segment_position": 1,
            "component": "code",
            "source_text": "render_visual()",
            "output_text": "",
            "metadata": {
                "image_evidence": [{
                    "region_index": 1,
                    "ocr_text": "Rendered visual",
                    "ocr_confidence": 88.0,
                    "is_authoritative": False,
                }],
            },
        }]
        selected, complete = select_evidence_blocks(requirement, blocks)
        self.assertFalse(complete)
        self.assertEqual(selected[0]["visual_evidence"][0]["ocr_text"], "Rendered visual")
        self.assertFalse(selected[0]["visual_evidence"][0]["is_authoritative"])

    def test_fixture_reports_missing_required_component_without_inventing_evidence(self):
        requirement = {
            "requirement_id": "missing-output",
            "atomic_text": "Provide an implementation and result.",
            "requirement_type": "action",
            "linked_evidence_block_ids": [],
            "mapping_diagnostics": {"category": "output_only"},
        }
        blocks = [{
            "block_id": "only_code",
            "segment_id": "cell1",
            "segment_position": 1,
            "component": "code",
            "source_text": "implementation()",
            "output_text": "",
        }]
        selected, complete = select_evidence_blocks(requirement, blocks)
        self.assertFalse(complete)
        self.assertEqual(selected, [])


if __name__ == "__main__":
    unittest.main()
