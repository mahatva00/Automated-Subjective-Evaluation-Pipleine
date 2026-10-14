import unittest

from src.evaluation.evidence_mapper import build_evidence_map, extract_submission_segments


class EvidenceMapperTests(unittest.TestCase):
    def setUp(self):
        self.qmap = {
            "schema_version": "1.1",
            "source_filename": "Example-QP.pdf",
            "questions": [
                {"question_id": "q1_occurrence_1", "number": "1", "occurrence": 1, "text": "1. Data preparation\nPrepare data and report missing values.", "source_pages": [1]},
                {"question_id": "q2_occurrence_1", "number": "2", "occurrence": 1, "text": "2. Query design\nWrite a query and explain it.", "source_pages": [2]},
            ],
            "standalone_tasks": [],
        }
        self.reqs = {
            "schema_version": "1.1", "source_filename": "Example-QP.pdf",
            "requirements": [
                {"requirement_id": "q1_r1", "question_id": "q1_occurrence_1", "question_number": "1", "requirement_type": "action", "atomic_text": "Prepare data."},
                {"requirement_id": "q2_r1", "question_id": "q2_occurrence_1", "question_number": "2", "requirement_type": "action", "atomic_text": "Write a query."},
            ],
            "global_requirements": [{"requirement_id": "global_1", "question_id": None, "requirement_type": "constraint", "atomic_text": "Show your work."}],
        }

    def test_notebook_cells_inherit_question_until_next_heading(self):
        submission = {
            "segments": [
                {"segment_id": "c1", "position": 0, "original_type": "markdown", "content": "## Question 1: Data preparation"},
                {"segment_id": "c2", "position": 1, "original_type": "code", "content": "df = load_data()", "outputs": [{"output_type": "stream", "text": "missing values: 2\n"}]},
                {"segment_id": "c3", "position": 2, "original_type": "markdown", "content": "## Question 2: Query design"},
                {"segment_id": "c4", "position": 3, "original_type": "code", "content": "SELECT COUNT(*) FROM data;"},
            ]
        }
        output = build_evidence_map(self.qmap, self.reqs, submission, "student01.json")
        q1, q2 = output["questions"]
        self.assertEqual([b["segment_id"] for b in q1["evidence_blocks"]], ["c1", "c2"])
        self.assertEqual([b["segment_id"] for b in q2["evidence_blocks"]], ["c3", "c4"])
        self.assertIn("missing values: 2", q1["evidence_blocks"][1]["output_text"])
        self.assertEqual(q1["requirements"][0]["requirement_id"], "q1_r1")
        self.assertEqual(output["student_code_executed"], False)

    def test_pdf_page_with_multiple_question_headings_is_split(self):
        submission = {"segments": [{"segment_id": "page_1", "position": 0, "original_type": "pdf_page", "source_location": {"page_number": 1}, "content": "Cover text\n1. Data preparation\nCode and explanation for first answer.\n2. Query design\nSQL answer and explanation."}]}
        output = build_evidence_map(self.qmap, self.reqs, submission, "answer.pdf.json")
        self.assertEqual(output["summary"]["evidence_blocks"], 3)
        self.assertEqual(output["summary"]["unassigned_blocks"], 1)
        self.assertEqual(output["questions"][0]["evidence_block_count"], 1)
        self.assertEqual(output["questions"][1]["evidence_block_count"], 1)
        self.assertIn("SQL answer", output["questions"][1]["evidence_blocks"][0]["source_text"])

    def test_unanchored_text_is_not_guessed_into_a_question(self):
        submission = {"segments": [{"segment_id": "c1", "position": 0, "original_type": "markdown", "content": "Here is my answer with no question number."}]}
        output = build_evidence_map(self.qmap, self.reqs, submission, "student.json")
        self.assertEqual(output["summary"]["mapped_blocks"], 0)
        self.assertEqual(output["summary"]["unassigned_blocks"], 1)
        self.assertTrue(all(q["mapping_status"] == "no_evidence_found" for q in output["questions"]))

    def test_source_cell_is_never_executed_and_retains_code(self):
        marker = "raise RuntimeError('must not run')"
        submission = {"cells": [{"cell_id": "code1", "cell_index": 0, "cell_type": "markdown", "source": "Question 1: Data preparation"}, {"cell_id": "code2", "cell_index": 1, "cell_type": "code", "source": marker}]}
        output = build_evidence_map(self.qmap, self.reqs, submission, "student.ipynb.json")
        code_blocks = [b for q in output["questions"] for b in q["evidence_blocks"] if b["source_text"] == marker]
        self.assertEqual(len(code_blocks), 1)
        self.assertEqual(output["student_code_executed"], False)

    def test_flat_and_nested_global_requirements_are_not_duplicated(self):
        payload = {"segments": [{"segment_id": "c1", "position": 0, "original_type": "markdown", "content": "Question 1: Data preparation"}]}
        output = build_evidence_map(self.qmap, self.reqs, payload, "student.json")
        ids = [r["requirement_id"] for r in output["global_requirements"]]
        self.assertEqual(ids, ["global_1"])

    def test_standalone_task_requirements_are_attached_without_question_number(self):
        qmap = {"source_filename": "Task.pdf", "questions": [], "standalone_tasks": [
            {"task_id": "task_unnumbered_1", "text": "Generate a dataset with numeric columns."}
        ]}
        reqs = {"requirements": [{"requirement_id": "task_unnumbered_1_r1", "question_id": None, "source_node_id": "task_unnumbered_1", "source_category": "standalone_task", "atomic_text": "Generate a dataset."}], "global_requirements": []}
        submission = {"segments": [{"segment_id": "c1", "position": 0, "original_type": "markdown", "content": "No explicit task label on this answer."}]}
        output = build_evidence_map(qmap, reqs, submission, "student.json")
        self.assertEqual(output["questions"][0]["requirements"][0]["requirement_id"], "task_unnumbered_1_r1")
        self.assertIn("source_node_id", output["questions"][0]["requirements"][0])

    def test_refuses_question_map_and_requirements_from_different_papers(self):
        submission = {"segments": [{"segment_id": "c1", "position": 0, "original_type": "markdown", "content": "Question 1: answer"}]}
        qmap = dict(self.qmap, source_filename="DBMS-QP.pdf")
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_evidence_map(qmap, self.reqs, submission, "student.json")

    def test_accepts_wrapped_source_list_and_page_metadata(self):
        parsed = {"segments": [{"segment_id": "p2", "position": 1, "original_type": "pdf_page", "source_location": {"page_number": "2"}, "content": ["Question 2: Query design", "\nSELECT 1;"]}]}
        segments = extract_submission_segments(parsed, "x.json")
        self.assertEqual(segments[0].page_number, 2)
        self.assertEqual(segments[0].source_text, "Question 2: Query design\nSELECT 1;")


    def test_code_cell_q1_variable_is_not_split_or_marked_as_question_heading(self):
        submission = {"segments": [
            {"segment_id": "c1", "position": 0, "original_type": "markdown", "content": "1. Data preparation"},
            {"segment_id": "c2", "position": 1, "original_type": "code", "content": "q1 = df[col].quantile(0.25)\nq3 = df[col].quantile(0.75)\nprint(q1, q3)"},
            {"segment_id": "c3", "position": 2, "original_type": "markdown", "content": "2. Query design"},
        ]}
        output = build_evidence_map(self.qmap, self.reqs, submission, "student.json")
        q1, q2 = output["questions"]
        code_blocks = [b for b in q1["evidence_blocks"] if b["segment_id"] == "c2"]
        self.assertEqual(len(code_blocks), 1)
        self.assertEqual(code_blocks[0]["source_text"], submission["segments"][1]["content"])
        self.assertEqual(code_blocks[0]["mapping_method"], "follows_explicit_question_heading")
        self.assertEqual(code_blocks[0]["heading_number"], None)
        self.assertNotEqual(q1["mapping_status"], "ambiguous_repeated_number")
        self.assertEqual(q2["evidence_blocks"][0]["segment_id"], "c3")

    def test_question_heading_in_code_comment_does_not_change_assignment(self):
        submission = {"segments": [
            {"segment_id": "c1", "position": 0, "original_type": "markdown", "content": "1. Data preparation"},
            {"segment_id": "c2", "position": 1, "original_type": "code", "content": "# Question 2: Query design\nprint('still evidence for question 1')"},
            {"segment_id": "c3", "position": 2, "original_type": "markdown", "content": "2. Query design"},
        ]}
        output = build_evidence_map(self.qmap, self.reqs, submission, "student.json")
        q1, q2 = output["questions"]
        self.assertEqual([b["segment_id"] for b in q1["evidence_blocks"]], ["c1", "c2"])
        self.assertEqual([b["segment_id"] for b in q2["evidence_blocks"]], ["c3"])

    def test_q_number_assignment_in_page_text_is_not_a_heading(self):
        submission = {"segments": [{"segment_id": "p1", "position": 0, "original_type": "pdf_page", "content": "1. Data preparation\nq1 = df[col].quantile(0.25)\n2. Query design\nSELECT 1;"}]}
        output = build_evidence_map(self.qmap, self.reqs, submission, "student.pdf.json")
        q1, q2 = output["questions"]
        self.assertEqual(q1["evidence_block_count"], 1)
        self.assertIn("q1 = df[col].quantile(0.25)", q1["evidence_blocks"][0]["source_text"])
        self.assertEqual(q2["evidence_block_count"], 1)
        self.assertEqual(q2["evidence_blocks"][0]["source_text"], "2. Query design\nSELECT 1;")

    def test_duplicate_question_number_does_not_drop_evidence(self):
        qmap = {"source_filename": "Repeated.pdf", "questions": [
            {"question_id": "q1_occurrence_1", "number": "1", "occurrence": 1, "text": "1. First version"},
            {"question_id": "q1_occurrence_2", "number": "1", "occurrence": 2, "text": "1. Second version"},
        ]}
        submission = {"segments": [{"segment_id": "c1", "position": 0, "original_type": "markdown", "content": "1. First version answer\n1. Second version answer"}]}
        output = build_evidence_map(qmap, {}, submission, "x.json")
        self.assertEqual(output["summary"]["mapped_blocks"], 2)
        self.assertEqual(output["questions"][0]["evidence_block_count"], 1)
        self.assertEqual(output["questions"][1]["evidence_block_count"], 1)


if __name__ == "__main__":
    unittest.main()
