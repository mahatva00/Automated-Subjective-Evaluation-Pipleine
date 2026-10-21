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

    def test_pdf_notebook_export_separates_code_and_output_blocks(self):
        submission = {
            "segments": [{
                "segment_id": "page_1",
                "position": 0,
                "original_type": "pdf_page",
                "source_location": {"page_number": 1},
                "content": (
                    "Question 1: Data preparation\n"
                    "In [1]:\n"
                    "df = load_data()\n"
                    "Out [1]:\n"
                    "rows: 10\n"
                ),
            }]
        }
        output = build_evidence_map(self.qmap, self.reqs, submission, "answer.pdf.json")
        question = output["questions"][0]
        self.assertEqual(
            [block["component"] for block in question["component_blocks"]["code"]],
            ["code"],
        )
        self.assertEqual(
            [block["output_text"] for block in question["component_blocks"]["output"]],
            ["rows: 10"],
        )
        self.assertEqual(question["missing_components"], ["explanation"])

    def test_pdf_notebook_export_separates_explanation_and_ai_log_after_output(self):
        submission = {
            "segments": [{
                "segment_id": "page_1",
                "position": 0,
                "original_type": "pdf_page",
                "content": (
                    "Question 1: Data preparation\n"
                    "In [1]:\n"
                    "df = load_data()\n"
                    "Out [1]:\n"
                    "rows: 10\n"
                    "we verify dataset size using shape.\n"
                    "no AI tool used for 1(a)\n"
                ),
            }]
        }
        output = build_evidence_map(self.qmap, self.reqs, submission, "answer.pdf.json")
        question = output["questions"][0]
        self.assertEqual(len(question["component_blocks"]["explanation"]), 1)
        self.assertEqual(len(question["component_blocks"]["ai_log"]), 1)
        self.assertEqual(question["missing_components"], [])

    def test_compact_known_question_heading_is_recognized_but_numbered_prose_is_not(self):
        submission = {
            "segments": [{
                "segment_id": "page_1",
                "position": 0,
                "original_type": "pdf_page",
                "content": (
                    "1.DATA PREPARATION\n"
                    "In [1]: df = load_data()\n"
                    "2. unrelated inference statement\n"
                ),
            }]
        }
        output = build_evidence_map(self.qmap, self.reqs, submission, "answer.pdf.json")
        self.assertEqual(output["questions"][0]["question_number"], "1")
        self.assertTrue(any("load_data" in block["source_text"] for block in output["questions"][0]["evidence_blocks"]))
        self.assertFalse(any("unrelated inference" in block["source_text"] for block in output["questions"][1]["evidence_blocks"]))

    def test_unanchored_text_is_not_guessed_into_a_question(self):
        submission = {"segments": [{"segment_id": "c1", "position": 0, "original_type": "markdown", "content": "Here is my answer with no question number."}]}
        output = build_evidence_map(self.qmap, self.reqs, submission, "student.json")
        self.assertEqual(output["summary"]["mapped_blocks"], 0)
        self.assertEqual(output["summary"]["unassigned_blocks"], 1)
        self.assertTrue(all(q["mapping_status"] == "no_evidence_found" for q in output["questions"]))

    def test_pdf_continuation_inherits_previous_anchor_until_next_question(self):
        submission = {"segments": [
            {
                "segment_id": "cover",
                "position": 0,
                "original_type": "pdf_page",
                "content": "Student name and declaration.",
            },
            {
                "segment_id": "page_1",
                "position": 1,
                "original_type": "pdf_page",
                "content": "1. Data preparation\nAnswer to question 1.",
            },
            {
                "segment_id": "page_2",
                "position": 2,
                "original_type": "pdf_page",
                "content": "Continuation without an explicit question heading.",
            },
            {
                "segment_id": "page_3",
                "position": 3,
                "original_type": "pdf_page",
                "content": "2. Query design\nAnswer to question 2.",
            },
        ]}
        output = build_evidence_map(self.qmap, self.reqs, submission, "student.pdf.json")

        self.assertEqual(output["summary"]["mapped_blocks"], 3)
        self.assertEqual(output["summary"]["unassigned_blocks"], 1)
        self.assertEqual(
            [block["segment_id"] for block in output["questions"][0]["evidence_blocks"]],
            ["page_1", "page_2"],
        )
        self.assertEqual(
            [block["segment_id"] for block in output["questions"][1]["evidence_blocks"]],
            ["page_3"],
        )
        unassigned = output["unassigned_evidence_blocks"][0]
        self.assertEqual(unassigned["segment_id"], "cover")
        self.assertEqual(unassigned["mapping_method"], "no_question_anchor")
        continuation = output["questions"][0]["evidence_blocks"][1]
        self.assertEqual(continuation["mapping_method"], "follows_explicit_question_heading")
        self.assertEqual(continuation["mapping_status"], "mapped_by_document_order")
        self.assertEqual(
            output["mapping_method"],
            "generic_explicit_question_anchors_with_document_order_continuations",
        )

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
        parsed = {"segments": [{
            "segment_id": "p2",
            "position": 1,
            "original_type": "pdf_page",
            "source_location": {"page_number": "2"},
            "content": ["Question 2: Query design", "\nSELECT 1;"],
            "extraction_method": "pdf_text+image_ocr",
            "extraction_status": "partial",
            "embedded_image_count": 2,
            "diagnostics": ["Embedded image OCR needs verification."],
        }]}
        segments = extract_submission_segments(parsed, "x.json")
        self.assertEqual(segments[0].page_number, 2)
        self.assertEqual(segments[0].source_text, "Question 2: Query design\nSELECT 1;")
        self.assertEqual(segments[0].metadata["extraction_method"], "pdf_text+image_ocr")
        self.assertEqual(segments[0].metadata["embedded_image_count"], 2)
        self.assertIn("verif", segments[0].metadata["extraction_diagnostics"][0].lower())
        evidence_map = build_evidence_map(
            self.qmap, self.reqs, parsed, "x.json"
        )
        q2_block = evidence_map["questions"][1]["evidence_blocks"][0]
        self.assertEqual(
            q2_block["metadata"]["extraction_method"], "pdf_text+image_ocr"
        )
        self.assertEqual(q2_block["metadata"]["embedded_image_count"], 2)


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

    def test_notebook_component_blocks_are_mapped_by_explicit_question_number(self):
        submission = {
            "segments": [
                {"segment_id": "c0", "position": 0, "original_type": "markdown", "content": "# Q1: Data preparation"},
            ],
            "question_blocks": {
                "1": [
                    {"block_id": "c0_question", "segment_id": "c0", "component": "question",
                     "source_text": "# Q1: Data preparation", "output_text": "",
                     "source_location": {"cell_index": 0}},
                    {"block_id": "c1_code", "segment_id": "c1", "component": "code",
                     "source_text": "df = load_data()", "output_text": "",
                     "source_location": {"cell_index": 1}},
                    {"block_id": "c1_output", "segment_id": "c1", "component": "output",
                     "source_text": "", "output_text": "missing values: 2",
                     "source_location": {"cell_index": 1}},
                    {"block_id": "c2_explanation", "segment_id": "c2", "component": "explanation",
                     "source_text": "The missing values were reported.", "output_text": "",
                     "source_location": {"cell_index": 2}},
                    {"block_id": "c3_ai", "segment_id": "c3", "component": "ai_log",
                     "source_text": "No AI tools were used.", "output_text": "",
                     "source_location": {"cell_index": 3}},
                ]
            },
        }
        output = build_evidence_map(self.qmap, self.reqs, submission, "student.ipynb.json")
        question = output["questions"][0]

        self.assertEqual(output["mapping_method"], "notebook_explicit_question_component_blocks")
        self.assertEqual(output["summary"]["mapped_blocks"], 5)
        self.assertEqual(question["mapping_status"], "evidence_found")
        self.assertEqual(question["missing_components"], [])
        self.assertTrue(question["ai_log_present"])
        self.assertEqual(
            {key: len(value) for key, value in question["component_blocks"].items()},
            {"question": 1, "question_copy": 0, "code": 1, "output": 1, "explanation": 1, "ai_log": 1},
        )
        self.assertNotIn("No AI tools", " ".join(
            block["source_text"] for block in question["component_blocks"]["explanation"]
        ))

    def test_notebook_component_mapping_reports_missing_scored_components(self):
        submission = {
            "segments": [],
            "question_blocks": {
                "1": [
                    {"block_id": "c0_question", "segment_id": "c0", "component": "question",
                     "source_text": "# Q1", "output_text": "", "source_location": {"cell_index": 0}},
                    {"block_id": "c1_code", "segment_id": "c1", "component": "code",
                     "source_text": "print('done')", "output_text": "",
                     "source_location": {"cell_index": 1}},
                ]
            },
        }
        output = build_evidence_map(self.qmap, self.reqs, submission, "student.ipynb.json")
        question = output["questions"][0]

        self.assertEqual(question["missing_components"], ["output", "explanation"])
        self.assertFalse(question["ai_log_present"])

    def test_requirement_evidence_links_follow_notebook_subquestion_anchors(self):
        submission = {
            "question_blocks": {
                "1": [
                    {"block_id": "q1", "segment_id": "q1", "source_location": {"cell_index": 0},
                     "component": "question", "question_number": "1",
                     "source_text": "# 1. Data preparation\n# (a) Prepare data.\n# (b) Report missing values.",
                     "output_text": ""},
                    {"block_id": "a_code", "segment_id": "a", "source_location": {"cell_index": 1},
                     "component": "code", "question_number": "1",
                     "source_text": "df = prepare_data()", "output_text": ""},
                    {"block_id": "b_exp", "segment_id": "b", "source_location": {"cell_index": 2},
                     "component": "question_copy", "question_number": "1",
                     "source_text": "# (b) Report missing values.", "output_text": ""},
                    {"block_id": "b_code", "segment_id": "b2", "source_location": {"cell_index": 3},
                     "component": "code", "question_number": "1",
                     "source_text": "df.isna().sum()", "output_text": ""},
                ]
            },
            "segments": [],
        }
        reqs = {
            "requirements": [
                {"requirement_id": "a_r1", "question_id": "q1_occurrence_1",
                 "source_node_id": "q1_occurrence_1_part_a", "atomic_text": "Prepare data."},
                {"requirement_id": "b_r1", "question_id": "q1_occurrence_1",
                 "source_node_id": "q1_occurrence_1_part_b", "atomic_text": "Report missing values."},
            ],
            "global_requirements": [],
            "source_filename": "Example-QP.pdf",
        }
        output = build_evidence_map(self.qmap, reqs, submission, "student.ipynb")
        requirements = {r["requirement_id"]: r for r in output["questions"][0]["requirements"]}
        self.assertEqual(requirements["a_r1"]["linked_evidence_block_ids"], ["a_code"])
        self.assertEqual(requirements["b_r1"]["linked_evidence_block_ids"], ["b_code"])
        self.assertEqual(output["questions"][0]["evidence_blocks"][1]["requirement_ids"], ["a_r1"])

    def test_only_atomic_code_matches_are_linked_not_other_components(self):
        submission = {
            "question_blocks": {
                "1": [
                    {"block_id": "q1", "segment_id": "q1", "source_location": {"cell_index": 0},
                     "component": "question", "question_number": "1",
                     "source_text": "# 1. Data preparation\n# (a) Prepare data.\n# (b) Report missing values.",
                     "output_text": ""},
                    {"block_id": "a_code", "segment_id": "a", "source_location": {"cell_index": 1},
                     "component": "code", "question_number": "1",
                     "source_text": "df = prepare_data()", "output_text": ""},
                    {"block_id": "a_output", "segment_id": "a", "source_location": {"cell_index": 1},
                     "component": "output", "question_number": "1",
                     "source_text": "", "output_text": "prepared successfully"},
                    {"block_id": "a_explanation", "segment_id": "a", "source_location": {"cell_index": 1},
                     "component": "explanation", "question_number": "1",
                     "source_text": "I prepared the data.", "output_text": ""},
                ]
            },
            "segments": [],
        }
        reqs = {
            "requirements": [
                {"requirement_id": "a_r1", "question_id": "q1_occurrence_1",
                 "source_node_id": "q1_occurrence_1_part_a", "atomic_text": "Prepare data."},
            ],
            "global_requirements": [],
            "source_filename": "Example-QP.pdf",
        }
        output = build_evidence_map(self.qmap, reqs, submission, "student.ipynb")
        requirement = output["questions"][0]["requirements"][0]
        self.assertEqual(requirement["linked_code_block_ids"], ["a_code"])
        self.assertEqual(requirement["linked_evidence_block_ids"], ["a_code"])
        self.assertEqual(output["questions"][0]["evidence_blocks"][2]["requirement_ids"], [])
        self.assertEqual(
            output["questions"][0]["evidence_blocks"][2]["requirement_link_status"],
            "component_presence_only",
        )

    def test_requirement_gets_same_subquestion_bundle_and_mapping_classification(self):
        submission = {
            "question_blocks": {
                "1": [
                    {"block_id": "q1", "segment_id": "q1", "component": "question",
                     "source_text": "Question 1", "output_text": ""},
                    {"block_id": "a_copy", "segment_id": "a_copy", "component": "question_copy",
                     "subquestion_anchor": "a", "source_text": "(a) Explain the result.", "output_text": ""},
                    {"block_id": "a_explanation", "segment_id": "a_cell", "component": "explanation",
                     "subquestion_anchor": "a", "source_text": "The result is stable.", "output_text": ""},
                    {"block_id": "a_output", "segment_id": "a_cell", "component": "output",
                     "subquestion_anchor": "a", "source_text": "", "output_text": "stable"},
                    {"block_id": "b_explanation", "segment_id": "b_cell", "component": "explanation",
                     "subquestion_anchor": "b", "source_text": "Unrelated explanation.", "output_text": ""},
                ]
            },
            "segments": [],
        }
        reqs = {
            "requirements": [{
                "requirement_id": "a_r1", "question_id": "q1_occurrence_1",
                "source_node_id": "q1_occurrence_1_part_a",
                "atomic_text": "Explain the result.",
            }],
            "global_requirements": [],
            "source_filename": "Example-QP.pdf",
        }
        output = build_evidence_map(self.qmap, reqs, submission, "student.ipynb")
        requirement = output["questions"][0]["requirements"][0]
        self.assertEqual(requirement["mapping_diagnostics"]["category"], "explanation_only")
        self.assertEqual(
            requirement["evidence_bundle"]["block_ids"],
            ["a_explanation", "a_output"],
        )
        self.assertNotIn("b_explanation", requirement["evidence_bundle"]["block_ids"])

    def test_atomic_matching_does_not_use_full_subquestion_source_text(self):
        submission = {
            "question_blocks": {
                "1": [
                    {"block_id": "q1", "segment_id": "q1", "source_location": {"cell_index": 0},
                     "component": "question", "question_number": "1",
                     "source_text": "# 1. Experiments\n# (a) Train a linear model.\n# (b) Plot the validation error.",
                     "output_text": ""},
                    {"block_id": "a_code", "segment_id": "a", "source_location": {"cell_index": 1},
                     "component": "code", "question_number": "1",
                     "source_text": "model = LinearRegression().fit(X_train, y_train)", "output_text": ""},
                    {"block_id": "b_copy", "segment_id": "b", "source_location": {"cell_index": 2},
                     "component": "question_copy", "question_number": "1",
                     "source_text": "# (b) Plot the validation error.", "output_text": ""},
                    {"block_id": "b_code", "segment_id": "b2", "source_location": {"cell_index": 3},
                     "component": "code", "question_number": "1",
                     "source_text": "plt.plot(validation_errors)", "output_text": ""},
                ]
            },
            "segments": [],
        }
        reqs = {
            "requirements": [
                {"requirement_id": "a_r1", "question_id": "q1_occurrence_1",
                 "source_node_id": "q1_occurrence_1_part_a",
                 "atomic_text": "Train a linear model.",
                 "source_text": "(a) Train a linear model. Plot the validation error."},
                {"requirement_id": "b_r1", "question_id": "q1_occurrence_1",
                 "source_node_id": "q1_occurrence_1_part_b",
                 "atomic_text": "Plot the validation error.",
                 "source_text": "(b) Plot the validation error. Train a linear model."},
            ],
            "global_requirements": [],
            "source_filename": "Example-QP.pdf",
        }
        output = build_evidence_map(self.qmap, reqs, submission, "student.ipynb")
        requirements = {r["requirement_id"]: r for r in output["questions"][0]["requirements"]}
        self.assertEqual(requirements["a_r1"]["linked_code_block_ids"], ["a_code"])
        self.assertEqual(requirements["b_r1"]["linked_code_block_ids"], ["b_code"])

    def test_atomic_matching_recognizes_common_plot_prediction_and_swap_idioms(self):
        submission = {
            "question_blocks": {
                "1": [
                    {"block_id": "q1", "segment_id": "q1", "source_location": {"cell_index": 0},
                     "component": "question", "question_number": "1",
                     "source_text": "# 1. Experiments", "output_text": ""},
                    {"block_id": "plot", "segment_id": "a", "source_location": {"cell_index": 1},
                     "component": "code", "question_number": "1",
                     "source_text": "plt.xlabel('samples')\nplt.ylabel('error')\nplt.legend()", "output_text": ""},
                    {"block_id": "predict", "segment_id": "a", "source_location": {"cell_index": 2},
                     "component": "code", "question_number": "1",
                     "source_text": "x_unknown @ theta_e1", "output_text": ""},
                    {"block_id": "b_copy", "segment_id": "b", "source_location": {"cell_index": 3},
                     "component": "question_copy", "question_number": "1",
                     "source_text": "(b) Swap the training and test datasets.", "output_text": ""},
                    {"block_id": "swap", "segment_id": "b", "source_location": {"cell_index": 3},
                     "component": "code", "question_number": "1",
                     "source_text": "X_train, X_test = X_test, X_train", "output_text": "",
                     "subquestion_anchor": "b"},
                ]
            },
            "segments": [],
        }
        reqs = {
            "requirements": [
                {"requirement_id": "plot_r1", "question_id": "q1_occurrence_1",
                 "source_node_id": "q1_occurrence_1_part_a",
                 "atomic_text": "Clearly label the axes and include a legend."},
                {"requirement_id": "predict_r1", "question_id": "q1_occurrence_1",
                 "source_node_id": "q1_occurrence_1_part_a",
                 "atomic_text": "Predict the target output for an unknown input."},
                {"requirement_id": "swap_r1", "question_id": "q1_occurrence_1",
                 "source_node_id": "q1_occurrence_1_part_b",
                 "atomic_text": "Swap the training and test datasets."},
            ],
            "global_requirements": [],
            "source_filename": "Example-QP.pdf",
        }
        output = build_evidence_map(self.qmap, reqs, submission, "student.ipynb")
        requirements = {r["requirement_id"]: r for r in output["questions"][0]["requirements"]}
        self.assertEqual(requirements["plot_r1"]["linked_code_block_ids"], ["plot"])
        self.assertEqual(requirements["predict_r1"]["linked_code_block_ids"], ["predict"])
        self.assertEqual(requirements["swap_r1"]["linked_code_block_ids"], ["swap"])

    def test_pdf_prose_preserves_inference_explanation_and_ai_log_components(self):
        submission = {
            "segments": [{
                "segment_id": "page_10",
                "position": 9,
                "original_type": "pdf_page",
                "content": (
                    "1. Data preparation\n"
                    "E1:\n1.0\nE2:\n0.5\nAI prompt: higher MSE means bad or good?\n"
                    "(b) Based on the results, list all possible inferences.\n"
                    "E2 performed better than E1."
                ),
            }],
        }
        reqs = {
            "requirements": [
                {"requirement_id": "q1_r1", "question_id": "q1_occurrence_1",
                 "source_node_id": "q1_occurrence_1_part_b",
                 "atomic_text": "List all possible inferences."},
            ],
            "global_requirements": [],
            "source_filename": "Example-QP.pdf",
        }
        output = build_evidence_map(self.qmap, reqs, submission, "student.pdf")
        question = output["questions"][0]
        self.assertTrue(any(b["component"] == "output" for b in question["evidence_blocks"]))
        self.assertTrue(any(b["component"] == "ai_log" for b in question["evidence_blocks"]))
        explanation = next(b for b in question["evidence_blocks"] if b["component"] == "explanation")
        self.assertEqual(explanation["subquestion_anchor"], "b")
        self.assertIn("E2 performed better", explanation["source_text"])


if __name__ == "__main__":
    unittest.main()
