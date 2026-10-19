import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.requirements.requirement_extractor import extract_atomic_requirements


class RequirementExtractorTests(unittest.TestCase):
    def setUp(self):
        self.parsed = {
            "schema_version": "1.0",
            "source_filename": "sample.pdf",
            "global_instructions": [{"text": "Use a dataset with at least 500 instances.", "source_pages": [1]}],
            "questions": [{
                "question_id": "q1_occurrence_1", "number": "1", "occurrence": 1,
                "text": "1. Clean the data and train a model.",
                "raw_lines": ["1. Clean the data and train a model.", "Use the same dataset for both tasks.", "(a) Calculate accuracy; plot the confusion matrix."],
                "source_pages": [2], "alternative_markers": [],
                "subparts": [{"label": "a", "text": "(a) Calculate accuracy; plot the confusion matrix.", "raw_lines": ["(a) Calculate accuracy; plot the confusion matrix."], "source_pages": [2], "nested_subparts": []}],
            }],
            "notes": [], "alternatives": [], "diagnostics": [],
        }

    def test_output_schema_and_no_execution(self):
        result = extract_atomic_requirements(self.parsed)
        self.assertEqual(result["schema_version"], "1.1")
        self.assertFalse(result["extraction"]["student_code_executed"])
        self.assertEqual(result["source_filename"], "sample.pdf")

    def test_global_instruction_preserved(self):
        result = extract_atomic_requirements(self.parsed)
        self.assertEqual(result["global_requirements"][0]["source_text"], "Use a dataset with at least 500 instances.")
        self.assertEqual(result["global_requirements"][0]["requirement_type"], "constraint")

    def test_splits_obvious_independent_actions(self):
        result = extract_atomic_requirements(self.parsed)
        q_requirements = result["questions"][0]["requirements"]
        self.assertEqual([r["atomic_text"] for r in q_requirements[:2]], ["Clean the data", "train a model."])
        part_requirements = result["questions"][0]["subparts"][0]["requirements"]
        self.assertEqual([r["atomic_text"] for r in part_requirements], ["Calculate accuracy", "plot the confusion matrix."])

    def test_keeps_dependent_same_model_clause_together(self):
        parsed = {
            "schema_version": "1.0",
            "source_filename": "dependent.pdf",
            "global_instructions": [],
            "questions": [{
                "question_id": "q1_occurrence_1", "number": "1", "occurrence": 1,
                "text": "1. Swap the training and test datasets and train the same model.",
                "raw_lines": ["1. Swap the training and test datasets and train the same model."],
                "source_pages": [1], "alternative_markers": [],
                "subparts": [],
            }],
            "notes": [], "alternatives": [], "diagnostics": [],
        }
        result = extract_atomic_requirements(parsed)
        requirements = result["questions"][0]["requirements"]
        self.assertEqual(
            [item["atomic_text"] for item in requirements],
            ["Swap the training and test datasets and train the same model."],
        )

    def test_parent_context_is_preserved_without_subpart_duplication(self):
        result = extract_atomic_requirements(self.parsed)
        parent_requirements = [
            r for r in result["questions"][0]["requirements"]
            if r["source_category"] in {"question_prompt", "question_context"}
        ]
        parent_sources = [r["source_text"] for r in parent_requirements]
        self.assertIn("Use the same dataset for both tasks.", parent_sources)
        self.assertNotIn("(a) Calculate accuracy; plot the confusion matrix.", parent_sources)

    def test_keeps_source_text_and_provenance(self):
        result = extract_atomic_requirements(self.parsed)
        req = result["questions"][0]["subparts"][0]["requirements"][0]
        self.assertEqual(req["source_text"], "(a) Calculate accuracy; plot the confusion matrix.")
        self.assertEqual(req["source_pages"], [2])
        start, end = req["source_span"]["start"], req["source_span"]["end"]
        self.assertEqual(req["source_text"][start:end], req["atomic_text"])
        self.assertEqual(req["status"], "unassessed")

    def test_multiple_requirements_have_unique_ids(self):
        result = extract_atomic_requirements(self.parsed)
        ids = [r["requirement_id"] for r in result["requirements"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_duplicate_sibling_labels_are_disambiguated_in_requirement_ids(self):
        parsed = {
            "source_filename": "duplicate-labels.pdf",
            "global_instructions": [],
            "questions": [{
                "question_id": "q1_occurrence_1", "number": "1", "occurrence": 1,
                "text": "1. Experiments\n(a) Run the experiments.\ni. First experiment.\ni. Second experiment.",
                "raw_lines": ["1. Experiments", "(a) Run the experiments.", "i. First experiment.", "i. Second experiment."],
                "source_pages": [1], "alternative_markers": [],
                "subparts": [{
                    "label": "a", "text": "(a) Run the experiments.",
                    "raw_lines": ["(a) Run the experiments."], "source_pages": [1],
                    "nested_subparts": [
                        {"label": "i", "text": "i. First experiment.", "raw_lines": ["i. First experiment."], "source_pages": [1], "nested_subparts": []},
                        {"label": "i", "text": "i. Second experiment.", "raw_lines": ["i. Second experiment."], "source_pages": [1], "nested_subparts": []},
                    ],
                }],
            }],
            "notes": [], "alternatives": [], "diagnostics": [],
        }
        result = extract_atomic_requirements(parsed)
        all_ids = [record["requirement_id"] for record in result["requirements"]]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        nested_ids = [node["source_node_id"] for node in result["questions"][0]["subparts"][0]["nested_subparts"]]
        self.assertEqual(nested_ids, [
            "q1_occurrence_1_part_a_part_i_occurrence_1",
            "q1_occurrence_1_part_a_part_i_occurrence_2",
        ])

    def test_decimal_values_are_not_split_at_decimal_points(self):
        text = "Construct values (3.1, 4.5, 0.1, 8.1, 9.0) within the stated bounds."
        parsed = {
            "source_filename": "decimals.pdf",
            "global_instructions": [{"text": text, "source_pages": [1]}],
            "questions": [], "notes": [], "alternatives": [], "diagnostics": [],
        }
        result = extract_atomic_requirements(parsed)
        reqs = result["global_requirements"]
        self.assertEqual(len(reqs), 1, [req["atomic_text"] for req in reqs])
        self.assertEqual(reqs[0]["atomic_text"], text)
        start, end = reqs[0]["source_span"]["start"], reqs[0]["source_span"]["end"]
        self.assertEqual(reqs[0]["source_text"][start:end], reqs[0]["atomic_text"])

    def test_uppercase_alphabetic_marker_is_not_a_standalone_requirement(self):
        text = "B. Experiment 2 (E2): Optimization strategy should be Batch Gradient Descent."
        parsed = {
            "source_filename": "alpha-label.pdf",
            "global_instructions": [{"text": text, "source_pages": [1]}],
            "questions": [], "notes": [], "alternatives": [], "diagnostics": [],
        }
        result = extract_atomic_requirements(parsed)
        reqs = result["global_requirements"]
        self.assertEqual(len(reqs), 1, [req["atomic_text"] for req in reqs])
        self.assertTrue(reqs[0]["atomic_text"].startswith("Experiment 2 (E2):"))
        start, end = reqs[0]["source_span"]["start"], reqs[0]["source_span"]["end"]
        self.assertEqual(reqs[0]["source_text"][start:end], reqs[0]["atomic_text"])

    def test_alphabetic_labels_inside_multiline_blocks_are_not_split_off(self):
        text = "i. Predicting the new output\nA. Append predicted values as columns.\nB. Change the numerals to labels."
        parsed = {
            "source_filename": "nested-alpha-labels.pdf",
            "global_instructions": [{"text": text, "source_pages": [1]}],
            "questions": [], "notes": [], "alternatives": [], "diagnostics": [],
        }
        result = extract_atomic_requirements(parsed)
        reqs = result["global_requirements"]
        self.assertFalse(any(req["atomic_text"].strip() in {"A.", "B."} for req in reqs))
        self.assertTrue(any("A. Append predicted values as columns." in req["atomic_text"] for req in reqs))
        self.assertTrue(any("B. Change the numerals to labels." in req["atomic_text"] for req in reqs))
        for req in reqs:
            start, end = req["source_span"]["start"], req["source_span"]["end"]
            self.assertEqual(req["source_text"][start:end], req["atomic_text"])

    def test_empty_questions_supported(self):
        result = extract_atomic_requirements({"source_filename": "empty.pdf", "questions": [], "global_instructions": []})
        self.assertEqual(result["requirements"], [])

    def test_pdf_wrapped_parent_text_is_joined_without_losing_source_wording(self):
        parsed = {
            "source_filename": "wrapped.pdf",
            "global_instructions": [],
            "questions": [{
                "question_id": "q1_occurrence_1", "number": "1", "occurrence": 1,
                "text": "1. Take a dataset\nper-\nform classification.",
                "raw_lines": ["1. Take a dataset", "per-", "form classification."],
                "source_pages": [1], "subparts": [], "alternative_markers": [],
            }],
            "notes": [], "alternatives": [], "diagnostics": [],
        }
        result = extract_atomic_requirements(parsed)
        reqs = result["questions"][0]["requirements"]
        self.assertEqual(len(reqs), 1)
        self.assertEqual(reqs[0]["atomic_text"], "Take a dataset\nper-\nform classification.")
        self.assertEqual(reqs[0]["requirement_type"], "action")
        for req in reqs:
            start, end = req["source_span"]["start"], req["source_span"]["end"]
            self.assertEqual(req["source_text"][start:end], req["atomic_text"])

    def test_structural_global_heading_is_not_emitted_as_requirement(self):
        parsed = {
            "source_filename": "heading.pdf",
            "global_instructions": [
                {"text": "Instructions:", "source_pages": [1]},
                {"text": "Add an explanation of your logic.", "source_pages": [1]},
            ],
            "questions": [], "notes": [], "alternatives": [], "diagnostics": [],
        }
        result = extract_atomic_requirements(parsed)
        self.assertEqual(len(result["global_requirements"]), 1)
        self.assertEqual(result["global_requirements"][0]["atomic_text"], "Add an explanation of your logic.")
        self.assertEqual(result["global_requirements"][0]["requirement_type"], "reasoning")

    def test_wrap_with_lowercase_continuation_uses_one_source_block(self):
        parsed = {
            "source_filename": "wrapped.pdf", "global_instructions": [],
            "questions": [{
                "question_id": "q2_occurrence_1", "number": "2", "occurrence": 1,
                "text": "2. Use the panda tutorial sheet, which is uploaded in ms-team\nclass-room materials folder.",
                "raw_lines": ["2. Use the panda tutorial sheet, which is uploaded in ms-team", "class-room materials folder."],
                "source_pages": [1], "subparts": [], "alternative_markers": [],
            }],
            "notes": [], "alternatives": [], "diagnostics": [],
        }
        result = extract_atomic_requirements(parsed)
        reqs = result["questions"][0]["requirements"]
        self.assertEqual(len(reqs), 1)
        self.assertIn("\nclass-room materials folder.", reqs[0]["atomic_text"])
        start, end = reqs[0]["source_span"]["start"], reqs[0]["source_span"]["end"]
        self.assertEqual(reqs[0]["source_text"][start:end], reqs[0]["atomic_text"])

    def test_invalid_input_rejected(self):
        with self.assertRaises(TypeError):
            extract_atomic_requirements([])


if __name__ == "__main__":
    unittest.main()
