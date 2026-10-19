
import json
import tempfile
import unittest
from pathlib import Path

from src.ingestion.notebook_parser import parse_notebook


class TestNotebookParser(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()
        self.notebook_path = (
            self.input_dir / "PAMCA504-25MCA1034-PAT2.ipynb"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_notebook(self, cells):
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3"}},
            "cells": cells
        }
        self.notebook_path.write_text(
            json.dumps(notebook),
            encoding="utf-8"
        )

    def test_extracts_filename_metadata(self):
        self.write_notebook([])
        result = parse_notebook(self.notebook_path, self.input_dir)
        submission = result["submission"]

        self.assertEqual(submission["student_roll_no"], "25MCA1034")
        self.assertEqual(submission["course_name"], "PAMCA504")
        self.assertEqual(submission["assessment_name"], "PAT2")

    def test_preserves_cells_and_outputs(self):
        cells = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["Question 1\n", "Explain the result."]
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "source": ["print(1 + 1)"],
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": ["2\n"]
                    }
                ]
            }
        ]
        self.write_notebook(cells)

        result = parse_notebook(self.notebook_path, self.input_dir)

        self.assertEqual(result["submission"]["extraction_status"], "success")
        self.assertEqual(len(result["segments"]), 2)
        self.assertEqual(
            result["segments"][0]["content"],
            "Question 1\nExplain the result."
        )
        self.assertEqual(result["segments"][1]["content"], "print(1 + 1)")
        self.assertEqual(
            result["segments"][1]["outputs"],
            cells[1]["outputs"]
        )
        self.assertEqual(
            result["segments"][1]["original_cell"],
            cells[1]
        )

    def test_classifies_question_components_without_merging_ai_log_into_explanation(self):
        cells = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Q1: Build the model"],
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "source": ["model.fit(X, y)"],
                "outputs": [{"output_type": "stream", "text": ["accuracy: 0.8\n"]}],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["The model is trained on the prepared features."],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## AI Log\nNo AI tools were used."],
            },
        ]
        self.write_notebook(cells)

        result = parse_notebook(self.notebook_path, self.input_dir)

        blocks = result["question_blocks"]["1"]
        self.assertEqual(
            [block["component"] for block in blocks],
            ["question", "code", "output", "explanation", "ai_log"],
        )
        self.assertEqual(result["question_component_summary"]["1"]["missing_components"], [])
        self.assertTrue(result["question_component_summary"]["1"]["ai_log_present"])
        self.assertEqual(result["component_contract"]["scored_components"], ["code", "output", "explanation"])

    def test_reports_missing_output_and_explanation_components(self):
        self.write_notebook([
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["Question 2: Explain the result"],
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "source": ["print('done')"],
                "outputs": [],
            },
        ])

        result = parse_notebook(self.notebook_path, self.input_dir)

        summary = result["question_component_summary"]["2"]
        self.assertEqual(summary["present_components"], ["code", "question"])
        self.assertEqual(summary["missing_components"], ["output", "explanation"])
        self.assertFalse(summary["ai_log_present"])

    def test_copied_subquestion_markdown_is_not_counted_as_explanation(self):
        self.write_notebook([
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Q1: Build the model"],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# (a) Train the model and report its accuracy [2 marks]"],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["The model uses the prepared features and is evaluated on held-out data."],
            },
        ])

        result = parse_notebook(self.notebook_path, self.input_dir)

        self.assertEqual(
            [block["component"] for block in result["question_blocks"]["1"]],
            ["question", "question_copy", "explanation"],
        )
        self.assertEqual(
            result["question_component_summary"]["1"]["present_components"],
            ["explanation", "question", "question_copy"],
        )

    def test_question_heading_can_follow_notebook_preamble_lines(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "PAMCA504-25MCA1029-PAT1.ipynb"
            path.write_text(json.dumps({
                "cells": [
                    {
                        "cell_type": "markdown",
                        "source": [
                            "# Instructions before the answer\n",
                            "# 1. Data Preprocessing\n",
                            "# (a) Check the data\n",
                        ],
                    },
                    {
                        "cell_type": "code",
                        "source": ["print('ok')"],
                        "outputs": [{"output_type": "stream", "text": ["ok"]}],
                    },
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }), encoding="utf-8")
            result = parse_notebook(path, root)
            self.assertIn("1", result["question_blocks"])
            self.assertEqual(result["segments"][0]["question_number"], "1")
            self.assertEqual(result["segments"][1]["question_number"], "1")

    def test_compact_markdown_headings_define_question_boundaries(self):
        self.write_notebook([
            {"cell_type": "markdown", "source": ["### 1.DATA PREPROCESSING"]},
            {"cell_type": "code", "source": ["answer_one()"], "outputs": []},
            {"cell_type": "markdown", "source": ["### 2. Training the models"]},
            {"cell_type": "code", "source": ["answer_two()"], "outputs": []},
            {"cell_type": "markdown", "source": ["### 3.Inferences:"]},
            {"cell_type": "markdown", "source": ["Conclusion for the experiments."]},
        ])

        result = parse_notebook(self.notebook_path, self.input_dir)

        self.assertEqual(
            {
                number: [block["component"] for block in blocks]
                for number, blocks in result["question_blocks"].items()
            },
            {
                "1": ["question", "code"],
                "2": ["question", "code"],
                "3": ["question", "explanation"],
            },
        )

    def test_numbered_inference_list_is_not_a_new_question(self):
        self.write_notebook([
            {"cell_type": "markdown", "source": ["### 3.Inferences:"]},
            {"cell_type": "markdown", "source": [
                "1. First inference.\n",
                "2. Second inference.",
            ]},
        ])

        result = parse_notebook(self.notebook_path, self.input_dir)

        self.assertEqual(set(result["question_blocks"]), {"3"})
        self.assertEqual(result["question_blocks"]["3"][1]["component"], "explanation")

    def test_nested_mime_output_and_ai_prompt_are_preserved(self):
        self.write_notebook([
            {"cell_type": "markdown", "source": ["# Q1: Results"]},
            {
                "cell_type": "code",
                "source": ["result()"],
                "outputs": [{
                    "output_type": "execute_result",
                    "data": {
                        "text/plain": ["value: 1\n"],
                        "text/html": ["<b>value: 1</b>"],
                    },
                }],
            },
            {"cell_type": "markdown", "source": ["No AI used for this answer."]},
            {"cell_type": "markdown", "source": ["AI prompt: summarize the result."]},
        ])

        result = parse_notebook(self.notebook_path, self.input_dir)
        blocks = result["question_blocks"]["1"]

        self.assertEqual(
            [block["component"] for block in blocks],
            ["question", "code", "output", "ai_log", "ai_log"],
        )
        self.assertIn("value: 1", blocks[2]["output_text"])

    def test_empty_cell_is_warning_not_extraction_failure(self):
        self.write_notebook([
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [],
                "outputs": []
            }
        ])

        result = parse_notebook(self.notebook_path, self.input_dir)

        self.assertEqual(result["submission"]["extraction_status"], "success")
        self.assertTrue(result["segments"][0]["diagnostics"])

    def test_unsupported_source_structure_is_partial(self):
        self.write_notebook([
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": 123,
                "outputs": []
            }
        ])

        result = parse_notebook(self.notebook_path, self.input_dir)

        self.assertEqual(result["submission"]["extraction_status"], "partial")
        self.assertTrue(result["diagnostics"])

    def test_malformed_notebook_is_failed(self):
        self.notebook_path.write_text(
            '{"cells": [',
            encoding="utf-8"
        )

        result = parse_notebook(self.notebook_path, self.input_dir)

        self.assertEqual(result["submission"]["extraction_status"], "failed")
        self.assertTrue(result["diagnostics"])

    def test_parser_does_not_execute_code(self):
        marker = self.root / "student_code_executed.txt"
        code = (
            f"from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('executed')"
        )
        self.write_notebook([
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": code.splitlines(keepends=True),
                "outputs": []
            }
        ])

        parse_notebook(self.notebook_path, self.input_dir)

        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
