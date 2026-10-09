
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
