import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_question_requirements import process_file


class BuildQuestionRequirementsTests(unittest.TestCase):
    def test_builds_question_map_and_requirements_without_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            source = temp_path / "DEMO-QP.json"
            source.write_text(json.dumps({
                "submission": {"source_filename": "DEMO-QP.pdf"},
                "segments": [{
                    "source_location": {"page_number": 1},
                    "content": "Instructions:\n- Use a dataset with at least 10 rows.\n1. Clean the data and train a model.\n(a) Calculate accuracy; plot a confusion matrix.",
                }],
            }), encoding="utf-8")
            map_dir = temp_path / "questions"
            req_dir = temp_path / "requirements"
            summary = process_file(source, map_dir, req_dir)

            question_map = json.loads(Path(summary["question_map"]).read_text(encoding="utf-8"))
            requirements = json.loads(Path(summary["requirements_file"]).read_text(encoding="utf-8"))
            self.assertEqual(question_map["source_filename"], "DEMO-QP.pdf")
            self.assertEqual(summary["questions"], 1)
            self.assertGreater(summary["requirements"], 0)
            self.assertFalse(requirements["extraction"]["student_code_executed"])
            self.assertTrue((map_dir / "DEMO-QP.question-map.json").is_file())
            self.assertTrue((req_dir / "DEMO-QP.requirements.json").is_file())

    def test_rejects_raw_json_without_segments(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            source = temp_path / "not_pdf_parser.json"
            source.write_text(json.dumps({"questions": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "segments"):
                process_file(source, temp_path / "q", temp_path / "r")


if __name__ == "__main__":
    unittest.main()
