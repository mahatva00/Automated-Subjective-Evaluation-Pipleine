import unittest

from src.evaluation.evidence_mapper import build_evidence_map


class FormatParityTests(unittest.TestCase):
    def setUp(self):
        self.qmap = {
            "schema_version": "1.1",
            "source_filename": "Example-QP.pdf",
            "questions": [{
                "question_id": "q1_occurrence_1",
                "number": "1",
                "occurrence": 1,
                "text": "1. Data preparation\n(a) Prepare data.",
                "source_pages": [1],
            }],
            "standalone_tasks": [],
        }
        self.reqs = {
            "schema_version": "1.1",
            "source_filename": "Example-QP.pdf",
            "requirements": [{
                "requirement_id": "q1_occurrence_1_part_a_r1",
                "question_id": "q1_occurrence_1",
                "question_number": "1",
                "source_node_id": "q1_occurrence_1_part_a",
                "atomic_text": "Prepare data.",
                "requirement_type": "action",
            }],
            "global_requirements": [],
        }

    @staticmethod
    def _signature(result):
        question = result["questions"][0]
        return {
            "question_numbers": [item["question_number"] for item in result["questions"]],
            "component_presence": {
                component: bool(question["component_blocks"].get(component))
                for component in ("code", "output", "explanation")
            },
            "requirement_count": len(question["requirements"]),
            "linked_code_count": len(question["requirements"][0]["linked_code_block_ids"]),
            "output_text": " ".join(
                block.get("output_text", "")
                for block in question["component_blocks"]["output"]
            ).strip(),
        }

    def test_equivalent_notebook_and_pdf_submission_preserve_semantic_evidence(self):
        notebook = {
            "question_blocks": {
                "1": [
                    {"block_id": "cell_01_question", "segment_id": "cell_01",
                     "source_location": {"cell_index": 0}, "component": "question",
                     "source_text": "### 1. Data preparation", "output_text": ""},
                    {"block_id": "cell_02_question_copy", "segment_id": "cell_02",
                     "source_location": {"cell_index": 1}, "component": "question_copy",
                     "subquestion_anchor": "a", "source_text": "(a) Prepare data.", "output_text": ""},
                    {"block_id": "cell_03_code", "segment_id": "cell_03",
                     "source_location": {"cell_index": 2}, "component": "code",
                     "subquestion_anchor": "a", "source_text": "df = prepare_data()",
                     "output_text": "prepared successfully"},
                    {"block_id": "cell_03_output", "segment_id": "cell_03",
                     "source_location": {"cell_index": 2}, "component": "output",
                     "subquestion_anchor": "a", "source_text": "",
                     "output_text": "prepared successfully"},
                    {"block_id": "cell_04_explanation", "segment_id": "cell_04",
                     "source_location": {"cell_index": 3}, "component": "explanation",
                     "subquestion_anchor": "a",
                     "source_text": "The data was prepared successfully.", "output_text": ""},
                ]
            },
            "segments": [],
        }
        pdf = {
            "segments": [{
                "segment_id": "page_01",
                "position": 0,
                "original_type": "pdf_page",
                "content": (
                    "1. Data preparation\n"
                    "(a) Prepare data.\n"
                    "In [1]:\n"
                    "df = prepare_data()\n"
                    "Out [1]:\n"
                    "prepared successfully\n"
                    "Explanation: The data was prepared successfully."
                ),
            }]
        }
        notebook_map = build_evidence_map(self.qmap, self.reqs, notebook, "student.ipynb")
        pdf_map = build_evidence_map(self.qmap, self.reqs, pdf, "student.pdf")
        self.assertEqual(self._signature(notebook_map), self._signature(pdf_map))
        self.assertTrue(notebook_map["questions"][0]["requirements"][0]["linked_code_block_ids"])
        self.assertTrue(pdf_map["questions"][0]["requirements"][0]["linked_code_block_ids"])


if __name__ == "__main__":
    unittest.main()
