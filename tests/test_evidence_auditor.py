import unittest

from scripts.audit_evidence_maps import audit_one
from src.evaluation.evidence_mapper import build_evidence_map


class EvidenceAuditSemanticChecks(unittest.TestCase):
    def setUp(self):
        self.qmap = {
            "schema_version": "1.1",
            "source_filename": "Example-QP.pdf",
            "questions": [
                {"question_id": "q1_occurrence_1", "number": "1", "occurrence": 1, "text": "1. Data preparation", "source_pages": [1]},
                {"question_id": "q2_occurrence_1", "number": "2", "occurrence": 1, "text": "2. Query design", "source_pages": [2]},
            ],
            "standalone_tasks": [],
        }
        self.reqs = {
            "schema_version": "1.1", "source_filename": "Example-QP.pdf",
            "requirements": [
                {"requirement_id": "q1_r1", "question_id": "q1_occurrence_1", "question_number": "1", "requirement_type": "action", "atomic_text": "Prepare data."},
                {"requirement_id": "q2_r1", "question_id": "q2_occurrence_1", "question_number": "2", "requirement_type": "action", "atomic_text": "Write a query."},
            ],
            "global_requirements": [],
        }
        self.submission = {"segments": [
            {"segment_id": "c1", "position": 0, "original_type": "markdown", "content": "1. Data preparation"},
            {"segment_id": "c2", "position": 1, "original_type": "code", "content": "q1 = df[col].quantile(0.25)"},
            {"segment_id": "c3", "position": 2, "original_type": "markdown", "content": "2. Query design"},
        ]}

    def test_audit_rejects_code_cell_claimed_as_explicit_heading(self):
        output = build_evidence_map(self.qmap, self.reqs, self.submission, "student.json")
        block = next(b for q in output["questions"] for b in q["evidence_blocks"] if b["segment_id"] == "c2")
        block["mapping_method"] = "explicit_question_heading"
        block["heading_number"] = "1"
        block["heading_title"] = "= df[col].quantile(0.25)"
        result = audit_one(self.qmap, self.reqs, self.submission, "student.json", output)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("code segment c2" in error for error in result["errors"]))

    def test_audit_rejects_assignment_like_heading_in_page_text(self):
        submission = {"segments": [{"segment_id": "p1", "position": 0, "original_type": "pdf_page", "content": "1. Data preparation\nq1 = df[col].quantile(0.25)\n2. Query design"}]}
        output = build_evidence_map(self.qmap, self.reqs, submission, "student.pdf.json")
        # Simulate the old mapper incorrectly cutting the page at the q1 assignment.
        q1_blocks = output["questions"][0]["evidence_blocks"]
        original = q1_blocks[0]
        source_text = original["source_text"]
        assignment = "q1 = df[col].quantile(0.25)\n"
        assignment_start = source_text.index(assignment)
        first_text = source_text[:assignment_start]
        second_text = source_text[assignment_start:]
        first_block = dict(original)
        first_block.update({"block_id": "p1_block_001", "source_text": first_text, "segment_start_offset": 0, "segment_end_offset": len(first_text), "mapping_method": "explicit_question_heading", "mapping_status": "mapped", "heading_number": "1", "heading_title": "Data preparation"})
        bad_block = dict(original)
        bad_block.update({"block_id": "p1_block_002", "source_text": second_text, "segment_start_offset": assignment_start, "segment_end_offset": len(source_text), "mapping_method": "explicit_question_heading", "mapping_status": "mapped", "heading_number": "1", "heading_title": "= df[col].quantile(0.25)"})
        output["questions"][0]["evidence_blocks"] = [first_block, bad_block]
        output["questions"][0]["evidence_block_count"] = 2
        output["summary"]["evidence_blocks"] = 3
        output["summary"]["mapped_blocks"] = 3
        result = audit_one(self.qmap, self.reqs, submission, "student.pdf.json", output)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("variable assignment" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
