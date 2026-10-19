import copy
import hashlib
import unittest

from src.annotation.coverage import build_coverage, validate_coverage, coverage_summary
from src.annotation.scope import build_scope_review


def make_doc(text="Title\nQuestion 1. Do task A.\nContext line.\n"):
    return {
        "schema_version": "1.0.0",
        "document_id": "test-doc-001",
        "source": {"filename": "test.pdf", "format": "pdf", "document_type": "question_paper"},
        "text": text,
        "page_ranges": [],
        "annotations": [],
        "relations": [],
        "annotation_metadata": {"status": "draft", "guidelines_version": "1.0.0", "annotator_ids": [], "notes": ""},
    }


def mark_all(coverage):
    for chunk in coverage["chunks"]:
        chunk["structure_status"] = "reviewed"
        chunk["semantic_status"] = "reviewed"
        chunk["structure_reviewer"] = "test-reviewer"
        chunk["structure_review_date"] = "2026-10-13"
        chunk["semantic_reviewer"] = "test-reviewer"
        chunk["semantic_review_date"] = "2026-10-13"


class CoverageWorkflowTests(unittest.TestCase):
    def test_chunks_partition_source_without_losing_characters(self):
        text = ("A sentence.\n" * 75) + "tail"
        doc = make_doc(text)
        cov = build_coverage(doc, "test.annotation.json", max_chars=100)
        self.assertEqual(cov["chunks"][0]["start_char"], 0)
        self.assertEqual(cov["chunks"][-1]["end_char"], len(text))
        self.assertEqual("".join(c["text"] for c in cov["chunks"]), text)
        cursor = 0
        for chunk in cov["chunks"]:
            self.assertEqual(chunk["start_char"], cursor)
            cursor = chunk["end_char"]
        self.assertEqual(validate_coverage(doc, cov), [])

    def test_all_pending_is_valid_but_not_complete(self):
        doc = make_doc()
        cov = build_coverage(doc, "test.annotation.json", max_chars=100)
        self.assertEqual(validate_coverage(doc, cov), [])
        self.assertFalse(coverage_summary(cov)["complete_both_layers"])

    def test_changed_source_text_fails_hash_and_length_checks(self):
        doc = make_doc()
        cov = build_coverage(doc, "test.annotation.json", max_chars=100)
        doc["text"] += "changed"
        errors = validate_coverage(doc, cov)
        self.assertTrue(any("sha256 mismatch" in e for e in errors))
        self.assertTrue(any("text_length mismatch" in e for e in errors))

    def test_gap_in_ranges_fails(self):
        doc = make_doc()
        cov = build_coverage(doc, "test.annotation.json", max_chars=100)
        if len(cov["chunks"]) == 1:
            cov["chunks"][0]["start_char"] = 1
        else:
            cov["chunks"][1]["start_char"] += 1
        self.assertTrue(any("does not continue coverage" in e for e in validate_coverage(doc, cov)))

    def test_reviewed_document_requires_complete_coverage(self):
        doc = make_doc()
        cov = build_coverage(doc, "test.annotation.json", max_chars=100)
        doc["annotation_metadata"]["status"] = "reviewed"
        errors = validate_coverage(doc, cov)
        self.assertTrue(any("full structure and semantic coverage of source text is not reviewed" in e for e in errors))

    def test_fully_reviewed_coverage_passes_for_reviewed_document(self):
        doc = make_doc()
        cov = build_coverage(doc, "test.annotation.json", max_chars=100)
        mark_all(cov)
        doc["annotation_metadata"]["status"] = "reviewed"
        self.assertEqual(validate_coverage(doc, cov), [])
        self.assertTrue(coverage_summary(cov)["complete_both_layers"])

    def test_reviewed_annotation_cannot_overlap_pending_chunk(self):
        doc = make_doc()
        doc["annotations"] = [{
            "annotation_id": "sem-1", "layer": "semantic", "label": "context",
            "start_char": 0, "end_char": 5, "text": "Title", "attributes": {},
            "review_status": "reviewed", "annotator_id": "reviewer-1"
        }]
        cov = build_coverage(doc, "test.annotation.json", max_chars=100)
        errors = validate_coverage(doc, cov)
        self.assertTrue(any("marked reviewed but overlaps a pending semantic coverage chunk" in e for e in errors))

    def test_out_of_scope_legacy_annotation_does_not_require_coverage_review_when_scope_is_supplied(self):
        doc = make_doc("GLOBAL note.\n\nQuestion 1: do task A.\n")
        start, end = 0, len("GLOBAL note.")
        doc["annotations"] = [{
            "annotation_id": "legacy-global", "layer": "semantic", "label": "other",
            "start_char": start, "end_char": end, "text": doc["text"][start:end],
            "attributes": {}, "review_status": "reviewed", "annotator_id": "reviewer"
        }]
        scope, _ = build_scope_review(doc, "test.annotation.json", max_chars=100)
        for seg in scope["segments"]:
            if seg["scope_status"] == "separator":
                continue
            out = "GLOBAL" in seg["text"]
            seg.update(scope_status="out_of_scope" if out else "in_scope",
                       question_id=None if out else "q1", reason="reviewed",
                       review_status="reviewed", reviewer="reviewer", review_date="2026-10-13")
        cov = build_coverage(doc, "test.annotation.json", max_chars=100)
        # Keep source-only coverage pending: it is irrelevant for the excluded annotation.
        errors = validate_coverage(doc, cov, scope=scope)
        self.assertFalse(any("legacy-global" in error for error in errors), errors)

    def test_reviewed_layer_requires_reviewer_and_iso_date(self):
        doc = make_doc()
        cov = build_coverage(doc, "test.annotation.json", max_chars=100)
        chunk = cov["chunks"][0]
        chunk["semantic_status"] = "reviewed"
        errors = validate_coverage(doc, cov)
        self.assertTrue(any("semantic_reviewer is required" in e for e in errors))
        self.assertTrue(any("semantic_review_date is required" in e for e in errors))
        chunk["semantic_reviewer"] = "tester"
        chunk["semantic_review_date"] = "not-a-date"
        errors = validate_coverage(doc, cov)
        self.assertTrue(any("must be an ISO date" in e for e in errors))

    def test_chunk_text_or_hash_tampering_fails(self):
        doc = make_doc()
        cov = build_coverage(doc, "test.annotation.json", max_chars=100)
        cov = copy.deepcopy(cov)
        cov["chunks"][0]["text"] += "tamper"
        self.assertTrue(any("text does not exactly match" in e for e in validate_coverage(doc, cov)))


if __name__ == "__main__":
    unittest.main()
