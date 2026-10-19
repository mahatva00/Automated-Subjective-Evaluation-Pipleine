import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.annotation.scope import (
    build_scope_review, validate_scope_ledger, classify_span_scope,
    validate_annotations_against_scope, project_token_labels, filter_document_annotations_to_scope, IGNORE_INDEX,
)
from src.annotation.coverage import build_coverage


def make_doc(text):
    return {
        "schema_version": "1.0.0", "document_id": "scope-test-001",
        "source": {"filename": "scope-test.pdf", "format": "pdf", "document_type": "question_paper"},
        "text": text, "page_ranges": [], "annotations": [], "relations": [],
        "annotation_metadata": {"status": "draft", "guidelines_version": "1.0.0", "annotator_ids": [], "notes": ""},
    }


def decide_all(ledger, question_id="q1"):
    """Test helper: treat first nonblank segment as out-of-scope, second as in-scope by text marker."""
    for seg in ledger["segments"]:
        if seg["scope_status"] == "separator":
            continue
        seg["scope_status"] = "out_of_scope" if "GLOBAL" in seg["text"] else "in_scope"
        seg["question_id"] = None if seg["scope_status"] == "out_of_scope" else question_id
        seg["review_status"] = "reviewed"
        seg["reviewer"] = "test-reviewer"
        seg["review_date"] = "2026-10-13"
        seg["reason"] = "test decision"
    return ledger


class QuestionScopeTests(unittest.TestCase):
    def test_builder_partitions_every_character_and_whitespace_is_separator(self):
        doc = make_doc("GLOBAL note.\n\nQuestion 1: do task A.\nLast line")
        ledger, rows = build_scope_review(doc, "test.annotation.json", max_chars=100)
        self.assertEqual("".join(s["text"] for s in ledger["segments"]), doc["text"])
        self.assertEqual(ledger["text_length"], len(doc["text"]))
        self.assertTrue(any(s["scope_status"] == "separator" for s in ledger["segments"]))
        self.assertEqual(len(rows), len(ledger["segments"]))
        self.assertEqual(validate_scope_ledger(doc, ledger), [])

    def test_scope_review_requires_reviewer_metadata(self):
        doc = make_doc("Question 1: task\n")
        ledger, _ = build_scope_review(doc, "x.annotation.json")
        seg = next(s for s in ledger["segments"] if s["scope_status"] == "pending")
        seg.update(scope_status="in_scope", question_id="q1", review_status="reviewed")
        self.assertTrue(any("reviewer is required" in e for e in validate_scope_ledger(doc, ledger)))

    def test_hash_or_source_mismatch_fails(self):
        doc = make_doc("Question 1: do task\n")
        ledger, _ = build_scope_review(doc, "x.annotation.json")
        ledger["segments"][0]["text"] += "x"
        self.assertTrue(any("does not exactly match source slice" in e for e in validate_scope_ledger(doc, ledger)))

    def test_out_of_scope_annotation_is_rejected_by_scope_auditor(self):
        doc = make_doc("GLOBAL note.\n\nQuestion 1: calculate mean.\n")
        ledger, _ = build_scope_review(doc, "x.annotation.json")
        for seg in ledger["segments"]:
            if seg["scope_status"] == "separator": continue
            seg.update(scope_status="out_of_scope" if "GLOBAL" in seg["text"] else "in_scope", question_id=None if "GLOBAL" in seg["text"] else "q1", review_status="reviewed", reviewer="reviewer", review_date="2026-10-13")
        start, end = 0, len("GLOBAL note.")
        doc["annotations"] = [{"annotation_id":"g1","layer":"semantic","label":"context","start_char":start,"end_char":end,"text":doc["text"][start:end],"attributes":{},"review_status":"reviewed","annotator_id":"reviewer"}]
        errs = validate_annotations_against_scope(doc, ledger)
        self.assertTrue(any("out_of_scope" in e for e in errs))

    def test_unresolved_scope_is_never_background(self):
        text = "GLOBAL note.\n\nQuestion 1: calculate mean.\n"
        doc = make_doc(text)
        ledger, _ = build_scope_review(doc, "x.annotation.json")
        # First line reviewed outside; second line intentionally remains pending.
        for seg in ledger["segments"]:
            if "GLOBAL" in seg["text"]:
                seg.update(scope_status="out_of_scope", review_status="reviewed", reviewer="r", review_date="2026-10-13")
        cov = build_coverage(doc, "x.annotation.json", max_chars=200)
        for c in cov["chunks"]:
            c.update(structure_status="reviewed", semantic_status="reviewed", structure_reviewer="r", structure_review_date="2026-10-13", semantic_reviewer="r", semantic_review_date="2026-10-13")
        offsets = [(0, 6), (13, 21), (22, 31), (32, 37)]
        result = project_token_labels(doc, ledger, cov, offsets, "semantic", labels={"requirement"})
        self.assertEqual(result["labels"][0], IGNORE_INDEX)
        self.assertTrue(all(x == IGNORE_INDEX for x in result["labels"][1:]))

    def test_in_scope_reviewed_tokens_can_receive_bio_and_background(self):
        text = "GLOBAL note.\n\nQuestion 1: calculate mean.\n"
        doc = make_doc(text)
        ledger, _ = build_scope_review(doc, "x.annotation.json")
        for seg in ledger["segments"]:
            if seg["scope_status"] == "separator": continue
            status = "out_of_scope" if "GLOBAL" in seg["text"] else "in_scope"
            seg.update(scope_status=status, question_id=None if status == "out_of_scope" else "q1", review_status="reviewed", reviewer="r", review_date="2026-10-13")
        task_start = text.index("calculate mean")
        doc["annotations"] = [{"annotation_id":"req1","layer":"semantic","label":"requirement","start_char":task_start,"end_char":task_start+len("calculate mean"),"text":"calculate mean","attributes":{"requirement_type":"action","atomicity":"atomic","assessment_relevance":"yes"},"review_status":"reviewed","annotator_id":"r"}]
        cov=build_coverage(doc,"x.annotation.json",max_chars=200)
        for c in cov["chunks"]:
            c.update(structure_status="reviewed",semantic_status="reviewed",structure_reviewer="r",structure_review_date="2026-10-13",semantic_reviewer="r",semantic_review_date="2026-10-13")
        offsets=[(0,6),(14,22),(23,25),(26,35),(36,40)]
        result=project_token_labels(doc,ledger,cov,offsets,"semantic",labels={"requirement"})
        self.assertEqual(result["labels"][0],IGNORE_INDEX)
        self.assertEqual(result["labels"][1],"O")
        self.assertEqual(result["labels"][2],"O")
        self.assertEqual(result["labels"][3],"B-requirement")
        self.assertEqual(result["labels"][4],"I-requirement")

    def test_downstream_filter_removes_global_annotations_and_relations_without_mutating_source(self):
        text = "GLOBAL remark.\n\nQuestion 1: calculate mean.\n"
        doc = make_doc(text)
        ledger, _ = build_scope_review(doc, "x.annotation.json")
        for seg in ledger["segments"]:
            if seg["scope_status"] == "separator":
                continue
            state = "out_of_scope" if "GLOBAL" in seg["text"] else "in_scope"
            seg.update(scope_status=state, question_id=None if state == "out_of_scope" else "q1",
                       reason="test", review_status="reviewed", reviewer="r", review_date="2026-10-13")
        global_start, global_end = 0, len("GLOBAL remark.")
        task_start = text.index("calculate mean")
        doc["annotations"] = [
            {"annotation_id":"global-span","layer":"semantic","label":"other","start_char":global_start,"end_char":global_end,"text":text[global_start:global_end],"attributes":{},"review_status":"reviewed","annotator_id":"r"},
            {"annotation_id":"task-span","layer":"semantic","label":"requirement","start_char":task_start,"end_char":task_start+len("calculate mean"),"text":"calculate mean","attributes":{"requirement_type":"action","atomicity":"atomic","assessment_relevance":"yes"},"review_status":"reviewed","annotator_id":"r"},
        ]
        doc["relations"] = [{"relation_id":"global-rel","source_annotation_id":"global-span","target_annotation_id":"task-span","type":"constrains","review_status":"reviewed"}]
        filtered, audit = filter_document_annotations_to_scope(doc, ledger)
        self.assertEqual([a["annotation_id"] for a in filtered["annotations"]], ["task-span"])
        self.assertEqual(filtered["relations"], [])
        self.assertEqual(audit["excluded_annotation_count_by_scope_state"], {"out_of_scope": 1})
        self.assertEqual(audit["excluded_relation_count"], 1)
        self.assertEqual(len(doc["annotations"]), 2)
        self.assertEqual(len(doc["relations"]), 1)

    def test_mixed_in_scope_and_out_of_scope_span_is_not_in_scope(self):
        doc=make_doc("GLOBAL text.\n\nQuestion 1 task\n")
        ledger,_=build_scope_review(doc,"x.annotation.json")
        for seg in ledger["segments"]:
            if seg["scope_status"] == "separator": continue
            status="out_of_scope" if "GLOBAL" in seg["text"] else "in_scope"
            seg.update(scope_status=status,question_id=None if status=="out_of_scope" else "q1",review_status="reviewed",reviewer="r",review_date="2026-10-13")
        self.assertEqual(classify_span_scope(0,len(doc["text"]),ledger,doc["text"])[0],"mixed")

if __name__ == "__main__": unittest.main()
