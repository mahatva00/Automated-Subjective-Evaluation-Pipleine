import json
import tempfile
import unittest
from pathlib import Path

from src.annotation.scope import build_scope_review
from scripts.prepare_distilbert_span_dataset import build_rows


def document(doc_id, split):
    text = "GLOBAL note.\n\nQuestion 1: calculate mean.\n"
    global_start = 0
    task_start = text.index("calculate mean")
    return {
        "schema_version": "1.0.0", "document_id": doc_id,
        "source": {"filename": f"{doc_id}.pdf", "format": "pdf", "document_type": "question_paper"},
        "text": text, "page_ranges": [], "relations": [],
        "annotations": [
            {"annotation_id":"global-context","layer":"semantic","label":"context","start_char":global_start,"end_char":len("GLOBAL note."),"text":"GLOBAL note.","attributes":{},"review_status":"reviewed","annotator_id":"reviewer"},
            {"annotation_id":"question-requirement","layer":"semantic","label":"requirement","start_char":task_start,"end_char":task_start+len("calculate mean"),"text":"calculate mean","attributes":{"requirement_type":"action","atomicity":"atomic","assessment_relevance":"yes"},"review_status":"reviewed","annotator_id":"reviewer"},
        ],
        "annotation_metadata": {"status":"draft","guidelines_version":"1.0.0","annotator_ids":["reviewer"],"notes":"test"},
    }


def scoped(doc):
    ledger, _ = build_scope_review(doc, doc["source"]["filename"] + ".annotation.json")
    for s in ledger["segments"]:
        if s["scope_status"] == "separator":
            continue
        in_scope = "GLOBAL" not in s["text"]
        s.update(scope_status="in_scope" if in_scope else "out_of_scope", question_id="q1" if in_scope else None,
                 reason="test scope", review_status="reviewed", reviewer="reviewer", review_date="2026-10-13")
    return ledger


class ScopeAwareSpanDatasetTests(unittest.TestCase):
    def test_out_of_scope_approved_spans_never_enter_classifier_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); input_dir=base/"pilot"; scope_dir=base/"scope"; input_dir.mkdir();scope_dir.mkdir()
            docs=[document("doc-A","train"),document("doc-B","validation")]
            manifest={"manifest_version":"1.0.0","documents":[{"document_id":"doc-A","annotation_path":"doc-A.annotation.json","split":"train","group_id":"doc-A"},{"document_id":"doc-B","annotation_path":"doc-B.annotation.json","split":"validation","group_id":"doc-B"}]}
            mp=base/"manifest.json";mp.write_text(json.dumps(manifest),encoding="utf-8")
            for doc in docs:
                (input_dir/f"{doc['document_id']}.annotation.json").write_text(json.dumps(doc),encoding="utf-8")
                (scope_dir/f"{doc['document_id']}.scope.json").write_text(json.dumps(scoped(doc)),encoding="utf-8")
            rows, _, counts=build_rows(input_dir,mp,scope_dir)
            self.assertEqual(len(rows),2)
            self.assertTrue(all(r["text"]=="calculate mean" for r in rows))
            self.assertEqual(counts["excluded_out_of_scope"],2)
            self.assertEqual(counts["included_in_scope"],2)

    def test_missing_scope_sidecar_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); inp=base/"pilot"; inp.mkdir()
            doc=document("doc-A","train"); (inp/"doc-A.annotation.json").write_text(json.dumps(doc),encoding="utf-8")
            manifest={"manifest_version":"1.0.0","documents":[{"document_id":"doc-A","annotation_path":"doc-A.annotation.json","split":"train","group_id":"doc-A"},{"document_id":"doc-B","annotation_path":"doc-B.annotation.json","split":"validation","group_id":"doc-B"}]}
            mp=base/"manifest.json";mp.write_text(json.dumps(manifest),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"scope sidecar missing"):
                build_rows(inp,mp,base/"missing")

if __name__=="__main__":unittest.main()
