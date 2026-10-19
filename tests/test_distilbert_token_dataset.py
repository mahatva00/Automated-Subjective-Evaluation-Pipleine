from __future__ import annotations
import hashlib
import json
import re
import unittest
import tempfile
from pathlib import Path

from src.annotation.token_dataset import (
    DatasetError, compile_document, coverage_template, label_maps, read_manifest,
    sha256_text, validate_approved_document, validate_coverage,
)

class FakeFastTokenizer:
    is_fast = True
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True, truncation=False):
        found = list(re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE))
        return {"input_ids": [i + 10 for i, _ in enumerate(found)],
                "offset_mapping": [(m.start(), m.end()) for m in found]}
    def num_special_tokens_to_add(self, pair=False):
        return 2
    def prepare_for_model(self, ids, add_special_tokens=True, padding=False, truncation=False,
                          return_attention_mask=True, return_special_tokens_mask=True,
                          return_token_type_ids=False):
        out_ids = ([101] + list(ids) + [102]) if add_special_tokens else list(ids)
        special = ([1] + [0] * len(ids) + [1]) if add_special_tokens else [0] * len(ids)
        return {"input_ids": out_ids, "attention_mask": [1] * len(out_ids), "special_tokens_mask": special}
    def convert_ids_to_tokens(self, ids):
        return [str(x) for x in ids]

def make_doc(text, annotations):
    return {
        "schema_version": "1.0.0",
        "document_id": "doc-1",
        "source": {"filename": "doc.pdf", "format": "pdf", "document_type": "question_paper"},
        "text": text,
        "page_ranges": [{"source_segment_id":"p1","start_char":0,"end_char":len(text),"page_number":1}],
        "annotations": annotations,
        "relations": [],
        "annotation_metadata": {"status":"reviewed","guidelines_version":"1.0.0","annotator_ids":["teacher"],"notes":""},
    }

def ann(aid, layer, label, text, start, end, attributes=None):
    return {"annotation_id":aid,"layer":layer,"label":label,"start_char":start,"end_char":end,
            "text":text[start:end],"attributes":attributes or {},"review_status":"reviewed","annotator_id":"teacher"}

def make_coverage(doc, spans):
    return {"coverage_schema_version":"1.0.0","document_id":doc["document_id"],
            "source_text_sha256":sha256_text(doc["text"]),"text_length":len(doc["text"]),
            "ledger_status":"reviewed","segments":spans}

def covseg(doc, cid, start, end, status, layers=None, scope_status=None):
    return {"coverage_id":cid,"start_char":start,"end_char":end,
            "text_sha256":sha256_text(doc["text"][start:end]),"status":status,
            "reviewed_layers":layers or [],"scope_status":scope_status,"source_segment_id":cid}

class TokenDatasetTests(unittest.TestCase):
    def test_coverage_template_never_auto_approves_in_scope(self):
        text="Global. Question text."
        doc=make_doc(text, [])
        split=len("Global. ")
        scope={"document_id":"doc-1","source_text_sha256":sha256_text(text),
               "policy_version":"test","segments":[
                   {"segment_id":"s1","start_char":0,"end_char":split,"text":text[:split],
                    "text_sha256":sha256_text(text[:split]),"scope_status":"out_of_scope","question_id":None,"page_numbers":[1]},
                   {"segment_id":"s2","start_char":split,"end_char":len(text),"text":text[split:],
                    "text_sha256":sha256_text(text[split:]),"scope_status":"in_scope","question_id":"Q1","page_numbers":[1]}]}
        cov=coverage_template(doc, scope)
        self.assertEqual(cov["segments"][0]["status"],"out_of_scope")
        self.assertEqual(cov["segments"][1]["status"],"pending")
        self.assertEqual(cov["ledger_status"],"draft")
        self.assertTrue(all(not x["reviewed_layers"] for x in cov["segments"]))

    def test_draft_document_is_rejected(self):
        text="Do task."
        doc=make_doc(text,[])
        doc["annotation_metadata"]["status"]="draft"
        with self.assertRaisesRegex(DatasetError,"refusing to train"):
            validate_approved_document(doc)

    def test_unreviewed_annotation_is_rejected(self):
        text="Do task."
        a=ann("a1","semantic","requirement",text,0,len(text),{"requirement_type":"action","atomicity":"atomic","assessment_relevance":"yes"})
        a["review_status"]="unreviewed"
        doc=make_doc(text,[a])
        with self.assertRaisesRegex(DatasetError,"all annotations must be reviewed"):
            validate_approved_document(doc)

    def test_bio_ids_and_ignore_mask_follow_reviewed_coverage(self):
        text="Admin. Do task now. Footer."
        start=text.index("Do task now.")
        end=start+len("Do task now.")
        a=ann("req1","semantic","requirement",text,start,end,{"requirement_type":"action","atomicity":"atomic","assessment_relevance":"yes"})
        st=ann("body1","structure","question_body",text,start,end)
        doc=make_doc(text,[a,st])
        cov=make_coverage(doc,[
            covseg(doc,"c1",0,start,"out_of_scope",scope_status="out_of_scope"),
            covseg(doc,"c2",start,end,"reviewed",["structure","semantic"],"in_scope"),
            covseg(doc,"c3",end,len(text),"pending",scope_status="in_scope")
        ])
        manifest={"split":"train","group_id":"doc-1"}
        windows, stats=compile_document(doc,cov,manifest,FakeFastTokenizer(),max_length=32,stride=4)
        row=windows[0]
        smap=label_maps()["semantic"]["tag_to_id"]
        first_content=next(i for i,x in enumerate(row["offset_mapping"]) if x[0] == start)
        self.assertEqual(row["semantic_labels"][first_content],smap["B-requirement"])
        # Source text before and after reviewed coverage is masked, not O.
        pre = next(i for i,x in enumerate(row["offset_mapping"]) if x[0] == 0)
        post = next(i for i,x in enumerate(row["offset_mapping"]) if x[0] >= end and x[1] > x[0])
        self.assertEqual(row["semantic_labels"][pre],-100)
        self.assertEqual(row["semantic_labels"][post],-100)
        self.assertEqual(row["semantic_labels"][0],-100)  # special CLS
        self.assertEqual(row["semantic_labels"][-1],-100) # special SEP
        self.assertEqual(row["structure_labels"][first_content],label_maps()["structure"]["tag_to_id"]["B-question_body"])
        self.assertEqual(stats["approved_annotations"],2)

    def test_o_is_emitted_only_inside_reviewed_coverage(self):
        text="Context. Do task now. Footer."
        start=text.index("Do task now.")
        end=start+len("Do task now.")
        a=ann("req1","semantic","requirement",text,start,end,{"requirement_type":"action","atomicity":"atomic","assessment_relevance":"yes"})
        doc=make_doc(text,[a])
        cov=make_coverage(doc,[
            covseg(doc,"c1",0,end,"reviewed",["semantic"]),
            covseg(doc,"c2",end,len(text),"pending")
        ])
        windows,_=compile_document(doc,cov,{"split":"train","group_id":"doc-1"},FakeFastTokenizer(),max_length=32,stride=4)
        row=windows[0]
        tmap=label_maps()["semantic"]["tag_to_id"]
        context_token=next(i for i,x in enumerate(row["offset_mapping"]) if x == [0,7] or tuple(x)==(0,7))
        self.assertEqual(row["semantic_labels"][context_token],tmap["O"])

    def test_coverage_hash_mismatch_fails_closed(self):
        text="Do task."
        a=ann("r","semantic","requirement",text,0,len(text),{"requirement_type":"action","atomicity":"atomic","assessment_relevance":"yes"})
        doc=make_doc(text,[a])
        cov=make_coverage(doc,[covseg(doc,"c",0,len(text),"reviewed",["semantic"])])
        cov["source_text_sha256"]="bad"
        with self.assertRaisesRegex(DatasetError,"hash/length mismatch"):
            validate_coverage(cov,doc)

    def test_approved_annotation_must_be_inside_reviewed_coverage(self):
        text="Do task. Global."
        a=ann("r","semantic","requirement",text,0,8,{"requirement_type":"action","atomicity":"atomic","assessment_relevance":"yes"})
        doc=make_doc(text,[a])
        cov=make_coverage(doc,[covseg(doc,"c",0,len(text),"pending")])
        with self.assertRaisesRegex(DatasetError,"not fully inside reviewed coverage"):
            compile_document(doc,cov,{"split":"train","group_id":"doc-1"},FakeFastTokenizer())

    def test_same_layer_overlap_rejected(self):
        text="Do task now."
        a=ann("a","semantic","requirement",text,0,7,{"requirement_type":"action","atomicity":"atomic","assessment_relevance":"yes"})
        b=ann("b","semantic","context",text,3,11)
        doc=make_doc(text,[a,b])
        with self.assertRaisesRegex(DatasetError,"overlapping annotations"):
            validate_approved_document(doc)

    def test_token_windows_overlap_and_retain_annotation(self):
        text="One two three four five six seven eight nine ten."
        start=text.index("seven")
        end=text.index(" eight")
        a=ann("r","semantic","requirement",text,start,end,{"requirement_type":"action","atomicity":"atomic","assessment_relevance":"yes"})
        doc=make_doc(text,[a])
        cov=make_coverage(doc,[covseg(doc,"c",0,len(text),"reviewed",["semantic","structure"])])
        windows,stats=compile_document(doc,cov,{"split":"train","group_id":"doc-1"},FakeFastTokenizer(),max_length=7,stride=2)
        self.assertGreater(len(windows),1)
        self.assertTrue(any(any(t["annotation_id"]=="r" for t in w["annotation_targets"]) for w in windows))
        self.assertEqual(stats["window_count"],len(windows))

    def test_too_long_annotation_fails_instead_of_dropping(self):
        text="one two three four five six seven eight nine"
        a=ann("r","semantic","requirement",text,0,len(text),{"requirement_type":"action","atomicity":"compound","assessment_relevance":"yes"})
        doc=make_doc(text,[a])
        cov=make_coverage(doc,[covseg(doc,"c",0,len(text),"reviewed",["semantic"])])
        with self.assertRaisesRegex(DatasetError,"does not fit wholly"):
            compile_document(doc,cov,{"split":"train","group_id":"doc-1"},FakeFastTokenizer(),max_length=6,stride=1)


    def test_manifest_group_leakage_is_rejected(self):
        manifest = {"manifest_version":"1.0.0","documents":[
            {"document_id":"a","annotation_path":"a.annotation.json","split":"train","group_id":"same-paper"},
            {"document_id":"b","annotation_path":"b.annotation.json","split":"validation","group_id":"same-paper"}
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(DatasetError,"group leakage"):
                read_manifest(path)

    def test_crossing_annotation_boundary_token_is_masked_and_reported(self):
        text = "Calculate value now."
        # Deliberately start inside the first tokenizer token to verify fail-closed alignment behavior.
        start = 2
        end = text.index(" now.")
        a = ann("r","semantic","requirement",text,start,end,{"requirement_type":"action","atomicity":"atomic","assessment_relevance":"yes"})
        doc = make_doc(text,[a])
        cov = make_coverage(doc,[covseg(doc,"c",0,len(text),"reviewed",["semantic"])])
        windows, stats = compile_document(doc,cov,{"split":"train","group_id":"doc-1"},FakeFastTokenizer(),max_length=32,stride=4)
        row = windows[0]
        calc_idx = next(i for i,x in enumerate(row["offset_mapping"]) if tuple(x) == (0,9))
        self.assertEqual(row["semantic_labels"][calc_idx],-100)
        self.assertGreater(stats["alignment_warning_count"],0)

    def test_coverage_must_preserve_manual_scope(self):
        text="Out. In."
        doc=make_doc(text, [])
        cov=make_coverage(doc,[
            covseg(doc,"c1",0,5,"reviewed",["semantic"],"out_of_scope"),
            covseg(doc,"c2",5,len(text),"pending",scope_status="in_scope")
        ])
        with self.assertRaisesRegex(DatasetError,"cannot be promoted"):
            validate_coverage(cov,doc)

if __name__ == "__main__":
    unittest.main()
