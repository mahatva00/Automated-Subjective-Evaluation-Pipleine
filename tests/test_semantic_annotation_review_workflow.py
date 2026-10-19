import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.annotation.review_workflow import (
    ReviewError, make_review_rows, write_review_tsv, _read_review,
    import_reviewed_sheet, _load_scope, one_line, TSV_FIELDS,
)

def sha(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()

def fixture():
    text = "Q1.\nCalculate sum.\nIntro context.\n"
    doc = {
      "schema_version":"1.0.0", "document_id":"doc-1",
      "source":{"filename":"test.pdf","format":"pdf","document_type":"question_paper","source_hash":"pdf-hash"},
      "text":text,
      "page_ranges":[{"source_segment_id":"p1","page_number":1,"sequence_index":0,"start_char":0,"end_char":len(text)}],
      "annotations":[],"relations":[],
      "annotation_metadata":{"status":"draft","guidelines_version":"1.0.0","annotator_ids":[],"notes":"draft"}
    }
    qseg={"segment_id":"scope_q","start_char":0,"end_char":23,"text":text[:23],"text_sha256":sha(text[:23]),"page_numbers":[1],"scope_status":"in_scope","question_id":"Q1","review_status":"reviewed"}
    ctext="Calculate sum."
    cand={"proposal_id":"SEM0001","start_char":4,"end_char":len("Q1.\nCalculate sum."),"text":ctext,"text_sha256":sha(ctext),"label_proposal":"requirement","question_id":"Q1","scope_segment_id":"scope_q","proposal_confidence":"medium","proposal_rationale":"cue","review_flags":[],"requirement_attributes_proposal":{"requirement_type":"action","atomicity":"atomic","assessment_relevance":"yes"}}
    proposal={"schema_version":"semantic-annotation-proposal-v1.0","document_id":"doc-1","annotation_filename":"test.annotation.json","source_text_sha256":sha(text),"text_length":len(text),"candidates":[cand]}
    scope={"document_id":"doc-1","source_text_sha256":sha(text),"segments":[qseg, {"segment_id":"scope_ctx","start_char":23,"end_char":len(text),"text":text[23:],"text_sha256":sha(text[23:]),"page_numbers":[1],"scope_status":"out_of_scope","question_id":None,"review_status":"reviewed"}]}
    return doc,proposal,scope

class ReviewWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.doc,self.prop,self.scope=fixture()
        self.scope_idx=_load_scope_from_obj(self.scope,self.doc)
    def test_prepare_never_approves(self):
        rows=make_review_rows(self.doc,self.prop,self.scope_idx)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['review_action'],'pending')
        self.assertEqual(rows[0]['final_label'],'')
    def test_accept_creates_span_reviewed_but_document_stays_draft(self):
        rows=make_review_rows(self.doc,self.prop,self.scope_idx)
        rows[0]['review_action']='accept'
        out,audit=import_reviewed_sheet(self.doc,self.prop,self.scope_idx,rows,'teacher1','2026-10-13')
        self.assertEqual(len(out['annotations']),1)
        ann=out['annotations'][0]
        self.assertEqual(ann['text'],'Calculate sum.')
        self.assertEqual(ann['review_status'],'reviewed')
        self.assertEqual(out['annotation_metadata']['status'],'draft')
        self.assertFalse(audit['document_finalized'])
    def test_proposal_not_auto_accepted(self):
        rows=make_review_rows(self.doc,self.prop,self.scope_idx)
        out,_=import_reviewed_sheet(self.doc,self.prop,self.scope_idx,rows,'teacher1','2026-10-13')
        self.assertEqual(out['annotations'],[])
    def test_unclassified_cannot_accept(self):
        rows=make_review_rows(self.doc,self.prop,self.scope_idx)
        self.prop['candidates'][0]['label_proposal']=None
        rows=make_review_rows(self.doc,self.prop,self.scope_idx)
        rows[0]['review_action']='accept'
        with self.assertRaises(ReviewError): import_reviewed_sheet(self.doc,self.prop,self.scope_idx,rows,'teacher1','2026-10-13')
    def test_revise_requires_label(self):
        rows=make_review_rows(self.doc,self.prop,self.scope_idx)
        rows[0]['review_action']='revise'
        with self.assertRaises(ReviewError): import_reviewed_sheet(self.doc,self.prop,self.scope_idx,rows,'teacher1','2026-10-13')
    def test_revise_validates_requirement_attrs(self):
        rows=make_review_rows(self.doc,self.prop,self.scope_idx)
        rows[0].update(review_action='revise',final_label='requirement',final_requirement_type='bad',final_atomicity='atomic',final_assessment_relevance='yes')
        with self.assertRaises(ReviewError): import_reviewed_sheet(self.doc,self.prop,self.scope_idx,rows,'teacher1','2026-10-13')
    def test_out_of_scope_candidate_rejected(self):
        p=copy.deepcopy(self.prop); p['candidates'][0]['scope_segment_id']='scope_ctx'
        with self.assertRaises(ReviewError): make_review_rows(self.doc,p,self.scope_idx)
    def test_pending_scope_candidate_rejected(self):
        pending=copy.deepcopy(self.scope_idx)
        pending['scope_q']['review_status']='pending'
        with self.assertRaises(ReviewError): make_review_rows(self.doc,self.prop,pending)
    def test_edited_immutable_text_rejected(self):
        rows=make_review_rows(self.doc,self.prop,self.scope_idx)
        rows[0]['candidate_text']='edited text'
        with self.assertRaises(ReviewError):
            import_reviewed_sheet(self.doc,self.prop,self.scope_idx,rows,'teacher1','2026-10-13')
    def test_deleted_proposal_row_rejected(self):
        with self.assertRaises(ReviewError):
            import_reviewed_sheet(self.doc,self.prop,self.scope_idx,[],'teacher1','2026-10-13')
    def test_edited_proposed_attributes_rejected(self):
        rows=make_review_rows(self.doc,self.prop,self.scope_idx)
        rows[0]['proposed_attributes_json']='{}'
        with self.assertRaises(ReviewError):
            import_reviewed_sheet(self.doc,self.prop,self.scope_idx,rows,'teacher1','2026-10-13')
    def test_proposal_source_hash_mismatch_rejected(self):
        rows=make_review_rows(self.doc,self.prop,self.scope_idx)
        proposal=copy.deepcopy(self.prop); proposal['source_text_sha256']='wrong-hash'
        with self.assertRaises(ReviewError):
            import_reviewed_sheet(self.doc,proposal,self.scope_idx,rows,'teacher1','2026-10-13')
    def test_revised_span_must_stay_in_scope(self):
        rows=make_review_rows(self.doc,self.prop,self.scope_idx)
        rows[0].update(review_action='revise',final_label='requirement',final_start_char='23',final_end_char=str(len(self.doc['text'])),final_requirement_type='action',final_atomicity='atomic',final_assessment_relevance='yes')
        with self.assertRaises(ReviewError): import_reviewed_sheet(self.doc,self.prop,self.scope_idx,rows,'teacher1','2026-10-13')
    def test_split_creates_nonoverlapping_exact_source_slices(self):
        rows=make_review_rows(self.doc,self.prop,self.scope_idx)
        rows[0]['review_action']='split'
        rows[0]['split_spans_json']=json.dumps([
          {"start_char":4,"end_char":13,"label":"requirement","requirement_type":"action","atomicity":"atomic","assessment_relevance":"yes"},
          {"start_char":13,"end_char":18,"label":"requirement","requirement_type":"output_specification","atomicity":"atomic","assessment_relevance":"yes"}
        ])
        out,_=import_reviewed_sheet(self.doc,self.prop,self.scope_idx,rows,'teacher1','2026-10-13')
        self.assertEqual([a['text'] for a in out['annotations']], ['Calculate', ' sum.'])
        self.assertEqual(out['annotations'][0]['review_status'], 'reviewed')
    def test_overlapping_split_is_rejected(self):
        rows=make_review_rows(self.doc,self.prop,self.scope_idx)
        rows[0]['review_action']='split'
        rows[0]['split_spans_json']=json.dumps([
          {"start_char":4,"end_char":14,"label":"requirement","requirement_type":"action","atomicity":"atomic","assessment_relevance":"yes"},
          {"start_char":13,"end_char":18,"label":"requirement","requirement_type":"output_specification","atomicity":"atomic","assessment_relevance":"yes"}
        ])
        with self.assertRaises(ReviewError):
            import_reviewed_sheet(self.doc,self.prop,self.scope_idx,rows,'teacher1','2026-10-13')
    def test_invalid_date_rejected(self):
        rows=make_review_rows(self.doc,self.prop,self.scope_idx); rows[0]['review_action']='accept'
        with self.assertRaises(ReviewError): import_reviewed_sheet(self.doc,self.prop,self.scope_idx,rows,'teacher1','13-10-2026')
    def test_tsv_roundtrip(self):
        rows=make_review_rows(self.doc,self.prop,self.scope_idx)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'review.tsv'; write_review_tsv(p,rows); read=_read_review(p)
            self.assertEqual(read,rows)
    def test_input_document_not_mutated(self):
        before=copy.deepcopy(self.doc); rows=make_review_rows(self.doc,self.prop,self.scope_idx); rows[0]['review_action']='accept'
        import_reviewed_sheet(self.doc,self.prop,self.scope_idx,rows,'teacher1','2026-10-13')
        self.assertEqual(self.doc,before)
    def test_duplicate_annotation_overlap_rejected(self):
        rows=make_review_rows(self.doc,self.prop,self.scope_idx); rows[0]['review_action']='accept'
        doc=copy.deepcopy(self.doc)
        doc['annotations']=[{"annotation_id":"old","layer":"semantic","label":"context","start_char":5,"end_char":9,"text":doc['text'][5:9],"review_status":"reviewed"}]
        with self.assertRaises(ReviewError): import_reviewed_sheet(doc,self.prop,self.scope_idx,rows,'teacher1','2026-10-13')
    def test_accept_with_proposed_attributes(self):
        rows=make_review_rows(self.doc,self.prop,self.scope_idx); rows[0]['review_action']='accept'
        out,_=import_reviewed_sheet(self.doc,self.prop,self.scope_idx,rows,'teacher1','2026-10-13')
        self.assertEqual(out['annotations'][0]['attributes']['atomicity'],'atomic')
        self.assertEqual(out['annotations'][0]['attributes']['source_proposal_id'],'SEM0001')

def _load_scope_from_obj(scope, doc):
    # Use same validations as disk loader without requiring temporary files.
    segs={}
    assert scope['source_text_sha256']==sha(doc['text'])
    for s in scope['segments']:
        assert doc['text'][s['start_char']:s['end_char']]==s['text']
        assert sha(s['text'])==s['text_sha256']
        segs[s['segment_id']]=s
    return segs

if __name__ == '__main__': unittest.main()
