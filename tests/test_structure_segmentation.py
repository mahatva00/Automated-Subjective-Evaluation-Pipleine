import hashlib
import unittest
from collections import Counter
from src.annotation.structure_segmentation import propose_structure, sha256_text


def mk_scope(text, statuses, doc='doc-1'):
    segments=[]; pos=0
    for i,(piece,status) in enumerate(statuses,1):
        start=text.index(piece, pos); end=start+len(piece); pos=end
        segments.append({"segment_id":f"s{i}","start_char":start,"end_char":end,
          "text_sha256":sha256_text(text[start:end]),"text":text[start:end],
          "page_numbers":[1],"scope_status":status,"question_id":None,
          "reason":"","review_status":"reviewed" if status in ('in_scope','out_of_scope') else 'pending',
          "reviewer":"test","review_date":"2026-10-13"})
    return {"scope_version":"1.0.0","document_id":doc,"annotation_filename":"x.annotation.json",
      "source_filename":"x.pdf","source_text_sha256":sha256_text(text),"text_length":len(text),
      "policy_version":"question_scope_policy_v1.2","segments":segments}


def mk_ann(text, doc='doc-1'):
    return {"document_id":doc,"text":text,"source":{"filename":"x.pdf"},"page_ranges":[
      {"page_number":1,"start_char":0,"end_char":len(text)}]}


class StructureSegmentationTests(unittest.TestCase):
    def test_detects_numbered_question_and_subquestion(self):
        text="Header\n1. First question\n  (a) First task\n  (b) Second task\n2. Second question\n"
        sc=mk_scope(text,[("Header\n","out_of_scope"),("1. First question\n","in_scope"),("  (a) First task\n","in_scope"),("  (b) Second task\n","in_scope"),("2. Second question\n","in_scope")])
        out=propose_structure(mk_ann(text),sc)
        types=[c['candidate_type'] for c in out['candidates']]
        self.assertIn('question_header',types); self.assertIn('subquestion_header',types)
        self.assertEqual([q['question_id'] for q in out['question_nodes']],['Q1','Q2'])
        self.assertTrue(all(c['review_status']=='pending' and c['training_eligible'] is False for c in out['candidates']))
        self.assertFalse(any('Header' in c['text'] for c in out['candidates']))

    def test_body_spans_are_separate_and_do_not_duplicate_header_spans(self):
        text="1. Main question\n  (a) Part prompt\n    i. Method\n      A. Experiment 1: fit the model\n        • Compute score\n        • Print report\n"
        sc=mk_scope(text,[(text,'in_scope')]); out=propose_structure(mk_ann(text),sc)
        by_id={c['candidate_id']:c for c in out['candidates']}
        for c in out['candidates']:
            if c['span_role']=='header' and c['paired_candidate_id']:
                body=by_id[c['paired_candidate_id']]
                self.assertEqual(c['end_char'],body['start_char'])
                self.assertNotEqual(c['text'],body['text'])
                self.assertEqual(body['parent_candidate_id'],c['candidate_id'])
        self.assertIn('question_header',[c['candidate_type'] for c in out['candidates']])
        self.assertIn('roman_item_header',[c['candidate_type'] for c in out['candidates']])
        self.assertIn('experiment_header',[c['candidate_type'] for c in out['candidates']])
        self.assertIn('list_item_header',[c['candidate_type'] for c in out['candidates']])

    def test_detects_experiment_and_bullets(self):
        text="1. Models\n  (a) Work\n    A. Experiment 1: do this\n      • Compute result\n      • Print result\n"
        sc=mk_scope(text,[(text,'in_scope')]); out=propose_structure(mk_ann(text),sc)
        types=[c['candidate_type'] for c in out['candidates']]
        self.assertIn('experiment_header',types)
        self.assertGreaterEqual(types.count('list_item_header'),2)
        self.assertFalse(any(c['candidate_type']=='list_item_block' and c['text']==next(x['text'] for x in out['candidates'] if x['candidate_type']=='list_item_header') for c in out['candidates']))

    def test_out_of_scope_never_candidate(self):
        text="Ignore this\n1. Actual question\nAnswer it\n"
        sc=mk_scope(text,[("Ignore this\n",'out_of_scope'),("1. Actual question\n",'in_scope'),("Answer it\n",'in_scope')])
        out=propose_structure(mk_ann(text),sc)
        self.assertFalse(any('Ignore this' in c['text'] for c in out['candidates']))
        self.assertEqual(out['scope_summary']['out_of_scope_segments_excluded'],1)

    def test_pending_never_candidate(self):
        text="1. Actual question\nPending text\n"
        sc=mk_scope(text,[("1. Actual question\n",'in_scope'),("Pending text\n",'pending')])
        out=propose_structure(mk_ann(text),sc)
        self.assertFalse(any('Pending text' in c['text'] for c in out['candidates']))
        self.assertEqual(out['scope_summary']['pending_scope_segments_excluded'],1)

    def test_exact_offsets_and_hashes(self):
        text="1. Question\n  (a) task with exact spaces  \n"
        sc=mk_scope(text,[(text,'in_scope')]); out=propose_structure(mk_ann(text),sc)
        for c in out['candidates']:
            self.assertEqual(text[c['start_char']:c['end_char']],c['text'])
            self.assertEqual(hashlib.sha256(c['text'].encode()).hexdigest(),c['text_sha256'])

    def test_hierarchy_runs_through_header_nodes(self):
        text = "1. Main question\n  (a) Part\n    i. Method\n      A. Experiment 1: fit the model\n        • Compute score\n        • Print report\n"
        sc = mk_scope(text, [(text, "in_scope")]); out = propose_structure(mk_ann(text), sc)
        by_type_and_text=lambda kind,sub: next(c for c in out['candidates'] if c['candidate_type']==kind and sub in c['text'])
        q=by_type_and_text('question_header','Main question')
        part=by_type_and_text('subquestion_header','Part')
        roman=by_type_and_text('roman_item_header','Method')
        exp=by_type_and_text('experiment_header','Experiment 1')
        bullet=by_type_and_text('list_item_header','Compute score')
        self.assertIsNone(q['parent_candidate_id'])
        self.assertEqual(part['parent_candidate_id'],q['candidate_id'])
        self.assertEqual(roman['parent_candidate_id'],part['candidate_id'])
        self.assertEqual(exp['parent_candidate_id'],roman['candidate_id'])
        self.assertEqual(bullet['parent_candidate_id'],exp['candidate_id'])

    def test_candidate_type_counts_include_every_candidate(self):
        text="1. Q\n  (a) Task\n    A. Experiment 1: run\n      • Print output\n"
        out=propose_structure(mk_ann(text),mk_scope(text,[(text,'in_scope')]))
        self.assertEqual(out['candidate_type_counts'],dict(sorted(Counter(c['candidate_type'] for c in out['candidates']).items())))
        self.assertEqual(sum(out['candidate_type_counts'].values()),out['total_candidate_count'])
        self.assertIn('anchor_type_counts',out)

    def test_continuation_is_linked_across_adjacent_scope_fragments(self):
        text="1. Question\n  (a) Task starts\n  continued task text\n"
        sc=mk_scope(text,[("1. Question\n",'in_scope'),("  (a) Task starts\n",'in_scope'),("  continued task text\n",'in_scope')])
        out=propose_structure(mk_ann(text),sc)
        cont=next(c for c in out['candidates'] if c['candidate_type']=='unclassified_in_scope_fragment')
        parent=next(c for c in out['candidates'] if c['candidate_type']=='subquestion_header')
        self.assertTrue(cont['linked_to_previous_scope_fragment'])
        self.assertEqual(cont['continuation_of_candidate_id'],parent['candidate_id'])
        self.assertTrue(cont['continuation_group_id'])
        self.assertEqual(cont['question_id'],'Q1')

    def test_whitespace_separator_can_preserve_continuation(self):
        text="1. Question\n continued\n"
        sc=mk_scope(text,[("1. Question\n",'in_scope'),(" ",'separator'),("continued\n",'in_scope')])
        out=propose_structure(mk_ann(text),sc)
        frag=next(c for c in out['candidates'] if c['candidate_type']=='unclassified_in_scope_fragment')
        self.assertTrue(frag['linked_to_previous_scope_fragment'])
        self.assertEqual(frag['question_id'],'Q1')

    def test_context_does_not_cross_out_of_scope_gap(self):
        text="1. Question\nExcluded global text\ncontinued but unrelated\n"
        sc=mk_scope(text,[("1. Question\n",'in_scope'),("Excluded global text\n",'out_of_scope'),("continued but unrelated\n",'in_scope')])
        out=propose_structure(mk_ann(text),sc)
        frag=next(c for c in out['candidates'] if 'continued but unrelated' in c['text'])
        self.assertIsNone(frag['question_id'])
        self.assertIsNone(frag['parent_candidate_id'])
        self.assertIsNone(frag['continuation_of_candidate_id'])

    def test_numeric_page_footer_is_excluded_but_preserves_question_context(self):
        text="1. Question\npart one\n  4\n(b) Next part\n"
        sc=mk_scope(text,[("1. Question\n",'in_scope'),("part one\n",'in_scope'),("  4\n",'out_of_scope'),("(b) Next part\n",'in_scope')])
        out=propose_structure(mk_ann(text),sc)
        next_part=next(c for c in out['candidates'] if '(b) Next part' in c['text'])
        self.assertEqual(next_part['question_id'],'Q1')
        self.assertFalse(any(c['text'].strip()=='4' for c in out['candidates']))

    def test_unattached_intro_fragment_remains_unassigned(self):
        text="Introductory global instruction\n1. Actual question\n"
        sc=mk_scope(text,[("Introductory global instruction\n",'in_scope'),("1. Actual question\n",'in_scope')])
        out=propose_structure(mk_ann(text),sc)
        intro=next(c for c in out['candidates'] if 'Introductory global' in c['text'])
        self.assertEqual(intro['candidate_type'],'unattached_in_scope_fragment')
        self.assertIsNone(intro['question_id'])
        self.assertIsNone(intro['parent_candidate_id'])
        self.assertEqual(sum(1 for c in out['candidates'] if 'Introductory global' in c['text']),1)

    def test_source_hash_mismatch_fails_closed(self):
        text="1. Q\n"; sc=mk_scope(text,[(text,'in_scope')]); sc['source_text_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'SHA-256 mismatch'):
            propose_structure(mk_ann(text),sc)

    def test_document_id_mismatch_fails_closed(self):
        text="1. Q\n"; sc=mk_scope(text,[(text,'in_scope')],doc='other')
        with self.assertRaisesRegex(ValueError,'Document ID mismatch'):
            propose_structure(mk_ann(text),sc)

    def test_does_not_split_code_like_false_heading(self):
        text="1. Main question\n    q1 = df[col].quantile(0.25)\n"
        sc=mk_scope(text,[(text,'in_scope')]); out=propose_structure(mk_ann(text),sc)
        self.assertEqual(len(out['question_nodes']),1)
        self.assertEqual(sum(c['candidate_type']=='question_header' for c in out['candidates']),1)
        self.assertNotIn('q1 = df',next(c['text'] for c in out['candidates'] if c['candidate_type']=='question_header'))

    def test_scope_segment_text_mismatch_fails_closed(self):
        text="1. Q\n"; sc=mk_scope(text,[(text,'in_scope')]); sc['segments'][0]['text']='tampered'
        with self.assertRaisesRegex(ValueError,'Exact text mismatch'):
            propose_structure(mk_ann(text),sc)

if __name__=='__main__': unittest.main()
