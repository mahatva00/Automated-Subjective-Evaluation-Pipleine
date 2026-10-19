import copy
import hashlib
import unittest
from src.annotation.semantic_proposals import propose_semantic_annotations


def sha(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def fixture(text, pieces, candidates, scope_statuses=None):
    # pieces = list of (start, end, segment_id, status)
    doc_id = 'demo-q-paper'
    annotation = {
        'schema_version': '1.0.0', 'document_id': doc_id,
        'source': {'filename': 'demo.pdf', 'format': 'pdf', 'document_type': 'question_paper'},
        'text': text, 'page_ranges': [{'source_segment_id': 'p1', 'page_number': 1, 'sequence_index': 0, 'start_char': 0, 'end_char': len(text)}],
        'annotations': [], 'relations': [], 'annotation_metadata': {'status': 'draft', 'guidelines_version': '1.0.0'},
    }
    scope_segments = []
    for start, end, sid, status in pieces:
        part = text[start:end]
        scope_segments.append({'segment_id': sid, 'start_char': start, 'end_char': end, 'text': part,
                               'text_sha256': sha(part), 'page_numbers': [1], 'scope_status': status,
                               'question_id': None, 'review_status': 'reviewed'})
    scope = {'scope_version': '1.0', 'document_id': doc_id, 'source_text_sha256': sha(text),
             'text_length': len(text), 'policy_version': 'question_scope_policy_v1.2', 'segments': scope_segments}
    norm_candidates = []
    for i, c in enumerate(candidates, 1):
        start, end = c['start_char'], c['end_char']
        item = {'candidate_id': f'SC{i:04d}', 'candidate_type': c.get('candidate_type', 'subquestion_header'),
                'span_role': c.get('span_role', 'header'), 'start_char': start, 'end_char': end,
                'text': text[start:end], 'text_sha256': sha(text[start:end]), 'scope_segment_id': c['scope_segment_id'],
                'scope_status': 'in_scope', 'question_id': c.get('question_id'), 'parent_candidate_id': c.get('parent_candidate_id'),
                'continuation_of_candidate_id': c.get('continuation_of_candidate_id')}
        norm_candidates.append(item)
    question_nodes = []
    for c in norm_candidates:
        if c['candidate_type'] == 'question_header' and c.get('question_id'):
            question_nodes.append({'question_id': c['question_id'], 'header_start_char': c['start_char'],
                                   'header_end_char': c['end_char'], 'header_sha256': sha(c['text']),
                                   'title_preview': c['text'], 'review_status': 'pending', 'training_eligible': False})
    if not question_nodes:
        q_candidates = [c for c in norm_candidates if c.get('question_id')]
        if q_candidates:
            first = min(q_candidates, key=lambda c: c['start_char'])
            question_nodes.append({'question_id': first['question_id'], 'header_start_char': first['start_char'],
                                   'header_end_char': first['end_char'], 'header_sha256': sha(first['text']),
                                   'title_preview': first['text'], 'review_status': 'pending', 'training_eligible': False})
    structure = {'schema_version': 'structure-segmentation-proposal-v1.1', 'document_id': doc_id,
                 'source_text_sha256': sha(text), 'text_length': len(text), 'annotation_filename': 'demo.annotation.json',
                 'scope_ledger_policy_version': 'question_scope_policy_v1.2', 'question_nodes': question_nodes,
                 'candidates': norm_candidates}
    return annotation, scope, structure


class SemanticProposalTests(unittest.TestCase):
    def test_requirement_and_marks_are_non_overlapping(self):
        text = '1. Compute the value and display the result. [2 marks]\n'
        pieces = [(0, len(text), 's1', 'in_scope')]
        ann, scope, struct = fixture(text, pieces, [{'start_char': 0, 'end_char': len(text), 'scope_segment_id': 's1', 'question_id': 'Q1', 'candidate_type': 'question_header'}])
        out = propose_semantic_annotations(ann, scope, struct)
        labels = [(c['label_proposal'], c['text']) for c in out['candidates']]
        self.assertIn(('requirement', 'Compute the value and display the result.'), labels)
        self.assertIn(('grading_instruction', '[2 marks]'), labels)
        spans = sorted((c['start_char'], c['end_char']) for c in out['candidates'])
        for a, b in zip(spans, spans[1:]): self.assertLessEqual(a[1], b[0])

    def test_global_unassigned_segment_is_omitted(self):
        text = 'Global instructions.\n1. Explain the method.\n'
        cut = text.index('1.')
        pieces = [(0, cut, 'global', 'in_scope'), (cut, len(text), 'q1', 'in_scope')]
        ann, scope, struct = fixture(text, pieces, [
            {'start_char': 0, 'end_char': cut-1, 'scope_segment_id': 'global', 'question_id': None},
            {'start_char': cut, 'end_char': len(text), 'scope_segment_id': 'q1', 'question_id': 'Q1', 'candidate_type': 'question_header'}])
        out = propose_semantic_annotations(ann, scope, struct)
        self.assertEqual(len(out['ignored_unassigned_regions']), 1)
        self.assertTrue(all('Global instructions' not in c['text'] for c in out['candidates']))

    def test_prefix_before_question_heading_left_unassigned_inside_same_segment(self):
        text = 'Global note before the first question.\n1. Calculate the result.\n'
        start = text.index('1.')
        ann, scope, struct = fixture(text, [(0, len(text), 'whole-page', 'in_scope')], [
            {'start_char': start, 'end_char': len(text), 'scope_segment_id': 'whole-page', 'question_id': 'Q1', 'candidate_type': 'question_header'}])
        out = propose_semantic_annotations(ann, scope, struct)
        self.assertFalse(any('Global note before' in c['text'] for c in out['candidates']))
        self.assertEqual(len(out['ignored_unassigned_regions']), 1)

    def test_out_of_scope_text_never_candidate(self):
        text = 'Administrative instructions.\n1. Calculate x.\n'
        cut = text.index('1.')
        pieces = [(0, cut, 'global', 'out_of_scope'), (cut, len(text), 'q1', 'in_scope')]
        ann, scope, struct = fixture(text, pieces, [{'start_char': cut, 'end_char': len(text), 'scope_segment_id': 'q1', 'question_id': 'Q1', 'candidate_type': 'question_header'}])
        out = propose_semantic_annotations(ann, scope, struct)
        self.assertFalse(any('Administrative' in c['text'] for c in out['candidates']))

    def test_example_is_separate_from_requirement(self):
        text = '1. Prepare the data. For example, use sample values.\n'
        pieces = [(0, len(text), 's1', 'in_scope')]
        ann, scope, struct = fixture(text, pieces, [{'start_char': 0, 'end_char': len(text), 'scope_segment_id': 's1', 'question_id': 'Q1', 'candidate_type': 'question_header'}])
        out = propose_semantic_annotations(ann, scope, struct)
        self.assertTrue(any(c['label_proposal'] == 'requirement' and c['text'] == 'Prepare the data.' for c in out['candidates']))
        self.assertTrue(any(c['label_proposal'] == 'example' and c['text'].startswith('For example') for c in out['candidates']))

    def test_multiple_actions_flag_atomicity(self):
        text = '(a) Check the shape and display the records.\n'
        pieces = [(0, len(text), 's1', 'in_scope')]
        ann, scope, struct = fixture(text, pieces, [{'start_char': 0, 'end_char': len(text), 'scope_segment_id': 's1', 'question_id': 'Q1', 'candidate_type': 'subquestion_header'}])
        out = propose_semantic_annotations(ann, scope, struct)
        req = next(c for c in out['candidates'] if c['label_proposal'] == 'requirement')
        self.assertEqual(req['requirement_attributes_proposal']['atomicity'], 'compound')
        self.assertIn('multiple_action_verbs_review_atomicity', req['review_flags'])

    def test_nonrequirement_heading_left_uncertain(self):
        text = '1. Data preprocessing [5 marks]\n'
        pieces = [(0, len(text), 's1', 'in_scope')]
        ann, scope, struct = fixture(text, pieces, [{'start_char': 0, 'end_char': len(text), 'scope_segment_id': 's1', 'question_id': 'Q1', 'candidate_type': 'question_header'}])
        out = propose_semantic_annotations(ann, scope, struct)
        self.assertEqual([c['label_proposal'] for c in out['candidates']].count('requirement'), 0)
        self.assertTrue(any(c['label_proposal'] is None for c in out['candidates']))

    def test_all_candidates_are_pending_not_training_eligible(self):
        text = '1. Calculate the result.\n'
        ann, scope, struct = fixture(text, [(0, len(text), 's1', 'in_scope')], [{'start_char': 0, 'end_char': len(text), 'scope_segment_id': 's1', 'question_id': 'Q1', 'candidate_type': 'question_header'}])
        out = propose_semantic_annotations(ann, scope, struct)
        self.assertTrue(out['candidates'])
        for c in out['candidates']:
            self.assertEqual(c['review_status'], 'pending')
            self.assertIsNone(c['annotation_label'])
            self.assertIsNone(c['training_label'])
            self.assertFalse(c['training_eligible'])

    def test_exact_source_text_and_hashes(self):
        text = '1. Calculate the result.\n'
        ann, scope, struct = fixture(text, [(0, len(text), 's1', 'in_scope')], [{'start_char': 0, 'end_char': len(text), 'scope_segment_id': 's1', 'question_id': 'Q1', 'candidate_type': 'question_header'}])
        out = propose_semantic_annotations(ann, scope, struct)
        for c in out['candidates']:
            self.assertEqual(c['text'], text[c['start_char']:c['end_char']])
            self.assertEqual(c['text_sha256'], sha(c['text']))

    def test_question_reconstruction_uses_only_in_scope_fragments(self):
        text = 'global\n1. Calculate x.\n2. Explain y.\n'
        p1 = text.index('1.')
        p2 = text.index('2.')
        pieces = [(0, p1, 'global', 'out_of_scope'), (p1, p2, 'q1', 'in_scope'), (p2, len(text), 'q2', 'in_scope')]
        ann, scope, struct = fixture(text, pieces, [
            {'start_char': p1, 'end_char': p2, 'scope_segment_id': 'q1', 'question_id': 'Q1', 'candidate_type': 'question_header'},
            {'start_char': p2, 'end_char': len(text), 'scope_segment_id': 'q2', 'question_id': 'Q2', 'candidate_type': 'question_header'}])
        out = propose_semantic_annotations(ann, scope, struct)
        self.assertEqual([q['question_id'] for q in out['question_reconstructions']], ['Q1', 'Q2'])
        self.assertTrue(all('global' not in q['reconstructed_text'] for q in out['question_reconstructions']))
        self.assertTrue(all(not q['reconstructed_text_is_literal_source_slice'] for q in out['question_reconstructions']))

    def test_mismatched_hash_fails_closed(self):
        text = '1. Calculate x.\n'
        ann, scope, struct = fixture(text, [(0, len(text), 's1', 'in_scope')], [{'start_char': 0, 'end_char': len(text), 'scope_segment_id': 's1', 'question_id': 'Q1', 'candidate_type': 'question_header'}])
        scope['source_text_sha256'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'source_text_sha256'):
            propose_semantic_annotations(ann, scope, struct)

    def test_candidate_spans_do_not_overlap(self):
        text = '1. Compute x. For example, use 2.\n'
        ann, scope, struct = fixture(text, [(0, len(text), 's1', 'in_scope')], [{'start_char': 0, 'end_char': len(text), 'scope_segment_id': 's1', 'question_id': 'Q1', 'candidate_type': 'question_header'}])
        out = propose_semantic_annotations(ann, scope, struct)
        spans = sorted((c['start_char'], c['end_char']) for c in out['candidates'])
        for (_, end), (start2, _) in zip(spans, spans[1:]): self.assertLessEqual(end, start2)

    def test_table_column_heading_does_not_trigger_requirement(self):
        text = 'ModelAccuracy        Train data   Validation data\nExperiment 1 (Regression mode)\n'
        ann, scope, struct = fixture(text, [(0, len(text), 's1', 'in_scope')], [
            {'start_char': 0, 'end_char': len(text), 'scope_segment_id': 's1', 'question_id': 'Q1', 'candidate_type': 'unclassified_in_scope_fragment', 'span_role': 'fragment'}])
        out = propose_semantic_annotations(ann, scope, struct)
        self.assertTrue(all(c['label_proposal'] != 'requirement' for c in out['candidates']))

    def test_literal_continuation_relation_only_for_explicit_group(self):
        text = '1. Compute the result\nand display the output.\n'
        split = text.index('and display')
        ann, scope, struct = fixture(text, [(0, split, 's1', 'in_scope'), (split, len(text), 's2', 'in_scope')], [
            {'start_char': 0, 'end_char': split, 'scope_segment_id': 's1', 'question_id': 'Q1', 'candidate_type': 'question_header'},
            {'start_char': split, 'end_char': len(text), 'scope_segment_id': 's2', 'question_id': 'Q1', 'candidate_type': 'continuation_fragment', 'span_role': 'fragment', 'continuation_of_candidate_id': 'SC0001'}])
        # Explicit continuation-group metadata is the gate for proposing a semantic continues relation.
        struct['candidates'][1]['continuation_group_id'] = 'CONT::SC0001'
        out = propose_semantic_annotations(ann, scope, struct)
        self.assertEqual(len(out['relation_proposals']), 1)
        self.assertEqual(out['relation_proposals'][0]['type_proposal'], 'continues')

    def test_inputs_are_not_mutated(self):
        text = '1. Compute x.\n'
        ann, scope, struct = fixture(text, [(0, len(text), 's1', 'in_scope')], [{'start_char': 0, 'end_char': len(text), 'scope_segment_id': 's1', 'question_id': 'Q1', 'candidate_type': 'question_header'}])
        snapshots = copy.deepcopy((ann, scope, struct))
        propose_semantic_annotations(ann, scope, struct)
        self.assertEqual((ann, scope, struct), snapshots)

    def test_only_question_paper_action_cues_used_not_domain_keywords(self):
        text = '1. Define an arbitrary term and prove the claim.\n'
        ann, scope, struct = fixture(text, [(0, len(text), 's1', 'in_scope')], [{'start_char': 0, 'end_char': len(text), 'scope_segment_id': 's1', 'question_id': 'Q1', 'candidate_type': 'question_header'}])
        out = propose_semantic_annotations(ann, scope, struct)
        req = next(c for c in out['candidates'] if c['label_proposal'] == 'requirement')
        self.assertEqual(req['requirement_attributes_proposal']['atomicity'], 'compound')

if __name__ == '__main__':
    unittest.main()
