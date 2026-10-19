import json
import unittest
from pathlib import Path
from src.annotation.semantic_proposals import propose_semantic_annotations

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'examples' / 'input'

class FatQPPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.annotation = json.loads((INPUT / 'FAT-QP.annotation.json').read_text(encoding='utf-8'))
        cls.scope = json.loads((INPUT / 'FAT-QP.scope.json').read_text(encoding='utf-8'))
        cls.structure = json.loads((INPUT / 'FAT-QP.structure-proposal.json').read_text(encoding='utf-8'))
        cls.proposal = propose_semantic_annotations(cls.annotation, cls.scope, cls.structure)

    def test_only_questions_with_structure_assignment_are_reconstructed(self):
        self.assertEqual([x['question_id'] for x in self.proposal['question_reconstructions']], ['Q1', 'Q2'])
        self.assertEqual(self.proposal['scope_summary']['unassigned_in_scope_regions_omitted'], 1)

    def test_global_introduction_stays_unassigned(self):
        self.assertFalse(any('You are provided with a dataset that bears your roll.no' in c['text'] for c in self.proposal['candidates']))

    def test_every_candidate_is_an_exact_slice_and_pending(self):
        source = self.annotation['text']
        for c in self.proposal['candidates']:
            self.assertEqual(c['text'], source[c['start_char']:c['end_char']])
            self.assertEqual(c['review_status'], 'pending')
            self.assertIsNone(c['annotation_label'])
            self.assertIsNone(c['training_label'])
            self.assertFalse(c['training_eligible'])

    def test_pilot_candidates_are_nonoverlapping(self):
        spans = sorted((c['start_char'], c['end_char']) for c in self.proposal['candidates'])
        for (_, end), (start2, _) in zip(spans, spans[1:]):
            self.assertLessEqual(end, start2)

    def test_counts_reconcile_and_task_candidates_exist(self):
        self.assertEqual(sum(self.proposal['label_proposal_counts'].values(), 0), self.proposal['candidate_count'])
        self.assertGreater(sum(c['label_proposal'] == 'requirement' for c in self.proposal['candidates']), 0)
        self.assertGreater(sum(c['label_proposal'] == 'grading_instruction' for c in self.proposal['candidates']), 0)
        self.assertGreater(sum(c['label_proposal'] == 'example' for c in self.proposal['candidates']), 0)

if __name__ == '__main__':
    unittest.main()
