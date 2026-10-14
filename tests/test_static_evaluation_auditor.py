import unittest
from scripts.audit_static_evaluations import audit

class StaticEvaluationAuditTests(unittest.TestCase):
    def setUp(self):
        self.source = {
            "student_code_executed": False,
            "questions": [{
                "question_id": "q1",
                "requirements": [{
                    "requirement_id": "r1",
                    "question_id": "q1",
                    "atomic_text": "Show the number of missing values."
                }],
                "evidence_blocks": [{
                    "block_id": "b1",
                    "source_text": "df.isnull().sum()",
                    "output_text": "colA 0"
                }]
            }],
            "global_requirements": [{
                "requirement_id": "g1",
                "atomic_text": "Include an explanation."
            }]
        }
        def result(rid, status="complete", quote="df.isnull().sum()", context_complete=True, evidence_sufficient=True):
            evidence = [{"block_id": "b1", "quote": quote, "purpose": "direct evidence"}] if quote else []
            return {
                "requirement_id": rid,
                "status": status,
                "rationale": "Auditor test rationale.",
                "confidence": 0.8,
                "context_complete": context_complete,
                "evidence_sufficient": evidence_sufficient,
                "evidence": evidence
            }
        self.evaluation = {
            "student_code_executed": False,
            "grading_policy_applied": False,
            "global_evaluations": [result("g1", status="needs_review", quote="")],
            "questions": [{"question_id":"q1","evaluations":[result("r1")]}]
        }

    def test_valid_result_passes(self):
        self.assertTrue(audit(self.source, self.evaluation)["passed"])

    def test_missing_requirement_fails(self):
        self.evaluation["questions"][0]["evaluations"] = []
        self.assertFalse(audit(self.source, self.evaluation)["passed"])

    def test_non_verbatim_quote_fails(self):
        self.evaluation["questions"][0]["evaluations"][0]["evidence"][0]["quote"] = "This calculates missing values"
        self.assertFalse(audit(self.source, self.evaluation)["passed"])

    def test_absent_with_incomplete_context_fails(self):
        row = self.evaluation["questions"][0]["evaluations"][0]
        row.update(status="absent", evidence=[], context_complete=False, evidence_sufficient=False)
        self.assertFalse(audit(self.source, self.evaluation)["passed"])

    def test_complete_without_citation_fails(self):
        row = self.evaluation["questions"][0]["evaluations"][0]
        row["evidence"] = []
        self.assertFalse(audit(self.source, self.evaluation)["passed"])

if __name__ == "__main__":
    unittest.main()
