import unittest
from unittest.mock import patch
from src.evaluation.static_answer_evaluator import (
    select_evidence_blocks, validate_assessment, evaluate_evidence_map, GeminiClient
)

class FakeClient:
    model = "fake-test-model"
    def chat_json(self, payload):
        reqs = payload["requirements_to_assess"]
        return {"assessments": [
            {
                "requirement_id": r["requirement_id"],
                "status": "complete",
                "rationale": "The supplied cell contains direct evidence.",
                "confidence": 0.81,
                "evidence_sufficient": True,
                "evidence": [{
                    "block_id": r["evidence_blocks"][0]["block_id"],
                    "quote": r["evidence_blocks"][0]["source_text"].split(" [")[0],
                    "purpose": "Directly supports the requirement."
                }]
            } for r in reqs if r["evidence_blocks"]
        ]}

class StaticAnswerEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.blocks = [
            {"block_id":"b1", "segment_id":"cell1", "segment_position":1, "segment_type":"code",
             "source_text":"df.isnull().sum()", "output_text":"A    0\nB    1\n"},
            {"block_id":"b2", "segment_id":"cell2", "segment_position":2, "segment_type":"markdown",
             "source_text":"This section explains the result.", "output_text":""},
        ]
        self.req = {"requirement_id":"r1","question_id":"q1","question_number":"1",
                    "atomic_text":"Count the null values in each column.","requirement_type":"action",
                    "source_text":"Count the null values in each column."}

    def test_selects_relevant_block_and_marks_partial_context(self):
        selected, complete = select_evidence_blocks(self.req, self.blocks, max_blocks=1)
        self.assertEqual(selected[0]["block_id"], "b1")
        self.assertFalse(complete)

    def test_complete_requires_exact_evidence_quote(self):
        result = validate_assessment({
            "requirement_id":"r1","status":"complete","rationale":"Evidence found.",
            "confidence":0.9,"evidence_sufficient":True,
            "evidence":[{"block_id":"b1","quote":"df.isnull().sum()","purpose":"Direct code evidence."}]
        }, self.req, self.blocks, {"b1"}, False, "test")
        self.assertEqual(result["status"], "complete")

    def test_invalid_quote_is_downgraded_to_needs_review(self):
        result = validate_assessment({
            "requirement_id":"r1","status":"complete","rationale":"Evidence found.",
            "confidence":0.9,"evidence_sufficient":True,
            "evidence":[{"block_id":"b1","quote":"This code prints all data","purpose":"Claim."}]
        }, self.req, self.blocks, {"b1"}, False, "test")
        self.assertEqual(result["status"], "needs_review")
        self.assertTrue(result["validation_errors"])

    def test_absent_refused_when_context_is_incomplete(self):
        result = validate_assessment({
            "requirement_id":"r1","status":"absent","rationale":"Not present.",
            "confidence":0.8,"evidence_sufficient":True,"evidence":[]
        }, self.req, self.blocks, {"b1"}, False, "test")
        self.assertEqual(result["status"], "needs_review")

    def test_evidence_map_pipeline_preserves_all_requirement_ids(self):
        em = {
            "artifact_type":"student_question_evidence_map",
            "student_code_executed":False,
            "source_submission_filename":"sample.json",
            "source_question_map":"sample-QP.pdf",
            "source_requirements_filename":"sample-QP.pdf",
            "questions":[{
                "question_id":"q1","question_number":"1","question_title":"Null values",
                "mapping_status":"evidence_found","evidence_blocks":self.blocks,
                "requirements":[self.req]
            }]
        }
        result = evaluate_evidence_map(em, FakeClient(), batch_size=2)
        self.assertEqual(result["summary"]["total_requirements_assessed"], 1)
        self.assertEqual(result["questions"][0]["evaluations"][0]["requirement_id"], "r1")
        self.assertFalse(result["student_code_executed"])
        self.assertFalse(result["grading_policy_applied"])

    def test_empty_evidence_is_needs_review(self):
        em = {
            "student_code_executed":False,
            "questions":[{
                "question_id":"q1","question_number":"1","evidence_blocks":[],
                "requirements":[self.req]
            }]
        }
        result = evaluate_evidence_map(em, FakeClient())
        self.assertEqual(result["questions"][0]["evaluations"][0]["status"], "needs_review")

    def test_progress_callback_receives_checkpointable_results(self):
        calls = []
        evidence_map = {"student_code_executed": False, "questions": [{"question_id": "q1", "question_number": "1", "question_title": "", "evidence_blocks": self.blocks, "requirements": [self.req]}]}
        result = evaluate_evidence_map(evidence_map, FakeClient(), batch_size=1, progress_callback=lambda rows, total: calls.append((len(rows), total)))
        self.assertEqual(result["summary"]["total_requirements_assessed"], 1)
        self.assertEqual(calls[-1], (1, 1))


    def test_resume_skips_completed_requirement(self):
        class ExplodingClient:
            model = "test"
            def chat_json(self, payload):
                raise AssertionError("completed requirement should not be called again")
        evidence_map = {"student_code_executed": False, "questions": [{"question_id": "q1", "question_number": "1", "question_title": "", "evidence_blocks": self.blocks, "requirements": [self.req]}]}
        saved = {"requirement_id": "r1", "question_id": "q1", "question_number": "1", "requirement_text": self.req["atomic_text"], "requirement_type": "action", "status": "needs_review", "rationale": "saved checkpoint", "confidence": 0.0, "evidence_sufficient": False, "context_complete": False, "evidence": [], "evaluator_model": "test", "validation_errors": []}
        result = evaluate_evidence_map(evidence_map, ExplodingClient(), batch_size=1, completed_results=[saved])
        self.assertEqual(result["summary"]["total_requirements_assessed"], 1)
        self.assertEqual(result["questions"][0]["evaluations"][0]["rationale"], "saved checkpoint")


    def test_gemini_interactions_client_parses_model_output(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self):
                return b'{"steps":[{"type":"model_output","content":[{"type":"text","text":"{\\"assessments\\": []}"}]}]}'
        client = GeminiClient(model="gemini-3.8-flash", api_key="test-key", timeout=2)
        with patch("src.evaluation.static_answer_evaluator.urllib.request.urlopen", return_value=Response()) as call:
            result = client.chat_json({"requirements_to_assess": []})
        self.assertEqual(result, {"assessments": []})
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "https://generativelanguage.googleapis.com/v1beta/interactions")
        self.assertEqual(request.get_header("X-goog-api-key"), "test-key")

if __name__ == "__main__":
    unittest.main()
