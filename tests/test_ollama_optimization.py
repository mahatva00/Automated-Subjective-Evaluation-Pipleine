import json
import unittest
from unittest.mock import patch, Mock

from src.evaluation.static_answer_evaluator import OllamaClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def read(self):
        return self.payload


class OllamaOptimizationTests(unittest.TestCase):
    def test_request_uses_bounded_deterministic_settings(self):
        response_payload = {
            "message": {"content": '{"assessments": []}'},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "eval_count": 12,
            "total_duration": 100000000,
        }
        with patch("src.evaluation.static_answer_evaluator.urllib.request.urlopen", return_value=FakeResponse(response_payload)) as mocked:
            client = OllamaClient(model="qwen2.5:3b-instruct", num_ctx=8192, num_predict=2048, keep_alive="30m")
            result = client.chat_json({"requirements_to_assess": []})
        self.assertEqual(result, {"assessments": []})
        req = mocked.call_args.args[0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["model"], "qwen2.5:3b-instruct")
        self.assertFalse(body["stream"])
        self.assertEqual(body["format"], "json")
        self.assertEqual(body["keep_alive"], "30m")
        self.assertEqual(body["options"]["temperature"], 0)
        self.assertEqual(body["options"]["num_ctx"], 8192)
        self.assertEqual(body["options"]["num_predict"], 2048)

    def test_qwen3_reasoning_is_disabled(self):
        client = OllamaClient(model="qwen3:4b")
        self.assertIs(client._request_body({})["think"], False)

    def test_qwen25_does_not_require_think_parameter(self):
        client = OllamaClient(model="qwen2.5:3b-instruct")
        self.assertNotIn("think", client._request_body({}))

    def test_empty_output_fails_with_actionable_message(self):
        response_payload = {"message": {"content": ""}, "done": True, "done_reason": "length", "eval_count": 2048}
        with patch("src.evaluation.static_answer_evaluator.urllib.request.urlopen", return_value=FakeResponse(response_payload)):
            with self.assertRaisesRegex(RuntimeError, "empty message.content"):
                OllamaClient(model="qwen2.5:3b-instruct").chat_json({})

if __name__ == "__main__":
    unittest.main()
