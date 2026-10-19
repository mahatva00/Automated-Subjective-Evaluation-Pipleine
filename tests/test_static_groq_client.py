import json
import sys
import types
import unittest
from unittest.mock import Mock, patch
from src.evaluation.static_answer_evaluator import GroqClient

class GroqClientTests(unittest.TestCase):
    def setUp(self):
        self.response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content='{"assessments": []}'),
                finish_reason="stop",
            )],
            usage={"total_tokens": 20},
        )
        self.create = Mock(return_value=self.response)
        fake_sdk = types.ModuleType("groq")
        fake_sdk.Groq = Mock(return_value=types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))
        ))
        self.sdk_patch = patch.dict(sys.modules, {"groq": fake_sdk})
        self.sdk_patch.start()
        self.addCleanup(self.sdk_patch.stop)

    def test_parses_json_and_sends_strict_schema_via_sdk(self):
        client = GroqClient(model="openai/gpt-oss-20b", api_key="test-key", timeout=2)
        result = client.chat_json({"requirements_to_assess": []})
        self.assertEqual(result, {"assessments": []})
        kwargs = self.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "openai/gpt-oss-20b")
        self.assertEqual(kwargs["reasoning_effort"], "low")
        self.assertFalse(kwargs["include_reasoning"])
        self.assertEqual(kwargs["response_format"]["type"], "json_schema")
        self.assertTrue(kwargs["response_format"]["json_schema"]["strict"])

    def test_empty_content_reports_diagnostics(self):
        self.response.choices[0].message.content = ""
        client = GroqClient(model="openai/gpt-oss-20b", api_key="test-key", timeout=2)
        with self.assertRaisesRegex(RuntimeError, "finish_reason"):
            client.chat_json({"requirements_to_assess": []})

    def test_retries_tpm_429_then_returns_json(self):
        from unittest.mock import patch
        error = RuntimeError("Rate limit reached on tokens per minute (TPM). Please try again in 0.01s.")
        error.status_code = 429
        error.response = types.SimpleNamespace(headers={})
        self.create.side_effect = [error, self.response]
        client = GroqClient(model="openai/gpt-oss-20b", api_key="test-key", timeout=2)
        with patch("src.evaluation.static_answer_evaluator.time.sleep") as sleeper:
            result = client.chat_json({"requirements_to_assess": []})
        self.assertEqual(result, {"assessments": []})
        self.assertEqual(self.create.call_count, 2)
        sleeper.assert_called_once()

    def test_does_not_retry_daily_quota_429(self):
        error = RuntimeError("Your project has exceeded its daily quota.")
        error.status_code = 429
        error.response = types.SimpleNamespace(headers={})
        self.create.side_effect = error
        client = GroqClient(model="openai/gpt-oss-20b", api_key="test-key", timeout=2)
        from unittest.mock import patch
        with patch("src.evaluation.static_answer_evaluator.time.sleep") as sleeper:
            with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
                client.chat_json({"requirements_to_assess": []})
        self.assertEqual(self.create.call_count, 1)
        sleeper.assert_not_called()

    def test_missing_key_fails_clearly(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "GROQ_API_KEY"):
                GroqClient(api_key="")

if __name__ == "__main__":
    unittest.main()
