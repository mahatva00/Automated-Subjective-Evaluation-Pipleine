import io
import json
import unittest
import urllib.error
from types import SimpleNamespace
from unittest.mock import patch
from src.evaluation.static_answer_evaluator import (
    build_question_component_payloads,
    select_evidence_blocks, validate_assessment, evaluate_evidence_map, GeminiClient, GroqClient, OllamaClient,
    IncompleteModelResponseError,
)
from src.evaluation.static_answer_evaluator import (
    _evidence_diagnostic, normalize_assessment_response,
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

    def test_component_payload_keeps_ai_log_separate_from_explanation(self):
        evidence_map = {
            "student_code_executed": False,
            "questions": [{
                "question_id": "q1",
                "question_number": "1",
                "question_title": "Data preparation",
                "mapping_status": "evidence_found",
                "missing_components": [],
                "ai_log_present": True,
                "component_blocks": {
                    "question": [],
                    "code": [{"block_id": "code1", "source_text": "df = load_data()"}],
                    "output": [{"block_id": "out1", "output_text": "rows: 10"}],
                    "explanation": [{"block_id": "exp1", "source_text": "I removed nulls."}],
                    "ai_log": [{"block_id": "ai1", "source_text": "ChatGPT suggested this."}],
                },
                "requirements": [self.req],
            }],
        }
        payload = build_question_component_payloads(evidence_map)
        self.assertEqual(payload[0]["components"]["explanation"][0]["block_id"], "exp1")
        self.assertEqual(payload[0]["components"]["ai_log"][0]["block_id"], "ai1")
        self.assertFalse(payload[0]["ai_log_is_scoring_evidence"])
        self.assertEqual(payload[0]["requirements"][0]["requirement_text"], self.req["atomic_text"])

    def test_requirement_evidence_selection_uses_linked_code_blocks_only(self):
        requirement = {
            "requirement_id": "r1",
            "atomic_text": "Prepare data.",
            "linked_evidence_block_ids": ["code1", "code2"],
        }
        blocks = [
            {"block_id": "code1", "segment_id": "cell1", "component": "code",
             "segment_position": 1, "source_text": "prepare_data()", "output_text": ""},
            {"block_id": "code1_output", "segment_id": "cell1", "component": "output",
             "segment_position": 1, "source_text": "", "output_text": "prepared"},
            {"block_id": "code1_explanation", "segment_id": "cell1", "component": "explanation",
             "segment_position": 1, "source_text": "I prepared data.", "output_text": ""},
        ]
        selected, complete = select_evidence_blocks(requirement, blocks)
        self.assertEqual(
            [block["block_id"] for block in selected],
            ["code1", "code1_output"],
        )
        self.assertTrue(complete)
        self.assertNotIn("code1_explanation", [block["block_id"] for block in selected])

    def test_selected_pdf_block_exposes_visual_evidence_as_auxiliary_context(self):
        block = {
            "block_id": "chart",
            "segment_position": 1,
            "source_text": "plt.legend()\nplt.xlabel('Index')",
            "output_text": "",
            "metadata": {
                "image_evidence": [{
                    "region_index": 1,
                    "ocr_text": "Bar chart: x1 vs r Index",
                    "ocr_confidence": 81.5,
                    "is_authoritative": False,
                }],
            },
        }
        selected, _ = select_evidence_blocks(self.req, [block])
        self.assertEqual(selected[0]["visual_evidence"][0]["ocr_text"], "Bar chart: x1 vs r Index")
        self.assertFalse(selected[0]["visual_evidence"][0]["is_authoritative"])

    def test_unlinked_code_requirement_recovers_same_question_code_and_output(self):
        requirement = {
            "requirement_id": "r-unlinked",
            "atomic_text": "Predict the target output for an unknown input.",
            "requirement_type": "deliverable",
            "linked_evidence_block_ids": [],
        }
        blocks = [
            {
                "block_id": "prediction_code",
                "segment_id": "cell1",
                "segment_position": 1,
                "component": "code",
                "source_text": "prediction = model.predict(x_unknown)",
                "output_text": "",
            },
            {
                "block_id": "prediction_output",
                "segment_id": "cell1",
                "segment_position": 1,
                "component": "output",
                "source_text": "",
                "output_text": "prediction: 42.5",
            },
            {
                "block_id": "unrelated_explanation",
                "segment_id": "cell2",
                "segment_position": 2,
                "component": "explanation",
                "source_text": "The model was trained on the dataset.",
                "output_text": "",
            },
        ]
        selected, complete = select_evidence_blocks(requirement, blocks)
        self.assertEqual(
            [block["block_id"] for block in selected],
            ["prediction_code", "prediction_output"],
        )
        self.assertFalse(complete)

    def test_unlinked_visual_requirement_recovers_plot_code_and_image_metadata(self):
        requirement = {
            "requirement_id": "r-visual",
            "atomic_text": "Draw a bar chart for each independent variable.",
            "requirement_type": "deliverable",
            "linked_evidence_block_ids": [],
        }
        blocks = [
            {
                "block_id": "plot_code",
                "segment_id": "cell1",
                "segment_position": 1,
                "component": "code",
                "source_text": "plt.bar(df.index, df['r'])\nplt.legend()",
                "output_text": "",
                "metadata": {
                    "image_evidence": [{
                        "region_index": 1,
                        "ocr_text": "Bar chart",
                        "is_authoritative": False,
                    }],
                },
            },
            {
                "block_id": "plot_explanation",
                "segment_id": "cell2",
                "segment_position": 2,
                "component": "explanation",
                "source_text": "This chart compares the variables.",
                "output_text": "",
            },
        ]
        selected, complete = select_evidence_blocks(requirement, blocks)
        self.assertEqual([block["block_id"] for block in selected], ["plot_code"])
        self.assertEqual(selected[0]["visual_evidence"][0]["ocr_text"], "Bar chart")
        self.assertFalse(complete)

    def test_short_visual_requirement_inherits_parent_code_context(self):
        requirement = {
            "requirement_id": "r-axis",
            "source_node_id": "q2_part_a_r5",
            "atomic_text": "Clearly label the axes.",
            "requirement_type": "action",
            "linked_evidence_block_ids": [],
        }
        blocks = [
            {
                "block_id": "parent_plot",
                "segment_id": "cell1",
                "segment_position": 1,
                "subquestion_anchor": "a",
                "component": "code",
                "source_text": "plt.plot(samples, train_error)\nplt.xlabel('samples')",
                "output_text": "",
            },
        ]
        selected, complete = select_evidence_blocks(requirement, blocks)
        self.assertEqual([block["block_id"] for block in selected], ["parent_plot"])
        self.assertFalse(complete)

    def test_unlinked_inference_requirement_uses_explanation_and_output_only(self):
        requirement = {
            "requirement_id": "r-inference",
            "atomic_text": "Conclude which experiment learned the coefficients better.",
            "requirement_type": "reasoning",
            "linked_evidence_block_ids": [],
        }
        blocks = [
            {
                "block_id": "inference_code",
                "segment_id": "cell1",
                "segment_position": 1,
                "component": "code",
                "source_text": "best = min(errors)",
                "output_text": "",
            },
            {
                "block_id": "inference_output",
                "segment_id": "cell1",
                "segment_position": 1,
                "component": "output",
                "source_text": "",
                "output_text": "E2 has the lowest coefficient error.",
            },
            {
                "block_id": "inference_explanation",
                "segment_id": "cell2",
                "segment_position": 2,
                "component": "explanation",
                "source_text": "E2 learned the coefficients better because its error is lowest.",
                "output_text": "",
            },
        ]
        selected, complete = select_evidence_blocks(requirement, blocks)
        self.assertEqual(
            [block["block_id"] for block in selected],
            ["inference_output", "inference_explanation"],
        )
        self.assertFalse(complete)

    def test_evidence_bundle_is_included_for_unmapped_requirement_context(self):
        requirement = {
            "requirement_id": "r-bundle",
            "atomic_text": "Explain the result.",
            "requirement_type": "reasoning",
            "linked_evidence_block_ids": [],
        }
        blocks = [
            {
                "block_id": "out1", "segment_id": "cell1", "segment_position": 1,
                "component": "output", "source_text": "", "output_text": "result",
            },
            {
                "block_id": "exp1", "segment_id": "cell2", "segment_position": 2,
                "component": "explanation", "source_text": "The result is explained.",
                "output_text": "",
            },
        ]
        selected, complete = select_evidence_blocks(requirement, blocks)
        self.assertFalse(complete)
        self.assertEqual(
            {block["component"] for block in selected},
            {"output", "explanation"},
        )

    def test_selects_relevant_block_and_marks_partial_context(self):
        selected, complete = select_evidence_blocks(self.req, self.blocks, max_blocks=1)
        self.assertEqual(selected[0]["block_id"], "b1")
        self.assertFalse(complete)

    def test_truncated_block_includes_relevant_excerpt_and_pdf_metadata(self):
        source = "unrelated preface\n" + ("unrelated words " * 100) + "Count null values"
        block = {
            "block_id": "pdf-page-8",
            "segment_id": "page_0008",
            "segment_position": 8,
            "segment_type": "pdf_page",
            "source_text": source,
            "page_number": 8,
            "metadata": {
                "extraction_method": "pdf_text+image_ocr",
                "extraction_status": "partial",
                "embedded_image_count": 1,
            },
        }

        selected, context_complete = select_evidence_blocks(
            self.req, [block], max_block_chars=100
        )

        self.assertFalse(context_complete)
        self.assertEqual(selected[0]["page_number"], 8)
        self.assertIn("null values", selected[0]["source_text"])
        self.assertIn("EXCERPT", selected[0]["source_text"])
        self.assertEqual(
            selected[0]["metadata"]["extraction_method"], "pdf_text+image_ocr"
        )

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

    def test_diagnostic_classifies_invalid_model_quote(self):
        result = validate_assessment({
            "requirement_id": "r1", "status": "complete",
            "rationale": "Evidence found.", "confidence": 0.9,
            "evidence_sufficient": True,
            "evidence": [{"block_id": "b1", "quote": "invented text", "purpose": "Claim."}],
        }, self.req, self.blocks, {"b1"}, True, "test")
        diagnostic = _evidence_diagnostic(self.req, self.blocks[:1], True, result)
        self.assertEqual(diagnostic["category"], "invalid_model_quote")

    def test_diagnostic_classifies_invalid_model_status(self):
        result = validate_assessment({
            "requirement_id": "r1", "status": "unsupported",
            "rationale": "Unknown status.", "confidence": 0.9,
            "evidence_sufficient": True, "evidence": [],
        }, self.req, self.blocks, set(), True, "test")
        diagnostic = _evidence_diagnostic(self.req, self.blocks[:1], True, result)
        self.assertEqual(diagnostic["category"], "invalid_model_status")

    def test_normalize_assessment_response_rejects_unknown_duplicates_and_exposes_missing(self):
        normalized = normalize_assessment_response(
            {
                "assessments": [
                    {"requirement_id": "r1", "status": "complete"},
                    {"requirement_id": "r1", "status": "partial"},
                    {"requirement_id": "unknown", "status": "complete"},
                ]
            },
            ["r1", "r2"],
        )
        self.assertEqual(normalized["assessments"][0]["status"], "complete")
        self.assertTrue(normalized["assessments"][1]["_response_missing"])

    def test_diagnostic_classifies_missing_model_assessment(self):
        result = validate_assessment(
            {"requirement_id": "r1", "_response_missing": True},
            self.req, self.blocks, set(), True, "test",
        )
        diagnostic = _evidence_diagnostic(self.req, self.blocks[:1], True, result)
        self.assertEqual(diagnostic["category"], "missing_model_assessment")

    def test_diagnostic_classifies_ambiguous_judgment(self):
        requirement = dict(self.req)
        requirement["linked_code_block_ids"] = ["b1"]
        result = {
            "status": "needs_review",
            "validation_errors": [],
        }
        diagnostic = _evidence_diagnostic(requirement, self.blocks[:1], True, result)
        self.assertEqual(diagnostic["category"], "ambiguous_model_judgment")

    def test_diagnostic_classifies_missing_required_component(self):
        requirement = dict(self.req)
        requirement["mapping_diagnostics"] = {"category": "true_no_evidence"}
        diagnostic = _evidence_diagnostic(requirement, [], False, {"validation_errors": []})
        self.assertEqual(diagnostic["category"], "missing_required_component")
        self.assertIn("missing_required_component", diagnostic["context_reasons"])

    def test_diagnostic_explains_truncated_and_omitted_context(self):
        requirement = dict(self.req)
        requirement["evidence_bundle"] = {"block_ids": ["b1", "b2"]}
        selected = [
            {
                "block_id": "b1",
                "component": "code",
                "source_text": "df.isnull() [SOURCE TEXT EXCERPT; ORIGINAL CONTENT OMITTED]",
                "output_text": "",
            },
        ]
        diagnostic = _evidence_diagnostic(
            requirement, selected, False, {"validation_errors": []}
        )
        self.assertIn("evidence_truncated", diagnostic["context_reasons"])
        self.assertIn("bundle_blocks_omitted", diagnostic["context_reasons"])
        self.assertEqual(diagnostic["omitted_bundle_block_ids"], ["b2"])

    def test_context_cap_retains_output_and_explanation_components(self):
        requirement = {
            "requirement_id": "r-context",
            "atomic_text": "Explain the result.",
            "linked_evidence_block_ids": ["code1", "code2"],
            "evidence_bundle": {"block_ids": ["code1", "code2", "out1", "exp1"]},
        }
        blocks = [
            {"block_id": "code1", "component": "code", "segment_position": 1,
             "source_text": "first()", "output_text": ""},
            {"block_id": "code2", "component": "code", "segment_position": 2,
             "source_text": "second()", "output_text": ""},
            {"block_id": "out1", "component": "output", "segment_position": 2,
             "source_text": "", "output_text": "result"},
            {"block_id": "exp1", "component": "explanation", "segment_position": 3,
             "source_text": "The result is stable.", "output_text": ""},
        ]
        selected, complete = select_evidence_blocks(requirement, blocks, max_blocks=3)
        self.assertFalse(complete)
        self.assertEqual(
            {block["component"] for block in selected},
            {"code", "output", "explanation"},
        )
        self.assertIn("code1", [block["block_id"] for block in selected])

    def test_invalid_quote_does_not_discard_other_valid_evidence(self):
        result = validate_assessment({
            "requirement_id": "r1", "status": "partial",
            "rationale": "The first block supports part of the requirement.",
            "confidence": 0.7, "evidence_sufficient": True,
            "evidence": [
                {"block_id": "b1", "quote": "df.isnull().sum()", "purpose": "Valid code."},
                {"block_id": "b2", "quote": "invented text", "purpose": "Invalid quote."},
            ],
        }, self.req, self.blocks, {"b1", "b2"}, False, "test")
        self.assertEqual(result["status"], "partial")
        self.assertEqual([item["block_id"] for item in result["evidence"]], ["b1"])
        self.assertTrue(result["validation_warnings"])
        self.assertFalse(result["validation_errors"])

    def test_line_wrapped_quote_is_recovered_as_verbatim_source(self):
        block = {
            "block_id": "pdf1",
            "source_text": "The analysis uses class-\nseparation to compare groups.",
            "output_text": "",
        }
        result = validate_assessment(
            {
                "requirement_id": "r1",
                "status": "partial",
                "rationale": "The report describes class separation.",
                "confidence": 0.8,
                "evidence_sufficient": True,
                "evidence": [{
                    "block_id": "pdf1",
                    "quote": "class-separation",
                    "purpose": "Describes the analysis.",
                }],
            },
            self.req,
            [block],
            {"pdf1"},
            True,
            "test",
        )

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["evidence"][0]["quote"], "class-\nseparation")
        self.assertTrue(result["validation_warnings"])

    def test_invalid_confidence_is_warning_not_a_lost_assessment(self):
        result = validate_assessment(
            {
                "requirement_id": "r1",
                "status": "complete",
                "rationale": "Evidence found.",
                "confidence": "not-a-number",
                "evidence_sufficient": True,
                "evidence": [{
                    "block_id": "b1",
                    "quote": "df.isnull().sum()",
                    "purpose": "Direct code evidence.",
                }],
            },
            self.req,
            self.blocks,
            {"b1"},
            True,
            "test",
        )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["confidence"], 0.0)
        self.assertTrue(result["validation_warnings"])
        self.assertFalse(result["validation_errors"])

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

    def test_missing_assessment_is_retried_as_single_requirement(self):
        class PartialThenRepairClient:
            model = "test"

            def __init__(self):
                self.calls = []

            def chat_json(self, payload):
                self.calls.append(payload)
                req = payload["requirements_to_assess"][0]
                if len(payload["requirements_to_assess"]) > 1:
                    return {"assessments": [{
                        "requirement_id": req["requirement_id"],
                        "status": "complete",
                        "rationale": "Evidence found.",
                        "confidence": 0.8,
                        "evidence_sufficient": True,
                        "evidence": [{
                            "block_id": req["evidence_blocks"][0]["block_id"],
                            "quote": req["evidence_blocks"][0]["source_text"],
                            "purpose": "Direct evidence.",
                        }],
                    }]}
                return FakeClient().chat_json(payload)

        second = dict(self.req)
        second["requirement_id"] = "r2"
        client = PartialThenRepairClient()
        evidence_map = {
            "student_code_executed": False,
            "questions": [{
                "question_id": "q1",
                "question_number": "1",
                "evidence_blocks": self.blocks,
                "requirements": [self.req, second],
            }],
        }
        result = evaluate_evidence_map(evidence_map, client, batch_size=2)
        evaluations = result["questions"][0]["evaluations"]
        self.assertEqual([row["status"] for row in evaluations], ["complete", "complete"])
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(client.calls[1]["requirements_to_assess"]), 1)

    def test_truncated_multi_requirement_batch_retries_only_that_batch_singly(self):
        class TruncatedThenCompleteClient:
            model = "test"

            def __init__(self):
                self.calls = []

            def chat_json(self, payload):
                requirements = payload["requirements_to_assess"]
                self.calls.append([item["requirement_id"] for item in requirements])
                if len(requirements) > 1:
                    raise IncompleteModelResponseError("response reached num_predict")
                return FakeClient().chat_json(payload)

        second = dict(self.req, requirement_id="r2")
        client = TruncatedThenCompleteClient()
        requests = []
        evidence_map = {
            "student_code_executed": False,
            "questions": [{
                "question_id": "q1",
                "question_number": "1",
                "evidence_blocks": self.blocks,
                "requirements": [self.req, second],
            }],
        }

        result = evaluate_evidence_map(
            evidence_map,
            client,
            batch_size=2,
            before_request_callback=requests.append,
        )

        self.assertEqual(client.calls, [["r1", "r2"], ["r1"], ["r2"]])
        self.assertEqual(requests, [2, 1, 1])
        self.assertEqual(
            [row["status"] for row in result["questions"][0]["evaluations"]],
            ["complete", "complete"],
        )

    def test_truncated_single_requirement_request_is_not_retried(self):
        class AlwaysTruncatedClient:
            model = "test"

            def __init__(self):
                self.calls = 0

            def chat_json(self, payload):
                self.calls += 1
                raise IncompleteModelResponseError("response reached num_predict")

        client = AlwaysTruncatedClient()
        evidence_map = {
            "student_code_executed": False,
            "questions": [{
                "question_id": "q1",
                "question_number": "1",
                "evidence_blocks": self.blocks,
                "requirements": [self.req],
            }],
        }
        with self.assertRaises(IncompleteModelResponseError):
            evaluate_evidence_map(evidence_map, client, batch_size=1)
        self.assertEqual(client.calls, 1)

    def test_invalid_quote_is_retried_as_single_requirement(self):
        class InvalidThenRepairClient:
            model = "test"

            def __init__(self):
                self.calls = []

            def chat_json(self, payload):
                self.calls.append(payload)
                req = payload["requirements_to_assess"][0]
                quote = "not in stored evidence"
                if len(self.calls) > 1:
                    quote = req["evidence_blocks"][0]["source_text"]
                return {"assessments": [{
                    "requirement_id": req["requirement_id"],
                    "status": "complete",
                    "rationale": "Evidence found.",
                    "confidence": 0.8,
                    "evidence_sufficient": True,
                    "evidence": [{
                        "block_id": req["evidence_blocks"][0]["block_id"],
                        "quote": quote,
                        "purpose": "Direct evidence.",
                    }],
                }]}

        client = InvalidThenRepairClient()
        evidence_map = {
            "student_code_executed": False,
            "questions": [{
                "question_id": "q1",
                "question_number": "1",
                "evidence_blocks": self.blocks,
                "requirements": [self.req],
            }],
        }
        result = evaluate_evidence_map(evidence_map, client, batch_size=1)
        self.assertEqual(result["questions"][0]["evaluations"][0]["status"], "complete")
        self.assertEqual(len(client.calls), 2)


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

    def test_gemini_retries_temporary_rate_limit_and_honors_retry_after(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self):
                return b'{"steps":[{"type":"model_output","content":[{"type":"text","text":"{\\"assessments\\": []}"}]}]}'

        rate_limited = urllib.error.HTTPError(
            "https://generativelanguage.googleapis.com/v1beta/interactions",
            429,
            "Too Many Requests",
            {"Retry-After": "2"},
            io.BytesIO(b"too many requests"),
        )
        client = GeminiClient(model="gemini-3.8-flash", api_key="test-key", timeout=2)
        with patch(
            "src.evaluation.static_answer_evaluator.urllib.request.urlopen",
            side_effect=[rate_limited, Response()],
        ) as request, patch("src.evaluation.static_answer_evaluator.time.sleep") as sleep:
            result = client.chat_json({"requirements_to_assess": []})
        self.assertEqual(result, {"assessments": []})
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once_with(2.5)

    def test_ollama_retries_http_429_and_stops_retrying_after_daily_quota_error(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self):
                payload = {
                    "message": {"content": '{"assessments": []}'},
                    "done": True,
                    "prompt_eval_count": 1,
                    "eval_count": 1,
                }
                return json.dumps(payload).encode("utf-8")

        client = OllamaClient(model="qwen2.5:7b-instruct", timeout=2)
        temporary = urllib.error.HTTPError(
            "http://localhost:11434/api/chat",
            429,
            "Too Many Requests",
            {},
            io.BytesIO(b"too many requests"),
        )
        with patch(
            "src.evaluation.static_answer_evaluator.urllib.request.urlopen",
            side_effect=[temporary, Response()],
        ) as request, patch("src.evaluation.static_answer_evaluator.time.sleep") as sleep:
            result = client.chat_json({"requirements_to_assess": []})
        self.assertEqual(result, {"assessments": []})
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once_with(1.0)

        quota = urllib.error.HTTPError(
            "http://localhost:11434/api/chat",
            429,
            "Too Many Requests",
            {},
            io.BytesIO(b"daily quota exceeded"),
        )
        with patch(
            "src.evaluation.static_answer_evaluator.urllib.request.urlopen",
            side_effect=quota,
        ) as request, patch("src.evaluation.static_answer_evaluator.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "non-temporary quota limit"):
                client.chat_json({"requirements_to_assess": []})
        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()

    def test_groq_retries_temporary_429_and_preserves_daily_quota_stop(self):
        class RateLimitError(Exception):
            status_code = 429
            response = SimpleNamespace(headers={"Retry-After": "3"})

        message = SimpleNamespace(content='{"assessments": []}')
        response = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

        class Completions:
            def __init__(self):
                self.calls = 0

            def create(self, **_):
                self.calls += 1
                if self.calls == 1:
                    raise RateLimitError("too many requests")
                return response

        completions = Completions()
        client = GroqClient.__new__(GroqClient)
        client.model = "test-model"
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with patch("src.evaluation.static_answer_evaluator.time.sleep") as sleep:
            result = client.chat_json({"requirements_to_assess": []})
        self.assertEqual(result, {"assessments": []})
        self.assertEqual(completions.calls, 2)
        sleep.assert_called_once_with(3.5)
        quota = RateLimitError("daily quota exceeded")
        self.assertIsNone(GroqClient._retry_delay_for_rate_limit(quota, 1))

if __name__ == "__main__":
    unittest.main()
