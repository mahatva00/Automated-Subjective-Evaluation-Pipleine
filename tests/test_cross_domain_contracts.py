import unittest

from src.evaluation.evidence_mapper import build_evidence_map
from src.ingestion.question_parser import parse_question_paper
from src.requirements.requirement_extractor import extract_atomic_requirements


class CrossDomainContractTests(unittest.TestCase):
    def setUp(self):
        self.domain_questions = [
            "1. Write a SQL query that joins the orders and customers tables.",
            "2. Configure a virtual network and deploy an application to a private subnet.",
            "3. Calculate compound interest and show the formula used.",
            "4. Draw a network topology and explain the routing path.",
            "5. Compare two payment schedules and report the total cost.",
        ]
        self.question_paper = {
            "submission": {"source_filename": "CrossDomain-QP.pdf"},
            "segments": [{
                "content": "\n".join(self.domain_questions),
                "source_location": {"page_number": 1},
            }],
        }

    def test_question_and_requirement_parsing_preserve_unfamiliar_domain_terms(self):
        question_map = parse_question_paper(self.question_paper)
        requirements = extract_atomic_requirements(question_map)

        self.assertEqual(
            [question["number"] for question in question_map["questions"]],
            ["1", "2", "3", "4", "5"],
        )
        self.assertEqual(
            [question["text"] for question in question_map["questions"]],
            self.domain_questions,
        )

        extracted_by_number = {
            question["question_number"]: question
            for question in requirements["questions"]
        }
        cloud_requirements = extracted_by_number["2"]["requirements"]
        self.assertEqual(
            [requirement["atomic_text"] for requirement in cloud_requirements],
            [
                "Configure a virtual network",
                "deploy an application to a private subnet.",
            ],
        )
        self.assertEqual(
            [requirement["requirement_type"] for requirement in cloud_requirements],
            ["action", "action"],
        )
        terms_by_number = {
            "1": ("SQL", "orders", "customers"),
            "2": ("virtual network", "private subnet"),
            "3": ("compound interest", "formula"),
            "4": ("network topology", "routing path"),
            "5": ("payment schedules", "total cost"),
        }
        for number, expected_terms in terms_by_number.items():
            with self.subTest(question=number):
                question = extracted_by_number[number]
                source_texts = [
                    requirement["source_text"]
                    for requirement in question["requirements"]
                ]
                combined_source = "\n".join(source_texts)
                for term in expected_terms:
                    self.assertIn(term.casefold(), combined_source.casefold())
                for requirement in question["requirements"]:
                    start = requirement["source_span"]["start"]
                    end = requirement["source_span"]["end"]
                    self.assertEqual(
                        requirement["source_text"][start:end],
                        requirement["atomic_text"],
                    )

    def test_evidence_components_remain_question_scoped_across_domains(self):
        question_map = parse_question_paper(self.question_paper)
        requirements = extract_atomic_requirements(question_map)
        question_blocks = {}
        code_samples = {
            "1": "SELECT orders.id FROM orders JOIN customers ON orders.customer_id = customers.id;",
            "2": (
                'resource "virtual_network" "main" { '
                'resource "private_subnet" "app" { deploy_application(app) } }'
            ),
            "3": "formula = principal * rate * time; compound_interest = formula",
            "4": "network_topology = router_a + router_b; routing_path = [router_a, router_b]",
            "5": "payment_schedule.total_cost = SUM(amount)",
        }
        for number, code in code_samples.items():
            question_blocks[number] = [
                {
                    "block_id": f"q{number}_heading",
                    "segment_id": f"q{number}_heading",
                    "component": "question",
                    "source_text": self.domain_questions[int(number) - 1],
                    "output_text": "",
                },
                {
                    "block_id": f"q{number}_code",
                    "segment_id": f"q{number}_code",
                    "component": "code",
                    "source_text": code,
                    "output_text": "",
                },
                {
                    "block_id": f"q{number}_output",
                    "segment_id": f"q{number}_output",
                    "component": "output",
                    "source_text": "",
                    "output_text": f"Result for question {number}",
                },
                {
                    "block_id": f"q{number}_explanation",
                    "segment_id": f"q{number}_explanation",
                    "component": "explanation",
                    "source_text": f"Explanation for question {number}",
                    "output_text": "",
                },
                {
                    "block_id": f"q{number}_ai_log",
                    "segment_id": f"q{number}_ai_log",
                    "component": "ai_log",
                    "source_text": "No AI tools used.",
                    "output_text": "",
                },
            ]

        evidence_map = build_evidence_map(
            question_map,
            requirements,
            {"question_blocks": question_blocks, "segments": []},
            "cross-domain-submission.ipynb",
        )

        self.assertFalse(evidence_map["student_code_executed"])
        self.assertEqual(len(evidence_map["questions"]), len(code_samples))
        for question in evidence_map["questions"]:
            with self.subTest(question=question["question_number"]):
                number = question["question_number"]
                self.assertEqual(question["mapping_status"], "evidence_found")
                self.assertEqual(question["missing_components"], [])
                self.assertEqual(
                    [block["source_text"] for block in question["component_blocks"]["code"]],
                    [code_samples[number]],
                )
                linked_code_ids = {
                    block_id
                    for requirement in question["requirements"]
                    for block_id in requirement["linked_code_block_ids"]
                }
                self.assertEqual(linked_code_ids, {f"q{number}_code"})
                for requirement in question["requirements"]:
                    self.assertEqual(
                        requirement["linked_code_block_ids"],
                        [f"q{number}_code"],
                        requirement["atomic_text"],
                    )
                self.assertEqual(
                    len(question["component_blocks"]["output"]),
                    1,
                )
                self.assertEqual(
                    len(question["component_blocks"]["explanation"]),
                    1,
                )
                self.assertTrue(question["ai_log_present"])
                for requirement in question["requirements"]:
                    self.assertNotIn(
                        f"q{number}_ai_log",
                        requirement["evidence_bundle"]["block_ids"],
                    )

    def test_pdf_heading_variants_map_questions_independently_of_subject(self):
        question_map = {
            "source_filename": "CrossDomain-QP.pdf",
            "questions": [
                {
                    "question_id": "q1",
                    "number": "1",
                    "title_text": "Data preparation",
                    "text": "1. Data preparation",
                },
                {
                    "question_id": "q2",
                    "number": "2",
                    "title_text": "Query design",
                    "text": "2. Query design",
                },
            ],
            "standalone_tasks": [],
        }
        submission = {
            "segments": [{
                "segment_id": "page_1",
                "position": 0,
                "original_type": "pdf_page",
                "content": (
                    "Q1: Data preparation\n"
                    "prepare_dataset(source)\n"
                    "(2) Query design\n"
                    "SELECT * FROM orders;\n"
                ),
            }],
        }

        evidence_map = build_evidence_map(
            question_map,
            {"requirements": [], "global_requirements": []},
            submission,
            "cross-domain-answers.pdf",
        )

        self.assertEqual(
            [
                [block["source_text"] for block in question["evidence_blocks"]]
                for question in evidence_map["questions"]
            ],
            [
                ["Q1: Data preparation\nprepare_dataset(source)\n"],
                ["(2) Query design\nSELECT * FROM orders;\n"],
            ],
        )

    def test_pdf_numbered_subpart_anchor_requires_matching_question_paper_part(self):
        question_map = {
            "source_filename": "Numbered-Subparts-QP.pdf",
            "questions": [
                {
                    "question_id": "q1",
                    "number": "1",
                    "title_text": "Query tasks",
                    "text": "1. Query tasks",
                    "subparts": [{"label": "a"}, {"label": "b"}],
                },
                {
                    "question_id": "q2",
                    "number": "2",
                    "title_text": "Query design",
                    "text": "2. Query design",
                    "subparts": [],
                },
                {
                    "question_id": "q3",
                    "number": "3",
                    "title_text": "Independent task",
                    "text": "3. Independent task",
                    "subparts": [],
                },
            ],
            "standalone_tasks": [],
        }
        submission_text = (
            "1 a\n"
            "SELECT orders.id FROM orders;\n"
            "2 a\n"
            "An ordinary numbered list item, not a question-paper subpart.\n"
            "1 b\n"
            "SELECT customers.name FROM customers;\n"
            "continued query explanation\n"
            "Question 2: Query design\n"
            "SELECT * FROM customers;\n"
            "Question 3: Independent task\n"
            "return status_code\n"
        )

        evidence_map = build_evidence_map(
            question_map,
            {"requirements": [], "global_requirements": []},
            {
                "segments": [{
                    "segment_id": "page_1",
                    "position": 0,
                    "original_type": "pdf_page",
                    "content": submission_text,
                }],
            },
            "numbered-subparts.pdf",
        )

        questions = {
            question["question_number"]: question
            for question in evidence_map["questions"]
        }
        self.assertEqual(
            [block["source_text"] for block in questions["1"]["evidence_blocks"]],
            [
                (
                    "1 a\n"
                    "SELECT orders.id FROM orders;\n"
                    "2 a\n"
                    "An ordinary numbered list item, not a question-paper subpart.\n"
                ),
                (
                    "1 b\n"
                    "SELECT customers.name FROM customers;\n"
                    "continued query explanation\n"
                ),
            ],
        )
        self.assertEqual(questions["1"]["mapping_status"], "evidence_found")
        self.assertTrue(all(
            block["mapping_method"] == "explicit_question_subpart_anchor"
            for block in questions["1"]["evidence_blocks"]
            if block["heading_number"] == "1"
        ))
        self.assertEqual(
            [block["source_text"] for block in questions["2"]["evidence_blocks"]],
            ["Question 2: Query design\nSELECT * FROM customers;\n"],
        )
        self.assertEqual(
            [block["source_text"] for block in questions["3"]["evidence_blocks"]],
            ["Question 3: Independent task\nreturn status_code\n"],
        )


if __name__ == "__main__":
    unittest.main()
