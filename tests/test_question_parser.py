import unittest

from src.ingestion.question_parser import parse_question_paper


def make_document(*pages, filename="sample-QP.pdf"):
    return {
        "submission": {"source_filename": filename},
        "segments": [
            {
                "content": content,
                "source_location": {"page_number": page_number},
            }
            for page_number, content in enumerate(pages, start=1)
        ],
    }


class TestQuestionParser(unittest.TestCase):
    def test_extracts_top_level_questions(self):
        result = parse_question_paper(make_document(
            "1. First question\nDetails for first question.",
            "2. Second question\nDetails for second question.\n3. Third question",
        ))
        self.assertEqual([q["number"] for q in result["questions"]], ["1", "2", "3"])
        self.assertEqual(len(result["questions"]), 3)

    def test_preserves_subparts_across_page_boundaries(self):
        result = parse_question_paper(make_document(
            "4. Main question\n(a) First part\n(b) Second part",
            "4(c) Third part begins on the next page",
        ))
        question = result["questions"][0]
        self.assertEqual([part["label"] for part in question["subparts"]], ["a", "b", "c"])
        self.assertEqual(question["subparts"][2]["source_pages"], [2])
        self.assertIn("4(c) Third part begins on the next page", question["text"])

    def test_keeps_global_dataset_constraint(self):
        result = parse_question_paper(make_document(
            "Instructions:\nUse the dataset with at least 500 instances.\n1. Load the dataset and inspect it."
        ))
        texts = [item["text"] for item in result["global_instructions"]]
        self.assertTrue(any("at least 500 instances" in text for text in texts))
        self.assertEqual(len(result["questions"]), 1)

    def test_does_not_treat_exam_administration_as_questions(self):
        result = parse_question_paper(make_document(
            "1. After completing the exam:\n2. Name your file as:\n3. Upload the single PDF file.",
            "1. Data Preprocessing\n2. Training models",
        ))
        self.assertEqual([q["number"] for q in result["questions"]], ["1", "2"])
        combined = "\n".join(q["text"] for q in result["questions"])
        self.assertNotIn("After completing the exam", combined)
        self.assertNotIn("Upload the single PDF", combined)

    def test_preserves_repeated_question_numbers(self):
        result = parse_question_paper(make_document(
            "1. First version of question one",
            "1. Second version of question one",
        ))
        self.assertEqual([q["number"] for q in result["questions"]], ["1", "1"])
        self.assertEqual([q["occurrence"] for q in result["questions"]], [1, 2])
        repeated = [d for d in result["diagnostics"] if d["type"] == "repeated_question_number"]
        self.assertEqual(len(repeated), 1)
        self.assertEqual(repeated[0]["occurrences"], 2)

    def test_preserves_malformed_wording(self):
        original = "10. Create a new dataframe from dg that ."
        result = parse_question_paper(make_document(original))
        self.assertEqual(result["questions"][0]["text"], original)
        self.assertTrue(any(d["type"] == "possibly_incomplete_question" for d in result["diagnostics"]))

    def test_keeps_notes_separate_from_questions(self):
        result = parse_question_paper(make_document(
            "1. First question\nNote: This note applies to the paper.\n2. Second question"
        ))
        self.assertEqual(len(result["notes"]), 1)
        self.assertEqual(result["notes"][0]["text"], "Note: This note applies to the paper.")
        self.assertNotIn("This note applies", result["questions"][0]["text"])

    def test_extracts_inline_roman_items_without_lettered_parent(self):
        result = parse_question_paper(make_document(
            "6. Implement a model and compute the following measures.\n"
            "Following measures: (i)Recall (ii)Precision\n"
            "(iii) Accuracy (iv) F-measure."
        ))
        question = result["questions"][0]
        self.assertEqual([part["label"] for part in question["subparts"]], ["i", "ii", "iii", "iv"])
        self.assertEqual(
            [part["text"] for part in question["subparts"]],
            ["(i)Recall", "(ii)Precision", "(iii) Accuracy", "(iv) F-measure."],
        )
        self.assertIn("(i)Recall (ii)Precision", question["text"])
        self.assertIn("(iii) Accuracy (iv) F-measure.", question["text"])
        self.assertTrue(all(part["source_pages"] == [1] for part in question["subparts"]))

    def test_only_flags_real_inline_measure_choice(self):
        result = parse_question_paper(make_document(
            "1. Impute missing values.\n"
            "Replace NaN with mean of all values of\n"
            "that attribute or mode of all values of that attribute. After imputation, confirm\n"
            "the result in the dataset.\n"
            "2. Check whether the dataset is balanced or not.\n"
            "3. Check whether the coefficient is 3 (or nearer to 3)."
        ))
        markers = result["alternatives"]
        self.assertEqual(len(markers), 1)
        self.assertEqual(markers[0]["question_id"], "q1_occurrence_1")

    def test_extracts_roman_items_on_separate_lines_without_lettered_parent(self):
        result = parse_question_paper(make_document(
            "1. Compute the following measures.\n(i) Recall\n(ii) Precision\n(iii) Accuracy"
        ))
        question = result["questions"][0]
        self.assertEqual([part["label"] for part in question["subparts"]], ["i", "ii", "iii"])
        self.assertEqual(question["subparts"][1]["text"], "(ii) Precision")

    def test_keeps_roman_items_nested_under_lettered_subpart(self):
        result = parse_question_paper(make_document(
            "1. Main question\n(a) Compute these measures.\n(i) Recall\n(ii) Precision"
        ))
        question = result["questions"][0]
        self.assertEqual([part["label"] for part in question["subparts"]], ["a"])
        self.assertEqual(
            [part["label"] for part in question["subparts"][0]["nested_subparts"]],
            ["i", "ii"],
        )


if __name__ == "__main__":
    unittest.main()
