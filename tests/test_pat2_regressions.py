import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.question_parser import parse_question_paper
from src.requirements.requirement_extractor import extract_atomic_requirements


PAT2_SAMPLE = """Instructions:
- Use Jupyter Notebook for this lab examination.
—-Question————- Generate a dataset D in xlsx format with 12 input attributes and one output
column.Name the input columns as L1, L2,...L12. Output column is named as Y. Size of the dataset should
only numbers. 10% dataset will be NaN. You have to generate the data set with your own code [5 marks]
1. Data Preprocessing (with the dataset D) [5 marks]
(a) Identify the NaN values in each of the input, output the number of NaN values in each of the input attributes. [1 mark]
(b) Do the imputation by replacing all the NaN values in an attribute with the mean of all the values of
that attribute or mode of all the values of that attribute. After executing the imputation, confirm that the imputation gets reflected in the dataset with appropriate outputs. [2 marks]
(c) Identify the outliers in each of the attribute and replace the outliers using the capping process. [2 marks]
2. Training the models [35 marks]
(a) With 75% of the dataset, Train a multiple linear regression model with a polynomial of degree 1 with all the terms by considering all input features and execute E1, E2, E3. [10 marks]
i. Experiment 1 (E1): Optimization strategy should be Stochastic Gradient Descent.
 Compute the error in E1 due to the training data and the error due to validation data.
 Based on the learning of your model, predict the output of 10 test data points (unknown data points). Test data points are to be chosen by you.
ii. Experiment 2 (E2): Optimization strategy should be Batch Gradient Descent.
 Compute the error in E2 due to training and validation data.
iii. Experiment 3 (E3): Optimization strategy should be Mini-batch Gradient Descent.
 Compute the error in E3 due to training and validation data.
iv. Experiment 4 (E4): Generate a new data set D1. [5 marks]
 Compute Q1, Q2, Q3 for output column D.
 For output values <= Q1, assign label A
 For output values > Q3, assign label D
Thus, D1 replaces output values with labels.
v. Experiment 5: Train a decision tree with information gain.
 Predict labels for the 10 input data points.
 Generate accuracy and classification report.
vi. Experiment 6: Train a tree-based classifier with Gini index.
 Predict labels for the 10 input data points.
vii. Experiment 7: Create a new dataframe D2.
 Output the new data set D2.
viii. Experiment 8 (E8) [10 marks]
 From dataset D, construct dataset D3 using any five input features.
 Convert D3 into classification dataset D4; explain the discretization strategy.
 Train a Decision Tree with entropy.
 Predict class labels for E1 test instances.
OR
Train a decision-tree model with the dataset D (without modifying D) and predict the output of the ten test-data points used in E1.
3. Inferences [5 marks]
Based on experiments E1 to E8, answer the following:
1. Of the three regression models, which learns best? [1 mark]
2. Of the two classification models, which performs better? [1 mark]
3. For each of the ten data points in D2, compute mode of labels and print that mode as the new class.
"""


def parsed_pat2():
    return parse_question_paper({
        "submission": {"source_filename": "PAT2-QP.pdf"},
        "segments": [{"content": PAT2_SAMPLE, "source_location": {"page_number": 1}}],
    })


class Pat2RegressionTests(unittest.TestCase):
    def test_dataset_generation_is_a_standalone_task_not_global_instruction(self):
        parsed = parsed_pat2()
        self.assertEqual(len(parsed["standalone_tasks"]), 1)
        task = parsed["standalone_tasks"][0]
        self.assertIn("Generate a dataset D in xlsx format", task["text"])
        self.assertFalse(any("Generate a dataset D" in i["text"] for i in parsed["global_instructions"]))

    def test_dotted_roman_experiments_and_bullets_form_tree(self):
        parsed = parsed_pat2()
        self.assertEqual([q["number"] for q in parsed["questions"]], ["1", "2", "3"])
        q2a = parsed["questions"][1]["subparts"][0]
        experiments = q2a["nested_subparts"]
        self.assertEqual([p["label"] for p in experiments], ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"])
        self.assertGreater(len(experiments[0]["nested_subparts"]), 0)
        self.assertTrue(any(child["label"].startswith("item_") for child in experiments[0]["nested_subparts"]))

    def test_or_branch_is_attached_to_experiment_eight(self):
        parsed = parsed_pat2()
        q2a = parsed["questions"][1]["subparts"][0]
        e8 = q2a["nested_subparts"][-1]
        self.assertEqual(len(e8["alternative_branches"]), 1)
        branch = e8["alternative_branches"][0]
        self.assertIn("without modifying D", branch["text"])
        self.assertEqual(e8["alternative_group_id"], branch["alternative_group_id"])
        self.assertEqual(parsed["questions"][1]["alternative_groups"][0]["kind"], "or_branch")

    def test_inference_numbering_is_nested_under_question_three(self):
        parsed = parsed_pat2()
        q3 = parsed["questions"][2]
        self.assertEqual([p["label"] for p in q3["subparts"]], ["1", "2", "3"])
        self.assertFalse(any(d["type"] == "repeated_question_number" for d in parsed["diagnostics"]))

    def test_inline_mean_mode_choice_is_detected_even_when_not_line_wrapped(self):
        parsed = parse_question_paper({
            "submission": {"source_filename": "choice.pdf"},
            "segments": [{"content": "1. Impute missing values.\n(a) Replace NaN with mean of values or mode of values.", "source_location": {"page_number": 1}}],
        })
        question = parsed["questions"][0]
        self.assertTrue(any(group.get("kind") == "inline_choice" for group in question["alternative_groups"]))
        result = extract_atomic_requirements(parsed)
        choice_records = [r for r in result["requirements"] if r["alternative_group_id"]]
        self.assertEqual(len(choice_records), 1)
        self.assertIn("mean", choice_records[0]["atomic_text"].lower())
        self.assertIn("mode", choice_records[0]["atomic_text"].lower())

    def test_fat_uppercase_experiment_I_is_not_confused_with_lowercase_roman_i(self):
        text = """2. Training models
(a) Carry out the following experiments.
i. Train the primary regression model.
A. Experiment 1: use stochastic descent.
\u0088 Compute training and validation error.
B. Experiment 2: use batch descent.
\u0088 Compute training and validation error.
I. Experiment 09: remove the target column and train K-means.
\u0088 Compute the silhouette score.
"""
        parsed = parse_question_paper({
            "submission": {"source_filename": "FAT-QP.pdf"},
            "segments": [{"content": text, "source_location": {"page_number": 1}}],
        })
        question = parsed["questions"][0]
        part_a = question["subparts"][0]
        self.assertEqual(part_a["label"], "a")
        roman_parent = part_a["nested_subparts"][0]
        self.assertEqual(roman_parent["label"], "i")
        alpha_items = [
            child for child in roman_parent["nested_subparts"]
            if child.get("node_kind") == "alpha_item"
        ]
        self.assertEqual([child["label"] for child in alpha_items], ["A", "B", "I"])
        self.assertIn("Experiment 09", alpha_items[-1]["text"])
        self.assertEqual(len([n for n in part_a["nested_subparts"] if n.get("label", "").lower() == "i"]), 1)

        extracted = extract_atomic_requirements(parsed)
        ids = [record["requirement_id"] for record in extracted["requirements"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_extractor_preserves_tree_alternatives_marks_and_source_spans(self):
        parsed = parsed_pat2()
        result = extract_atomic_requirements(parsed)
        self.assertEqual(len(result["standalone_tasks"]), 1)
        task_texts = [r["atomic_text"] for r in result["standalone_tasks"][0]["requirements"]]
        self.assertTrue(any(text.startswith("Generate a dataset D") for text in task_texts))
        self.assertFalse(any(text == "L12." for text in task_texts), task_texts)
        self.assertFalse(any(text.startswith("Question") for text in task_texts), task_texts)
        q2 = result["questions"][1]
        e8 = q2["subparts"][0]["nested_subparts"][-1]
        group_id = e8["alternative_group_id"]
        primary_ids = [r["alternative_group_id"] for r in e8["requirements"]]
        branch_ids = [r["alternative_group_id"] for r in e8["alternative_branches"][0]["requirements"]]
        for child in e8["nested_subparts"]:
            primary_ids.extend(r["alternative_group_id"] for r in child["requirements"])
        self.assertTrue(group_id)
        self.assertTrue(all(value == group_id for value in primary_ids + branch_ids))
        q1b = result["questions"][0]["subparts"][1]
        choice_reqs = [r for r in q1b["requirements"] if r["alternative_group_id"]]
        self.assertEqual(len(choice_reqs), 1)
        self.assertIn("mean", choice_reqs[0]["atomic_text"].lower())
        self.assertIn("mode", choice_reqs[0]["atomic_text"].lower())
        self.assertEqual(q1b["marks"], 2)
        self.assertTrue(all("[2 marks]" not in r["atomic_text"] for r in q1b["requirements"]))
        all_ids = [r["requirement_id"] for r in result["requirements"]]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        for req in result["requirements"]:
            start, end = req["source_span"]["start"], req["source_span"]["end"]
            self.assertEqual(req["source_text"][start:end], req["atomic_text"])


if __name__ == "__main__":
    unittest.main()
