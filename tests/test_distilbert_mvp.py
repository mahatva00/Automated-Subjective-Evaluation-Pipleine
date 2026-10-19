import csv
import json
import sys
import unittest
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_distilbert_span_dataset import build_rows
from src.annotation.scope import build_scope_review

PILOT = ROOT / "data" / "annotations" / "approved_pilot"
MANIFEST = PILOT / "span_classifier_manifest.json"


class DistilBertMvpDataTests(unittest.TestCase):
    def _all_annotations(self):
        docs = [json.loads(p.read_text(encoding="utf-8")) for p in PILOT.glob("*.annotation.json")]
        return docs, {(doc["document_id"], ann["annotation_id"]) for doc in docs for ann in doc["annotations"]}

    def _scope_dir(self, tempdir):
        """Test-only scope map: all nonblank rows in-scope to exercise exporter plumbing.

        Real project scope sidecars must be manually reviewed under policy v1.2;
        this synthetic test helper is not suitable for production exports.
        """
        scope_dir = Path(tempdir)
        for path in PILOT.glob("*.annotation.json"):
            doc = json.loads(path.read_text(encoding="utf-8"))
            ledger, _ = build_scope_review(doc, path.name)
            for seg in ledger["segments"]:
                if seg["scope_status"] == "separator":
                    continue
                seg.update(scope_status="in_scope", question_id="q_test", reason="synthetic test fixture", review_status="reviewed", reviewer="test", review_date="2026-10-13")
            (scope_dir / (path.name.removesuffix(".annotation.json") + ".scope.json")).write_text(json.dumps(ledger), encoding="utf-8")
        return scope_dir

    def test_all_approved_csv_rows_are_approved_and_map_to_annotations(self):
        with (PILOT / "pilot_review.csv").open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        docs, annotation_keys = self._all_annotations()
        self.assertEqual(len(rows), 89)
        self.assertTrue(all(row["review_decision"] == "approve" for row in rows))
        csv_keys = {(row["document_id"], row["annotation_id"]) for row in rows}
        self.assertEqual(csv_keys, annotation_keys)
        self.assertEqual(len(csv_keys), len(rows))

    def test_candidate_spans_are_reviewed_but_partial_documents_stay_draft(self):
        docs, _ = self._all_annotations()
        annotation_count = 0
        relation_count = 0
        for doc in docs:
            self.assertEqual(doc["annotation_metadata"]["status"], "draft")
            self.assertTrue(all(a["review_status"] == "reviewed" for a in doc["annotations"]))
            self.assertTrue(all(r.get("review_status", "unreviewed") == "unreviewed" for r in doc["relations"]))
            annotation_count += len(doc["annotations"])
            relation_count += len(doc["relations"])
        self.assertEqual(annotation_count, 89)
        self.assertGreater(relation_count, 0)

    def test_builder_exports_only_reviewed_semantic_spans_when_in_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            scope_dir = self._scope_dir(tmp)
            rows, manifest, scope_counts = build_rows(PILOT, MANIFEST, scope_dir)
        self.assertEqual(len(rows), 57)
        self.assertTrue(all(row["layer"] == "semantic" for row in rows))
        self.assertTrue(all(row["coverage_note"].endswith("unknown, not a negative example") for row in rows))
        self.assertEqual({row["split"] for row in rows}, {"train", "validation"})
        self.assertEqual({row["label"] for row in rows}, {
            "requirement", "instruction_or_assessment_constraint", "option_or_alternative", "context_or_example"
        })
        self.assertEqual(scope_counts["included_in_scope"], 57)
        self.assertEqual(len({row["document_id"] for row in rows if row["split"] == "train"}), 2)
        self.assertEqual(len({row["document_id"] for row in rows if row["split"] == "validation"}), 1)

    def test_every_exported_span_matches_exact_source_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, _manifest, _scope_counts = build_rows(PILOT, MANIFEST, self._scope_dir(tmp))
        docs = {doc["document_id"]: doc for doc in (json.loads(p.read_text(encoding="utf-8")) for p in PILOT.glob("*.annotation.json"))}
        for row in rows:
            doc = docs[row["document_id"]]
            self.assertEqual(doc["text"][row["start_char"]:row["end_char"]], row["text"])
            self.assertEqual(row["scope_status"], "in_scope")

    def test_training_and_validation_are_split_by_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, _manifest, _scope_counts = build_rows(PILOT, MANIFEST, self._scope_dir(tmp))
        train_ids = {row["document_id"] for row in rows if row["split"] == "train"}
        val_ids = {row["document_id"] for row in rows if row["split"] == "validation"}
        self.assertTrue(train_ids)
        self.assertTrue(val_ids)
        self.assertFalse(train_ids & val_ids)


if __name__ == "__main__":
    unittest.main()
