import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.annotation.validation import validate_document
from scripts.prepare_annotation_templates import build_template
from src.annotation.coverage import build_coverage
from src.annotation.scope import build_scope_review, filter_document_annotations_to_scope


class AnnotationValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example = json.loads((ROOT / "examples" / "example_annotation.json").read_text(encoding="utf-8"))

    def test_synthetic_example_is_valid_but_marked_non_training(self):
        self.assertEqual(validate_document(self.example), [])
        self.assertEqual(self.example["annotation_metadata"]["status"], "example")

    def test_annotation_offsets_must_match_source_text(self):
        doc = json.loads(json.dumps(self.example))
        doc["annotations"][0]["text"] = "wrong text"
        self.assertTrue(any("does not equal" in e for e in validate_document(doc)))

    def test_same_layer_overlaps_are_rejected(self):
        doc = json.loads(json.dumps(self.example))
        original = next(a for a in doc["annotations"] if a["annotation_id"] == "s_question_1")
        overlap = dict(original)
        overlap.update({"annotation_id": "s_overlapping", "start_char": 0, "end_char": 5, "text": doc["text"][0:5]})
        doc["annotations"].append(overlap)
        self.assertTrue(any("overlapping annotations in structure" in e for e in validate_document(doc)))

    def test_cross_layer_overlap_is_allowed(self):
        self.assertFalse(any("overlapping annotations" in e for e in validate_document(self.example)))

    def test_requirement_attributes_are_mandatory(self):
        doc = json.loads(json.dumps(self.example))
        req = next(a for a in doc["annotations"] if a["label"] == "requirement")
        req["attributes"].pop("requirement_type")
        self.assertTrue(any("requirement_type is invalid or missing" in e for e in validate_document(doc)))

    def test_relations_must_reference_existing_annotations(self):
        doc = json.loads(json.dumps(self.example))
        doc["relations"][0]["target_annotation_id"] = "missing"
        self.assertTrue(any("references missing target annotation" in e for e in validate_document(doc)))


class AnnotationTemplateTests(unittest.TestCase):
    def test_template_preserves_page_text_and_global_offsets(self):
        record = {
            "schema_version": "3.0.0",
            "submission": {"source_filename": "cross-domain-paper.pdf", "source_format": "pdf", "source_hash": "abcdef1234567890", "source_relative_path": "sample/paper.pdf"},
            "segments": [
                {"segment_id": "page_0001", "position": 0, "source_location": {"page_number": 1}, "original_type": "pdf_page", "content": "First page\nwrapped line", "extraction_method": "pdf_text"},
                {"segment_id": "page_0002", "position": 1, "source_location": {"page_number": 2}, "original_type": "pdf_page", "content": "Second page includes SQL, cloud, or another subject.", "extraction_method": "ocr"},
            ],
        }
        doc = build_template(record, Path("cross-domain-paper.json"))
        self.assertEqual(doc["text"], "First page\nwrapped line\n\f\nSecond page includes SQL, cloud, or another subject.")
        self.assertEqual(doc["annotations"], [])
        self.assertEqual(doc["annotation_metadata"]["status"], "draft")
        self.assertEqual(doc["page_ranges"][0]["start_char"], 0)
        self.assertEqual(doc["text"][doc["page_ranges"][1]["start_char"]:doc["page_ranges"][1]["end_char"]], "Second page includes SQL, cloud, or another subject.")


class TrainingExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.annotation_dir = self.base / "reviewed"
        self.coverage_dir = self.base / "coverage"
        self.scope_dir = self.base / "scope"
        self.annotation_dir.mkdir(); self.coverage_dir.mkdir(); self.scope_dir.mkdir()
        self.example = json.loads((ROOT / "examples" / "example_annotation.json").read_text(encoding="utf-8"))
        self.example["annotation_metadata"]["status"] = "reviewed"
        self.example["document_id"] = "reviewed-paper-001"
        self._write_doc_sidecars(self.example, "paper.annotation.json")
        self.manifest_path = self.base / "manifest.json"
        self.output_path = self.base / "training.jsonl"

    def _write_doc_sidecars(self, doc, filename):
        (self.annotation_dir / filename).write_text(json.dumps(doc), encoding="utf-8")
        cov = build_coverage(doc, filename)
        for chunk in cov["chunks"]:
            for layer in ("structure", "semantic"):
                chunk[f"{layer}_status"] = "reviewed"
                chunk[f"{layer}_reviewer"] = "test-reviewer"
                chunk[f"{layer}_review_date"] = "2026-10-13"
        (self.coverage_dir / (filename.removesuffix(".annotation.json") + ".coverage.json")).write_text(json.dumps(cov), encoding="utf-8")
        scope, _ = build_scope_review(doc, filename)
        for seg in scope["segments"]:
            if seg["scope_status"] == "separator":
                continue
            seg.update(scope_status="in_scope", question_id="q1", reason="test scope", review_status="reviewed", reviewer="test-reviewer", review_date="2026-10-13")
        (self.scope_dir / (filename.removesuffix(".annotation.json") + ".scope.json")).write_text(json.dumps(scope), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def run_export(self, manifest, legacy=False):
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        script = "build_training_jsonl.py" if legacy else "build_training_jsonl_coverage_safe.py"
        return subprocess.run([
            sys.executable, str(ROOT / "scripts" / script),
            "--input-dir", str(self.annotation_dir), "--coverage-dir", str(self.coverage_dir),
            "--scope-dir", str(self.scope_dir), "--manifest", str(self.manifest_path),
            "--output", str(self.output_path), "--recursive",
        ], cwd=ROOT, capture_output=True, text=True, check=False)

    def manifest(self, split="train", group="paper-001"):
        return {"manifest_version": "1.0.0", "documents": [{
            "document_id": "reviewed-paper-001", "annotation_path": "paper.annotation.json",
            "split": split, "group_id": group, "subject": "SQL"
        }]}

    def test_exports_reviewed_document_with_scope_and_coverage(self):
        result = self.run_export(self.manifest())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = self.output_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 1)
        row = json.loads(rows[0])
        self.assertEqual(row["dataset_assignment"]["split"], "train")
        self.assertEqual(row["document_id"], "reviewed-paper-001")
        self.assertEqual(row["scope_loss_contract"], "out_of_scope_pending_and_separator_tokens_must_use_ignore_index")
        self.assertIn("scope_review", row)

    def test_legacy_export_is_blocked_without_scope_and_coverage_flags(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_training_jsonl.py"),
                                 "--input-dir", str(self.annotation_dir), "--manifest", str(self.manifest_path),
                                 "--output", str(self.output_path)], cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unscoped training export is disabled", result.stdout)

    def test_export_filters_out_of_scope_legacy_annotations(self):
        # Add a pre-question global remark plus a valid legacy span over it. The
        # canonical input must remain untouched; only the emitted dataset copy filters it.
        doc = json.loads(json.dumps(self.example))
        prefix = "GLOBAL remark.\n\n"
        for ann in doc["annotations"]:
            ann["start_char"] += len(prefix)
            ann["end_char"] += len(prefix)
            ann["text"] = doc["text"][ann["start_char"] - len(prefix):ann["end_char"] - len(prefix)]
        doc["text"] = prefix + doc["text"]
        doc["annotations"].append({
            "annotation_id": "legacy-global-span", "layer": "semantic", "label": "other",
            "start_char": 0, "end_char": len("GLOBAL remark."), "text": "GLOBAL remark.",
            "attributes": {"label_note": "legacy global span for filtering test"}, "review_status": "reviewed", "annotator_id": "test-reviewer"
        })
        # Keep the demo file's page range/source offsets consistent for this test.
        doc["page_ranges"] = []
        (self.annotation_dir / "paper.annotation.json").write_text(json.dumps(doc), encoding="utf-8")
        cov = build_coverage(doc, "paper.annotation.json")
        for chunk in cov["chunks"]:
            chunk.update(structure_status="reviewed", semantic_status="reviewed",
                         structure_reviewer="test-reviewer", structure_review_date="2026-10-13",
                         semantic_reviewer="test-reviewer", semantic_review_date="2026-10-13")
        (self.coverage_dir / "paper.coverage.json").write_text(json.dumps(cov), encoding="utf-8")
        scope, _ = build_scope_review(doc, "paper.annotation.json")
        for seg in scope["segments"]:
            if seg["scope_status"] == "separator":
                continue
            is_global = seg["text"].strip() == "GLOBAL remark."
            seg.update(scope_status="out_of_scope" if is_global else "in_scope",
                       question_id=None if is_global else "q1", reason="test decision",
                       review_status="reviewed", reviewer="test-reviewer", review_date="2026-10-13")
        (self.scope_dir / "paper.scope.json").write_text(json.dumps(scope), encoding="utf-8")
        result = self.run_export(self.manifest())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        row = json.loads(self.output_path.read_text(encoding="utf-8").splitlines()[0])
        ids = {a["annotation_id"] for a in row["annotations"]}
        self.assertNotIn("legacy-global-span", ids)
        self.assertEqual(row["scope_filter_audit"]["excluded_annotation_count_by_scope_state"], {"out_of_scope": 1})
        # Canonical source still retains the original out-of-scope annotation.
        canonical = json.loads((self.annotation_dir / "paper.annotation.json").read_text(encoding="utf-8"))
        self.assertIn("legacy-global-span", {a["annotation_id"] for a in canonical["annotations"]})

    def test_refuses_unreviewed_draft(self):
        doc = json.loads((self.annotation_dir / "paper.annotation.json").read_text(encoding="utf-8"))
        doc["annotation_metadata"]["status"] = "draft"
        self._write_doc_sidecars(doc, "paper.annotation.json")
        result = self.run_export(self.manifest())
        self.assertEqual(result.returncode, 2)
        self.assertIn("expected reviewed/adjudicated", result.stdout)

    def test_refuses_same_group_in_different_splits(self):
        second = json.loads((self.annotation_dir / "paper.annotation.json").read_text(encoding="utf-8"))
        second["document_id"] = "reviewed-paper-002"
        self._write_doc_sidecars(second, "second.annotation.json")
        manifest = {"manifest_version": "1.0.0", "documents": [
            {"document_id": "reviewed-paper-001", "annotation_path": "paper.annotation.json", "split": "train", "group_id": "same-paper-family"},
            {"document_id": "reviewed-paper-002", "annotation_path": "second.annotation.json", "split": "test", "group_id": "same-paper-family"},
        ]}
        result = self.run_export(manifest)
        self.assertEqual(result.returncode, 2)
        self.assertIn("group leakage", result.stdout)


if __name__ == "__main__":
    unittest.main()
