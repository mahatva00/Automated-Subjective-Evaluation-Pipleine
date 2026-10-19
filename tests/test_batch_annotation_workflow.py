import csv, tempfile, unittest, json, hashlib
from pathlib import Path
import importlib.util

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "batch_annotation_workflow.py"
spec = importlib.util.spec_from_file_location("batch_annotation_workflow", SCRIPT)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

class BatchTsvTests(unittest.TestCase):
    def test_merge_split_round_trip_keeps_per_file_schemas_and_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); src = root / "src"; (src / "nested").mkdir(parents=True)
            with (src / "a.tsv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["document_id", "decision"], delimiter="\t"); w.writeheader(); w.writerow({"document_id":"A", "decision":"accept"})
            with (src / "nested" / "b.tsv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["document_id", "decision", "note"], delimiter="\t"); w.writeheader(); w.writerow({"document_id":"B", "decision":"revise", "note":"x"})
            master = root / "MASTER.tsv"
            self.assertEqual(mod.merge_tsvs(src, "*.tsv", master), 2)
            out = root / "split"
            self.assertEqual(mod.split_tsv(master, out), 2)
            with (out / "a.tsv").open(encoding="utf-8-sig", newline="") as f:
                r = csv.DictReader(f, delimiter="\t"); self.assertEqual(r.fieldnames, ["document_id", "decision"]); self.assertEqual(next(r)["decision"], "accept")
            with (out / "nested" / "b.tsv").open(encoding="utf-8-sig", newline="") as f:
                r = csv.DictReader(f, delimiter="\t"); self.assertEqual(r.fieldnames, ["document_id", "decision", "note"]); self.assertEqual(next(r)["note"], "x")

    def test_structure_label_suggestions_are_conservative_and_schema_valid(self):
        self.assertEqual(mod.suggested_structure_label({"candidate_type":"question_header", "span_role":"header", "level":"0"}), "question_heading")
        self.assertEqual(mod.suggested_structure_label({"candidate_type":"question_header", "span_role":"header", "level":"1"}), "subquestion_heading")
        self.assertEqual(mod.suggested_structure_label({"candidate_type":"unattached_in_scope_fragment", "span_role":"fragment"}), "continuation_block")
        self.assertEqual(mod.suggested_structure_label({"candidate_type":"unclear_new_thing", "span_role":"unknown"}), "")


    def test_apply_structure_imports_only_explicitly_accepted_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); run_dir = root / "run"; proposal_dir = run_dir / "structure_proposals"; proposal_dir.mkdir(parents=True)
            ann_root = root / "current_annotations"; ann_root.mkdir()
            text = "Question 1."
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            ann_path = ann_root / "Demo-QP.annotation.json"
            doc = {"schema_version":"1.0.0", "document_id":"demo-q", "source":{"filename":"Demo-QP.pdf","format":"pdf","document_type":"question_paper"}, "text":text, "page_ranges":[], "annotations":[], "relations":[], "annotation_metadata":{"status":"draft"}}
            ann_path.write_text(json.dumps(doc), encoding="utf-8")
            proposal = {"document_id":"demo-q", "source_text_sha256":digest, "candidates":[{"candidate_id":"SC0001", "start_char":0, "end_char":len(text), "text":text}]}
            (proposal_dir / "Demo-QP.structure-proposal.json").write_text(json.dumps(proposal), encoding="utf-8")
            tsv_dir = run_dir / "structure-proposals-tsv"; tsv_dir.mkdir()
            tsv = tsv_dir / "Demo-QP.structure-proposal.tsv"
            fields = ["document_id","candidate_id","candidate_type","span_role","question_id","start_char","end_char","text_sha256","text","proposal_confidence","annotation_label","review_decision","reviewer","review_date","review_notes","_proposal_start_char","_proposal_end_char","_proposal_text_sha256"]
            row = {"document_id":"demo-q","candidate_id":"SC0001","candidate_type":"question_header","span_role":"header","question_id":"Q1","start_char":"0","end_char":str(len(text)),"text_sha256":digest,"text":text,"proposal_confidence":"high","annotation_label":"question_heading","review_decision":"accept","reviewer":"reviewer-1","review_date":"2026-10-13","review_notes":"","_proposal_start_char":"0","_proposal_end_char":str(len(text)),"_proposal_text_sha256":digest}
            with tsv.open("w", encoding="utf-8", newline="") as f:
                w=csv.DictWriter(f, fieldnames=fields, delimiter="\t"); w.writeheader(); w.writerow(row)
            master = run_dir / "MASTER_STRUCTURE_PROPOSALS.tsv"; mod.merge_tsvs(tsv_dir, "*.structure-proposal.tsv", master)
            out = root / "out"
            result = mod.cmd_apply_structure(type("Args", (), {"root":str(root),"run_dir":str(run_dir),"master":str(master),"annotation_dir":str(ann_root),"reviewer":"reviewer-1","review_date":"2026-10-13","output_dir":str(out)})())
            self.assertEqual(result, 0)
            output_doc = json.loads((out / ann_path.name).read_text(encoding="utf-8"))
            self.assertEqual(len(output_doc["annotations"]),1)
            self.assertEqual(output_doc["annotations"][0]["label"],"question_heading")
            self.assertEqual(output_doc["annotation_metadata"]["status"],"draft")

    def test_split_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); src = root / "src"; src.mkdir()
            with (src / "a.tsv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["document_id"], delimiter="\t"); w.writeheader(); w.writerow({"document_id":"A"})
            master = root / "MASTER.tsv"; mod.merge_tsvs(src, "*.tsv", master)
            out = root / "split"; mod.split_tsv(master, out)
            with self.assertRaises(FileExistsError): mod.split_tsv(master, out)

if __name__ == "__main__": unittest.main()
