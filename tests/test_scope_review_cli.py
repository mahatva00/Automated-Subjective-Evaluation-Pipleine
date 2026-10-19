import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def call(script, *args):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)], cwd=ROOT, env=env, capture_output=True, text=True)


class ScopeReviewCliTests(unittest.TestCase):
    def test_prepare_import_and_audit_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); inp = base / "drafts"; review = base / "review"; out = base / "scope"; report = base / "report.json"
            inp.mkdir()
            text = "GLOBAL instructions\n\nQuestion 1: calculate the mean.\n"
            doc = {"schema_version":"1.0.0","document_id":"scope-cli-1","source":{"filename":"test.pdf","format":"pdf","document_type":"question_paper"},"text":text,"page_ranges":[],"annotations":[],"relations":[],"annotation_metadata":{"status":"draft","guidelines_version":"1.0.0","annotator_ids":[],"notes":""}}
            (inp / "T.annotation.json").write_text(json.dumps(doc),encoding="utf-8")
            p = call("prepare_question_scope_review.py", "--input-dir", inp, "--output-dir", review)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            tsv = review / "T.scope-review.tsv"
            with tsv.open("r",encoding="utf-8-sig",newline="") as f:
                reader=csv.DictReader(f,delimiter="\t"); fields=reader.fieldnames; rows=list(reader)
            for r in rows:
                if r["scope_status"] == "separator":
                    continue
                if "GLOBAL" in json.loads(r["text_preview_json"]):
                    r.update(scope_status="out_of_scope", question_id="", reason="Global instruction outside question", reviewer="tester", review_date="2026-10-13")
                else:
                    r.update(scope_status="in_scope", question_id="q1", reason="Question content", reviewer="tester", review_date="2026-10-13")
            with tsv.open("w",encoding="utf-8-sig",newline="") as f:
                w=csv.DictWriter(f,fieldnames=fields,delimiter="\t",lineterminator="\n");w.writeheader();w.writerows(rows)
            p=call("import_question_scope_review.py","--input-dir",inp,"--review-dir",review,"--output-dir",out)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            p=call("audit_question_scope.py","--input-dir",inp,"--scope-dir",out,"--report",report,"--require-complete")
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            result=json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(result["passed"])
            self.assertEqual(result["documents_checked"],1)
            ledger=json.loads((out/"T.scope.json").read_text(encoding="utf-8"))
            self.assertTrue(any(s["scope_status"]=="out_of_scope" for s in ledger["segments"]))
            self.assertTrue(any(s["scope_status"]=="in_scope" for s in ledger["segments"]))

if __name__ == "__main__": unittest.main()
