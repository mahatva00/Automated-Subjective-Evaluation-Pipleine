#!/usr/bin/env python3
"""Audit question-only scope sidecars against raw text and annotation spans."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.annotation.scope import validate_scope_ledger, validate_annotations_against_scope

def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input-dir",required=True);p.add_argument("--scope-dir",required=True);p.add_argument("--pattern",default="*.annotation.json");p.add_argument("--recursive",action="store_true");p.add_argument("--report",required=True);p.add_argument("--require-complete",action="store_true")
 a=p.parse_args();root=Path(a.input_dir);sroot=Path(a.scope_dir)
 if not root.is_dir() or not sroot.is_dir():print("ERROR: input-dir and scope-dir must exist");return 2
 paths=sorted(root.rglob(a.pattern) if a.recursive else root.glob(a.pattern))
 if not paths:print("ERROR: no annotation files found");return 2
 docs=[];all_errors=[]
 for path in paths:
  rel=path.relative_to(root);stem=path.name.removesuffix(".annotation.json");scope_path=sroot/rel.parent/f"{stem}.scope.json"
  if not scope_path.is_file():all_errors.append(f"{path.name}: missing scope sidecar {scope_path}");continue
  try:doc=json.loads(path.read_text(encoding="utf-8"));ledger=json.loads(scope_path.read_text(encoding="utf-8"))
  except Exception as exc:all_errors.append(f"{path.name}: cannot parse JSON: {exc}");continue
  errs=validate_scope_ledger(doc,ledger)+validate_annotations_against_scope(doc,ledger)
  if a.require_complete and any(s.get("scope_status")=="pending" for s in ledger.get("segments",[])):errs.append("scope review is incomplete: pending segments remain")
  chars={k:sum(s.get("end_char",0)-s.get("start_char",0) for s in ledger.get("segments",[]) if s.get("scope_status")==k) for k in ("in_scope","out_of_scope","pending","separator")}
  record={"filename":path.name,"document_id":doc.get("document_id"),"segment_count":len(ledger.get("segments",[])),"character_counts":chars,"scope_complete":chars["pending"]==0,"annotation_span_errors":[e for e in errs if "annotations[" in e],"errors":errs}
  docs.append(record)
  if errs:all_errors.extend(f"{path.name}: {e}" for e in errs)
 report={"report_version":"1.0.0","policy_version":"question_scope_policy_v1.2","documents":docs,"documents_checked":len(docs),"errors":all_errors,"passed":not all_errors,"student_code_executed":False}
 out=Path(a.report);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 for d in docs:print(f"{d['filename']}: segments={d['segment_count']} chars={d['character_counts']} scope_complete={d['scope_complete']} errors={len(d['errors'])}")
 print(f"Audit {'PASS' if report['passed'] else 'FAIL'}: documents={len(docs)}, errors={len(all_errors)}, report={out}")
 return 0 if report["passed"] else 1
if __name__=="__main__":raise SystemExit(main())
