#!/usr/bin/env python3
"""Import edited scope-review TSV rows into canonical scope JSON sidecars."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.annotation.scope import SCOPE_VERSION, sha256_text, validate_scope_ledger

REQUIRED = {"document_id", "segment_id", "start_char", "end_char", "scope_status", "question_id", "reason", "reviewer", "review_date", "text_preview_json"}

def _page_numbers(doc, start, end):
    result=[]
    for page in doc.get("page_ranges", []):
        if isinstance(page, dict) and isinstance(page.get("start_char"), int) and isinstance(page.get("end_char"), int) and start < page["end_char"] and end > page["start_char"]:
            pn=page.get("page_number")
            if pn is not None and pn not in result: result.append(pn)
    return result

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",required=True,help="Directory containing canonical *.annotation.json files")
    p.add_argument("--review-dir",required=True,help="Directory containing matching *.scope-review.tsv files")
    p.add_argument("--output-dir",required=True,help="Directory for imported *.scope.json files")
    p.add_argument("--pattern",default="*.annotation.json");p.add_argument("--recursive",action="store_true");p.add_argument("--overwrite",action="store_true")
    a=p.parse_args();root=Path(a.input_dir);reviewroot=Path(a.review_dir);outroot=Path(a.output_dir)
    if not root.is_dir() or not reviewroot.is_dir(): print("ERROR: input-dir and review-dir must exist");return 2
    paths=sorted(root.rglob(a.pattern) if a.recursive else root.glob(a.pattern))
    if not paths: print("ERROR: no annotation files found");return 2
    for path in paths:
        rel=path.relative_to(root); stem=path.name.removesuffix(".annotation.json")
        tsv=reviewroot/rel.parent/f"{stem}.scope-review.tsv"
        out=outroot/rel.parent/f"{stem}.scope.json"
        if not tsv.is_file(): print(f"ERROR: missing review TSV: {tsv}");return 2
        if out.exists() and not a.overwrite: print(f"ERROR: output exists: {out}; use --overwrite or a new output directory");return 2
        doc=json.loads(path.read_text(encoding="utf-8"));text=doc.get("text")
        if not isinstance(text,str): print(f"ERROR: {path} has no text string");return 2
        with tsv.open("r",encoding="utf-8-sig",newline="") as f:
            reader=csv.DictReader(f,delimiter="\t")
            if not reader.fieldnames or not REQUIRED.issubset(set(reader.fieldnames)):
                print(f"ERROR: {tsv} missing columns: {sorted(REQUIRED-set(reader.fieldnames or []))}");return 2
            rows=list(reader)
        if any((r.get("document_id") or "") != doc.get("document_id") for r in rows): print(f"ERROR: document_id mismatch in {tsv}");return 2
        segments=[];cursor=0;seen=set()
        for i,r in enumerate(rows,1):
            try: start,end=int(r["start_char"]),int(r["end_char"])
            except (ValueError,TypeError): print(f"ERROR: invalid offsets in {tsv} row {i+1}");return 2
            sid=(r.get("segment_id") or "").strip();status=(r.get("scope_status") or "").strip();piece=text[start:end] if 0 <= start < end <= len(text) else None
            if not sid or sid in seen: print(f"ERROR: blank/duplicate segment_id at {tsv} row {i+1}");return 2
            seen.add(sid)
            if piece is None: print(f"ERROR: invalid range {start}:{end} at {tsv} row {i+1}");return 2
            if start != cursor: print(f"ERROR: scope TSV has gap/overlap at row {i+1}; expected start {cursor}, got {start}");return 2
            try: preview=json.loads(r.get("text_preview_json") or "null")
            except json.JSONDecodeError: print(f"ERROR: text_preview_json is not valid JSON at row {i+1}");return 2
            if preview != piece: print(f"ERROR: text preview does not match source at row {i+1}; use split helper if a row crosses a scope boundary");return 2
            qid=(r.get("question_id") or "").strip() or None;reason=(r.get("reason") or "").strip();reviewer=(r.get("reviewer") or "").strip();review_date=(r.get("review_date") or "").strip()
            if status not in {"in_scope","out_of_scope","pending","separator"}: print(f"ERROR: invalid scope_status {status!r} at row {i+1}");return 2
            if status == "separator":
                if piece.strip(): print(f"ERROR: row {i+1} cannot be separator because it contains non-whitespace characters");return 2
                review="not_required";qid=None;reviewer="";review_date=""
            elif status == "pending":
                review="pending"
                if reviewer or review_date: print(f"ERROR: pending row {i+1} cannot have reviewer/date");return 2
            else:
                review="reviewed"
                if not reviewer or not review_date: print(f"ERROR: decided row {i+1} needs reviewer and review_date");return 2
                if status == "in_scope" and not qid: print(f"ERROR: in_scope row {i+1} needs question_id");return 2
                if status == "out_of_scope" and qid: print(f"ERROR: out_of_scope row {i+1} must not have question_id");return 2
            segments.append({"segment_id":sid,"start_char":start,"end_char":end,"text_sha256":sha256_text(piece),"text":piece,"page_numbers":_page_numbers(doc,start,end),"scope_status":status,"question_id":qid,"reason":reason,"review_status":review,"reviewer":reviewer,"review_date":review_date})
            cursor=end
        if cursor != len(text): print(f"ERROR: TSV only covers to char {cursor}; source length is {len(text)} for {path.name}");return 2
        ledger={"scope_version":SCOPE_VERSION,"document_id":doc["document_id"],"annotation_filename":path.name,"source_filename":doc.get("source",{}).get("filename"),"source_text_sha256":sha256_text(text),"text_length":len(text),"policy_version":"question_scope_policy_v1.2","policy_summary":"Only content inside an actual question/subquestion is in scope. All other non-whitespace text is excluded from question requirement extraction and training loss.","segments":segments}
        errors=validate_scope_ledger(doc,ledger)
        if errors:
            print(f"ERROR: invalid scope review for {path.name}:");[print("  - "+e) for e in errors];return 2
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(f"Imported {path.name}: {len(segments)} segments; {sum(s['scope_status']=='in_scope' for s in segments)} in-scope; {sum(s['scope_status']=='out_of_scope' for s in segments)} out-of-scope; {sum(s['scope_status']=='pending' for s in segments)} pending")
    print("Scope TSV import complete. No annotation labels were created.")
    return 0
if __name__=="__main__": raise SystemExit(main())
