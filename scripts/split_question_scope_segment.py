#!/usr/bin/env python3
"""Split one pending row in a scope-review TSV at an absolute character offset."""
from __future__ import annotations
import argparse, csv, json, hashlib
from pathlib import Path

def sha(text): return hashlib.sha256(text.encode("utf-8")).hexdigest()
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--tsv",required=True);p.add_argument("--annotation-json",required=True);p.add_argument("--segment-id",required=True);p.add_argument("--split-at",type=int,required=True);p.add_argument("--output-tsv",required=True)
 a=p.parse_args();tsv=Path(a.tsv);doc=json.loads(Path(a.annotation_json).read_text(encoding="utf-8"));text=doc.get("text","")
 with tsv.open("r",encoding="utf-8-sig",newline="") as f:
  reader=csv.DictReader(f,delimiter="\t");fields=reader.fieldnames;rows=list(reader)
 if not fields: print("ERROR: TSV has no header");return 2
 idx=next((i for i,r in enumerate(rows) if r.get("segment_id")==a.segment_id),None)
 if idx is None: print(f"ERROR: segment not found: {a.segment_id}");return 2
 row=rows[idx]
 if row.get("scope_status")!="pending": print("ERROR: only pending rows may be split; split before scope review");return 2
 try: start,end=int(row["start_char"]),int(row["end_char"])
 except Exception: print("ERROR: row has invalid offsets");return 2
 if not start < a.split_at < end: print(f"ERROR: split-at must be inside {start}:{end}");return 2
 if json.loads(row.get("text_preview_json") or '""') != text[start:end]: print("ERROR: TSV preview does not match annotation source");return 2
 def make_piece(s,e,suffix):
  piece=text[s:e];r=dict(row);r.update({"segment_id":row["segment_id"]+suffix,"start_char":s,"end_char":e,"scope_status":"pending","question_id":"","reason":"","reviewer":"","review_date":"","text_sha256":sha(piece),"text_preview_json":json.dumps(piece,ensure_ascii=False)})
  return r
 rows[idx:idx+1]=[make_piece(start,a.split_at,"a"),make_piece(a.split_at,end,"b")]
 out=Path(a.output_tsv);out.parent.mkdir(parents=True,exist_ok=True)
 with out.open("w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields,delimiter="\t",lineterminator="\n");w.writeheader();w.writerows(rows)
 print(f"Split {a.segment_id} at char {a.split_at} -> {a.segment_id}a, {a.segment_id}b; output={out}")
 return 0
if __name__=="__main__":raise SystemExit(main())
