#!/usr/bin/env python3
"""Independently verify the raw approved pilot files and prebuilt DistilBERT dataset.
Uses only the Python standard library; checks contents, not console claims."""
from __future__ import annotations
import argparse, csv, json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT = ROOT / 'data' / 'annotations' / 'approved_pilot'
DEFAULT_DATASET = ROOT / 'data' / 'annotations' / 'distilbert_mvp' / 'semantic_span_dataset.jsonl'
DEFAULT_SUMMARY = ROOT / 'data' / 'annotations' / 'distilbert_mvp' / 'semantic_span_dataset_summary.json'

def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

def verify(pilot: Path, dataset_path: Path, summary_path: Path):
    errors=[]
    docs={}
    ann_by_key={}
    n_annotations=0
    n_relations=0
    for path in sorted(pilot.glob('*.annotation.json')):
        try: d=read_json(path)
        except Exception as e:
            errors.append(f'{path.name}: JSON parse failed: {e}'); continue
        docid=d.get('document_id')
        if not docid or docid in docs:
            errors.append(f'{path.name}: missing/duplicate document_id {docid!r}'); continue
        docs[docid]=d
        if d.get('annotation_metadata',{}).get('status')!='draft': errors.append(f'{path.name}: partial document is not draft')
        for a in d.get('annotations',[]):
            key=(docid,a.get('annotation_id'))
            if key in ann_by_key: errors.append(f'duplicate annotation key: {key}')
            ann_by_key[key]=a; n_annotations+=1
            if a.get('review_status')!='reviewed': errors.append(f'{key}: annotation not reviewed')
            s,e=a.get('start_char'),a.get('end_char')
            if not isinstance(s,int) or not isinstance(e,int) or s<0 or e<s or e>len(d.get('text','')) or d['text'][s:e]!=a.get('text'):
                errors.append(f'{key}: raw source offset/text mismatch')
        for r in d.get('relations',[]):
            n_relations+=1
            if r.get('review_status','unreviewed')!='unreviewed': errors.append(f'{path.name}: relation unexpectedly marked reviewed')
    csv_path=pilot/'pilot_review.csv'
    csv_rows=[]
    if not csv_path.is_file(): errors.append('pilot_review.csv missing')
    else:
        with csv_path.open(encoding='utf-8-sig',newline='') as f: csv_rows=list(csv.DictReader(f))
        csv_keys=[(r.get('document_id'),r.get('annotation_id')) for r in csv_rows]
        if len(csv_rows)!=89: errors.append(f'expected 89 CSV rows, got {len(csv_rows)}')
        if any(r.get('review_decision')!='approve' for r in csv_rows): errors.append('CSV has a non-approve decision')
        if len(set(csv_keys))!=len(csv_keys): errors.append('duplicate identity in CSV')
        if set(csv_keys)!=set(ann_by_key): errors.append('CSV identities do not exactly match annotation JSON identities')
    if len(docs)!=3: errors.append(f'expected 3 annotation docs, got {len(docs)}')
    if n_annotations!=89: errors.append(f'expected 89 annotations, got {n_annotations}')
    manifest_path=pilot/'span_classifier_manifest.json'
    manifest=read_json(manifest_path) if manifest_path.is_file() else {}
    if not manifest_path.is_file(): errors.append('span_classifier_manifest.json missing')
    manifest_by_path={x.get('annotation_path'):x for x in manifest.get('documents',[])}
    if len(manifest_by_path)!=3: errors.append(f'manifest must list 3 documents, got {len(manifest_by_path)}')
    for p in pilot.glob('*.annotation.json'):
        d=read_json(p); m=manifest_by_path.get(p.name)
        if not m: errors.append(f'{p.name}: absent from manifest'); continue
        if m.get('document_id')!=d.get('document_id'): errors.append(f'{p.name}: manifest document ID mismatch')
        split=m.get('split')
        if split not in ('train','validation'): errors.append(f'{p.name}: invalid split {split!r}')
    if not dataset_path.is_file():
        errors.append(f'dataset file missing: {dataset_path}')
        dataset_rows=[]
    else:
        dataset_rows=[json.loads(line) for line in dataset_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    ann_semantic={key:a for key,a in ann_by_key.items() if a.get('layer')=='semantic'}
    seen=set(); split_docs=Counter(); labels=Counter(); splits=Counter()
    for row in dataset_rows:
        key=(row.get('document_id'),row.get('annotation_id'))
        if key in seen: errors.append(f'duplicate dataset row: {key}')
        seen.add(key)
        ann=ann_semantic.get(key)
        if ann is None: errors.append(f'dataset row has no approved semantic annotation: {key}'); continue
        d=docs[key[0]]
        if row.get('text')!=ann.get('text') or row.get('start_char')!=ann.get('start_char') or row.get('end_char')!=ann.get('end_char'):
            errors.append(f'dataset row content/span mismatch: {key}')
        if d['text'][row['start_char']:row['end_char']]!=row.get('text'):
            errors.append(f'dataset raw source slice mismatch: {key}')
        m=manifest_by_path.get(next((name for name in manifest_by_path if manifest_by_path[name].get('document_id')==key[0]),''),{})
        if row.get('split')!=m.get('split'): errors.append(f'dataset split differs from manifest: {key}')
        if row.get('layer')!='semantic': errors.append(f'non-semantic training row: {key}')
        if not row.get('coverage_note','').endswith('unknown, not a negative example'):
            errors.append(f'row lacks unknown-not-negative coverage note: {key}')
        labels[row.get('label','unknown')]+=1; splits[row.get('split','unknown')]+=1; split_docs[row.get('split','unknown')]+=0
    if len(dataset_rows)!=57: errors.append(f'expected 57 dataset rows, got {len(dataset_rows)}')
    if seen!=set(ann_semantic): errors.append(f'dataset identities do not exactly cover semantic annotations (dataset={len(seen)}, semantic={len(ann_semantic)})')
    train_docs={r.get('document_id') for r in dataset_rows if r.get('split')=='train'}
    val_docs={r.get('document_id') for r in dataset_rows if r.get('split')=='validation'}
    if len(train_docs)!=2 or len(val_docs)!=1 or train_docs & val_docs: errors.append('document split isolation/counts are wrong')
    if splits!=Counter({'train':45,'validation':12}): errors.append(f'unexpected split counts: {dict(splits)}')
    if summary_path.is_file():
        summary=read_json(summary_path)
        if summary.get('num_examples')!=len(dataset_rows): errors.append('summary num_examples does not match JSONL')
        if summary.get('by_split')!={'train':45,'validation':12}: errors.append('summary split counts mismatch')
        expected_labels=dict(Counter(r.get('label','unknown') for r in dataset_rows))
        if summary.get('by_label')!=expected_labels: errors.append('summary label counts mismatch JSONL')
    else: errors.append(f'dataset summary missing: {summary_path}')
    return {
        'passed': not errors,
        'annotation_documents':len(docs),
        'annotation_records':n_annotations,
        'approval_csv_records':len(csv_rows),
        'relation_proposals_unreviewed':n_relations,
        'semantic_training_records':len(dataset_rows),
        'split_counts':dict(splits),
        'label_counts':dict(labels),
        'train_document_ids':sorted(train_docs),
        'validation_document_ids':sorted(val_docs),
        'all_selected_spans_match_raw_text': not any('offset' in x or 'source slice' in x for x in errors),
        'errors':errors,
        'limitations':['Only the 89 selected spans were approved; documents are partial and remain draft.', 'Approval was time-boxed by the project owner; labels have not been independently adjudicated by a domain expert.', 'The pilot covers three Machine Learning papers only.']
    }

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pilot-dir', type=Path, default=DEFAULT_PILOT)
    ap.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    ap.add_argument('--summary', type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument('--report', type=Path, default=None)
    args=ap.parse_args()
    result=verify(args.pilot_dir,args.dataset,args.summary)
    report=args.report or args.pilot_dir.parent/'distilbert_mvp'/'raw_data_verification.json'
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    print(f'Raw data verification: {"PASS" if result["passed"] else "FAIL"}')
    print(f'Report saved: {report}')
    return 0 if result['passed'] else 1

if __name__=='__main__': raise SystemExit(main())
