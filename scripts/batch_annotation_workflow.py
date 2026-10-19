#!/usr/bin/env python3
"""Batch orchestrator and consolidated-review utilities for the notebook evaluator.

Safe defaults: reads canonical inputs; writes each run into a new directory; never
executes student code; never promotes pending proposals to approved labels.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, shutil, subprocess, sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

META_SOURCE = "_batch_source_file"
META_REL = "_batch_source_relpath"
META_COLS = "_batch_original_columns_json"


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def all_files(root: Path, pattern: str) -> list[Path]:
    return sorted(p for p in root.rglob(pattern) if p.is_file()) if root.exists() else []


def index_by_doc(paths: Iterable[Path]) -> dict[str, Path]:
    out = {}
    for p in paths:
        try: doc_id = read_json(p).get("document_id")
        except Exception: continue
        if doc_id:
            if doc_id in out:
                raise ValueError(f"Duplicate document_id {doc_id!r}: {out[doc_id]} and {p}")
            out[doc_id] = p
    return out


def scope_complete(doc_path: Path, scope_path: Path) -> tuple[bool, str]:
    try:
        doc, ledger = read_json(doc_path), read_json(scope_path)
        text = doc.get("text")
        if not isinstance(text, str): return False, "annotation has no source text"
        if ledger.get("document_id") != doc.get("document_id"): return False, "document_id mismatch"
        if ledger.get("text_length") != len(text): return False, "text length mismatch"
        if ledger.get("source_text_sha256") and ledger["source_text_sha256"] != sha256(text): return False, "source hash mismatch"
        segs = ledger.get("segments")
        if not isinstance(segs, list) or not segs: return False, "scope segments missing"
        if any(s.get("scope_status") not in {"in_scope", "out_of_scope", "separator"} for s in segs):
            return False, "scope has pending/unknown segments"
        cursor = 0
        for s in segs:
            if s.get("start_char") != cursor or not isinstance(s.get("end_char"), int): return False, "scope has gap/overlap"
            end = s["end_char"]
            if not (cursor < end <= len(text)): return False, "scope span out of range"
            if s.get("text") != text[cursor:end]: return False, "scope text mismatch"
            if s.get("scope_status") == "separator" and text[cursor:end].strip(): return False, "non-whitespace separator"
            cursor = end
        if cursor != len(text): return False, "scope does not cover entire source"
        return True, "complete"
    except Exception as e:
        return False, f"cannot validate scope: {e}"


def coverage_finalized(doc_path: Path, coverage_path: Path) -> tuple[bool, str]:
    try:
        doc, ledger = read_json(doc_path), read_json(coverage_path)
        text = doc.get("text")
        if not isinstance(text, str): return False, "annotation has no source text"
        if ledger.get("document_id") != doc.get("document_id"): return False, "document_id mismatch"
        if ledger.get("text_length") != len(text): return False, "text length mismatch"
        if ledger.get("source_text_sha256") != sha256(text): return False, "source hash mismatch"
        if ledger.get("ledger_status") != "reviewed": return False, "coverage ledger is not reviewed"
        segments = ledger.get("segments")
        if not isinstance(segments, list) or not segments: return False, "coverage segments missing"
        if any(s.get("status") not in {"reviewed", "pending", "out_of_scope", "separator"} for s in segments): return False, "coverage has invalid status"
        return True, "reviewed coverage ledger matches source"
    except Exception as e:
        return False, f"cannot validate coverage: {e}"

def merge_tsvs(input_dir: Path, pattern: str, output: Path) -> int:
    paths = [p for p in all_files(input_dir, pattern) if p.resolve() != output.resolve()]
    if not paths: raise ValueError(f"No TSV files matched {pattern!r} in {input_dir}")
    loaded = []
    union = []
    for p in paths:
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f, delimiter="\t")
            if not r.fieldnames: raise ValueError(f"Empty/missing TSV header: {p}")
            cols = list(r.fieldnames)
            for c in cols:
                if c not in union: union.append(c)
            rows = list(r)
        loaded.append((p, cols, rows))
    out_fields = union + [META_SOURCE, META_REL, META_COLS]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        count = 0
        for p, cols, rows in loaded:
            try: rel = p.relative_to(input_dir).as_posix()
            except ValueError: rel = p.name
            for row in rows:
                row[META_SOURCE] = p.name
                row[META_REL] = rel
                row[META_COLS] = json.dumps(cols, ensure_ascii=False)
                w.writerow(row); count += 1
    return count


def split_tsv(master: Path, outdir: Path, overwrite: bool = False) -> int:
    with master.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        if not r.fieldnames: raise ValueError("Master TSV has no header")
        if not {META_SOURCE, META_REL, META_COLS}.issubset(r.fieldnames):
            raise ValueError(f"Master TSV must contain {META_SOURCE}, {META_REL}, {META_COLS}; use this tool's merge command")
        rows = list(r); fields = list(r.fieldnames)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        rel = (row.get(META_REL) or "").strip()
        src = (row.get(META_SOURCE) or "").strip()
        if not rel or not src: raise ValueError("Master TSV has blank source metadata; do not delete hidden _batch_* columns")
        groups.setdefault(rel, []).append(row)
    count = 0
    for rel, group in groups.items():
        cols = json.loads(group[0][META_COLS])
        if not isinstance(cols, list) or any(c not in fields for c in cols): raise ValueError(f"Invalid original columns metadata for {rel}")
        if any(row[META_COLS] != group[0][META_COLS] for row in group): raise ValueError(f"Mixed original schema in {rel}")
        target = outdir / Path(rel)
        if target.exists() and not overwrite: raise FileExistsError(f"Refusing to overwrite {target}; choose a new output directory or pass --overwrite")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            for row in group: w.writerow({c: row.get(c, "") for c in cols})
        count += len(group)
    return count


def cmd_inventory(args) -> int:
    root = Path(args.root).resolve()
    drafts = root / "data/annotations/drafts"
    if not drafts.is_dir(): raise ValueError(f"Canonical draft directory not found: {drafts}")
    ann_paths = all_files(drafts, "*.annotation.json")
    if not ann_paths: raise ValueError(f"No *.annotation.json files under {drafts}")
    scopes = index_by_doc(all_files(root / "data/annotations/scope_reviews", "*.scope.json"))
    structures = index_by_doc(all_files(root / "data/annotations/structure_proposals", "*.structure-proposal.json"))
    semantics = index_by_doc(all_files(root / "data/annotations/semantic_proposals_v1", "*.semantic-proposal.json"))
    coverage = index_by_doc(all_files(root / "data/annotations/training_coverage_reviewed", "*.training-coverage.json"))
    out = Path(args.output) if args.output else root / "data/annotations/batch_workflow_inventory.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["document_id", "annotation_file", "source_text_chars", "source_text_sha256", "scope_file", "scope_complete", "scope_note", "structure_proposal", "semantic_proposal", "reviewed_coverage"]
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n"); w.writeheader()
        for p in ann_paths:
            doc = read_json(p); did = doc.get("document_id", ""); text = doc.get("text", "")
            scope = scopes.get(did); complete, note = scope_complete(p, scope) if scope else (False, "no scope ledger")
            w.writerow({"document_id": did, "annotation_file": p.relative_to(drafts).as_posix(), "source_text_chars": len(text) if isinstance(text, str) else "", "source_text_sha256": sha256(text) if isinstance(text, str) else "", "scope_file": str(scope.relative_to(root)) if scope else "", "scope_complete": str(complete).lower(), "scope_note": note, "structure_proposal": str(structures.get(did).relative_to(root)) if did in structures else "", "semantic_proposal": str(semantics.get(did).relative_to(root)) if did in semantics else "", "reviewed_coverage": str(coverage.get(did).relative_to(root)) if did in coverage else ""})
    print(f"Inventory: {len(ann_paths)} documents; scope-complete={sum(1 for p in ann_paths if scopes.get(read_json(p).get('document_id')) and scope_complete(p, scopes[read_json(p).get('document_id')])[0])}; output={out}")
    return 0


def _copy_preserve(src: Path, base: Path, dest: Path) -> Path:
    rel = src.relative_to(base)
    target = dest / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    return target



STRUCTURE_LABELS = {
    "document_title", "section_heading", "question_heading", "subquestion_heading",
    "question_body", "subquestion_body", "instruction_heading", "list_marker", "list_item",
    "table_heading", "table", "table_row", "table_cell", "code_block", "equation_block",
    "figure_caption", "page_header", "page_footer", "document_metadata", "continuation_block", "other",
}


def suggested_structure_label(row: dict) -> str:
    ctype = (row.get("candidate_type") or "").strip().lower()
    role = (row.get("span_role") or "").strip().lower()
    try: level = int(row.get("level") or 0)
    except (ValueError, TypeError): level = 0
    direct = {
        "document_title": "document_title", "section_heading": "section_heading",
        "instruction_heading": "instruction_heading", "list_marker": "list_marker",
        "list_item": "list_item", "table_heading": "table_heading", "table": "table",
        "table_row": "table_row", "table_cell": "table_cell", "code_block": "code_block",
        "equation_block": "equation_block", "figure_caption": "figure_caption",
        "page_header": "page_header", "page_footer": "page_footer", "document_metadata": "document_metadata",
        "continuation_block": "continuation_block", "question_body": "question_body",
        "subquestion_body": "subquestion_body", "question_header": "question_heading",
        "subquestion_header": "subquestion_heading",
    }
    if ctype in {"question_header", "question_heading"}:
        return "subquestion_heading" if level > 0 else "question_heading"
    if ctype in {"subquestion_header", "subquestion_heading"}:
        return "subquestion_heading"
    if role in {"header", "question_header"}:
        return "subquestion_heading" if level > 0 else "question_heading"
    if ctype in direct: return direct[ctype]
    if role in {"body", "question_body"}: return "subquestion_body" if level > 0 else "question_body"
    if role in {"list_marker", "list_item"}: return role
    if "table" in ctype: return "table"
    if "continuation" in ctype or "fragment" in ctype: return "continuation_block"
    return ""


def enrich_structure_review_tsvs(structure_dir: Path) -> None:
    for path in all_files(structure_dir, "*.structure-proposal.tsv"):
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if not reader.fieldnames: raise ValueError(f"Missing TSV header: {path}")
            fields = list(reader.fieldnames); rows = list(reader)
        extra = ["suggested_annotation_label", "review_decision", "reviewer", "review_date", "review_notes", "_proposal_start_char", "_proposal_end_char", "_proposal_text_sha256"]
        for col in extra:
            if col not in fields: fields.append(col)
        for row in rows:
            suggestion = suggested_structure_label(row)
            row["suggested_annotation_label"] = suggestion
            if not (row.get("annotation_label") or "").strip(): row["annotation_label"] = suggestion
            row["review_decision"] = ""
            row["reviewer"] = ""
            row["review_date"] = ""
            row["review_notes"] = ""
            row["_proposal_start_char"] = row.get("start_char", "")
            row["_proposal_end_char"] = row.get("end_char", "")
            row["_proposal_text_sha256"] = row.get("text_sha256", "")
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
            w.writeheader(); w.writerows(rows)


def cmd_apply_structure(args) -> int:
    root = Path(args.root).resolve(); run_dir = Path(args.run_dir).resolve(); master = Path(args.master).resolve()
    current_ann_root = Path(args.annotation_dir).resolve()
    review_dir = run_dir / "structure_review_edited"
    split_tsv(master, review_dir)
    current_docs = _documents_by_id(current_ann_root)
    proposals_root = run_dir / "structure_proposals"
    proposal_docs = index_by_doc(all_files(proposals_root, "*.structure-proposal.json"))
    out_root = Path(args.output_dir).resolve() if args.output_dir else root / "data/annotations/structure_review_imported_batch" / run_dir.name
    audit_root = out_root.parent / (out_root.name + "_audits")
    decisions_by_doc: dict[str, list[dict]] = {}
    for tsv in all_files(review_dir, "*.structure-proposal.tsv"):
        with tsv.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f, delimiter="\t")
            if not r.fieldnames or not {"document_id", "candidate_id", "review_decision", "annotation_label", "start_char", "end_char", "text"}.issubset(r.fieldnames):
                raise ValueError(f"Structure review TSV is missing required columns: {tsv}")
            rows = list(r)
        for row in rows:
            did = (row.get("document_id") or "").strip()
            if not did: raise ValueError(f"Blank document_id in {tsv}")
            decisions_by_doc.setdefault(did, []).append(row)
    if not decisions_by_doc: raise ValueError(f"No structure proposal review TSVs found under {review_dir}")
    summary = {"reviewer": args.reviewer, "review_date": args.review_date, "documents": [], "counts": {"accept": 0, "reject": 0, "defer": 0}}
    from datetime import date as _date
    _date.fromisoformat(args.review_date)
    for did, rows in decisions_by_doc.items():
        if did not in current_docs: raise ValueError(f"Current canonical annotation missing for {did}: {current_ann_root}")
        if did not in proposal_docs: raise ValueError(f"Structure proposal JSON missing for {did}: {proposals_root}")
        source_path = current_docs[did]; doc = read_json(source_path); source_text = doc.get("text")
        proposal = read_json(proposal_docs[did])
        if not isinstance(source_text, str): raise ValueError(f"No source text in {source_path}")
        if proposal.get("source_text_sha256") and proposal["source_text_sha256"] != sha256(source_text): raise ValueError(f"Source hash mismatch for structure proposal {did}")
        decision_counts = {"accept": 0, "reject": 0, "defer": 0}; accepted = []
        seen_candidate_ids = set()
        for i, row in enumerate(rows, 1):
            cid = (row.get("candidate_id") or "").strip()
            if not cid or cid in seen_candidate_ids: raise ValueError(f"Blank/duplicate candidate_id for {did}, TSV row {i}")
            seen_candidate_ids.add(cid)
            decision = (row.get("review_decision") or "").strip().lower()
            if decision not in {"accept", "reject", "defer"}: raise ValueError(f"Invalid/missing review_decision for {did}/{cid}: {decision!r}; use accept, reject, or defer")
            reviewer = (row.get("reviewer") or args.reviewer).strip()
            review_date = (row.get("review_date") or args.review_date).strip()
            _date.fromisoformat(review_date)
            if decision == "defer":
                decision_counts["defer"] += 1; summary["counts"]["defer"] += 1; continue
            if decision == "reject":
                decision_counts["reject"] += 1; summary["counts"]["reject"] += 1; continue
            label = (row.get("annotation_label") or "").strip()
            if label not in STRUCTURE_LABELS: raise ValueError(f"Accepted candidate {did}/{cid} has invalid annotation_label {label!r}")
            try: start, end = int(row["start_char"]), int(row["end_char"])
            except (TypeError, ValueError): raise ValueError(f"Invalid offsets for accepted candidate {did}/{cid}")
            if not (0 <= start < end <= len(source_text)): raise ValueError(f"Accepted candidate {did}/{cid} has out-of-range offsets")
            piece = source_text[start:end]
            if piece != (row.get("text") or ""): raise ValueError(f"Accepted candidate {did}/{cid} text does not match exact source offsets; update text with boundaries or reject/defer")
            source_hash = proposal.get("source_text_sha256")
            if source_hash and source_hash != sha256(source_text): raise ValueError(f"Source text hash mismatch for {did}/{cid}")
            accepted.append({"annotation_id": f"structure_{cid.lower()}", "layer": "structure", "label": label, "start_char": start, "end_char": end, "text": piece, "attributes": {}, "review_status": "reviewed", "annotator_id": reviewer, "note": (row.get("review_notes") or "").strip() or f"Accepted structure candidate {cid}; proposed role={row.get('span_role', '')}; confidence={row.get('proposal_confidence', '')}."})
            decision_counts["accept"] += 1; summary["counts"]["accept"] += 1
        # Validate the union of existing and newly accepted structure spans before writing any file.
        existing = [a for a in doc.get("annotations", []) if a.get("layer") == "structure"]
        all_structure = existing + accepted
        ordered = sorted(all_structure, key=lambda a: (int(a.get("start_char", -1)), int(a.get("end_char", -1)), a.get("annotation_id", "")))
        for left, right in zip(ordered, ordered[1:]):
            if int(right["start_char"]) < int(left["end_char"]):
                raise ValueError(f"Same-layer structure overlap for {did}: {left.get('annotation_id')} and {right.get('annotation_id')}; resolve in the master review before importing")
        ann_ids = {a.get("annotation_id") for a in doc.get("annotations", [])}
        for annotation in accepted:
            if annotation["annotation_id"] in ann_ids: raise ValueError(f"Annotation ID already exists for {did}: {annotation['annotation_id']}")
            ann_ids.add(annotation["annotation_id"])
        doc["annotations"] = list(doc.get("annotations", [])) + accepted
        rel = source_path.relative_to(current_ann_root)
        out_path = out_root / rel; out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        audit = {"document_id": did, "source_annotation": str(source_path), "output_annotation": str(out_path), "proposal_file": str(proposal_docs[did]), "source_text_sha256": sha256(source_text), "decision_counts": decision_counts, "accepted_annotations": len(accepted), "deferred_candidates_block_layer_finalization": decision_counts["defer"] > 0, "document_status_after_import": doc.get("annotation_metadata", {}).get("status", "unknown"), "student_code_executed": False}
        audit_root.mkdir(parents=True, exist_ok=True); (audit_root / f"{source_path.stem}.structure-review-audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary["documents"].append(audit)
    # Carry through current canonical files that had no structure candidates (for example, empty proposal TSVs).
    handled = set(decisions_by_doc)
    for did, source_path in current_docs.items():
        if did in handled: continue
        rel = source_path.relative_to(current_ann_root)
        out_path = out_root / rel
        if out_path.exists(): continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, out_path)
        summary["documents"].append({"document_id": did, "source_annotation": str(source_path), "output_annotation": str(out_path), "decision_counts": {"accept": 0, "reject": 0, "defer": 0}, "accepted_annotations": 0, "note": "No structure review rows existed; canonical file copied unchanged", "document_status_after_import": read_json(source_path).get("annotation_metadata", {}).get("status", "unknown"), "student_code_executed": False})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Structure annotations imported to separate output directory: {out_root}")
    print("Document-level annotation status and coverage are not finalized by this importer.")
    return 0

def cmd_prepare(args) -> int:
    root = Path(args.root).resolve()
    drafts_root = root / "data/annotations/drafts"
    if not drafts_root.is_dir(): raise ValueError(f"Canonical draft directory not found: {drafts_root}")
    drafts = all_files(drafts_root, "*.annotation.json")
    if not drafts: raise ValueError(f"No annotation drafts under {drafts_root}")
    scope_root = Path(args.scope_dir).resolve() if args.scope_dir else root / "data/annotations/scope_reviews"
    scope_index = index_by_doc(all_files(scope_root, "*.scope.json"))
    coverage_root = root / "data/annotations/training_coverage_reviewed"
    coverage_index = index_by_doc(all_files(coverage_root, "*.training-coverage.json"))
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = root / "data/annotations/batch_workflow_v1" / run_id
    if run_dir.exists(): raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    report = {"run_id": run_id, "project_root": str(root), "discovered_documents": len(drafts), "scope_ready": [], "scope_pending": [], "actions": [], "guardrails": ["No student code is executed", "No proposal is automatically approved", "Original files are not overwritten"]}

    # Make one scope queue for documents whose scope ledger is absent or incomplete.
    queue_inputs = run_dir / "scope_inputs"
    existing_review_dir = run_dir / "scope_review_files"
    queue_inputs.mkdir(parents=True); existing_review_dir.mkdir(parents=True)
    complete_doc_ids = set()
    missing_or_incomplete = []
    for p in drafts:
        doc = read_json(p); did = doc.get("document_id", "")
        cov = coverage_index.get(did)
        cov_ok, cov_note = coverage_finalized(p, cov) if cov else (False, "no finalized coverage")
        if cov_ok:
            report.setdefault("already_coverage_finalized", []).append({"document_id": did, "note": cov_note})
            # Preserve the reviewed corpus. Do not regenerate candidates for this document.
            continue
        sp = scope_index.get(did)
        ok, why = scope_complete(p, sp) if sp else (False, "no scope ledger")
        if ok:
            complete_doc_ids.add(did); report["scope_ready"].append(did)
        else:
            report["scope_pending"].append({"document_id": did, "reason": why})
            # If a matching review TSV already exists, preserve it verbatim rather than regenerate over decisions.
            stem = p.name.removesuffix(".annotation.json")
            current_tsv = None
            if scope_root.exists():
                matches = [q for q in scope_root.rglob(f"{stem}.scope-review.tsv")]
                current_tsv = matches[0] if len(matches) == 1 else None
            if current_tsv:
                rel_out = p.relative_to(drafts_root).parent / current_tsv.name
                target = existing_review_dir / rel_out; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(current_tsv, target)
            else:
                _copy_preserve(p, drafts_root, queue_inputs)
                missing_or_incomplete.append(p)

    if missing_or_incomplete:
        preparer = root / "scripts/prepare_question_scope_review.py"
        if not preparer.is_file():
            raise FileNotFoundError(f"Missing {preparer}. Install the existing scope-review workflow before running this batch command.")
        generated = run_dir / "generated_scope_reviews"
        cmd = [sys.executable, str(preparer), "--input-dir", str(queue_inputs), "--output-dir", str(generated), "--pattern", "*.annotation.json", "--recursive"]
        subprocess.run(cmd, cwd=root, check=True)
        # move/copy generated TSVs into the common set without touching original folders
        for tsv in all_files(generated, "*.scope-review.tsv"):
            rel = tsv.relative_to(generated); target = existing_review_dir / rel; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(tsv, target)
        report["actions"].append(f"scope review templates generated for {len(missing_or_incomplete)} documents")

    scope_master = run_dir / "MASTER_SCOPE_REVIEW.tsv"
    scope_rows = 0
    if all_files(existing_review_dir, "*.scope-review.tsv"):
        scope_rows = merge_tsvs(existing_review_dir, "*.scope-review.tsv", scope_master)
        report["actions"].append(f"scope master review has {scope_rows} rows")
    else:
        report["actions"].append("no scope review needed; all documents already have complete scope ledgers")

    # For currently scope-complete documents, make all structure and semantic proposals in one batch.
    eligible = [p for p in drafts if read_json(p).get("document_id") in complete_doc_ids]
    if eligible:
        batch_inputs = run_dir / "proposal_inputs"; ann_stage = batch_inputs / "drafts"; scope_stage = batch_inputs / "scopes"
        ann_stage.mkdir(parents=True); scope_stage.mkdir(parents=True)
        for p in eligible:
            doc = read_json(p); did = doc["document_id"]; scope_path = scope_index[did]
            dest_ann = ann_stage / p.name
            dest_scope = scope_stage / f"{p.name.removesuffix('.annotation.json')}.scope.json"
            if dest_ann.exists() or dest_scope.exists(): raise ValueError(f"Duplicate flattened paper stem in proposal stage: {p.stem}")
            shutil.copy2(p, dest_ann); shutil.copy2(scope_path, dest_scope)
        structure_script = root / "scripts/propose_structure_aware_segments.py"
        semantic_script = root / "scripts/propose_semantic_annotations.py"
        review_script = root / "scripts/prepare_semantic_annotation_review.py"
        for script in (structure_script, semantic_script, review_script):
            if not script.is_file(): raise FileNotFoundError(f"Missing required pipeline script: {script}")
        structure_out = run_dir / "structure_proposals"
        semantic_out = run_dir / "semantic_proposals"
        review_out = run_dir / "semantic_review_files"
        subprocess.run([sys.executable, str(structure_script), "--annotation-dir", str(ann_stage), "--scope-dir", str(scope_stage), "--output-dir", str(structure_out), "--pattern", "*.annotation.json"], cwd=root, check=True)
        subprocess.run([sys.executable, str(semantic_script), "--annotation-dir", str(ann_stage), "--scope-dir", str(scope_stage), "--structure-dir", str(structure_out), "--output-dir", str(semantic_out), "--pattern", "*.annotation.json"], cwd=root, check=True)
        subprocess.run([sys.executable, str(review_script), "--annotation-dir", str(ann_stage), "--proposal-dir", str(semantic_out), "--scope-dir", str(scope_stage), "--output-dir", str(review_out), "--pattern", "*.semantic-proposal.json"], cwd=root, check=True)
        master = run_dir / "MASTER_SEMANTIC_REVIEW.tsv"
        n = merge_tsvs(review_out, "*.semantic-review.tsv", master) if all_files(review_out, "*.semantic-review.tsv") else 0
        report["actions"].append(f"semantic review master has {n} rows for {len(eligible)} scope-ready documents")
        # Add explicit decisions, suggested labels, and original-boundary audit columns before consolidation.
        enrich_structure_review_tsvs(structure_out)
        struct_master = run_dir / "MASTER_STRUCTURE_PROPOSALS.tsv"
        struct_rows = merge_tsvs(structure_out, "*.structure-proposal.tsv", struct_master) if all_files(structure_out, "*.structure-proposal.tsv") else 0
        report["actions"].append(f"structure proposal master has {struct_rows} rows")
    else:
        report["actions"].append("no scope-complete documents yet; annotation proposal stage correctly deferred")

    report["outputs"] = {"scope_master": str(scope_master) if scope_master.exists() else None, "semantic_master": str(run_dir / "MASTER_SEMANTIC_REVIEW.tsv") if (run_dir / "MASTER_SEMANTIC_REVIEW.tsv").exists() else None, "structure_master": str(run_dir / "MASTER_STRUCTURE_PROPOSALS.tsv") if (run_dir / "MASTER_STRUCTURE_PROPOSALS.tsv").exists() else None}
    (run_dir / "BATCH_REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "BATCH_REPORT.md").write_text(make_report(report), encoding="utf-8")
    print(f"Batch preparation complete: {run_dir}")
    print(f"Documents discovered: {len(drafts)}; scope-ready for new proposals: {len(eligible)}; scope-review required: {len(report['scope_pending'])}; already coverage-finalized: {len(report.get('already_coverage_finalized', []))}")
    if scope_master.exists(): print(f"Review all scope decisions in one file: {scope_master}")
    if (run_dir / "MASTER_SEMANTIC_REVIEW.tsv").exists(): print(f"Review all semantic proposals in one file: {run_dir / 'MASTER_SEMANTIC_REVIEW.tsv'}")
    if (run_dir / "MASTER_STRUCTURE_PROPOSALS.tsv").exists(): print(f"Inspect all structure candidates in one file: {run_dir / 'MASTER_STRUCTURE_PROPOSALS.tsv'}")
    print("No original project file was overwritten and no candidate was auto-approved.")
    return 0


def make_report(r: dict) -> str:
    lines = ["# Batch annotation preparation report", "", f"**Run:** `{r['run_id']}`", f"**Project:** `{r['project_root']}`", f"**Documents discovered:** {r['discovered_documents']}", f"**Scope-ready for new proposals:** {len(r['scope_ready'])}", f"**Scope pending/incomplete:** {len(r['scope_pending'])}", f"**Already coverage-finalized and preserved:** {len(r.get('already_coverage_finalized', []))}"]
    if r.get("already_coverage_finalized"):
        lines.extend(["", "## Preserved documents with finalized coverage", *[f"- `{item['document_id']}` — {item['note']}" for item in r["already_coverage_finalized"]]])
    lines.extend(["", "## Scope pending/incomplete"])
    if r["scope_pending"]:
        lines.extend([f"- `{item['document_id']}` — {item['reason']}" for item in r["scope_pending"]])
    else: lines.append("None.")
    lines.extend(["", "## Actions", *[f"- {x}" for x in r["actions"]], "", "## Safety/validity", *[f"- {x}" for x in r["guardrails"]], "", "Scope review is a required prerequisite for structure/semantic proposals. Coverage review must be generated after annotation decisions are imported. This report does not claim training data is ready.", ""])
    return "\n".join(lines)


def cmd_prepare_coverage(args) -> int:
    root = Path(args.root).resolve()
    ann_dir = Path(args.annotation_dir).resolve()
    scope_dir = Path(args.scope_dir).resolve()
    script = root / "scripts/prepare_distilbert_coverage_templates.py"
    if not script.is_file(): raise FileNotFoundError(f"Missing {script}")
    run_id = datetime.now().strftime("coverage_%Y%m%d_%H%M%S")
    out = Path(args.output_dir).resolve() if args.output_dir else root / "data/annotations/batch_workflow_v1" / run_id / "coverage_templates"
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Refusing to overwrite non-empty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(script), "--annotation-dir", str(ann_dir), "--scope-dir", str(scope_dir), "--output-dir", str(out), "--pattern", "*.annotation.json", "--recursive"], cwd=root, check=True)
    master = out.parent / "MASTER_COVERAGE_REVIEW.tsv"
    n = merge_tsvs(out, "*.training-coverage.tsv", master)
    print(f"Coverage templates for all supplied reviewed/canonical annotations are ready: {out}")
    print(f"Single coverage review sheet: {master} ({n} rows)")
    print("The coverage sheet has not been approved or finalized.")
    return 0


def cmd_merge(args) -> int:
    n = merge_tsvs(Path(args.input_dir).resolve(), args.pattern, Path(args.output).resolve())
    print(f"Merged {n} rows into {Path(args.output).resolve()}")
    return 0


def cmd_split(args) -> int:
    n = split_tsv(Path(args.master).resolve(), Path(args.output_dir).resolve(), args.overwrite)
    print(f"Split {n} rows into {Path(args.output_dir).resolve()}")
    return 0



def _documents_by_id(root: Path) -> dict[str, Path]:
    return index_by_doc(all_files(root, "*.annotation.json"))


def _run_script(root: Path, script_rel: str, cmd_args: list[str]) -> None:
    script = root / script_rel
    if not script.is_file(): raise FileNotFoundError(f"Missing required script: {script}")
    subprocess.run([sys.executable, str(script), *cmd_args], cwd=root, check=True)


def cmd_apply_scope(args) -> int:
    root = Path(args.root).resolve()
    master = Path(args.master).resolve()
    run_stamp = datetime.now().strftime("scope_import_%Y%m%d_%H%M%S")
    base = root / "data/annotations/batch_workflow_v1" / run_stamp
    review_dir, input_dir, imported_dir = base / "scope_review_files", base / "scope_input_documents", base / "scope_imported"
    split_tsv(master, review_dir)
    # Get the exact document IDs represented in this master and stage only those documents.
    with master.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t"); rows = list(reader)
    doc_ids = {r.get("document_id", "").strip() for r in rows if r.get("document_id", "").strip()}
    docs = _documents_by_id(root / "data/annotations/drafts")
    missing = sorted(doc_ids - set(docs))
    if missing: raise ValueError(f"Master references document IDs not found in drafts: {missing}")
    drafts_root = root / "data/annotations/drafts"
    for did in sorted(doc_ids): _copy_preserve(docs[did], drafts_root, input_dir)
    _run_script(root, "scripts/import_question_scope_review.py", ["--input-dir", str(input_dir), "--review-dir", str(review_dir), "--output-dir", str(imported_dir), "--pattern", "*.annotation.json", "--recursive"])
    # Build a combined scope directory, preferring newly imported ledgers for docs in this review.
    active = base / "active_scopes"
    active.mkdir(parents=True, exist_ok=True)
    import_index = index_by_doc(all_files(imported_dir, "*.scope.json"))
    existing_scope_root = Path(args.existing_scope_dir).resolve() if args.existing_scope_dir else root / "data/annotations/scope_reviews"
    existing_index = index_by_doc(all_files(existing_scope_root, "*.scope.json"))
    for did, original in docs.items():
        chosen = import_index.get(did) or existing_index.get(did)
        if not chosen: continue
        stem = original.name.removesuffix(".annotation.json")
        dest = active / f"{stem}.scope.json"
        if dest.exists(): raise ValueError(f"Scope filename collision while building active scopes: {dest}")
        shutil.copy2(chosen, dest)
    print(f"Imported reviewed scope decisions for {len(import_index)} documents: {imported_dir}")
    print(f"Combined active scope ledger directory: {active}")
    print("Next, run the batch prepare command with --scope-dir pointing to this active_scopes directory to generate proposals for every newly scope-ready paper.")
    return 0


def cmd_apply_semantic(args) -> int:
    root = Path(args.root).resolve(); master = Path(args.master).resolve(); run_dir = Path(args.run_dir).resolve()
    review_dir = run_dir / "semantic_review_edited"
    split_tsv(master, review_dir)
    proposal_inputs = run_dir / "proposal_inputs"
    ann_stage, scope_stage = proposal_inputs / "drafts", proposal_inputs / "scopes"
    proposal_dir = run_dir / "semantic_proposals"
    out_dir = Path(args.output_dir).resolve() if args.output_dir else root / "data/annotations/semantic_review_imported_batch" / run_dir.name
    _run_script(root, "scripts/import_semantic_annotation_review.py", ["--annotation-dir", str(ann_stage), "--proposal-dir", str(proposal_dir), "--scope-dir", str(scope_stage), "--review-dir", str(review_dir), "--output-dir", str(out_dir), "--pattern", "*.semantic-review.tsv", "--recursive", "--reviewer", args.reviewer, "--review-date", args.review_date, "--confirm-reviewed-decisions"])
    print(f"Batch semantic import complete: {out_dir}")
    print("The importer deliberately leaves document-level status as draft. Finalize canonical annotations only after your annotation review/adjudication policy is satisfied.")
    return 0


def cmd_apply_coverage(args) -> int:
    root = Path(args.root).resolve(); master = Path(args.master).resolve(); templates = Path(args.template_dir).resolve()
    stamp = datetime.now().strftime("coverage_import_%Y%m%d_%H%M%S")
    base = root / "data/annotations/batch_workflow_v1" / stamp
    review_dir = base / "coverage_review_files"
    split_tsv(master, review_dir)
    # Copy same-stem JSON templates beside each split review TSV, preserving relative paths.
    template_index = {p.name: p for p in all_files(templates, "*.training-coverage.json")}
    copied = 0
    for tsv in all_files(review_dir, "*.training-coverage.tsv"):
        json_name = tsv.name.removesuffix(".tsv") + ".json"
        template = template_index.get(json_name)
        if not template: raise FileNotFoundError(f"Missing paired coverage JSON template {json_name} under {templates}")
        dest = tsv.parent / json_name
        shutil.copy2(template, dest); copied += 1
    out_dir = Path(args.output_dir).resolve() if args.output_dir else base / "training_coverage_reviewed"
    _run_script(root, "scripts/import_distilbert_coverage_review.py", ["--coverage-dir", str(review_dir), "--output-dir", str(out_dir), "--pattern", "*.training-coverage.tsv", "--recursive", "--finalize", "--reviewer", args.reviewer, "--review-date", args.review_date])
    print(f"Imported and finalized {copied} coverage ledgers: {out_dir}")
    print("Finalization is a reviewer attestation; do not run this command until the entire coverage master has been reviewed for each layer.")
    return 0

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare", help="Batch-create scope queues and proposals for all scope-ready papers")
    q.add_argument("--root", default="."); q.add_argument("--scope-dir", help="Optional active scope-ledger directory; defaults to data/annotations/scope_reviews"); q.set_defaults(func=cmd_prepare)
    q = sub.add_parser("inventory", help="Write a paper-by-paper inventory/status report")
    q.add_argument("--root", default="."); q.add_argument("--output"); q.set_defaults(func=cmd_inventory)
    q = sub.add_parser("prepare-coverage", help="Create coverage templates and one master TSV across all supplied canonical annotations")
    q.add_argument("--root", default=".", required=True); q.add_argument("--annotation-dir", required=True); q.add_argument("--scope-dir", required=True); q.add_argument("--output-dir"); q.set_defaults(func=cmd_prepare_coverage)
    q = sub.add_parser("apply-scope", help="Import a reviewed master scope TSV for all represented papers and build a combined active scope directory")
    q.add_argument("--root", default="."); q.add_argument("--master", required=True); q.add_argument("--existing-scope-dir"); q.set_defaults(func=cmd_apply_scope)
    q = sub.add_parser("apply-structure", help="Import reviewed structure candidates from the consolidated structure TSV")
    q.add_argument("--root", default="."); q.add_argument("--run-dir", required=True); q.add_argument("--master", required=True); q.add_argument("--annotation-dir", required=True, help="Latest canonical annotation output to extend, usually the semantic-import output"); q.add_argument("--reviewer", required=True); q.add_argument("--review-date", required=True); q.add_argument("--output-dir"); q.set_defaults(func=cmd_apply_structure)
    q = sub.add_parser("apply-semantic", help="Import reviewed decisions from a master semantic TSV in one batch")
    q.add_argument("--root", default="."); q.add_argument("--run-dir", required=True); q.add_argument("--master", required=True); q.add_argument("--reviewer", required=True); q.add_argument("--review-date", required=True); q.add_argument("--output-dir"); q.set_defaults(func=cmd_apply_semantic)
    q = sub.add_parser("apply-coverage", help="Import/finalize a reviewed master coverage TSV in one batch")
    q.add_argument("--root", default="."); q.add_argument("--master", required=True); q.add_argument("--template-dir", required=True); q.add_argument("--reviewer", required=True); q.add_argument("--review-date", required=True); q.add_argument("--output-dir"); q.set_defaults(func=cmd_apply_coverage)
    q = sub.add_parser("merge-tsv", help="Merge per-document TSVs into one editable master TSV")
    q.add_argument("--input-dir", required=True); q.add_argument("--pattern", required=True); q.add_argument("--output", required=True); q.set_defaults(func=cmd_merge)
    q = sub.add_parser("split-tsv", help="Split an edited master TSV back into per-document TSVs")
    q.add_argument("--master", required=True); q.add_argument("--output-dir", required=True); q.add_argument("--overwrite", action="store_true"); q.set_defaults(func=cmd_split)
    args = p.parse_args()
    try: return args.func(args)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: batch sub-command failed (exit code {e.returncode}). See output above; generated files remain in the run folder.", file=sys.stderr); return e.returncode or 1
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr); return 1

if __name__ == "__main__": raise SystemExit(main())
