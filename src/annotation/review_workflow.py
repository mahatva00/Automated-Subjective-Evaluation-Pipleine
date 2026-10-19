"""Human-in-the-loop import/export for semantic annotation proposals.

This module never treats proposals as labels. Only explicit TSV review actions
can create reviewed annotation spans, and document metadata remains draft until
separate whole-layer coverage review is completed.
"""
from __future__ import annotations

import csv
import copy
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

SEMANTIC_LABELS = {
    "requirement", "context", "example", "option", "alternative_marker",
    "grading_instruction", "submission_instruction", "administrative_instruction",
    "definition", "explanatory_note", "non_assessable", "other",
}
REQUIREMENT_TYPES = {
    "action", "constraint", "deliverable", "success_criterion", "input_specification",
    "output_specification", "method_or_tool", "evaluation_or_comparison",
    "explanation_or_justification", "other",
}
ATOMICITIES = {"atomic", "compound", "uncertain"}
ASSESSMENT_RELEVANCE = {"yes", "no", "uncertain"}
ACTIONS = {"pending", "accept", "revise", "split", "reject", "defer"}
TSV_FIELDS = [
    "document_id", "annotation_filename", "proposal_id", "source_text_sha256",
    "start_char", "end_char", "candidate_text", "candidate_text_sha256",
    "question_id", "scope_segment_id", "proposed_label", "proposed_attributes_json",
    "proposal_confidence", "proposal_rationale", "review_flags",
    "review_action", "final_label", "final_start_char", "final_end_char",
    "final_requirement_type", "final_atomicity", "final_assessment_relevance",
    "split_spans_json", "reviewer_note",
]

class ReviewError(ValueError):
    pass

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def one_line(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\t", "\\t").replace("\r", "\\r").replace("\n", "\\n")

def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReviewError(f"Could not read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReviewError(f"Expected a JSON object: {path}")
    return value

def _validate_document_source(document: dict[str, Any], proposal: dict[str, Any]) -> None:
    text = document.get("text")
    if not isinstance(text, str) or not text:
        raise ReviewError("Canonical annotation document has no source text")
    source_hash = sha256_text(text)
    if document.get("document_id") != proposal.get("document_id"):
        raise ReviewError("Document ID mismatch between canonical annotation and semantic proposal")
    if proposal.get("source_text_sha256") != source_hash:
        raise ReviewError("Source text hash mismatch between canonical annotation and semantic proposal")
    if proposal.get("text_length") != len(text):
        raise ReviewError("Source text length mismatch between canonical annotation and semantic proposal")

def _load_scope(scope_path: Path, document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scope = _read_json(scope_path)
    if scope.get("document_id") != document.get("document_id"):
        raise ReviewError("Scope ledger document_id mismatch")
    if scope.get("source_text_sha256") != sha256_text(document["text"]):
        raise ReviewError("Scope ledger source text hash mismatch")
    segments = scope.get("segments")
    if not isinstance(segments, list):
        raise ReviewError("Scope ledger must contain a segments list")
    index: dict[str, dict[str, Any]] = {}
    for seg in segments:
        sid = seg.get("segment_id")
        if not sid or sid in index:
            raise ReviewError(f"Missing or duplicate scope segment_id: {sid}")
        start, end = seg.get("start_char"), seg.get("end_char")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start or end > len(document["text"]):
            raise ReviewError(f"Invalid scope offsets for {sid}")
        if document["text"][start:end] != seg.get("text") or sha256_text(document["text"][start:end]) != seg.get("text_sha256"):
            raise ReviewError(f"Scope source slice/hash mismatch for {sid}")
        index[sid] = seg
    return index

def _candidate_index(proposal: dict[str, Any], document: dict[str, Any], scope_index: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    _validate_document_source(document, proposal)
    candidates = proposal.get("candidates")
    if not isinstance(candidates, list):
        raise ReviewError("Semantic proposal must contain candidates list")
    text = document["text"]
    result: dict[str, dict[str, Any]] = {}
    for c in candidates:
        pid = c.get("proposal_id")
        if not pid or pid in result:
            raise ReviewError(f"Missing or duplicate proposal_id: {pid}")
        start, end = c.get("start_char"), c.get("end_char")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or start >= end or end > len(text):
            raise ReviewError(f"Invalid source span for {pid}")
        exact = text[start:end]
        if exact != c.get("text") or sha256_text(exact) != c.get("text_sha256"):
            raise ReviewError(f"Source slice/hash mismatch for {pid}")
        sid = c.get("scope_segment_id")
        seg = scope_index.get(sid)
        if not seg:
            raise ReviewError(f"Unknown scope segment {sid} for {pid}")
        if seg.get("scope_status") != "in_scope" or seg.get("review_status") not in {"reviewed", "adjudicated"}:
            raise ReviewError(f"Candidate {pid} is not in reviewed in-scope material")
        if start < seg["start_char"] or end > seg["end_char"]:
            raise ReviewError(f"Candidate {pid} crosses scope-segment boundary")
        result[pid] = c
    return result

def make_review_rows(document: dict[str, Any], proposal: dict[str, Any], scope_index: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    candidates = _candidate_index(proposal, document, scope_index)
    rows: list[dict[str, str]] = []
    for pid, c in candidates.items():
        attrs = c.get("requirement_attributes_proposal") or {}
        row = {
            "document_id": document["document_id"],
            "annotation_filename": proposal.get("annotation_filename") or "",
            "proposal_id": pid,
            "source_text_sha256": proposal["source_text_sha256"],
            "start_char": str(c["start_char"]),
            "end_char": str(c["end_char"]),
            "candidate_text": one_line(c["text"]),
            "candidate_text_sha256": c["text_sha256"],
            "question_id": str(c.get("question_id") or ""),
            "scope_segment_id": str(c.get("scope_segment_id") or ""),
            "proposed_label": str(c.get("label_proposal") or ""),
            "proposed_attributes_json": json.dumps(attrs, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "proposal_confidence": str(c.get("proposal_confidence") or ""),
            "proposal_rationale": one_line(str(c.get("proposal_rationale") or "")),
            "review_flags": ";".join(str(x) for x in (c.get("review_flags") or [])),
            "review_action": "pending",
            "final_label": "", "final_start_char": "", "final_end_char": "",
            "final_requirement_type": "", "final_atomicity": "", "final_assessment_relevance": "",
            "split_spans_json": "", "reviewer_note": "",
        }
        rows.append(row)
    return rows

def write_review_tsv(path: Path, rows: list[dict[str, str]], overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise ReviewError(f"Refusing to overwrite existing review sheet: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def _read_review(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        if r.fieldnames != TSV_FIELDS:
            raise ReviewError(f"Review TSV columns do not match schema in {path}")
        rows = list(r)
    for i, row in enumerate(rows, start=2):
        action = (row.get("review_action") or "pending").strip().lower()
        if action not in ACTIONS:
            raise ReviewError(f"Invalid review_action {action!r} on row {i} of {path}")
        row["review_action"] = action
    return rows

def _validate_review_sheet(rows: list[dict[str, str]], document: dict[str, Any], proposal: dict[str, Any], scope_index: dict[str, dict[str, Any]]) -> dict[str, tuple[dict[str, str], dict[str, Any]]]:
    expected = _candidate_index(proposal, document, scope_index)
    ids: set[str] = set()
    result: dict[str, tuple[dict[str, str], dict[str, Any]]] = {}
    for row_num, row in enumerate(rows, start=2):
        pid = row.get("proposal_id", "")
        if not pid or pid in ids:
            raise ReviewError(f"Missing or duplicate proposal_id on TSV row {row_num}")
        ids.add(pid)
        c = expected.get(pid)
        if c is None:
            raise ReviewError(f"Unknown proposal_id {pid} on TSV row {row_num}")
        immutable = {
            "document_id": document["document_id"],
            "source_text_sha256": proposal["source_text_sha256"],
            "start_char": str(c["start_char"]),
            "end_char": str(c["end_char"]),
            "candidate_text": one_line(c["text"]),
            "candidate_text_sha256": c["text_sha256"],
            "scope_segment_id": str(c.get("scope_segment_id") or ""),
            "question_id": str(c.get("question_id") or ""),
            "proposed_label": str(c.get("label_proposal") or ""),
            "proposed_attributes_json": json.dumps(c.get("requirement_attributes_proposal") or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "proposal_confidence": str(c.get("proposal_confidence") or ""),
            "proposal_rationale": one_line(str(c.get("proposal_rationale") or "")),
            "review_flags": ";".join(str(x) for x in (c.get("review_flags") or [])),
        }
        for key, value in immutable.items():
            if row.get(key, "") != value:
                raise ReviewError(f"Immutable field {key} changed for {pid}; regenerate from source if input changed")
        if row.get("annotation_filename") != (proposal.get("annotation_filename") or ""):
            raise ReviewError(f"annotation_filename mismatch for {pid}")
        result[pid] = (row, c)
    if ids != set(expected):
        missing = sorted(set(expected) - ids)
        raise ReviewError(f"Review sheet omits {len(missing)} proposals; do not delete rows—use reject/defer. First missing: {missing[:5]}")
    return result

def _parse_int(value: str, field: str, pid: str) -> int | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ReviewError(f"{field} must be an integer for {pid}") from exc

def _attrs_for(label: str, row: dict[str, str], c: dict[str, Any], pid: str) -> dict[str, Any]:
    raw = row.get("proposed_attributes_json", "{}") or "{}"
    try:
        suggested = json.loads(raw)
    except Exception as exc:
        raise ReviewError(f"Invalid proposed_attributes_json for {pid}") from exc
    if not isinstance(suggested, dict):
        raise ReviewError(f"proposed_attributes_json must be an object for {pid}")
    attrs = {}
    if label == "requirement":
        fields = {
            "requirement_type": (row.get("final_requirement_type") or suggested.get("requirement_type") or "").strip(),
            "atomicity": (row.get("final_atomicity") or suggested.get("atomicity") or "").strip(),
            "assessment_relevance": (row.get("final_assessment_relevance") or suggested.get("assessment_relevance") or "").strip(),
        }
        for key, allowed in [("requirement_type", REQUIREMENT_TYPES), ("atomicity", ATOMICITIES), ("assessment_relevance", ASSESSMENT_RELEVANCE)]:
            if fields[key] not in allowed:
                raise ReviewError(f"Requirement {pid} needs valid {key}; allowed: {sorted(allowed)}")
        attrs.update(fields)
    qid = c.get("question_id")
    if qid:
        attrs["question_id"] = qid
    attrs["source_proposal_id"] = pid
    extra_note = (row.get("reviewer_note") or "").strip()
    if extra_note:
        attrs["label_note"] = extra_note
    return attrs

def _mk_annotation(document: dict[str, Any], c: dict[str, Any], pid: str, layer: str, label: str, start: int, end: int,
                   reviewer: str, reviewer_note: str, row: dict[str, str], suffix: str = "") -> dict[str, Any]:
    text = document["text"]
    if start < 0 or start >= end or end > len(text):
        raise ReviewError(f"Invalid final span {start}:{end} for {pid}")
    if not text[start:end].strip():
        raise ReviewError(f"Final span is whitespace-only for {pid}")
    if label not in SEMANTIC_LABELS:
        raise ReviewError(f"Invalid semantic label {label!r} for {pid}; allowed: {sorted(SEMANTIC_LABELS)}")
    attrs = _attrs_for(label, row, c, pid)
    # Existing semantic proposal layer is the first consumer; this importer emits semantic annotations only.
    return {
        "annotation_id": f"sem_{pid.lower()}{suffix}",
        "layer": layer,
        "label": label,
        "start_char": start,
        "end_char": end,
        "text": text[start:end],
        "attributes": attrs,
        "review_status": "reviewed",
        "annotator_id": reviewer,
        "note": f"Imported from explicitly reviewed semantic proposal {pid}. {reviewer_note.strip()}".strip(),
    }

def _parse_split_spans(row: dict[str, str], document: dict[str, Any], c: dict[str, Any], reviewer: str) -> list[dict[str, Any]]:
    raw = (row.get("split_spans_json") or "").strip()
    if not raw:
        raise ReviewError(f"split_spans_json required for split action on {c['proposal_id']}")
    try:
        specs = json.loads(raw)
    except Exception as exc:
        raise ReviewError(f"Invalid split_spans_json for {c['proposal_id']}") from exc
    if not isinstance(specs, list) or len(specs) < 2:
        raise ReviewError(f"Split must contain at least 2 spans for {c['proposal_id']}")
    parent_start, parent_end = c["start_char"], c["end_char"]
    specs = sorted(specs, key=lambda x: (x.get("start_char", -1), x.get("end_char", -1)))
    out = []
    previous_end = parent_start
    for ix, spec in enumerate(specs, start=1):
        if not isinstance(spec, dict):
            raise ReviewError(f"Each split span must be an object for {c['proposal_id']}")
        try:
            start, end = int(spec["start_char"]), int(spec["end_char"])
        except Exception as exc:
            raise ReviewError(f"Split span needs integer start_char/end_char for {c['proposal_id']}") from exc
        if start < parent_start or end > parent_end or start >= end:
            raise ReviewError(f"Split span must stay inside original proposal {c['proposal_id']}")
        if start < previous_end:
            raise ReviewError(f"Split spans overlap for {c['proposal_id']}")
        label = str(spec.get("label", "")).strip()
        if label not in SEMANTIC_LABELS:
            raise ReviewError(f"Invalid split label {label!r} for {c['proposal_id']}")
        split_row = dict(row)
        for k in ("final_requirement_type", "final_atomicity", "final_assessment_relevance"):
            split_row[k] = str(spec.get(k) or "")
        # Explicit values in split JSON take precedence over proposed attributes.
        if label == "requirement":
            for k in ("requirement_type", "atomicity", "assessment_relevance"):
                if spec.get(k):
                    split_row[{
                        "requirement_type": "final_requirement_type",
                        "atomicity": "final_atomicity",
                        "assessment_relevance": "final_assessment_relevance",
                    }[k]] = str(spec[k])
        note = str(spec.get("note") or row.get("reviewer_note") or "")
        out.append(_mk_annotation(document, c, c["proposal_id"], "semantic", label, start, end, reviewer, note, split_row, suffix=f"_s{ix}"))
        previous_end = end
    return out

def _check_nonoverlap(annotations: list[dict[str, Any]]) -> None:
    spans = sorted((a["start_char"], a["end_char"], a["annotation_id"]) for a in annotations if a.get("layer") == "semantic")
    prev_end = -1
    prev_id = None
    for start, end, aid in spans:
        if start < prev_end:
            raise ReviewError(f"Semantic annotation overlap: {prev_id} and {aid}")
        prev_end, prev_id = end, aid

def import_reviewed_sheet(document: dict[str, Any], proposal: dict[str, Any], scope_index: dict[str, dict[str, Any]],
                          rows: list[dict[str, str]], reviewer: str, review_date: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not reviewer.strip():
        raise ReviewError("Reviewer ID must not be blank")
    try:
        date.fromisoformat(review_date)
    except Exception as exc:
        raise ReviewError("review_date must be ISO YYYY-MM-DD") from exc
    matched = _validate_review_sheet(rows, document, proposal, scope_index)
    out_doc = copy.deepcopy(document)
    generated: list[dict[str, Any]] = []
    decisions: Counter[str] = Counter()
    for pid, (row, c) in matched.items():
        action = row["review_action"]
        decisions[action] += 1
        if action in {"pending", "defer", "reject"}:
            continue
        if action == "accept":
            label = str(c.get("label_proposal") or "").strip()
            if not label:
                raise ReviewError(f"Cannot accept unclassified proposal {pid}; use revise and select a label")
            start, end = c["start_char"], c["end_char"]
            if (row.get("final_label") or "").strip() and row["final_label"].strip() != label:
                raise ReviewError(f"Use review_action=revise if changing the label for {pid}")
            generated.append(_mk_annotation(out_doc, c, pid, "semantic", label, start, end, reviewer, row.get("reviewer_note", ""), row))
        elif action == "revise":
            label = (row.get("final_label") or "").strip()
            if not label:
                raise ReviewError(f"final_label required for revise action on {pid}")
            start = _parse_int(row.get("final_start_char", ""), "final_start_char", pid)
            end = _parse_int(row.get("final_end_char", ""), "final_end_char", pid)
            start = c["start_char"] if start is None else start
            end = c["end_char"] if end is None else end
            # Revised span must remain inside the same reviewed in-scope segment.
            seg = scope_index[c["scope_segment_id"]]
            if start < seg["start_char"] or end > seg["end_char"]:
                raise ReviewError(f"Revised span for {pid} crosses scope boundary")
            generated.append(_mk_annotation(out_doc, c, pid, "semantic", label, start, end, reviewer, row.get("reviewer_note", ""), row))
        elif action == "split":
            generated.extend(_parse_split_spans(row, out_doc, c, reviewer))
    existing = out_doc.get("annotations", [])
    all_annotations = existing + generated
    # Require globally unique ids and same-layer non-overlap for the full canonical document.
    ids = [a.get("annotation_id") for a in all_annotations]
    if len(ids) != len(set(ids)):
        raise ReviewError("Generated annotation_id collides with an existing canonical annotation")
    _check_nonoverlap(all_annotations)
    out_doc["annotations"] = all_annotations
    meta = out_doc.setdefault("annotation_metadata", {})
    meta["status"] = "draft"  # This operation reviews individual spans, not complete document coverage.
    meta["annotator_ids"] = sorted(set(meta.get("annotator_ids") or []) | {reviewer})
    prior_notes = meta.get("notes", "")
    note = f"Semantic proposal review import on {review_date} by {reviewer}; document remains draft until complete layer coverage is reviewed."
    meta["notes"] = (prior_notes + " " + note).strip()
    audit = {
        "audit_version": "1.0.0",
        "document_id": document["document_id"],
        "source_text_sha256": sha256_text(document["text"]),
        "reviewer": reviewer,
        "review_date": review_date,
        "input_proposal_count": len(matched),
        "decision_counts": dict(sorted(decisions.items())),
        "annotations_created": len(generated),
        "annotation_metadata_status_after_import": "draft",
        "document_finalized": False,
        "notes": "Only explicit accept/revise/split decisions create reviewed semantic annotations. The whole document remains draft until full layer coverage is reviewed separately.",
    }
    return out_doc, audit

def load_index(paths: list[Path], key: str) -> dict[str, Path]:
    result = {}
    for p in paths:
        d = _read_json(p)
        value = d.get(key)
        if value in result:
            raise ReviewError(f"Duplicate {key}={value} across files: {result[value]} and {p}")
        result[value] = p
    return result

def collect_json(root: Path, pattern: str) -> list[Path]:
    paths = sorted(root.rglob(pattern))
    if not paths:
        raise ReviewError(f"No files matched {pattern!r} under {root}")
    return paths

def prepare_one(annotation_path: Path, proposal_path: Path, scope_path: Path, output_path: Path, overwrite: bool = False) -> dict[str, Any]:
    doc = _read_json(annotation_path)
    prop = _read_json(proposal_path)
    scope_index = _load_scope(scope_path, doc)
    if prop.get("annotation_filename") and prop["annotation_filename"] != annotation_path.name:
        raise ReviewError(f"Proposal expects {prop['annotation_filename']}, found {annotation_path.name}")
    rows = make_review_rows(doc, prop, scope_index)
    write_review_tsv(output_path, rows, overwrite=overwrite)
    return {"document_id": doc["document_id"], "proposal_count": len(rows), "output": str(output_path)}

def import_one(annotation_path: Path, proposal_path: Path, scope_path: Path, review_path: Path, output_path: Path,
               reviewer: str, review_date: str, overwrite: bool = False) -> dict[str, Any]:
    audit_path = output_path.with_name(output_path.stem + ".review-audit.json")
    if not overwrite and (output_path.exists() or audit_path.exists()):
        raise ReviewError(f"Refusing to overwrite existing output or audit: {output_path} / {audit_path}")
    doc = _read_json(annotation_path)
    prop = _read_json(proposal_path)
    scope_index = _load_scope(scope_path, doc)
    if prop.get("annotation_filename") and prop["annotation_filename"] != annotation_path.name:
        raise ReviewError(f"Proposal expects {prop['annotation_filename']}, found {annotation_path.name}")
    rows = _read_review(review_path)
    reviewed_doc, audit = import_reviewed_sheet(doc, prop, scope_index, rows, reviewer, review_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(reviewed_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"document_id": doc["document_id"], "annotations_created": audit["annotations_created"], "decisions": audit["decision_counts"], "output": str(output_path), "audit": str(audit_path)}
