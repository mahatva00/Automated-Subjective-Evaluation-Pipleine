"""Scope-safe, offset-preserving structural segmentation proposals for question papers.

This module proposes boundaries only. It never executes student code and never emits
approved semantic labels or training labels.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA_VERSION = "structure-segmentation-proposal-v1.0"

# These are format-level patterns, not subject-specific vocabularies.  They are
# intentionally permissive; every result remains a pending proposal for review.
_PATTERNS: List[Tuple[str, re.Pattern[str], int]] = [
    ("question_header", re.compile(r"^\s{0,8}(?:question\s*)?(?P<label>\d{1,2})\s*[.)]\s+(?P<title>[^\n]{2,140})\s*$", re.I), 0),
    ("question_header", re.compile(r"^\s{0,8}Q(?:uestion)?\s*(?P<label>\d{1,2})\s*[:.)-]\s*(?P<title>[^\n]{1,140})\s*$", re.I), 0),
    ("subquestion_header", re.compile(r"^\s{0,24}\((?P<label>[a-z])\)\s+(?P<title>[^\n]{1,180})\s*$", re.I), 1),
    ("subquestion_header", re.compile(r"^\s{1,24}\((?P<label>\d{1,2})\)\s+(?P<title>[^\n]{1,180})\s*$"), 1),
    ("experiment_header", re.compile(r"^\s{0,32}(?P<label>[A-Z])\s*[.)]\s+(?P<title>[^\n]{0,180}\bexperiment\s*(?:\d+|[A-Z0-9]+)[^\n]{0,150})$", re.I), 3),
    ("roman_item_header", re.compile(r"^\s{1,28}\(?((?:i{1,3}|iv|v|vi{0,3}|ix|x))\)?[.)]\s+(?P<title>[^\n]{1,180})\s*$", re.I), 2),
    ("lettered_item_header", re.compile(r"^\s{1,32}(?P<label>[A-Z])\s*[.)]\s+(?P<title>[^\n]{2,180})\s*$"), 3),
    ("list_item_header", re.compile(r"^\s{1,36}(?:[\u2022\u25aa\u25cf\u25e6\u0088\uf0a7\uf0b7\uf0fc*]|[-–—])\s+(?P<title>[^\n]{1,220})\s*$"), 4),
]

@dataclass
class Anchor:
    kind: str
    start: int
    line_end: int
    end: int
    line: str
    indent: int
    level: int
    label: Optional[str] = None
    confidence: str = "medium"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _page_numbers(annotation: Dict[str, Any], start: int, end: int) -> List[int]:
    pages = []
    for p in annotation.get("page_ranges", []):
        ps, pe = p.get("start_char", 0), p.get("end_char", 0)
        # Use character overlap. Page extraction separators may sit between pages.
        if start < pe and end > ps:
            pages.append(p.get("page_number"))
    return [p for p in pages if isinstance(p, int)]


def _line_anchor(line: str, absolute_start: int, line_end: int) -> Optional[Anchor]:
    if not line.strip():
        return None
    indent = len(line) - len(line.lstrip(" \t"))
    for kind, pattern, level in _PATTERNS:
        m = pattern.match(line.rstrip("\r\n"))
        if m:
            label = m.groupdict().get("label")
            # A header needs a title-bearing line and should not be a decimal value,
            # code assignment, URL, or short numeric output. Review is still required.
            title = (m.groupdict().get("title") or "").strip()
            if kind == "question_header":
                if not title or re.match(r"^(?:\d+(?:\.\d+)+|[=<>]=?|[-+*/])\b", title):
                    continue
                # The one/two-digit marker is only treated as a question-level
                # candidate if the line looks like a numbered heading, not code.
                if re.search(r"(?:=|\b(?:print|return|df\[|len\(|range\())", title, re.I):
                    continue
            confidence = "medium"
            if kind in ("question_header", "subquestion_header", "experiment_header"):
                confidence = "medium"
            return Anchor(kind, absolute_start, absolute_start + len(line), line_end,
                          line.rstrip("\r\n"), indent, level, label, confidence)
    return None


def _anchors_for_region(text: str, start: int, end: int) -> List[Anchor]:
    region = text[start:end]
    anchors: List[Anchor] = []
    offset = start
    for raw_line in region.splitlines(keepends=True):
        line_end = offset + len(raw_line)
        a = _line_anchor(raw_line, offset, line_end)
        if a:
            anchors.append(a)
        offset = line_end
    return anchors


def _region_candidate(region: Dict[str, Any], question_id: Optional[str], candidate_id: str,
                      candidate_type: str, parent_id: Optional[str], text: str,
                      start: int, end: int, title: str, confidence: str,
                      rationale: str, level: int, annotation: Dict[str, Any]) -> Dict[str, Any]:
    exact = text[start:end]
    return {
        "candidate_id": candidate_id,
        "candidate_type": candidate_type,
        "parent_candidate_id": parent_id,
        "question_id": question_id,
        "start_char": start,
        "end_char": end,
        "text_sha256": sha256_text(exact),
        "text": exact,
        "title_preview": title[:180],
        "page_numbers": _page_numbers(annotation, start, end),
        "scope_segment_id": region.get("segment_id"),
        "scope_status": "in_scope",
        "proposal_confidence": confidence,
        "proposal_rationale": rationale,
        "review_status": "pending",
        "annotation_label": None,
        "training_label": None,
        "training_eligible": False,
        "level": level,
    }


def propose_structure(annotation: Dict[str, Any], scope: Dict[str, Any]) -> Dict[str, Any]:
    """Produce review-only structural proposals with distinct headers and body spans.

    A header candidate is the complete marker/title line. A paired block candidate,
    when there is substantive body text, covers only the text following that line.
    Hierarchy follows header nodes; body blocks are attached to their own header.
    """
    text = annotation.get("text")
    if not isinstance(text, str):
        raise ValueError("Annotation draft must contain a string field named 'text'.")
    if annotation.get("document_id") != scope.get("document_id"):
        raise ValueError("Document ID mismatch between annotation draft and scope ledger.")
    expected_hash = sha256_text(text)
    if scope.get("source_text_sha256") != expected_hash:
        raise ValueError("Source text SHA-256 mismatch. Stop; do not apply offsets to a different source.")
    if scope.get("text_length") != len(text):
        raise ValueError("Source text length mismatch between annotation and scope ledger.")

    scope_segments = sorted(scope.get("segments", []), key=lambda s: (s.get("start_char", -1), s.get("end_char", -1)))
    seen = set()
    previous_end = 0
    for s in scope_segments:
        sid, start, end = s.get("segment_id"), s.get("start_char"), s.get("end_char")
        if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start <= end <= len(text)):
            raise ValueError(f"Invalid offsets in scope segment {sid!r}.")
        if start < previous_end:
            raise ValueError(f"Overlapping scope segments near {sid!r}.")
        previous_end = end
        if text[start:end] != s.get("text", ""):
            raise ValueError(f"Exact text mismatch in scope segment {sid!r}.")
        if sha256_text(text[start:end]) != s.get("text_sha256"):
            raise ValueError(f"Segment hash mismatch in scope segment {sid!r}.")
        if sid in seen:
            raise ValueError(f"Duplicate scope segment ID: {sid!r}.")
        seen.add(sid)

    # Preserve neutral whitespace separators as transparent boundaries. Any
    # pending/out-of-scope/non-neutral region breaks continuity so later content
    # cannot inherit question membership or a parent across excluded text.
    entries: List[Dict[str, Any]] = []
    active_question_id: Optional[str] = None
    previous_in_scope_end: Optional[int] = None
    context_broken = False
    segment_statuses = {"in_scope", "out_of_scope", "pending", "separator"}
    question_nodes: List[Dict[str, Any]] = []
    seen_questions = set()

    for region in scope_segments:
        status = region.get("scope_status")
        start, end = region["start_char"], region["end_char"]
        segment_text = text[start:end]
        if status in ("out_of_scope", "pending"):
            # A reviewed-out pure numeric line is a common PDF page-footer form.
            # Exclude it from candidates but allow the surrounding logical question
            # context to continue across it. Other excluded text breaks context.
            neutral_page_footer = (
                status == "out_of_scope"
                and re.fullmatch(r"\s*\d{1,3}\s*", segment_text) is not None
            )
            if neutral_page_footer:
                continue
            active_question_id = None
            previous_in_scope_end = None
            context_broken = True
            continue
        if status == "separator":
            if segment_text.strip():
                active_question_id = None
                previous_in_scope_end = None
                context_broken = True
            # Whitespace-only separators may sit between neighboring segments
            # without breaking the logical node or continuation group.
            continue
        if status != "in_scope":
            active_question_id = None
            previous_in_scope_end = None
            context_broken = True
            continue

        reset_context_before = context_broken
        if reset_context_before:
            active_question_id = None
        linked_to_previous = previous_in_scope_end is not None and not reset_context_before
        anchors = _anchors_for_region(text, start, end)
        qid_before = active_question_id
        for a in anchors:
            if a.kind == "question_header":
                active_question_id = f"Q{a.label}"
                if active_question_id not in seen_questions:
                    seen_questions.add(active_question_id)
                    question_nodes.append({
                        "question_id": active_question_id,
                        "header_start_char": a.start,
                        "header_end_char": a.line_end,
                        "header_sha256": sha256_text(text[a.start:a.line_end]),
                        "title_preview": a.line.strip(),
                        "review_status": "pending",
                        "training_eligible": False,
                        "note": "Logical question node; body may span several reviewed in-scope fragments.",
                    })
        entries.append({
            "region": region,
            "anchors": anchors,
            "question_id_before": qid_before,
            "question_id_after": active_question_id,
            "reset_context_before": reset_context_before,
            "linked_to_previous_scope_fragment": linked_to_previous,
        })
        previous_in_scope_end = end
        context_broken = False

    candidates: List[Dict[str, Any]] = []
    candidate_counter = 0
    unclassified_regions = 0
    anchor_counts: Dict[str, int] = {}
    # Stack items are (nesting level, header candidate id). Header nodes form
    # the hierarchy; their optional body blocks are paired children, not nodes
    # that duplicate the header interval.
    node_stack: List[Tuple[int, str]] = []
    header_to_block: Dict[str, str] = {}
    continuation_sequences: Dict[str, int] = {}

    def next_id() -> str:
        nonlocal candidate_counter
        candidate_counter += 1
        return f"SC{candidate_counter:04d}"

    def append_fragment(region: Dict[str, Any], qid: Optional[str], start: int, end: int,
                        candidate_type: str, rationale: str, annotation_data: Dict[str, Any],
                        linked_to_previous: bool) -> None:
        nonlocal unclassified_regions
        fragment_text = text[start:end]
        if not fragment_text.strip():
            return
        parent_header_id = node_stack[-1][1] if qid is not None and node_stack else None
        candidate_id = next_id()
        if candidate_type == "unclassified_in_scope_fragment":
            unclassified_regions += 1
        cont_of = header_to_block.get(parent_header_id, parent_header_id) if (linked_to_previous and parent_header_id) else None
        group_id = f"CONT::{parent_header_id}" if cont_of else None
        seq = None
        if group_id:
            continuation_sequences[group_id] = continuation_sequences.get(group_id, 0) + 1
            seq = continuation_sequences[group_id]
        c = _region_candidate(region, qid, candidate_id, candidate_type, parent_header_id,
            text, start, end, fragment_text.strip(), "low", rationale, 5, annotation_data)
        c.update({
            "span_role": "fragment",
            "continuation_of_candidate_id": cont_of,
            "continuation_group_id": group_id,
            "continuation_sequence": seq,
            "linked_to_previous_scope_fragment": linked_to_previous,
        })
        candidates.append(c)

    for entry in entries:
        region = entry["region"]
        start, end = region["start_char"], region["end_char"]
        qid: Optional[str] = entry["question_id_before"]
        anchors: List[Anchor] = entry["anchors"]
        if entry["reset_context_before"]:
            node_stack.clear()
            header_to_block.clear()
        if not anchors:
            if text[start:end].strip():
                typ = "unattached_in_scope_fragment" if qid is None and not node_stack else "unclassified_in_scope_fragment"
                append_fragment(
                    region, qid, start, end, typ,
                    "No generic structural marker was recognized; keep unassigned if this is global/context text, otherwise review as a possible continuation or missed heading.",
                    annotation, entry["linked_to_previous_scope_fragment"])
            continue
        first_start = anchors[0].start
        prefix = text[start:first_start]
        if prefix.strip():
            typ = "unattached_in_scope_fragment" if qid is None and not node_stack else "continuation_fragment"
            append_fragment(
                region, qid, start, first_start, typ,
                "Text before the first recognized structural marker in this scope fragment; keep unassigned if it is introductory/global text, otherwise review as continuation.",
                annotation, entry["linked_to_previous_scope_fragment"])

        for i, a in enumerate(anchors):
            if a.kind == "question_header":
                qid = f"Q{a.label}"
                node_stack.clear()
                header_to_block.clear()
            while node_stack and node_stack[-1][0] >= a.level:
                node_stack.pop()
            parent_header_id = node_stack[-1][1] if node_stack else None
            next_end = end
            for b in anchors[i + 1:]:
                if b.level <= a.level:
                    next_end = b.start
                    break
            span_end = min(max(a.line_end, next_end), end)
            header_id = next_id()
            header_candidate = _region_candidate(region, qid, header_id, a.kind,
                parent_header_id, text, a.start, a.line_end, a.line.strip(), a.confidence,
                "Recognized generic structural marker; header span is distinct from any following body span. Confirm boundary and level during review.", a.level, annotation)
            header_candidate.update({
                "span_role": "header",
                "paired_candidate_id": None,
                "body_start_char": a.line_end,
                "body_end_char": span_end,
            })
            candidates.append(header_candidate)
            anchor_counts[a.kind] = anchor_counts.get(a.kind, 0) + 1
            node_stack.append((a.level, header_id))

            body_text = text[a.line_end:span_end]
            if body_text.strip():
                block_id = next_id()
                block_type = a.kind.replace("_header", "_block")
                block_candidate = _region_candidate(region, qid, block_id, block_type,
                    header_id, text, a.line_end, span_end, a.line.strip(), a.confidence,
                    "Body-only span paired with its header; header text is excluded from this interval. Nested items may lie within the body interval and remain separate pending proposals.", a.level, annotation)
                block_candidate.update({
                    "span_role": "block_body",
                    "paired_candidate_id": header_id,
                    "header_start_char": a.start,
                    "header_end_char": a.line_end,
                })
                candidates.append(block_candidate)
                header_candidate["paired_candidate_id"] = block_id
                header_to_block[header_id] = block_id

    # Scope, text alignment, and paired spans are validated before returning.
    by_id = {s["segment_id"]: s for s in scope_segments}
    by_candidate_id = {c["candidate_id"]: c for c in candidates}
    for c in candidates:
        s = by_id[c["scope_segment_id"]]
        if not (s["start_char"] <= c["start_char"] <= c["end_char"] <= s["end_char"]):
            raise AssertionError(f"Candidate {c['candidate_id']} crosses a scope boundary.")
        if text[c["start_char"]:c["end_char"]] != c["text"]:
            raise AssertionError(f"Candidate {c['candidate_id']} lost exact source alignment.")
        if c.get("paired_candidate_id"):
            if c["paired_candidate_id"] not in by_candidate_id:
                raise AssertionError(f"Candidate {c['candidate_id']} has a missing paired candidate.")
            paired = by_candidate_id[c["paired_candidate_id"]]
            if c["span_role"] == "header":
                if not (c["end_char"] <= paired["start_char"] <= paired["end_char"]):
                    raise AssertionError(f"Header/body spans overlap or are reversed for {c['candidate_id']}.")
            elif c["span_role"] == "block_body":
                if not (paired["end_char"] <= c["start_char"]):
                    raise AssertionError(f"Header/body spans overlap or are reversed for {c['candidate_id']}.")

    from collections import Counter
    candidate_type_counts = dict(sorted(Counter(c["candidate_type"] for c in candidates).items()))
    in_scope_count = sum(1 for s in scope_segments if s.get("scope_status") == "in_scope")
    out_scope_count = sum(1 for s in scope_segments if s.get("scope_status") == "out_of_scope")
    pending_count = sum(1 for s in scope_segments if s.get("scope_status") == "pending")
    separator_count = sum(1 for s in scope_segments if s.get("scope_status") == "separator")
    return {
        "schema_version": "structure-segmentation-proposal-v1.1",
        "document_id": annotation["document_id"],
        "source_filename": annotation.get("source", {}).get("filename") or scope.get("source_filename"),
        "source_text_sha256": expected_hash,
        "text_length": len(text),
        "annotation_filename": scope.get("annotation_filename", ""),
        "scope_ledger_policy_version": scope.get("policy_version", ""),
        "method": "generic_format_pattern_proposals_v2",
        "safety_contract": {
            "student_code_executed": False,
            "source_text_modified": False,
            "out_of_scope_text_used_for_candidates": False,
            "pending_scope_text_used_for_candidates": False,
            "labels_approved": False,
            "training_labels_created": False,
            "training_eligible_candidates_created": False,
        },
        "scope_summary": {
            "in_scope_segments": in_scope_count,
            "out_of_scope_segments_excluded": out_scope_count,
            "pending_scope_segments_excluded": pending_count,
            "separator_segments_excluded": separator_count,
        },
        "question_nodes": question_nodes,
        "total_candidate_count": len(candidates),
        "candidate_type_counts": candidate_type_counts,
        "anchor_type_counts": dict(sorted(anchor_counts.items())),
        "unclassified_in_scope_regions": unclassified_regions,
        "review_contract": "Every candidate is a proposal only. Header candidates contain their complete marker/title line; paired body candidates start after the header and never duplicate its character interval. Header nodes carry logical parent relationships; paired body spans are children of their own header. Whitespace-only scope separators and reviewed-out pure numeric page-footer lines may be bridged for logical continuity, but pending and other non-neutral out-of-scope regions break context. Unassigned global/context fragments may remain unassigned. Missed headings are acceptable in difficult formats and should be reviewed rather than force-fitted. Unreviewed/pending/out-of-scope text must not receive training labels.",
        "candidates": candidates,
    }
