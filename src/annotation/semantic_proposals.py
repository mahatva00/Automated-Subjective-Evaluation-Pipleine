from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from typing import Any

SCHEMA_VERSION = "semantic-annotation-proposal-v1.0"

# Generic request/action language across disciplines; intentionally not subject-specific.
_ACTIONS = {
    "action": re.compile(
        r"\b(check|display|show|identify|output|print|calculate|compute|find|determine|replace|impute|prepare|create|construct|generate|train|conduct|perform|execute|fit|predict|append|name|write|report|state|list|define|explain|describe|compare|contrast|discuss|justify|derive|prove|evaluate|assess|analy[sz]e|plot|draw|visuali[sz]e|demonstrate|validate|verify|confirm|select|choose|classify|cluster|estimate|measure|test(?!-)\b|implement|design|convert|transform|normalize|standardize|scale|balance|change|set|assign|add|remove|answer|summarize|interpret|infer|mention|include|use|record|illustrate|formulate|obtain|tabulate|present|investigate|explore|review|inspect)\b",
        re.IGNORECASE,
    ),
    "constraint": re.compile(r"\b(must|should|shall|required to|has to|have to|will be|not allowed|must not|should not|shall not|without|exactly|at least|at most|between|no\s+\w+)\b", re.I),
    "output": re.compile(r"\b(display|show|print|output|append|name|write|report|present|plot|draw|save|export|produce|return)\b", re.I),
    "input": re.compile(r"\b(given|provided|input|dataset|data set|from\s+the|using\s+the|for\s+each)\b", re.I),
    "evaluation": re.compile(r"\b(compare|contrast|evaluate|assess|error|accuracy|performance|difference|infer|interpret|validate|test(?!-)\b)", re.I),
    "explanation": re.compile(r"\b(explain|describe|discuss|justify|interpret|infer|reason|why|how)\b", re.I),
}

_MARKS_RE = re.compile(r"\[\s*\d+(?:\.\d+)?\s*(?:marks?|points?)\s*\]\s*$", re.I)
_EXAMPLE_RE = re.compile(r"\b(for example|e\.g\.)\s*[,.:]?", re.I)
_CONTEXT_RE = re.compile(r"\b(you are provided|provided with|given that|consider the following|suppose that|assume that|the dataset is|the data set is|you have the|is referred to as|given the)\b", re.I)
_DEFINITION_RE = re.compile(r"\b(means|is defined as|are defined as|refers to|is called|are called)\b", re.I)
_EXPLANATORY_RE = re.compile(r"\b(earlier|previously|in short|note that|in other words|as mentioned|let us call|we call|based on the above)\b", re.I)
_ALTERNATIVE_RE = re.compile(r"\b(or|alternatively|either|otherwise)\b", re.I)
_REQUEST_START_RE = re.compile(r"^\s*(what|which|who|where|when|why|how|whether|is|are|can|could|would|should|does|do|did|will|state|discuss|explain|describe|calculate|compute|identify|find|determine|show|write|list|compare|derive|prove|evaluate|plot|prepare|train|perform|conduct|create|design|implement|predict|report|justify|analy[sz]e|analyse|interpret|give|mention|tabulate|demonstrate|verify|confirm)\b", re.I)

# Marker patterns are generic document-format conventions, not domain rules.
_MARKERS = {
    "question_header": re.compile(r"^\s*\d+\s*[.)]\s*"),
    "subquestion_header": re.compile(r"^\s*(?:\([A-Za-z0-9ivxlcdmIVXLCDM]+\)|[A-Za-z]\))\s*"),
    "roman_item_header": re.compile(r"^\s*[ivxlcdmIVXLCDM]+\s*[.)]\s*"),
    "experiment_header": re.compile(r"^\s*[A-Z]\s*[.)]\s*"),
    "lettered_item_header": re.compile(r"^\s*(?:\([A-Za-z]\)|[A-Za-z]\))\s*"),
    "list_item_header": re.compile(r"^\s*(?:[•▪◦‣⁃\u0088\uf0b7\uf0a7\uf0a8\uf0b8\uf0a9\uf0aa\uf0b9\uf0c2\uf0b4–—-]+|\d+[.)]|[A-Za-z][.)])\s*"),
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _candidate_text(candidate: dict[str, Any], text: str) -> None:
    s, e = candidate.get("start_char"), candidate.get("end_char")
    if not isinstance(s, int) or not isinstance(e, int) or s < 0 or e <= s or e > len(text):
        raise ValueError(f"Invalid structure candidate interval: {candidate.get('candidate_id')}")
    sliced = text[s:e]
    if candidate.get("text") != sliced:
        raise ValueError(f"Structure candidate source mismatch: {candidate.get('candidate_id')}")
    if candidate.get("text_sha256") != sha256_text(sliced):
        raise ValueError(f"Structure candidate hash mismatch: {candidate.get('candidate_id')}")


def validate_inputs(annotation: dict[str, Any], scope: dict[str, Any], structure: dict[str, Any]) -> None:
    text = annotation.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError("Canonical annotation draft is missing non-empty text")
    text_hash = sha256_text(text)
    doc_id = annotation.get("document_id")
    if scope.get("document_id") != doc_id or structure.get("document_id") != doc_id:
        raise ValueError("document_id mismatch between annotation, scope ledger, and structure proposal")
    if scope.get("source_text_sha256") != text_hash:
        raise ValueError("Scope ledger source_text_sha256 does not match canonical annotation text")
    if structure.get("source_text_sha256") != text_hash:
        raise ValueError("Structure proposal source_text_sha256 does not match canonical annotation text")
    if scope.get("text_length") != len(text) or structure.get("text_length") != len(text):
        raise ValueError("text_length mismatch between inputs")
    seen_scope_ids: set[str] = set()
    for seg in scope.get("segments", []):
        sid, s, e = seg.get("segment_id"), seg.get("start_char"), seg.get("end_char")
        if not sid or sid in seen_scope_ids:
            raise ValueError(f"Missing or duplicate scope segment_id: {sid}")
        seen_scope_ids.add(sid)
        if not isinstance(s, int) or not isinstance(e, int) or s < 0 or e < s or e > len(text):
            raise ValueError(f"Invalid scope range: {sid}")
        part = text[s:e]
        if seg.get("text") != part or seg.get("text_sha256") != sha256_text(part):
            raise ValueError(f"Scope segment text/hash mismatch: {sid}")
    for c in structure.get("candidates", []):
        _candidate_text(c, text)
        if c.get("scope_status") != "in_scope":
            raise ValueError(f"Structure proposal contains a non-in-scope candidate: {c.get('candidate_id')}")
        if c.get("scope_segment_id") not in seen_scope_ids:
            raise ValueError(f"Structure candidate references unknown scope segment: {c.get('candidate_id')}")


def _marker_end(candidate: dict[str, Any]) -> int:
    text = candidate["text"]
    pattern = _MARKERS.get(candidate.get("candidate_type"))
    if not pattern:
        return 0
    match = pattern.match(text)
    return match.end() if match else 0


def _trim_content_interval(start: int, end: int, text: str) -> tuple[int, int]:
    return _trim_span(text, start, end)


def _split_examples(start: int, end: int, text: str) -> list[tuple[int, int, str]]:
    """Split off an explicitly signalled example while preserving source offsets."""
    value = text[start:end]
    match = _EXAMPLE_RE.search(value)
    if not match:
        return [(start, end, "")]
    cut = start + match.start()
    left = _trim_span(text, start, cut)
    right = _trim_span(text, cut, end)
    out: list[tuple[int, int, str]] = []
    if left[1] > left[0]:
        out.append((left[0], left[1], "before_explicit_example"))
    if right[1] > right[0]:
        out.append((right[0], right[1], "explicit_example"))
    return out or [(start, end, "")]


def _split_sentence_units(start: int, end: int, text: str) -> list[tuple[int, int, str]]:
    """Conservatively split on sentence-ending punctuation; preserve all source slices."""
    if end <= start:
        return []
    raw = text[start:end]
    # Example labels get their own non-overlapping span if explicit.
    ex_parts = _split_examples(start, end, text)
    if len(ex_parts) > 1:
        result: list[tuple[int, int, str]] = []
        for a, b, hint in ex_parts:
            if hint == "explicit_example":
                result.append((a, b, hint))
            else:
                result.extend(_split_sentence_units_no_example(a, b, text))
        return result
    return _split_sentence_units_no_example(start, end, text)


def _split_sentence_units_no_example(start: int, end: int, text: str) -> list[tuple[int, int, str]]:
    chunk = text[start:end]
    # Keep decimal points and do not attempt linguistic splitting at every line break.
    cut_positions = [m.end() for m in re.finditer(r"[.!?](?:[\]\)'\"”’]*)?(?=\s+(?=[A-Z0-9(“\"']))", chunk)]
    bounds = [0] + cut_positions + [len(chunk)]
    result = []
    for x, y in zip(bounds, bounds[1:]):
        a, b = _trim_span(text, start + x, start + y)
        if a < b:
            result.append((a, b, ""))
    return result



def _is_action_usage(value: str, match: re.Match[str]) -> bool:
    """Conservative generic filter to avoid interpreting table headings/nouns as commands."""
    prefix = value[:match.start()]
    stripped = prefix.strip()
    if not stripped:
        return True
    # A command can begin after a conventional clause/sequence marker.
    if re.search(r"[,;:]\s*$", prefix):
        return True
    if re.search(r"\b(?:and|or|then|to|by|please|must|should|shall|will|now|also)\s+$", prefix, re.I):
        return True
    # Request verbs at the beginning of a span are the strongest generic cue.
    return len(stripped) <= 18 and len(stripped.split()) <= 3 and bool(
        re.fullmatch(r"(?:based on .{0,60}|from .{0,60}|using .{0,60}|with .{0,60}|after .{0,60})?", stripped, re.I)
    )

def _classify(span: str, context_hint: str = "") -> tuple[str | None, dict[str, Any], str, list[str], str]:
    value = span.strip()
    low = value.lower()
    flags: list[str] = []
    attrs: dict[str, Any] = {}

    if _MARKS_RE.fullmatch(value):
        return "grading_instruction", attrs, "medium", flags, "Explicit marks/points notation attached to an in-scope question item."
    if context_hint == "explicit_example" or _EXAMPLE_RE.search(value):
        return "example", attrs, "medium", flags, "Explicit example language detected; verify whether the span is illustrative rather than required."
    if re.fullmatch(r"(?:or|alternatively|either|otherwise)[,.:;]?", low):
        return "alternative_marker", attrs, "medium", flags, "Standalone alternative marker."

    action_matches = [m for m in _ACTIONS["action"].finditer(value) if _is_action_usage(value, m)]
    actions = list(dict.fromkeys(m.group(1).lower() for m in action_matches))
    has_constraint = bool(_ACTIONS["constraint"].search(value))
    has_request = bool(_REQUEST_START_RE.search(value))
    has_output = bool(_ACTIONS["output"].search(value))
    has_input = bool(_ACTIONS["input"].search(value))
    has_eval = bool(_ACTIONS["evaluation"].search(value))
    has_explain = bool(_ACTIONS["explanation"].search(value))
    has_example = bool(_EXAMPLE_RE.search(value))
    has_context = bool(_CONTEXT_RE.search(value))
    has_definition = bool(_DEFINITION_RE.search(value))
    has_alternative = bool(_ALTERNATIVE_RE.search(value))

    # Task-specific obligations and explicit constraints take precedence over general context cues.
    obligation = bool(actions or has_constraint or has_request or re.search(r"\b(there (?:must|should|shall) be|will have|will not have|is required to|are required to|will be|(?:is|are)\s+(?:the|a|an|equal to|computed as|obtained as))\b", value, re.I))
    if obligation:
        if has_eval:
            req_type = "evaluation_or_comparison"
        elif has_constraint and ("should be" in low or "must be" in low or "will be" in low or "will have" in low or "will not have" in low or "not have" in low or "should not" in low or "must not" in low or "between" in low or "without" in low or "exactly" in low or "at least" in low or "at most" in low):
            req_type = "constraint"
        elif has_explain and not actions:
            req_type = "explanation_or_justification"
        elif has_output:
            req_type = "output_specification" if not actions or any(a in actions for a in ["display", "show", "print", "output", "append", "name", "write", "report", "present", "save", "export", "produce", "return", "plot", "draw", "tabulate"]) else "action"
        elif has_input and not actions:
            req_type = "input_specification"
        elif re.search(r"\b(using|by means of|based on)\b", value, re.I) and actions:
            req_type = "method_or_tool"
        elif has_input and re.search(r"\b(from|using|given|provided)\b", value, re.I):
            req_type = "input_specification"
        else:
            req_type = "action"
        atomicity = "atomic"
        if len(actions) > 1:
            atomicity = "compound" if re.search(r"\band\b|;|,", value, re.I) else "uncertain"
            flags.append("multiple_action_verbs_review_atomicity")
        elif has_alternative:
            atomicity = "uncertain"
            flags.append("contains_alternative_word_review_relationship")
        if has_example:
            flags.append("contains_example_phrase_review_boundary")
        attrs.update({"requirement_type": req_type, "atomicity": atomicity, "assessment_relevance": "yes"})
        if len(actions) > 1:
            rationale = "Generic task/action or constraint cues detected; multiple action verbs were retained in one source span for reviewer atomicity decision."
        elif has_constraint and not actions:
            rationale = "Generic obligation/constraint wording detected within an in-scope question."
        else:
            rationale = "Generic task/action wording detected within an in-scope question; this is a candidate label, not an approved annotation."
        return "requirement", attrs, "medium", flags, rationale

    if has_definition:
        return "definition", attrs, "low", ["verify_definition_role_in_context"], "Definition-like wording detected, but its role in the question needs context review."
    if _EXPLANATORY_RE.search(value):
        return "explanatory_note", attrs, "low", ["verify_explanatory_note_not_assessable"], "Generic discourse/transition wording detected; verify that the span is explanatory rather than assessable."
    if has_context:
        return "context", attrs, "low", flags, "Background/setup wording detected within an in-scope question; verify it is not itself assessable."
    if has_example:
        return "example", attrs, "low", flags, "Example-like wording detected; verify boundary and role."
    if has_alternative:
        flags.append("possible_alternative_or_choice")
        return None, attrs, "low", flags, "Alternative wording appears, but it is not safe to label the entire span as an alternative marker."
    if len(value) < 3 or not re.search(r"[A-Za-z0-9]", value):
        return None, attrs, "low", ["non_semantic_or_fragmentary"], "Span is too short or fragmentary for a reliable semantic proposal."
    return None, attrs, "low", ["no_reliable_generic_semantic_cue"], "No sufficiently reliable generic semantic cue; retain as pending/unclassified instead of forcing a label."


def _page_numbers(start: int, end: int, annotation: dict[str, Any], scope_segment: dict[str, Any]) -> list[int]:
    nums: set[int] = set(scope_segment.get("page_numbers") or [])
    for pr in annotation.get("page_ranges", []):
        if pr.get("start_char", 0) < end and pr.get("end_char", 0) > start and pr.get("page_number") is not None:
            nums.add(int(pr["page_number"]))
    return sorted(nums)


def propose_semantic_annotations(annotation: dict[str, Any], scope: dict[str, Any], structure: dict[str, Any]) -> dict[str, Any]:
    validate_inputs(annotation, scope, structure)
    text = annotation["text"]
    scope_by_id = {s["segment_id"]: s for s in scope["segments"]}
    structure_by_seg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    structure_by_id = {c["candidate_id"]: c for c in structure["candidates"]}
    for c in structure["candidates"]:
        structure_by_seg[c["scope_segment_id"]].append(c)

    question_ranges = _question_ranges(structure, text)
    candidates: list[dict[str, Any]] = []
    ignored_unassigned = []
    for seg in scope["segments"]:
        if seg.get("scope_status") != "in_scope":
            continue
        s0, e0, sid = seg["start_char"], seg["end_char"], seg["segment_id"]
        segment_candidates = structure_by_seg.get(sid, [])
        intersections = []
        for qstart, qend, qid in question_ranges:
            a, b = max(s0, qstart), min(e0, qend)
            if a < b:
                intersections.append((a, b, qid))
        intersections.sort(key=lambda item: (item[0], item[1]))
        # Preserve any in-scope region outside all recognized question boundaries as unassigned.
        covered = sorted((a, b) for a, b, _ in intersections)
        gap_cursor = s0
        for a, b in covered:
            if gap_cursor < a:
                ignored_unassigned.append(_ignored_region(text, seg, gap_cursor, a,
                    "Outside recognized top-level question boundaries; intentionally left unassigned."))
            gap_cursor = max(gap_cursor, b)
        if gap_cursor < e0:
            ignored_unassigned.append(_ignored_region(text, seg, gap_cursor, e0,
                "Outside recognized top-level question boundaries; intentionally left unassigned."))

        for rstart, rend, qid in intersections:
            anchors = [c for c in segment_candidates if c.get("span_role") == "header"
                       and c["start_char"] >= rstart and c["end_char"] <= rend]
            anchors.sort(key=lambda c: (c["start_char"], c["end_char"]))
            # Treat each header plus following unmarked text up to the next header as one
            # logical semantic region. This rejoins PDF line wraps (including split mark strings)
            # without changing source offsets. The next detected heading creates a hard boundary.
            if anchors and anchors[0]["start_char"] > rstart:
                pre_start, pre_end = _trim_span(text, rstart, anchors[0]["start_char"])
                if pre_start < pre_end:
                    leading_ids = [c["candidate_id"] for c in segment_candidates
                                   if c.get("candidate_type") == "continuation_fragment"
                                   and c["start_char"] < pre_end and c["end_char"] > pre_start]
                    _append_semantic_candidate(candidates, text, pre_start, pre_end, qid, sid, seg, annotation,
                                               leading_ids, "leading_fragment", None, "")
            for idx, anchor in enumerate(anchors):
                a = anchor["start_char"]
                next_start = anchors[idx + 1]["start_char"] if idx + 1 < len(anchors) else rend
                content_start = a + _marker_end(anchor)
                content_start, content_end = _trim_span(text, content_start, next_start)
                if content_start >= content_end:
                    continue
                mark_match = _MARKS_RE.search(text[content_start:content_end])
                task_end = content_end
                marks_span: tuple[int, int] | None = None
                if mark_match:
                    mstart = content_start + mark_match.start()
                    mend = content_start + mark_match.end()
                    task_end = _trim_span(text, content_start, mstart)[1]
                    marks_span = _trim_span(text, mstart, mend)
                if content_start < task_end:
                    for x, y, hint in _split_sentence_units(content_start, task_end, text):
                        _append_semantic_candidate(candidates, text, x, y, qid, sid, seg, annotation,
                                                   [anchor["candidate_id"]], "header_with_following_fragment", anchor, hint)
                if marks_span and marks_span[0] < marks_span[1]:
                    _append_semantic_candidate(candidates, text, marks_span[0], marks_span[1], qid, sid, seg, annotation,
                                               [anchor["candidate_id"]], "grading_marks", anchor, "marks_suffix")
            if not anchors:
                # Preserve question text where a heading was missed; keep its local structure status visible.
                a, b = _trim_span(text, rstart, rend)
                if a < b:
                    fragment_structures = [c for c in segment_candidates
                                           if c.get("span_role") == "fragment" or "fragment" in c.get("candidate_type", "")]
                    src_ids = [c["candidate_id"] for c in fragment_structures
                               if c["start_char"] < b and c["end_char"] > a]
                    _append_semantic_candidate(candidates, text, a, b, qid, sid, seg, annotation, src_ids,
                                               "body_fragment", None, "")

    # Assign stable IDs after deterministic source-order sorting. Semantic candidates never overlap.
    candidates.sort(key=lambda c: (c["start_char"], c["end_char"], c["proposal_id"]))
    for i, c in enumerate(candidates, 1):
        c["proposal_id"] = f"SEM{i:05d}"
    _validate_semantic_candidates(candidates, text)

    # Connect semantic candidates associated with structure continuation links. These are proposals only.
    by_structure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        for source_id in c.get("source_structure_candidate_ids", []):
            by_structure[source_id].append(c)
    relation_proposals: list[dict[str, Any]] = []
    seen_rel: set[tuple[str, str]] = set()
    for c in structure["candidates"]:
        prev = c.get("continuation_of_candidate_id")
        # Generic unclassified fragments can be context under a parent, not literal continuations.
        if not prev or c.get("candidate_type") != "continuation_fragment" or not c.get("continuation_group_id"):
            continue
        left = by_structure.get(prev, [])
        right = by_structure.get(c["candidate_id"], [])
        left = sorted(left, key=lambda x: (x.get("label_proposal") != "requirement", x["start_char"]))
        for l in left[:1]:
            for r in right[:1]:
                if l["proposal_id"] != r["proposal_id"] and (l["proposal_id"], r["proposal_id"]) not in seen_rel:
                    seen_rel.add((l["proposal_id"], r["proposal_id"]))
                    relation_proposals.append({"source_proposal_id": l["proposal_id"], "target_proposal_id": r["proposal_id"],
                                               "type_proposal": "continues", "review_status": "pending",
                                               "rationale": "Source structure proposal indicates continuation; verify semantic continuity before adopting."})

    question_reconstructions = _reconstruct_questions(text, scope, structure)
    label_counts = Counter(c["label_proposal"] or "unclassified" for c in candidates)
    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": annotation["document_id"],
        "source_filename": annotation.get("source", {}).get("filename"),
        "source_text_sha256": sha256_text(text),
        "text_length": len(text),
        "annotation_filename": structure.get("annotation_filename"),
        "scope_ledger_policy_version": scope.get("policy_version"),
        "structure_proposal_schema_version": structure.get("schema_version"),
        "method": "generic_lexical_semantic_proposals_v1",
        "safety_contract": {
            "student_code_executed": False,
            "source_text_modified": False,
            "out_of_scope_text_used_for_candidates": False,
            "unassigned_question_context_used_for_candidates": False,
            "labels_approved": False,
            "training_labels_created": False,
            "training_eligible_candidates_created": False,
            "semantic_candidate_spans_overlap": False,
        },
        "scope_summary": {
            "in_scope_segments": sum(s.get("scope_status") == "in_scope" for s in scope["segments"]),
            "out_of_scope_segments_excluded": sum(s.get("scope_status") == "out_of_scope" for s in scope["segments"]),
            "pending_scope_segments_excluded": sum(s.get("scope_status") == "pending" for s in scope["segments"]),
            "separator_segments_excluded": sum(s.get("scope_status") == "separator" for s in scope["segments"]),
            "unassigned_in_scope_regions_omitted": len(ignored_unassigned),
        },
        "candidate_count": len(candidates),
        "label_proposal_counts": dict(sorted(label_counts.items())),
        "question_reconstructions": question_reconstructions,
        "ignored_unassigned_regions": ignored_unassigned,
        "relation_proposals": relation_proposals,
        "review_contract": "All semantic labels are pending heuristic suggestions, not approved annotations. Every candidate text must equal the exact canonical source slice. Only reviewed in-scope text with a question_id is considered. Candidate gaps/uncertain labels are retained without forcing a semantic label. Same-layer candidate spans are non-overlapping. Reconstructed question text uses synthetic newline joins for display only; original fragment spans/offsets remain authoritative.",
        "candidates": candidates,
    }


def _append_semantic_candidate(out: list[dict[str, Any]], text: str, start: int, end: int, qid: str,
                               scope_segment_id: str, scope_segment: dict[str, Any], annotation: dict[str, Any],
                               structure_ids: list[str], unit_kind: str, anchor: dict[str, Any] | None,
                               split_hint: str) -> None:
    start, end = _trim_span(text, start, end)
    if start >= end:
        return
    value = text[start:end]
    label, attrs, confidence, flags, rationale = _classify(value, split_hint)
    if split_hint == "before_explicit_example":
        # Example boundary found later in the same source region, not in this preceding task span.
        flags.append("preceded_by_explicit_example_in_same_region")
    if label == "requirement" and _ALTERNATIVE_RE.search(value):
        flags.append("possible_choice_or_alternative_requires_review")
    anchor_id = anchor.get("candidate_id") if anchor else None
    out.append({
        "proposal_id": "TEMP",
        "layer": "semantic",
        "label_proposal": label,
        "label_proposal_candidates": [label] if label else [],
        "start_char": start,
        "end_char": end,
        "text": value,
        "text_sha256": sha256_text(value),
        "question_id": qid,
        "scope_segment_id": scope_segment_id,
        "page_numbers": _page_numbers(start, end, annotation, scope_segment),
        "source_structure_candidate_ids": structure_ids,
        "source_structure_candidate_type": anchor.get("candidate_type") if anchor else None,
        "source_span_role": anchor.get("span_role") if anchor else "fragment",
        "unit_kind": unit_kind,
        "requirement_attributes_proposal": attrs if label == "requirement" else {},
        "proposal_confidence": confidence,
        "proposal_rationale": rationale,
        "review_flags": sorted(set(flags)),
        "review_status": "pending",
        "annotation_label": None,
        "training_label": None,
        "training_eligible": False,
        "reviewer_decision": None,
        "note": "Heuristic proposal only; confirm the exact semantic role and requirement atomicity against question context.",
    })


def _validate_semantic_candidates(candidates: list[dict[str, Any]], text: str) -> None:
    seen = set()
    ordered = sorted(candidates, key=lambda c: (c["start_char"], c["end_char"]))
    prev_end = -1
    for c in ordered:
        pid = c["proposal_id"]
        if pid in seen:
            raise ValueError(f"Duplicate proposal ID: {pid}")
        seen.add(pid)
        s, e = c["start_char"], c["end_char"]
        if not (0 <= s < e <= len(text)):
            raise ValueError(f"Invalid semantic interval: {pid}")
        if c["text"] != text[s:e] or c["text_sha256"] != sha256_text(text[s:e]):
            raise ValueError(f"Semantic candidate source mismatch: {pid}")
        if s < prev_end:
            raise ValueError(f"Overlapping same-layer semantic candidates detected near {pid}")
        prev_end = e
        if c["review_status"] != "pending" or c["annotation_label"] is not None or c["training_label"] is not None or c["training_eligible"]:
            raise ValueError(f"Review/training contract violated: {pid}")


def _question_ranges(structure: dict[str, Any], text: str) -> list[tuple[int, int, str]]:
    nodes = sorted(structure.get("question_nodes", []), key=lambda q: q.get("header_start_char", -1))
    valid = []
    previous_start = -1
    for node in nodes:
        qid = node.get("question_id")
        start, end = node.get("header_start_char"), node.get("header_end_char")
        if not qid or not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(text)):
            raise ValueError(f"Invalid question node boundary: {node}")
        header = text[start:end]
        if node.get("header_sha256") != sha256_text(header):
            raise ValueError(f"Question node header hash mismatch: {qid}")
        if start <= previous_start:
            raise ValueError("Question nodes are not in strictly increasing source order")
        valid.append((start, qid))
        previous_start = start
    return [(start, valid[i + 1][0] if i + 1 < len(valid) else len(text), qid)
            for i, (start, qid) in enumerate(valid)]


def _ignored_region(text: str, segment: dict[str, Any], start: int, end: int, reason: str) -> dict[str, Any]:
    return {"scope_segment_id": segment["segment_id"], "start_char": start, "end_char": end,
            "text_sha256": sha256_text(text[start:end]), "page_numbers": segment.get("page_numbers", []),
            "text_preview": text[start:end][:240], "reason": reason,
            "review_status": "pending", "training_eligible": False}


def _reconstruct_questions(text: str, scope: dict[str, Any], structure: dict[str, Any]) -> list[dict[str, Any]]:
    # Top-level question nodes define the inclusion boundaries. Only reviewed in-scope intersections are used.
    q_ranges = _question_ranges(structure, text)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seg in scope["segments"]:
        if seg.get("scope_status") != "in_scope":
            continue
        for qstart, qend, qid in q_ranges:
            a, b = max(seg["start_char"], qstart), min(seg["end_char"], qend)
            if a >= b:
                continue
            groups[qid].append({"scope_segment_id": seg["segment_id"], "start_char": a, "end_char": b,
                                "text_sha256": sha256_text(text[a:b]), "page_numbers": seg.get("page_numbers", []),
                                "text": text[a:b]})
    result = []
    for qid, fragments in sorted(groups.items(), key=lambda item: min(x["start_char"] for x in item[1])):
        fragments.sort(key=lambda x: x["start_char"])
        assembled = "\n".join(f["text"] for f in fragments)
        result.append({"question_id": qid, "fragment_count": len(fragments), "source_fragments": fragments,
                       "reconstructed_text": assembled, "reconstructed_text_sha256": sha256_text(assembled),
                       "synthetic_join": "\n", "reconstructed_text_is_literal_source_slice": False,
                       "note": "Display/context reconstruction only. Synthetic newlines separate reviewed in-scope source fragments; individual offsets and fragment hashes remain authoritative. Text before the first recognized top-level question heading and out-of-scope text are not included."})
    return result

def write_tsv(proposal: dict[str, Any], path: str) -> None:
    fields = ["document_id", "proposal_id", "question_id", "scope_segment_id", "page_numbers", "layer", "label_proposal",
              "label_proposal_candidates", "requirement_type", "atomicity", "assessment_relevance", "start_char", "end_char",
              "text_sha256", "proposal_confidence", "review_status", "training_label", "training_eligible", "review_flags",
              "source_structure_candidate_ids", "unit_kind", "text", "proposal_rationale"]
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for c in proposal["candidates"]:
            attrs = c.get("requirement_attributes_proposal", {})
            row = dict(c)
            row.update({"requirement_type": attrs.get("requirement_type", ""), "atomicity": attrs.get("atomicity", ""),
                        "assessment_relevance": attrs.get("assessment_relevance", ""),
                        "page_numbers": ",".join(map(str, c.get("page_numbers", []))),
                        "label_proposal_candidates": ",".join(c.get("label_proposal_candidates", [])),
                        "review_flags": ";".join(c.get("review_flags", [])),
                        "source_structure_candidate_ids": ",".join(c.get("source_structure_candidate_ids", [])),
                        "training_label": "", "training_eligible": "false"})
            w.writerow(row)
