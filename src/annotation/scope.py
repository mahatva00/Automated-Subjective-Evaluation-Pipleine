"""Question-only source scope, exact-offset auditing, and scope-aware token labels.

Out-of-scope and unresolved text must never become model background/O labels.
Whitespace-only separator ranges contain no tokens and may bridge a multi-line span
only when the non-whitespace content on both sides belongs to the same question.
"""
from __future__ import annotations
import copy
import datetime as _datetime
import hashlib
from typing import Any, Iterable

SCOPE_VERSION = "1.0.0"
SCOPE_STATES = {"in_scope", "out_of_scope", "pending", "separator"}
REVIEW_STATES = {"pending", "reviewed", "not_required"}
LAYERS = {"structure", "semantic"}
IGNORE_INDEX = -100


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pages_for_range(doc: dict[str, Any], start: int, end: int) -> list[int]:
    result = []
    for page in doc.get("page_ranges", []):
        if not isinstance(page, dict):
            continue
        pstart, pend = page.get("start_char"), page.get("end_char")
        if isinstance(pstart, int) and isinstance(pend, int) and start < pend and end > pstart:
            pnum = page.get("page_number")
            if pnum is not None and pnum not in result:
                result.append(pnum)
    return result


def _scope_ranges(text: str, max_chars: int) -> list[tuple[int, int]]:
    """Partition text into paragraph-like ranges plus whitespace-only separators.

    PDF line wraps within a paragraph are grouped to reduce annotation workload.
    Long paragraphs are split near a line/whitespace boundary. Any mixed range can
    be split later at an absolute character offset using split_question_scope_segment.py.
    """
    lines = text.splitlines(keepends=True)
    raw_groups: list[tuple[int, int, bool]] = []  # start, end, whitespace-only
    cursor = 0
    content_start = None
    content_end = None
    blank_start = None
    for line in lines:
        end = cursor + len(line)
        if line.strip():
            if blank_start is not None:
                raw_groups.append((blank_start, cursor, True))
                blank_start = None
            if content_start is None:
                content_start = cursor
            content_end = end
        else:
            if content_start is not None:
                raw_groups.append((content_start, content_end, False))
                content_start = content_end = None
            if blank_start is None:
                blank_start = cursor
        cursor = end
    if content_start is not None:
        raw_groups.append((content_start, content_end, False))
    if blank_start is not None:
        raw_groups.append((blank_start, cursor, True))
    if cursor < len(text):
        raw_groups.append((cursor, len(text), not text[cursor:].strip()))

    ranges: list[tuple[int, int]] = []
    for start, end, is_blank in raw_groups:
        if end <= start:
            continue
        if is_blank:
            ranges.append((start, end))
            continue
        # Split a paragraph into manageable segments, preferring line and then whitespace.
        pos = start
        while pos < end:
            limit = min(pos + max_chars, end)
            if limit < end:
                floor = pos + max(1, max_chars // 2)
                newline = text.rfind("\n", floor, limit)
                space = max(text.rfind(" ", floor, limit), text.rfind("\t", floor, limit))
                boundary = newline if newline >= floor else space
                if boundary >= floor:
                    limit = boundary + 1
            ranges.append((pos, limit))
            pos = limit
    out = []
    cursor = 0
    for start, end in ranges:
        if start != cursor or end <= start:
            raise ValueError("internal scope segment partition error")
        out.append((start, end))
        cursor = end
    if cursor != len(text):
        raise ValueError("scope segment partition does not cover source text")
    return out


def build_scope_review(doc: dict[str, Any], annotation_filename: str, max_chars: int = 900) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(doc, dict) or not isinstance(doc.get("text"), str) or not doc["text"]:
        raise ValueError("annotation document must contain a non-empty text string")
    if not isinstance(doc.get("document_id"), str) or not doc["document_id"].strip():
        raise ValueError("document_id is missing")
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    text = doc["text"]
    segments, tsv_rows = [], []
    for i, (start, end) in enumerate(_scope_ranges(text, max_chars), 1):
        piece = text[start:end]
        blank = not piece.strip()
        row = {
            "segment_id": f"scope_{i:05d}", "start_char": start, "end_char": end,
            "text_sha256": sha256_text(piece), "text": piece,
            "page_numbers": _pages_for_range(doc, start, end),
            "scope_status": "separator" if blank else "pending",
            "question_id": None, "reason": "Whitespace-only separator; no token labels are assigned." if blank else "",
            "review_status": "not_required" if blank else "pending",
            "reviewer": "", "review_date": "",
        }
        segments.append(row)
        tsv_rows.append({
            "document_id": doc["document_id"], "segment_id": row["segment_id"],
            "start_char": start, "end_char": end, "scope_status": row["scope_status"],
            "question_id": "", "reason": row["reason"], "reviewer": "", "review_date": "",
            "text_sha256": row["text_sha256"],
            # JSON quoting protects tabs, line breaks, quotes, and unusual PDF chars in TSV.
            "text_preview_json": __import__("json").dumps(piece, ensure_ascii=False),
        })
    ledger = {
        "scope_version": SCOPE_VERSION,
        "document_id": doc["document_id"],
        "annotation_filename": annotation_filename,
        "source_filename": doc.get("source", {}).get("filename"),
        "source_text_sha256": sha256_text(text),
        "text_length": len(text),
        "policy_version": "question_scope_policy_v1.2",
        "policy_summary": "Only content inside an actual question/subquestion is in scope. All other non-whitespace text is excluded from question requirement extraction and training loss.",
        "segments": segments,
    }
    return ledger, tsv_rows


def validate_scope_ledger(doc: dict[str, Any], ledger: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(doc, dict) or not isinstance(doc.get("text"), str):
        return ["annotation document has no text string"]
    if not isinstance(ledger, dict):
        return ["scope ledger must be a JSON object"]
    text = doc["text"]
    if ledger.get("scope_version") != SCOPE_VERSION:
        errors.append(f"scope_version must be {SCOPE_VERSION!r}")
    if ledger.get("document_id") != doc.get("document_id"):
        errors.append("scope document_id does not match annotation document")
    if ledger.get("source_text_sha256") != sha256_text(text):
        errors.append("scope source_text_sha256 mismatch")
    if ledger.get("text_length") != len(text):
        errors.append("scope text_length mismatch")
    segments = ledger.get("segments")
    if not isinstance(segments, list) or not segments:
        errors.append("scope segments must be a non-empty array")
        return errors
    cursor, seen = 0, set()
    for i, seg in enumerate(segments):
        where = f"segments[{i}]"
        if not isinstance(seg, dict):
            errors.append(f"{where} must be an object")
            continue
        sid = seg.get("segment_id")
        if not isinstance(sid, str) or not sid or sid in seen:
            errors.append(f"{where}.segment_id is empty or duplicated")
        seen.add(sid)
        start, end = seg.get("start_char"), seg.get("end_char")
        if not isinstance(start, int) or not isinstance(end, int):
            errors.append(f"{where}.start_char/end_char must be integers")
            continue
        if start != cursor:
            errors.append(f"{where} does not continue source partition at {cursor}; starts at {start}")
        if start < 0 or end <= start or end > len(text):
            errors.append(f"{where} has invalid range {start}:{end}")
            continue
        piece = text[start:end]
        if seg.get("text") != piece:
            errors.append(f"{where}.text does not exactly match source slice")
        if seg.get("text_sha256") != sha256_text(piece):
            errors.append(f"{where}.text_sha256 mismatch")
        status, review = seg.get("scope_status"), seg.get("review_status")
        if status not in SCOPE_STATES:
            errors.append(f"{where}.scope_status is invalid")
        if review not in REVIEW_STATES:
            errors.append(f"{where}.review_status is invalid")
        if status == "separator":
            if piece.strip():
                errors.append(f"{where} is separator but contains non-whitespace text")
            if review != "not_required":
                errors.append(f"{where} separator must have review_status='not_required'")
        elif status == "pending":
            if review != "pending":
                errors.append(f"{where} pending scope must have review_status='pending'")
            if seg.get("reviewer") or seg.get("review_date"):
                errors.append(f"{where} pending scope cannot have reviewer/date")
        elif status in {"in_scope", "out_of_scope"}:
            if review != "reviewed":
                errors.append(f"{where} decided scope must have review_status='reviewed'")
            if not isinstance(seg.get("reviewer"), str) or not seg["reviewer"].strip():
                errors.append(f"{where} reviewer is required for decided scope")
            date = seg.get("review_date")
            try:
                _datetime.date.fromisoformat(date)
                if not isinstance(date, str):
                    raise ValueError
            except (ValueError, TypeError):
                errors.append(f"{where} review_date must be ISO date YYYY-MM-DD")
            if status == "in_scope":
                qid = seg.get("question_id")
                if not isinstance(qid, str) or not qid.strip():
                    errors.append(f"{where} in_scope segment requires question_id")
            elif seg.get("question_id") not in (None, ""):
                errors.append(f"{where} out_of_scope segment must not have question_id")
        cursor = end
    if cursor != len(text):
        errors.append(f"scope partition ends at {cursor}, source ends at {len(text)}")
    return errors


def classify_span_scope(start: int, end: int, ledger: dict[str, Any], source_text: str | None = None) -> tuple[str, str | None]:
    """Return (in_scope/out_of_scope/pending/mixed/separator, question_id)."""
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
        return "mixed", None
    relevant = []
    for seg in ledger.get("segments", []):
        if start < seg.get("end_char", -1) and end > seg.get("start_char", -1):
            if seg.get("scope_status") == "separator":
                continue
            relevant.append(seg)
    if not relevant:
        return "separator", None
    statuses = {s.get("scope_status") for s in relevant}
    qids = {s.get("question_id") for s in relevant if s.get("scope_status") == "in_scope"}
    if statuses == {"in_scope"} and len(qids) == 1:
        return "in_scope", next(iter(qids))
    if statuses == {"out_of_scope"}:
        return "out_of_scope", None
    if statuses == {"pending"}:
        return "pending", None
    return "mixed", None


def validate_annotations_against_scope(doc: dict[str, Any], ledger: dict[str, Any]) -> list[str]:
    errors = []
    for i, ann in enumerate(doc.get("annotations", [])):
        if not isinstance(ann, dict):
            continue
        start, end = ann.get("start_char"), ann.get("end_char")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        scope_status, qid = classify_span_scope(start, end, ledger, doc.get("text"))
        if scope_status != "in_scope":
            errors.append(f"annotations[{i}] ({ann.get('annotation_id')}) overlaps {scope_status} text; annotations may only be inside a reviewed question")
        attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
        if attrs.get("question_id") and qid and attrs["question_id"] != qid:
            errors.append(f"annotations[{i}] question_id does not match scope question_id {qid!r}")
    return errors



def coverage_chunk_requires_review(chunk: dict[str, Any], ledger: dict[str, Any] | None, source_text: str) -> bool:
    """Whether this coverage chunk contains any non-whitespace in-scope text.

    Out-of-scope and separator characters do not need structural/semantic annotation
    review. A mixed chunk is required if any part of it is in scope; reviewers should
    review only its in-scope question content under policy v1.2.
    """
    if ledger is None:
        return True
    start, end = chunk.get("start_char"), chunk.get("end_char")
    if not isinstance(start, int) or not isinstance(end, int):
        return True
    for seg in ledger.get("segments", []):
        if not isinstance(seg, dict) or seg.get("scope_status") != "in_scope":
            continue
        ss, se = seg.get("start_char"), seg.get("end_char")
        if not isinstance(ss, int) or not isinstance(se, int) or start >= se or end <= ss:
            continue
        a, b = max(start, ss), min(end, se)
        if a < b and source_text[a:b].strip():
            return True
    return False


def filter_document_annotations_to_scope(doc: dict[str, Any], ledger: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a copied document containing only annotation spans wholly in question scope.

    Canonical files are never modified. Spans in out-of-scope, pending, separator, or
    mixed ranges are removed from the downstream copy. Relations are retained only if
    both endpoint annotations survive. The audit summary records all removals.
    Callers must still reject incomplete/pending scope before using the result for training.
    """
    filtered = copy.deepcopy(doc)
    kept = []
    kept_ids = set()
    removed_by_state: dict[str, int] = {}
    removed_ids: list[str] = []
    for ann in doc.get("annotations", []):
        if not isinstance(ann, dict):
            continue
        start, end = ann.get("start_char"), ann.get("end_char")
        if not isinstance(start, int) or not isinstance(end, int):
            state = "invalid_span"
        else:
            state, _ = classify_span_scope(start, end, ledger, doc.get("text"))
        if state == "in_scope":
            kept.append(copy.deepcopy(ann))
            if isinstance(ann.get("annotation_id"), str):
                kept_ids.add(ann["annotation_id"])
        else:
            removed_by_state[state] = removed_by_state.get(state, 0) + 1
            if isinstance(ann.get("annotation_id"), str):
                removed_ids.append(ann["annotation_id"])
    filtered["annotations"] = kept
    original_relations = doc.get("relations", [])
    kept_relations = [copy.deepcopy(rel) for rel in original_relations if isinstance(rel, dict)
                      and rel.get("source_annotation_id") in kept_ids
                      and rel.get("target_annotation_id") in kept_ids]
    filtered["relations"] = kept_relations
    removed_relations = len(original_relations) - len(kept_relations)
    audit = {
        "policy_version": "question_scope_policy_v1.2",
        "original_annotation_count": len(doc.get("annotations", [])),
        "kept_in_scope_annotation_count": len(kept),
        "excluded_annotation_count_by_scope_state": dict(sorted(removed_by_state.items())),
        "excluded_annotation_ids": removed_ids,
        "original_relation_count": len(original_relations),
        "kept_relation_count": len(kept_relations),
        "excluded_relation_count": removed_relations,
        "canonical_source_modified": False,
    }
    return filtered, audit

def _range_is_reviewed_coverage(start: int, end: int, coverage: dict[str, Any], layer: str) -> bool:
    if layer not in LAYERS:
        raise ValueError("layer must be 'structure' or 'semantic'")
    spans = sorted((c.get("start_char"), c.get("end_char"), c.get(f"{layer}_status")) for c in coverage.get("chunks", []) if isinstance(c, dict))
    cursor = start
    for cstart, cend, status in spans:
        if not isinstance(cstart, int) or not isinstance(cend, int) or cend <= cursor or cstart >= end:
            continue
        if cstart > cursor or status != "reviewed":
            return False
        cursor = min(end, cend)
        if cursor >= end:
            return True
    return False


def project_token_labels(doc: dict[str, Any], ledger: dict[str, Any], coverage: dict[str, Any], offsets: Iterable[tuple[int, int]], layer: str, labels: set[str] | None = None) -> dict[str, Any]:
    """Project human spans to BIO labels; out-of-scope/pending/unreviewed tokens are -100.

    This function is independent of a tokenizer. Pass token offset_mapping values for a
    single untruncated sequence/window. Only reviewed/adjudicated annotations are used.
    """
    if layer not in LAYERS:
        raise ValueError("layer must be 'structure' or 'semantic'")
    offsets = list(offsets)
    annotations = [a for a in doc.get("annotations", []) if isinstance(a, dict) and a.get("layer") == layer and a.get("review_status") in {"reviewed", "adjudicated"}]
    annotations.sort(key=lambda a: (a.get("start_char", -1), a.get("end_char", -1)))
    names = set(labels or {a.get("label") for a in annotations})
    names.discard(None)
    unknown = sorted({a.get("label") for a in annotations if a.get("label") not in names})
    if unknown:
        raise ValueError(f"annotations use labels missing from label inventory: {unknown}")
    out, counts = [], {"ignored_out_of_scope_or_pending": 0, "ignored_unreviewed_coverage": 0, "ignored_boundary_token": 0, "positive_tokens": 0, "background_tokens": 0}
    previous_annotation_id = None
    for start, end in offsets:
        if not isinstance(start, int) or not isinstance(end, int) or start == end:
            out.append(IGNORE_INDEX); previous_annotation_id = None; continue
        scope_status, qid = classify_span_scope(start, end, ledger, doc.get("text"))
        if scope_status != "in_scope":
            out.append(IGNORE_INDEX); previous_annotation_id = None
            counts["ignored_out_of_scope_or_pending"] += 1
            continue
        if not _range_is_reviewed_coverage(start, end, coverage, layer):
            out.append(IGNORE_INDEX); previous_annotation_id = None
            counts["ignored_unreviewed_coverage"] += 1
            continue
        overlaps = [a for a in annotations if start < a["end_char"] and end > a["start_char"]]
        if not overlaps:
            out.append("O"); previous_annotation_id = None; counts["background_tokens"] += 1
            continue
        if len(overlaps) != 1 or start < overlaps[0]["start_char"] or end > overlaps[0]["end_char"]:
            out.append(IGNORE_INDEX); previous_annotation_id = None
            counts["ignored_boundary_token"] += 1
            continue
        ann = overlaps[0]
        prefix = "I" if previous_annotation_id == ann["annotation_id"] else "B"
        out.append(f"{prefix}-{ann['label']}")
        previous_annotation_id = ann["annotation_id"]
        counts["positive_tokens"] += 1
    return {"labels": out, "counts": counts, "label_inventory": sorted(names)}
