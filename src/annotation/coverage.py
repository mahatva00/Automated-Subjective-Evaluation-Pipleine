"""Coverage ledgers that distinguish reviewed background from unreviewed text.

The ledger is a sidecar: source annotation JSONs and their schema remain unchanged.
"""
from __future__ import annotations

import hashlib
import datetime as _datetime
from typing import Any

from src.annotation.scope import validate_scope_ledger, coverage_chunk_requires_review, classify_span_scope

COVERAGE_VERSION = "1.0.0"
VALID_STATES = {"pending", "reviewed"}
LAYERS = ("structure", "semantic")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _choose_end(text: str, start: int, max_chars: int) -> int:
    limit = min(start + max_chars, len(text))
    if limit >= len(text):
        return len(text)
    # Prefer a newline near the end of the window, but never create a tiny chunk
    # merely because the current paragraph begins with one or two newlines.
    floor = start + max(1, max_chars // 2)
    newline = text.rfind("\n", floor, limit)
    if newline >= floor:
        return newline + 1
    return limit


def build_coverage(doc: dict[str, Any], annotation_filename: str, max_chars: int = 900) -> dict[str, Any]:
    if not isinstance(doc, dict) or not isinstance(doc.get("text"), str):
        raise ValueError("annotation document must contain a text string")
    text = doc["text"]
    if not text:
        raise ValueError("source text is empty")
    if not isinstance(doc.get("document_id"), str) or not doc["document_id"].strip():
        raise ValueError("document_id is missing")
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")

    chunks = []
    start = 0
    index = 1
    while start < len(text):
        end = _choose_end(text, start, max_chars)
        if end <= start:
            raise ValueError("chunking failed to advance")
        piece = text[start:end]
        page_numbers = []
        for page_range in doc.get("page_ranges", []):
            if not isinstance(page_range, dict):
                continue
            pstart, pend = page_range.get("start_char"), page_range.get("end_char")
            if isinstance(pstart, int) and isinstance(pend, int) and start < pend and end > pstart:
                page_number = page_range.get("page_number")
                if page_number is not None and page_number not in page_numbers:
                    page_numbers.append(page_number)
        chunks.append({
            "chunk_id": f"chunk_{index:04d}",
            "start_char": start,
            "end_char": end,
            "page_numbers": page_numbers,
            "text_sha256": sha256_text(piece),
            "text": piece,
            "structure_status": "pending",
            "semantic_status": "pending",
            "structure_reviewer": "",
            "structure_review_date": "",
            "semantic_reviewer": "",
            "semantic_review_date": "",
            "review_notes": "",
        })
        start = end
        index += 1

    return {
        "coverage_version": COVERAGE_VERSION,
        "document_id": doc["document_id"],
        "annotation_filename": annotation_filename,
        "source_text_sha256": sha256_text(text),
        "text_length": len(text),
        "chunking": {"strategy": "newline-aware-fixed-window-v1", "max_chars": max_chars},
        "policy": "A layer may be marked reviewed only after every character in the chunk has been checked for spans in that layer. No span in a fully reviewed region means reviewed background, not an unreviewed negative.",
        "chunks": chunks,
    }


def validate_coverage(doc: dict[str, Any], coverage: dict[str, Any], scope: dict[str, Any] | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(doc, dict) or not isinstance(doc.get("text"), str):
        return ["annotation document has no text string"]
    if not isinstance(coverage, dict):
        return ["coverage ledger must be a JSON object"]
    text = doc["text"]
    if scope is not None:
        scope_errors = validate_scope_ledger(doc, scope)
        errors.extend(f"scope: {e}" for e in scope_errors)
    if coverage.get("coverage_version") != COVERAGE_VERSION:
        errors.append(f"coverage_version must be {COVERAGE_VERSION!r}")
    if coverage.get("document_id") != doc.get("document_id"):
        errors.append("coverage document_id does not match annotation document")
    if coverage.get("source_text_sha256") != sha256_text(text):
        errors.append("source_text_sha256 mismatch; source text may have changed")
    if coverage.get("text_length") != len(text):
        errors.append("text_length mismatch")
    chunks = coverage.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        errors.append("chunks must be a non-empty array")
        return errors

    cursor = 0
    seen_ids: set[str] = set()
    for i, chunk in enumerate(chunks):
        where = f"chunks[{i}]"
        if not isinstance(chunk, dict):
            errors.append(f"{where} must be an object")
            continue
        cid = chunk.get("chunk_id")
        if not isinstance(cid, str) or not cid:
            errors.append(f"{where}.chunk_id must be a non-empty string")
        elif cid in seen_ids:
            errors.append(f"duplicate chunk_id: {cid}")
        else:
            seen_ids.add(cid)
        if not isinstance(chunk.get("page_numbers"), list):
            errors.append(f"{where}.page_numbers must be an array")
        if not isinstance(chunk.get("review_notes"), str):
            errors.append(f"{where}.review_notes must be a string")
        for metadata_field in ("structure_reviewer", "structure_review_date", "semantic_reviewer", "semantic_review_date"):
            if not isinstance(chunk.get(metadata_field), str):
                errors.append(f"{where}.{metadata_field} must be a string")
        start, end = chunk.get("start_char"), chunk.get("end_char")
        if not isinstance(start, int) or not isinstance(end, int):
            errors.append(f"{where}.start_char/end_char must be integers")
            continue
        if start != cursor:
            errors.append(f"{where} does not continue coverage at char {cursor}; starts at {start}")
        if start < 0 or end <= start or end > len(text):
            errors.append(f"{where} has invalid char range {start}:{end}")
            continue
        piece = text[start:end]
        if chunk.get("text") != piece:
            errors.append(f"{where}.text does not exactly match source text slice")
        if chunk.get("text_sha256") != sha256_text(piece):
            errors.append(f"{where}.text_sha256 mismatch")
        for layer in LAYERS:
            state = chunk.get(f"{layer}_status")
            if state not in VALID_STATES:
                errors.append(f"{where}.{layer}_status must be 'pending' or 'reviewed'")
            elif state == "reviewed":
                reviewer = chunk.get(f"{layer}_reviewer")
                review_date = chunk.get(f"{layer}_review_date")
                if not isinstance(reviewer, str) or not reviewer.strip():
                    errors.append(f"{where}.{layer}_reviewer is required when layer status is reviewed")
                if not isinstance(review_date, str) or not review_date.strip():
                    errors.append(f"{where}.{layer}_review_date is required when layer status is reviewed")
                else:
                    try:
                        _datetime.date.fromisoformat(review_date)
                    except ValueError:
                        errors.append(f"{where}.{layer}_review_date must be an ISO date YYYY-MM-DD")
        cursor = end
    if cursor != len(text):
        errors.append(f"coverage ends at char {cursor}, but source text length is {len(text)}")

    metadata = doc.get("annotation_metadata", {})
    doc_status = metadata.get("status") if isinstance(metadata, dict) else None
    required_chunks = [
        chunk for chunk in chunks if isinstance(chunk, dict)
        and coverage_chunk_requires_review(chunk, scope, text)
    ]
    all_reviewed = bool(required_chunks) and not errors and all(
        chunk.get("structure_status") == "reviewed" and chunk.get("semantic_status") == "reviewed"
        for chunk in required_chunks
    )
    if doc_status in {"reviewed", "adjudicated"} and not all_reviewed:
        scope_note = " in-scope question text" if scope is not None else " source text"
        errors.append(f"document status is {doc_status!r}, but full structure and semantic coverage of{scope_note} is not reviewed")

    # A reviewed annotation must be supported by a reviewed chunk in its own layer.
    annotations = doc.get("annotations", [])
    if isinstance(annotations, list):
        for i, ann in enumerate(annotations):
            if not isinstance(ann, dict) or ann.get("review_status") == "unreviewed":
                continue
            layer = ann.get("layer")
            start, end = ann.get("start_char"), ann.get("end_char")
            if layer not in LAYERS or not isinstance(start, int) or not isinstance(end, int):
                continue
            # Canonical drafts may contain legacy spans on global instructions. When a
            # question-scope ledger is supplied, these excluded spans do not create review
            # obligations or downstream labels. They are filtered in the exported copy.
            if scope is not None:
                scope_state, _ = classify_span_scope(start, end, scope, text)
                if scope_state != "in_scope":
                    continue
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                cstart, cend = chunk.get("start_char"), chunk.get("end_char")
                if isinstance(cstart, int) and isinstance(cend, int) and start < cend and end > cstart:
                    if chunk.get(f"{layer}_status") != "reviewed":
                        errors.append(f"annotations[{i}] is marked reviewed but overlaps a pending {layer} coverage chunk {chunk.get('chunk_id')}")
                        break
    return errors


def coverage_summary(coverage: dict[str, Any], scope: dict[str, Any] | None = None, source_text: str | None = None) -> dict[str, Any]:
    chunks = coverage.get("chunks", []) if isinstance(coverage, dict) else []
    all_total = len(chunks)
    if scope is not None and isinstance(source_text, str):
        eligible = [c for c in chunks if isinstance(c, dict) and coverage_chunk_requires_review(c, scope, source_text)]
    else:
        eligible = [c for c in chunks if isinstance(c, dict)]
    total = len(eligible)
    def count(layer: str) -> int:
        return sum(1 for c in eligible if c.get(f"{layer}_status") == "reviewed")
    structure = count("structure")
    semantic = count("semantic")
    return {
        "document_id": coverage.get("document_id") if isinstance(coverage, dict) else None,
        "total_chunks": all_total,
        "chunks_requiring_review": total,
        "scope_aware_coverage": scope is not None,
        "structure_reviewed_chunks": structure,
        "structure_coverage_percent": round(100 * structure / total, 2) if total else 0.0,
        "semantic_reviewed_chunks": semantic,
        "semantic_coverage_percent": round(100 * semantic / total, 2) if total else 0.0,
        "complete_both_layers": bool(total and structure == total and semantic == total),
    }
