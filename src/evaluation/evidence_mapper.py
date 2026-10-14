"""Map statically parsed submission segments to question-paper questions.

This module only reads JSON and text. It never imports or executes submitted code.
It uses generic numbered-question anchors and document order; uncertain/unanchored
content is retained as unassigned evidence rather than guessed into a question.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "1.0"
QUESTION_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:question(?:\s+no\.?\s*)?\s*|q\s*[:#.-]?\s*)"
    r"(?P<number>\d{1,3})\s*(?:[:.)\-]\s*|\s+)(?P<title>.*\S)?\s*$",
    re.IGNORECASE,
)
NUMERIC_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?P<number>\d{1,3})[.)]\s+(?P<title>\S.*)$"
)
PAREN_QUESTION_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?\((?P<number>\d{1,3})\)\s+(?P<title>\S.*)$"
)
TEXT_KEYS = ("content", "text", "source_text", "page_text", "raw_text", "source", "markdown", "code")
OUTPUT_KEYS = ("output_text", "outputs_text", "rendered_output", "stdout")
CONTAINER_KEYS = ("segments", "cells", "pages", "items", "parsed_segments", "extracted_segments")


@dataclass
class Segment:
    segment_id: str
    position: int
    segment_type: str
    source_text: str
    output_text: str
    page_number: int | None
    metadata: dict[str, Any]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_text(item) for item in value]
        return "".join(parts)
    if isinstance(value, dict):
        # Common notebook output representation: {"text/plain": ["..."]}
        for key in ("text/plain", "text/html", "text/markdown", "text"):
            if key in value:
                rendered = _text(value[key])
                if rendered:
                    return rendered
        for key in TEXT_KEYS + OUTPUT_KEYS:
            if key in value:
                rendered = _text(value[key])
                if rendered:
                    return rendered
    return ""


def _output_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for output in value:
            if isinstance(output, dict):
                text = ""
                for key in OUTPUT_KEYS + ("text", "data", "content"):
                    if key in output:
                        text = _text(output[key])
                        if text:
                            break
                if text:
                    parts.append(text)
            else:
                text = _text(output)
                if text:
                    parts.append(text)
        return "\n".join(parts)
    if isinstance(value, dict):
        return _text(value)
    return ""


def _first_nonempty(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            value = record[key]
            if not isinstance(value, (str, list, dict)) or _text(value).strip():
                if isinstance(value, (str, list, dict)) and not _text(value).strip():
                    continue
                return value
    return None


def _segment_records(payload: Any) -> list[dict[str, Any]]:
    """Read common parser schemas without recursively duplicating nested text."""
    if isinstance(payload, list):
        if all(isinstance(item, dict) for item in payload):
            return payload
        raise ValueError("Submission JSON list must contain segment/cell objects")
    if not isinstance(payload, dict):
        raise ValueError("Submission JSON root must be an object or a list of segment objects")

    for key in CONTAINER_KEYS:
        value = payload.get(key)
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            return value
        if isinstance(value, list) and not value:
            # Keep checking other candidates; some schemas include an empty `items` wrapper.
            continue
    # Single-page/single-cell fallback, but don't walk arbitrary nested dictionaries.
    if any(key in payload for key in TEXT_KEYS):
        return [payload]
    raise ValueError(
        "No recognized segment collection found. Expected one of: " + ", ".join(CONTAINER_KEYS)
    )


def extract_submission_segments(payload: Any, filename: str) -> list[Segment]:
    records = _segment_records(payload)
    segments: list[Segment] = []
    for index, record in enumerate(records):
        source_value = _first_nonempty(record, TEXT_KEYS)
        source_text = _text(source_value)
        # Some parser outputs place source under `data.source`.
        if not source_text and isinstance(record.get("data"), dict):
            source_text = _text(record["data"].get("source"))
        outputs_value = _first_nonempty(record, ("outputs", "output", "rendered_outputs"))
        out_text = _output_text(outputs_value)
        if not out_text:
            out_text = _text(_first_nonempty(record, OUTPUT_KEYS))
        if not source_text and not out_text:
            # Ignore metadata-only records. Keep only meaningful evidence.
            continue

        source_location = record.get("source_location")
        if not isinstance(source_location, dict):
            source_location = {}
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        page = source_location.get("page_number", record.get("page_number", metadata.get("page_number")))
        try:
            page_number = int(page) if page is not None else None
        except (TypeError, ValueError):
            page_number = None
        segment_id = str(record.get("segment_id") or record.get("cell_id") or record.get("id") or f"segment_{index + 1:04d}")
        try:
            position = int(record.get("position", record.get("cell_index", index)))
        except (TypeError, ValueError):
            position = index
        segment_type = str(
            record.get("original_type") or record.get("cell_type") or record.get("segment_type")
            or record.get("type") or record.get("semantic_role") or "unknown"
        )
        safe_meta = {
            "execution_count": record.get("execution_count"),
            "extraction_status": record.get("extraction_status"),
        }
        # Include stable source metadata, not arbitrary large/raw objects.
        if "attachments" in record:
            safe_meta["attachment_count"] = len(record.get("attachments") or []) if isinstance(record.get("attachments"), list) else None
        segments.append(Segment(segment_id, position, segment_type, source_text, out_text, page_number, safe_meta))
    if not segments:
        raise ValueError(f"No text-bearing segments found in submission: {filename}")
    return sorted(segments, key=lambda item: (item.position, item.segment_id))


def _question_inventory(question_map: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in question_map.get("questions", []) or []:
        if not isinstance(entry, dict):
            continue
        qid = entry.get("question_id") or entry.get("id")
        if not qid:
            continue
        number = entry.get("number", entry.get("question_number"))
        title = entry.get("title_text") or entry.get("heading_text")
        if not title:
            title = next((line.strip() for line in str(entry.get("text", "")).splitlines() if line.strip()), "")
        result.append({
            "question_id": str(qid),
            "number": str(number) if number is not None else None,
            "occurrence": entry.get("occurrence", 1),
            "title": str(title or ""),
            "source_kind": "question",
            "source_text": str(entry.get("text", "")),
            "source_pages": entry.get("source_pages", []),
        })
    for entry in question_map.get("standalone_tasks", []) or []:
        if not isinstance(entry, dict):
            continue
        qid = entry.get("task_id") or entry.get("question_id")
        if not qid:
            continue
        title = entry.get("title") or next((line.strip() for line in str(entry.get("text", "")).splitlines() if line.strip()), "")
        result.append({
            "question_id": str(qid),
            "number": entry.get("number"),
            "occurrence": entry.get("occurrence", 1),
            "title": str(title or ""),
            "source_kind": "standalone_task",
            "source_text": str(entry.get("text", "")),
            "source_pages": entry.get("source_pages", []),
        })
    return result


def _is_code_segment(segment_type: str) -> bool:
    """Return True for parser segment types that represent executable source code.

    Code cells must remain atomic evidence blocks: identifiers such as q1, comments,
    strings, and experiment names inside them are not document question headings.
    """
    normalized = re.sub(r"[^a-z0-9]+", "_", str(segment_type or "").casefold()).strip("_")
    return (
        "code" in normalized.split("_")
        or normalized in {"python", "ipython", "jupyter_code", "source_code", "program"}
    )


def _heading(line: str, valid_numbers: set[str]) -> tuple[str, str] | None:
    # Compact names such as `q1 = df.quantile(...)` are common in code and can
    # also appear in OCR text. They are assignments, not question headings.
    if re.match(r"^\s*(?:#\s*)?q\s*\d{1,3}\s*=", line, re.IGNORECASE):
        return None
    for pattern in (QUESTION_HEADING_RE, NUMERIC_HEADING_RE, PAREN_QUESTION_RE):
        match = pattern.match(line)
        if match:
            number = match.group("number")
            if number in valid_numbers:
                return number, (match.groupdict().get("title") or "").strip()
    return None


def _split_segment(segment: Segment, valid_numbers: set[str]) -> list[dict[str, Any]]:
    """Split prose/page segments at recognizable top-level question headings.

    Notebook/code segments are intentionally kept whole. Headings are document
    structure; strings and comments inside code are evidence, not structure.
    """
    lines = segment.source_text.splitlines(keepends=True)
    if not lines:
        lines = [segment.source_text]
    boundaries: list[tuple[int, str, str]] = []
    offset = 0
    for line in lines:
        match = None if _is_code_segment(segment.segment_type) else _heading(line.rstrip("\r\n"), valid_numbers)
        if match:
            boundaries.append((offset, match[0], match[1]))
        offset += len(line)

    start_offsets = [item[0] for item in boundaries]
    cuts = sorted(set(start_offsets))
    # Text before the first recognized heading stays as a preamble block.
    ranges: list[tuple[int, int, str | None, str]] = []
    if not cuts:
        ranges.append((0, len(segment.source_text), None, ""))
    else:
        if cuts[0] > 0:
            ranges.append((0, cuts[0], None, ""))
        for idx, start in enumerate(cuts):
            end = cuts[idx + 1] if idx + 1 < len(cuts) else len(segment.source_text)
            _, number, title = next(item for item in boundaries if item[0] == start)
            ranges.append((start, end, number, title))
    blocks: list[dict[str, Any]] = []
    for subidx, (start, end, number, title) in enumerate(ranges, 1):
        text = segment.source_text[start:end]
        if not text.strip() and not segment.output_text.strip():
            continue
        blocks.append({
            "block_id": f"{segment.segment_id}_block_{subidx:03d}",
            "segment_id": segment.segment_id,
            "segment_position": segment.position,
            "segment_type": segment.segment_type,
            "source_text": text,
            # Outputs are kept as evidence, but never considered a question-heading source.
            "output_text": segment.output_text,
            "page_number": segment.page_number,
            "segment_start_offset": start,
            "segment_end_offset": end,
            "explicit_question_number": number,
            "heading_title": title,
            "metadata": segment.metadata,
        })
    return blocks


def _requirement_index(requirements: dict[str, Any], valid_qids: set[str]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_question: dict[str, list[dict[str, Any]]] = {qid: [] for qid in valid_qids}
    globals_out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    # The extractor's flat `requirements` array includes global requirements too;
    # `global_requirements` is a hierarchy-specific duplicate representation.
    # Consume the flat list first, then add only previously unseen IDs from the nested list.
    for key in ("requirements", "global_requirements"):
        values = requirements.get(key, []) or []
        if not isinstance(values, list):
            continue
        for record in values:
            if not isinstance(record, dict):
                continue
            rid = record.get("requirement_id")
            if rid is not None and str(rid) in seen_ids:
                continue
            if rid is not None:
                seen_ids.add(str(rid))
            # Preserve the full requirement record so later evaluation can use marks,
            # dependencies, extraction metadata, alternatives, and provenance.
            selected = dict(record)
            qid = record.get("question_id")
            if qid is None or str(qid) not in by_question:
                # Standalone tasks may have no `question_id`; their source node is the task ID.
                source_node_id = record.get("source_node_id")
                parent_id = record.get("parent_id")
                fallback_qid = source_node_id if str(source_node_id) in by_question else parent_id
                if fallback_qid is not None and str(fallback_qid) in by_question:
                    by_question[str(fallback_qid)].append(selected)
                elif key == "global_requirements" or record.get("source_category") == "global_instruction":
                    globals_out.append(selected)
            else:
                by_question[str(qid)].append(selected)
    return by_question, globals_out


def build_evidence_map(question_map: dict[str, Any], requirements: dict[str, Any], submission_payload: Any, submission_filename: str) -> dict[str, Any]:
    q_source = Path(str(question_map.get("source_filename", ""))).name.casefold()
    r_source = Path(str(requirements.get("source_filename", ""))).name.casefold()
    if q_source and r_source and q_source != r_source:
        raise ValueError(f"Question map source {q_source!r} does not match requirements source {r_source!r}")
    inventory = _question_inventory(question_map)
    if not inventory:
        raise ValueError("Question map contains no usable questions or standalone tasks")
    ids = [q["question_id"] for q in inventory]
    if len(ids) != len(set(ids)):
        raise ValueError("Question map contains duplicate question/task IDs")
    valid_numbers = {q["number"] for q in inventory if q["number"] is not None and q["number"].isdigit()}
    segments = extract_submission_segments(submission_payload, submission_filename)
    all_blocks: list[dict[str, Any]] = []
    for seg in segments:
        all_blocks.extend(_split_segment(seg, valid_numbers))

    by_number: dict[str, list[int]] = {}
    for index, question in enumerate(inventory):
        if question["number"] is not None:
            by_number.setdefault(str(question["number"]), []).append(index)
    assigned_blocks: dict[str, list[dict[str, Any]]] = {qid: [] for qid in ids}
    unassigned_blocks: list[dict[str, Any]] = []
    current_index: int | None = None
    last_explicit_index = -1
    occurrence_cursor: dict[str, int] = {}
    explicit_ambiguous: set[str] = set()

    for block in all_blocks:
        number = block.pop("explicit_question_number")
        title = block.pop("heading_title")
        if number is not None:
            choices = by_number.get(number, [])
            forward = [idx for idx in choices if idx > last_explicit_index]
            if forward:
                selected_index = forward[0]
            elif choices:
                # Repeated question numbering: do not silently claim certainty.
                selected_index = choices[min(occurrence_cursor.get(number, 0), len(choices) - 1)]
                explicit_ambiguous.add(inventory[selected_index]["question_id"])
            else:
                selected_index = None
            if selected_index is None:
                block["mapping_method"] = "unrecognized_question_number"
                block["mapping_confidence"] = 0.0
                block["mapping_status"] = "unassigned"
                block["heading_number"] = number
                block["heading_title"] = title
                unassigned_blocks.append(block)
                continue
            current_index = selected_index
            last_explicit_index = max(last_explicit_index, selected_index)
            occurrence_cursor[number] = occurrence_cursor.get(number, 0) + 1
            q = inventory[selected_index]
            block["mapping_method"] = "explicit_question_heading"
            block["mapping_confidence"] = 1.0 if len(choices) == 1 else 0.75
            block["mapping_status"] = "mapped" if len(choices) == 1 else "mapped_repeated_number_review"
            block["heading_number"] = number
            block["heading_title"] = title
            assigned_blocks[q["question_id"]].append(block)
        elif current_index is not None:
            q = inventory[current_index]
            block["mapping_method"] = "follows_explicit_question_heading"
            block["mapping_confidence"] = 0.75
            block["mapping_status"] = "mapped_by_document_order"
            block["heading_number"] = None
            block["heading_title"] = ""
            assigned_blocks[q["question_id"]].append(block)
        else:
            block["mapping_method"] = "no_question_anchor"
            block["mapping_confidence"] = 0.0
            block["mapping_status"] = "unassigned"
            block["heading_number"] = None
            block["heading_title"] = ""
            unassigned_blocks.append(block)

    by_question, globals_out = _requirement_index(requirements, set(ids))
    mapped_questions: list[dict[str, Any]] = []
    for question in inventory:
        qid = question["question_id"]
        blocks = assigned_blocks[qid]
        ambiguous = qid in explicit_ambiguous or any(b["mapping_status"] == "mapped_repeated_number_review" for b in blocks)
        mapped_questions.append({
            "question_id": qid,
            "question_number": question["number"],
            "occurrence": question["occurrence"],
            "question_kind": question["source_kind"],
            "question_title": question["title"],
            "question_source_pages": question["source_pages"],
            "mapping_status": "ambiguous_repeated_number" if ambiguous else ("evidence_found" if blocks else "no_evidence_found"),
            "evidence_block_count": len(blocks),
            "evidence_blocks": blocks,
            "requirements": by_question[qid],
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "student_question_evidence_map",
        "mapping_method": "generic_explicit_question_anchors_plus_document_order",
        "student_code_executed": False,
        "source_submission_filename": submission_filename,
        "source_question_map": question_map.get("source_filename"),
        "source_requirements_filename": requirements.get("source_filename"),
        "question_map_schema_version": question_map.get("schema_version"),
        "requirements_schema_version": requirements.get("schema_version"),
        "questions": mapped_questions,
        "global_requirements": globals_out,
        "unassigned_evidence_blocks": unassigned_blocks,
        "summary": {
            "question_or_task_count": len(inventory),
            "submission_segments": len(segments),
            "evidence_blocks": len(all_blocks),
            "mapped_blocks": sum(len(items) for items in assigned_blocks.values()),
            "unassigned_blocks": len(unassigned_blocks),
            "questions_with_evidence": sum(bool(items) for items in assigned_blocks.values()),
            "questions_without_evidence": sum(not items for items in assigned_blocks.values()),
            "student_code_executed": False,
        },
    }


def build_from_paths(question_map_path: Path, requirements_path: Path, submission_path: Path) -> dict[str, Any]:
    question_map = json.loads(question_map_path.read_text(encoding="utf-8"))
    requirements = json.loads(requirements_path.read_text(encoding="utf-8"))
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    return build_evidence_map(question_map, requirements, submission, submission_path.name)
