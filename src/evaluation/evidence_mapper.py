"""Map statically parsed submission segments to question-paper questions.

This module only reads JSON and text. It never imports or executes submitted code.
It uses generic numbered-question anchors, notebook document order, and page-local
PDF continuation in document order; content before the first question anchor remains
unassigned.
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
NUMBERED_SUBPART_HEADING_RE = re.compile(
    r"^\s*(?P<number>\d{1,3})\s+(?P<subpart>[a-z])\s*$",
    re.IGNORECASE,
)
SUBPART_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:\((?P<paren>[a-z])\)|(?P<plain>[a-z])[.)])\s+"
)
MAPPING_STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "by", "for", "from", "give",
    "in", "into", "of", "on", "or", "the", "to", "using", "with",
    "write", "code", "also", "call", "this", "that", "all", "each",
}
MAPPING_ALIASES = {
    "dimension": {"shape", "dimensions"},
    "records": {"head", "tail", "rows", "record"},
    "record": {"head", "tail", "rows", "records"},
    "datatype": {"dtypes", "dtype", "info"},
    "columns": {"column", "cols", "dtypes"},
    "statistical": {"mean", "median", "var", "std", "min", "max", "agg", "describe"},
    "summary": {"info", "describe", "summary", "agg"},
    "missing": {"isna", "isnull", "null", "na"},
    "values": {"sum", "count", "value"},
    "dataset": {"df", "data", "dataset"},
    "create": {"copy", "dataframe", "array"},
    "size": {"shape", "len", "rows", "cols"},
    "zero": {"zeros", "iat", "fill", "0"},
    "index": {"iat", "iloc", "index", "for"},
    "chart": {"bar", "plot", "figure"},
    "draw": {"bar", "plot", "figure"},
    "independent": {"x1", "x2", "features", "input"},
    "variable": {"x1", "x2", "features"},
    "partition": {"split", "train_test_split", "train", "test"},
    "ratio": {"split", "train", "test"},
    "training": {"train", "fit", "model"},
    "trainingdata": {"train", "fit", "model"},
    "test": {"test", "validation", "predict"},
    "error": {"error", "mse", "mean_squared_error", "loss"},
    "predict": {"predict", "prediction", "predictor"},
    "target": {"target", "y", "predict"},
    "plot": {"plot", "figure", "xlabel", "ylabel", "legend"},
    "label": {"xlabel", "ylabel", "legend", "label", "set_xlabel", "set_ylabel"},
    "axes": {"xlabel", "ylabel", "set_xlabel", "set_ylabel"},
    "axis": {"xlabel", "ylabel", "set_xlabel", "set_ylabel"},
    "legend": {"legend", "label"},
    "swap": {"swap", "train", "test", "iloc", "x_train", "x_test", "y_train", "y_test"},
    "unknown": {"x_unknown", "unknown_input", "new_input"},
    "input": {"x_unknown", "unknown_input", "new_input"},
    "experiment": {"e1", "e2", "e3", "e4", "model"},
    "linear": {"linearregression", "linear_regression"},
}
MAPPING_GENERIC_TOKENS = {
    "data", "dataset", "train", "training", "test", "model", "error",
    "experiment", "output", "input", "target", "unknown", "number",
    "numbers", "sample", "samples", "result", "results", "set", "same",
    "learning", "due", "compute", "computed",
}
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
            "extraction_method": record.get("extraction_method", metadata.get("extraction_method")),
            "canonical_text": record.get("canonical_text", metadata.get("canonical_text", True)),
            "image_evidence": metadata.get("image_evidence", []),
            "embedded_image_count": record.get(
                "embedded_image_count", metadata.get("embedded_image_count")
            ),
            "extraction_diagnostics": record.get("diagnostics", []),
        }
        if page_number is not None:
            safe_meta["page_number"] = page_number
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
            "subpart_labels": {
                str(part.get("label") or "").casefold()
                for part in entry.get("subparts", []) or []
                if isinstance(part, dict) and part.get("label") is not None
            },
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


def _heading(
    line: str,
    valid_numbers: set[str],
    question_titles: dict[str, str] | None = None,
    question_subparts: dict[str, set[str]] | None = None,
) -> tuple[str, str] | None:
    # Compact names such as `q1 = df.quantile(...)` are common in code and can
    # also appear in OCR text. They are assignments, not question headings.
    if re.match(r"^\s*(?:#\s*)?q\s*\d{1,3}\s*=", line, re.IGNORECASE):
        return None
    for pattern in (QUESTION_HEADING_RE, NUMERIC_HEADING_RE, PAREN_QUESTION_RE):
        match = pattern.match(line)
        if match:
            number = match.group("number")
            if number in valid_numbers:
                title = (match.groupdict().get("title") or "").strip()
                if pattern is not QUESTION_HEADING_RE and question_titles:
                    expected = re.sub(
                        r"[^a-z0-9]+", " ", question_titles.get(number, "").casefold()
                    ).strip()
                    actual = re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()
                    expected_tokens = set(expected.split())
                    actual_tokens = set(actual.split())
                    if expected_tokens and not (
                        len(expected_tokens & actual_tokens)
                        >= max(1, min(2, len(expected_tokens)))
                    ):
                        continue
                return number, title
    numbered_subpart = NUMBERED_SUBPART_HEADING_RE.fullmatch(line)
    if numbered_subpart:
        number = numbered_subpart.group("number")
        subpart = numbered_subpart.group("subpart").casefold()
        if (
            number in valid_numbers
            and subpart in (question_subparts or {}).get(number, set())
        ):
            return number, f"subquestion {subpart}"
    compact = re.match(
        r"^\s*(?P<number>\d{1,3})[.)\-:]\s*(?P<title>[A-Za-z][^:]{2,}?)[\s:]*$",
        line,
    )
    if compact and compact.group("number") in valid_numbers and question_titles:
        number = compact.group("number")
        title = compact.group("title").strip()
        expected = re.sub(r"[^a-z0-9]+", " ", question_titles.get(number, "").casefold()).strip()
        actual = re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()
        expected_tokens = set(expected.split())
        actual_tokens = set(actual.split())
        if expected_tokens and len(expected_tokens & actual_tokens) >= max(1, min(2, len(expected_tokens))):
            return number, title
    return None


def _split_segment(
    segment: Segment,
    valid_numbers: set[str],
    question_titles: dict[str, str] | None = None,
    question_subparts: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
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
        match = None if _is_code_segment(segment.segment_type) else _heading(
            line.rstrip("\r\n"), valid_numbers, question_titles, question_subparts
        )
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


def _notebook_component_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the notebook parser's explicit component blocks without reclassifying text."""
    question_blocks = payload.get("question_blocks")
    if not isinstance(question_blocks, dict):
        return []
    result: list[dict[str, Any]] = []
    for number, blocks in question_blocks.items():
        if not isinstance(blocks, list):
            continue
        current_anchor: str | None = None
        for index, item in enumerate(blocks):
            if not isinstance(item, dict):
                continue
            component = str(item.get("component") or "")
            if component not in {"question", "question_copy", "code", "output", "explanation", "ai_log"}:
                continue
            source_text = str(item.get("source_text") or "")
            output_text = str(item.get("output_text") or "")
            if not source_text.strip() and not output_text.strip():
                continue
            cell_index = item.get("source_location", {}).get("cell_index")
            try:
                position = int(cell_index)
            except (TypeError, ValueError):
                position = index
            normalized = {
                "block_id": str(item.get("block_id") or f"cell_{position:04d}_{component}"),
                "segment_id": str(item.get("segment_id") or f"cell_{position:04d}"),
                "segment_position": position,
                "segment_type": component,
                "component": component,
                "source_text": source_text,
                "output_text": output_text,
                "page_number": None,
                "metadata": {
                    "question_number": str(number),
                    "component": component,
                    "source_location": item.get("source_location", {}),
                },
                "explicit_question_number": str(number),
                "heading_title": "",
            }
            inferred_anchor = item.get("subquestion_anchor")
            if inferred_anchor is None and component == "question_copy":
                inferred_anchor = _block_anchor(item)
            if inferred_anchor:
                current_anchor = str(inferred_anchor).casefold()
            if current_anchor:
                normalized["subquestion_anchor"] = current_anchor
            for key in ("subquestion_anchor", "next_subquestion_anchor"):
                if item.get(key) is not None:
                    normalized[key] = str(item[key]).casefold()
            result.append(normalized)
    return sorted(result, key=lambda block: (block["segment_position"], block["block_id"]))


def _component_fields(blocks: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    components = {name: [] for name in ("question", "question_copy", "code", "output", "explanation", "ai_log")}
    for block in blocks:
        component = block.get("component")
        if component in components:
            components[component].append(block)
    missing = [
        component for component in ("code", "output", "explanation")
        if not components[component]
    ]
    return components, missing


def _requirement_anchor(requirement: dict[str, Any]) -> str | None:
    source_node = str(requirement.get("source_node_id") or "")
    match = re.search(r"(?:^|_)part_([a-z])(?:_|$)", source_node, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    text = str(requirement.get("source_text") or "")
    match = SUBPART_RE.search(text)
    return (match.group("paren") or match.group("plain")).lower() if match else None


def _block_anchor(block: dict[str, Any]) -> str | None:
    explicit_anchor = block.get("subquestion_anchor")
    if explicit_anchor:
        return str(explicit_anchor).lower()
    text = str(block.get("source_text") or "")
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    match = SUBPART_RE.match(first_line)
    return (match.group("paren") or match.group("plain")).lower() if match else None


def _same_cell(block: dict[str, Any]) -> tuple[Any, ...]:
    metadata = block.get("metadata")
    execution_count = metadata.get("execution_count") if isinstance(metadata, dict) else None
    if execution_count is not None:
        return ("execution", block.get("segment_id"), execution_count)
    block_id = re.sub(
        r"_(?:output|explanation|ai_log)$",
        "",
        str(block.get("block_id") or ""),
    )
    return ("block", block.get("segment_id"), block_id)


def _requirement_context_bundle(
    requirement: dict[str, Any],
    blocks: list[dict[str, Any]],
    linked_ids: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select non-AI context within the requirement's subquestion boundary."""
    anchor = _requirement_anchor(requirement)
    usable = [
        block for block in blocks
        if str(block.get("component") or "") not in {"question", "question_copy", "ai_log"}
    ]
    anchored = [
        block for block in usable
        if anchor and _block_anchor(block) == anchor
    ]
    if anchored:
        candidates = anchored
        basis = "same_subquestion_anchor"
    elif linked_ids:
        linked = [
            block for block in usable
            if str(block.get("block_id") or "") in set(linked_ids)
        ]
        identities = {_same_cell(block) for block in linked}
        candidates = [
            block for block in usable
            if _same_cell(block) in identities
        ]
        basis = "same_execution_cell"
    else:
        candidates = []
        basis = "no_context"

    ordered = sorted(
        candidates,
        key=lambda block: (
            int(block.get("segment_position") or 0),
            str(block.get("block_id") or ""),
        ),
    )
    component_counts = {
        component: sum(
            1 for block in ordered
            if str(block.get("component") or "") == component
        )
        for component in ("code", "output", "explanation")
    }
    if component_counts["code"]:
        category = "mapped_code" if linked_ids else "contextual_evidence"
    elif component_counts["explanation"] and component_counts["output"]:
        category = "explanation_only"
    elif component_counts["explanation"]:
        category = "explanation_only"
    elif component_counts["output"]:
        category = "output_only"
    else:
        category = "true_no_evidence"
    bundle = {
        "category": category,
        "selection_basis": basis,
        "subquestion_anchor": anchor,
        "block_ids": [str(block.get("block_id") or "") for block in ordered],
        "component_counts": component_counts,
    }
    return bundle, ordered


def _mapping_tokens(text: str) -> set[str]:
    raw_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text)
    tokens = {
        value
        for token in raw_tokens
        for value in (
            token.casefold(),
            *(
                part.casefold()
                for part in re.split(r"_+", token)
                if part
            ),
        )
    }
    return {token for token in tokens if token not in MAPPING_STOPWORDS and len(token) > 1}


def _requirement_code_score(requirement: dict[str, Any], code: str) -> int:
    requirement_tokens = _mapping_tokens(
        str(requirement.get("atomic_text") or requirement.get("requirement_text") or "")
    )
    code_tokens = _mapping_tokens(code)
    specific_tokens = requirement_tokens - MAPPING_GENERIC_TOKENS
    expanded_specific = set(specific_tokens)
    for token in specific_tokens:
        expanded_specific.update(MAPPING_ALIASES.get(token, set()))
    specific_matches = expanded_specific & code_tokens
    if specific_matches:
        return len(specific_matches) * 2

    # Generic words are useful only when the code contains a distinctive
    # compound operation such as mean_squared_error, train_test_split, or
    # a named plotting/model API. This prevents an entire experiment's cells
    # from being linked to every requirement mentioning train/test/model.
    expanded_generic = set(requirement_tokens & MAPPING_GENERIC_TOKENS)
    for token in tuple(expanded_generic):
        expanded_generic.update(MAPPING_ALIASES.get(token, set()))
    distinctive_generic_matches = {
        token for token in expanded_generic & code_tokens
        if "_" in token or token in {
            "mse", "predict", "prediction", "fit", "split", "plot",
            "figure", "xlabel", "ylabel", "legend",
        }
    }
    return len(distinctive_generic_matches)


def _link_requirement_evidence(
    requirements: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach evidence blocks to subquestion requirements without executing content."""
    if not requirements:
        return []
    groups: dict[str | None, list[dict[str, Any]]] = {}
    for requirement in requirements:
        groups.setdefault(_requirement_anchor(requirement), []).append(requirement)
    ordered_anchors = list(groups)
    current_anchor: str | None = ordered_anchors[0] if ordered_anchors else None
    linked: dict[str, list[str]] = {str(r.get("requirement_id") or ""): [] for r in requirements}
    confidence: dict[str, str] = {}
    ordered_blocks = sorted(
        blocks,
        key=lambda item: (int(item.get("segment_position") or 0), str(item.get("block_id") or "")),
    )
    def cell_group_key(block: dict[str, Any]) -> tuple[Any, ...]:
        metadata = block.get("metadata")
        execution_count = metadata.get("execution_count") if isinstance(metadata, dict) else None
        if execution_count is not None:
            return ("execution", block.get("segment_id"), execution_count)
        block_id = str(block.get("block_id") or "")
        block_id = re.sub(r"_(?:output|explanation|ai_log)$", "", block_id)
        return ("block", block.get("segment_id"), block_id)

    index = 0
    while index < len(ordered_blocks):
        group_key = cell_group_key(ordered_blocks[index])
        end = index + 1
        while end < len(ordered_blocks) and cell_group_key(ordered_blocks[end]) == group_key:
            end += 1
        group_blocks = ordered_blocks[index:end]
        explicit_anchor = next(
            (anchor for anchor in (_block_anchor(block) for block in group_blocks) if anchor in groups),
            None,
        )
        if explicit_anchor:
            current_anchor = explicit_anchor
        for block in group_blocks:
            component = str(block.get("component") or "")
            if component in {"question", "question_copy"}:
                block["requirement_ids"] = []
                block["requirement_link_status"] = "question_context"
                continue
            if current_anchor not in groups:
                block["requirement_ids"] = []
                block["requirement_link_status"] = "unlinked"
                continue
            if component != "code":
                block["requirement_ids"] = []
                block["requirement_link_status"] = "component_presence_only"
                continue
            candidates = groups[current_anchor]
            scored = [
                (requirement, _requirement_code_score(requirement, str(block.get("source_text") or "")))
                for requirement in candidates
            ]
            matched = [
                requirement for requirement, score in scored
                if score > 0
            ]
            requirement_ids = [
                str(requirement.get("requirement_id") or "")
                for requirement in matched
            ]
            block["requirement_ids"] = requirement_ids
            block["requirement_link_status"] = "atomic_code_match" if requirement_ids else "no_atomic_code_match"
            block["requirement_link_confidence"] = 0.8 if requirement_ids else 0.0
            for rid in requirement_ids:
                linked[rid].append(str(block.get("block_id") or ""))
                confidence[rid] = "atomic_code_match"
        next_anchor = next(
            (
                str(block.get("next_subquestion_anchor") or "").lower()
                for block in reversed(group_blocks)
                if block.get("next_subquestion_anchor") in groups
            ),
            None,
        )
        if next_anchor:
            current_anchor = next_anchor
        index = end
    output = []
    for requirement in requirements:
        item = dict(requirement)
        rid = str(requirement.get("requirement_id") or "")
        linked_ids = linked.get(rid, [])
        evidence_bundle, _ = _requirement_context_bundle(item, blocks, linked_ids)
        item["linked_code_block_ids"] = linked_ids
        item["linked_evidence_block_ids"] = linked_ids
        link_status = confidence.get(rid, "no_linked_evidence")
        item["evidence_link_status"] = link_status
        item["evidence_link_confidence"] = 1.0 if linked_ids else 0.0
        item["evidence_bundle"] = evidence_bundle
        item["mapping_diagnostics"] = {
            "category": evidence_bundle["category"],
            "linked_code_block_count": len(linked_ids),
            "question_block_count": len(blocks),
            "question_components": {
                component: sum(
                    1 for block in blocks
                    if str(block.get("component") or "") == component
                )
                for component in ("code", "output", "explanation", "ai_log")
            },
            "evidence_link_status": link_status,
            "context_block_ids": evidence_bundle["block_ids"],
            "context_component_counts": evidence_bundle["component_counts"],
            "context_selection_basis": evidence_bundle["selection_basis"],
        }
        output.append(item)
    return output


def _pdf_component_blocks(block: dict[str, Any]) -> list[dict[str, Any]]:
    """Split notebook-export PDF text into explicit code/output/prose blocks."""
    text = str(block.get("source_text") or "")
    if not re.search(r"^\s*In\s*\[\s*\d+\s*\]\s*:?", text, re.IGNORECASE | re.MULTILINE):
        if (
            str(block.get("segment_type") or "") == "pdf_page"
            and (
                SUBPART_RE.search(text)
                or re.search(r"(?im)^\s*(?:no\s+ai\b|ai\s+prompt\b|chatgpt\b|copilot\b)", text)
            )
        ):
            return _pdf_prose_blocks(block)
        return [block]

    cell_starts = list(re.finditer(
        r"^\s*In\s*\[\s*(\d+)\s*\]\s*:?",
        text,
        re.IGNORECASE | re.MULTILINE,
    ))
    result: list[dict[str, Any]] = []
    for index, start in enumerate(cell_starts):
        end = cell_starts[index + 1].start() if index + 1 < len(cell_starts) else len(text)
        cell_text = text[start.end():end].strip()
        if not cell_text:
            continue
        preceding_text = text[:start.start()]
        preceding_subparts = list(SUBPART_RE.finditer(preceding_text))
        subquestion_anchor = None
        if preceding_subparts:
            match = preceding_subparts[-1]
            subquestion_anchor = (match.group("paren") or match.group("plain")).lower()
        output_match = re.search(
            r"^\s*Out\s*\[\s*\d+\s*\]\s*:?",
            cell_text,
            re.IGNORECASE | re.MULTILINE,
        )
        code_text = cell_text[:output_match.start()].strip() if output_match else cell_text
        output_text = cell_text[output_match.end():].strip() if output_match else ""
        explanation_text = ""
        ai_log_text = ""
        if output_text:
            lines = output_text.splitlines()
            ai_start = next(
                (index for index, line in enumerate(lines)
                 if re.search(r"\b(?:no\s+ai|ai\s+(?:prompt|tool|assistance)|chatgpt|copilot)\b", line, re.I)),
                None,
            )
            if ai_start is not None:
                ai_log_text = "\n".join(lines[ai_start:]).strip()
                output_text = "\n".join(lines[:ai_start]).strip()
            explanation_start = next(
                (index for index, line in enumerate(output_text.splitlines())
                 if re.search(r"^\s*(?:we\s+verify|this\s+code|explanation|approach|conclusion|because|therefore)\b", line, re.I)),
                None,
            )
            if explanation_start is not None:
                output_lines = output_text.splitlines()
                explanation_text = "\n".join(output_lines[explanation_start:]).strip()
                output_text = "\n".join(output_lines[:explanation_start]).strip()
        base = dict(block)
        base["block_id"] = f"{block['block_id']}_cell_{index + 1:03d}"
        base["segment_start_offset"] = start.start()
        base["segment_end_offset"] = end
        base["source_text"] = code_text
        base["output_text"] = output_text
        base["component"] = "code" if code_text else "output"
        base["metadata"] = dict(block.get("metadata") or {})
        base["metadata"].update({
            "pdf_notebook_export": True,
            "execution_count": int(cell_starts[index].group(1)),
        })
        if subquestion_anchor:
            base["subquestion_anchor"] = subquestion_anchor
        following_subparts = list(SUBPART_RE.finditer(cell_text))
        if following_subparts:
            match = following_subparts[-1]
            following_anchor = (match.group("paren") or match.group("plain")).lower()
            if following_anchor != subquestion_anchor:
                base["next_subquestion_anchor"] = following_anchor
        result.append(base)
        if output_text:
            output_block = dict(base)
            output_block["block_id"] = f"{base['block_id']}_output"
            output_block["source_text"] = ""
            output_block["output_text"] = output_text
            output_block["component"] = "output"
            result.append(output_block)
        if explanation_text:
            explanation_block = dict(base)
            explanation_block["block_id"] = f"{base['block_id']}_explanation"
            explanation_block["source_text"] = explanation_text
            explanation_block["output_text"] = ""
            explanation_block["component"] = "explanation"
            result.append(explanation_block)
        if ai_log_text:
            ai_block = dict(base)
            ai_block["block_id"] = f"{base['block_id']}_ai_log"
            ai_block["source_text"] = ai_log_text
            ai_block["output_text"] = ""
            ai_block["component"] = "ai_log"
            result.append(ai_block)
    return result or [block]


def _pdf_prose_blocks(block: dict[str, Any]) -> list[dict[str, Any]]:
    """Classify PDF prose/results that occur outside an ``In [n]`` cell."""
    text = str(block.get("source_text") or "").strip()
    if not text:
        return [block]
    boundaries = list(SUBPART_RE.finditer(text))
    ranges: list[tuple[int, int, str | None]] = []
    if boundaries:
        if boundaries[0].start() > 0:
            ranges.append((0, boundaries[0].start(), None))
        for index, match in enumerate(boundaries):
            end = boundaries[index + 1].start() if index + 1 < len(boundaries) else len(text)
            ranges.append((match.start(), end, (match.group("paren") or match.group("plain")).lower()))
    else:
        ranges = [(0, len(text), None)]

    result: list[dict[str, Any]] = []
    for index, (start, end, anchor) in enumerate(ranges, start=1):
        content = text[start:end].strip()
        if not content:
            continue
        chunks: list[tuple[str, str]] = []
        ai_match = re.search(r"(?im)^\s*(?:no\s+ai\b|ai\s+prompt\b|chatgpt\b|copilot\b)", content)
        if ai_match:
            if ai_match.start() > 0:
                chunks.append(("output", content[:ai_match.start()].strip()))
            chunks.append(("ai_log", content[ai_match.start():].strip()))
        else:
            looks_like_result = bool(re.search(
                r"(?im)^\s*(?:E[1-4]\s*:|(?:[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*$|"
                r"np\.float|dtype:\s*\w+|rangeindex:|data columns:)",
                content,
            ))
            chunks.append(("output" if looks_like_result else "explanation", content))
        for chunk_index, (component, chunk) in enumerate(chunks, start=1):
            child = dict(block)
            child["block_id"] = f"{block['block_id']}_prose_{index:03d}_{chunk_index:02d}"
            child["source_text"] = chunk if component != "output" else chunk
            child["output_text"] = ""
            child["component"] = component
            child["subquestion_anchor"] = anchor or block.get("subquestion_anchor")
            child["metadata"] = dict(block.get("metadata") or {})
            child["metadata"]["pdf_prose_component"] = True
            result.append(child)
    return result or [block] or [block]


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
    number_counts = {
        number: sum(1 for item in inventory if item.get("number") == number)
        for number in {q.get("number") for q in inventory}
    }
    question_titles = {
        str(q["number"]): str(q.get("title") or "")
        for q in inventory
        if q.get("number") is not None and number_counts.get(q.get("number")) == 1
    }
    question_subparts = {
        str(q["number"]): q["subpart_labels"]
        for q in inventory
        if q.get("number") is not None
        and number_counts.get(q.get("number")) == 1
        and q.get("subpart_labels")
    }
    notebook_blocks = (
        _notebook_component_blocks(submission_payload)
        if isinstance(submission_payload, dict)
        else []
    )
    if notebook_blocks:
        raw_segments = submission_payload.get("segments", [])
        segments = raw_segments if isinstance(raw_segments, list) else []
        all_blocks = notebook_blocks
    else:
        segments = extract_submission_segments(submission_payload, submission_filename)
        all_blocks = []
        for seg in segments:
            for block in _split_segment(
                seg, valid_numbers, question_titles, question_subparts
            ):
                all_blocks.extend(_pdf_component_blocks(block))

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
            is_subpart_anchor = title.startswith("subquestion ")
            if (notebook_blocks or is_subpart_anchor) and len(choices) == 1:
                selected_index = choices[0]
            else:
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
            if not is_subpart_anchor:
                occurrence_cursor[number] = occurrence_cursor.get(number, 0) + 1
            q = inventory[selected_index]
            block["mapping_method"] = (
                "explicit_question_component_number"
                if notebook_blocks
                else "explicit_question_subpart_anchor"
                if is_subpart_anchor
                else "explicit_question_heading"
            )
            block["mapping_confidence"] = 1.0 if len(choices) == 1 else 0.75
            block["mapping_status"] = (
                "mapped"
                if len(choices) == 1 or is_subpart_anchor
                else "mapped_repeated_number_review"
            )
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
        components, missing_components = _component_fields(blocks)
        bundle_summary = {
            component: {
                "block_count": len(components[component]),
                "block_ids": [str(block.get("block_id") or "") for block in components[component]],
            }
            for component in ("question", "question_copy", "code", "output", "explanation", "ai_log")
        }
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
            "component_blocks": components,
            "evidence_bundle": bundle_summary,
            "missing_components": missing_components,
            "ai_log_present": bool(components["ai_log"]),
            "requirements": _link_requirement_evidence(by_question[qid], blocks),
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "student_question_evidence_map",
        "mapping_method": (
            "notebook_explicit_question_component_blocks"
            if notebook_blocks
            else "generic_explicit_question_anchors_with_document_order_continuations"
        ),
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
