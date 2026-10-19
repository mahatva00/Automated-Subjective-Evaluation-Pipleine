"""Compile explicitly reviewed annotations into masked, windowed BIO data for DistilBERT.

This module only reads source text/annotations/coverage and returns dataset records.
It never executes student code or approves labels.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

IGNORE_INDEX = -100
APPROVED_STATUSES = {"reviewed", "adjudicated"}
SPLITS = {"train", "validation", "test"}
LAYERS = {"structure", "semantic"}
COVERAGE_STATUSES = {"reviewed", "out_of_scope", "pending", "separator"}

STRUCTURE_LABELS = sorted({
    "document_title", "section_heading", "question_heading", "subquestion_heading",
    "question_body", "subquestion_body", "instruction_heading", "list_marker", "list_item",
    "table_heading", "table", "table_row", "table_cell", "code_block", "equation_block",
    "figure_caption", "page_header", "page_footer", "document_metadata", "continuation_block", "other",
})
SEMANTIC_LABELS = sorted({
    "requirement", "context", "example", "option", "alternative_marker",
    "grading_instruction", "submission_instruction", "administrative_instruction",
    "definition", "explanatory_note", "non_assessable", "other",
})
REQUIREMENT_TYPES = {
    "action", "constraint", "deliverable", "success_criterion", "input_specification",
    "output_specification", "method_or_tool", "evaluation_or_comparison",
    "explanation_or_justification", "other",
}
ATOMICITY = {"atomic", "compound", "uncertain"}
ASSESSMENT_RELEVANCE = {"yes", "no", "uncertain"}


class DatasetError(ValueError):
    """Input data cannot safely be compiled."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DatasetError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise DatasetError(f"{path}: expected a JSON object")
    return result


def index_documents(input_dir: Path, pattern: str = "*.annotation.json", recursive: bool = True) -> dict[str, tuple[Path, dict]]:
    if not input_dir.is_dir():
        raise DatasetError(f"annotation directory does not exist: {input_dir}")
    files = sorted(input_dir.rglob(pattern) if recursive else input_dir.glob(pattern))
    if not files:
        raise DatasetError(f"no annotation files matched {pattern!r} in {input_dir}")
    result = {}
    for path in files:
        doc = read_json(path)
        doc_id = doc.get("document_id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise DatasetError(f"{path}: missing document_id")
        if doc_id in result:
            raise DatasetError(f"duplicate document_id {doc_id!r}: {result[doc_id][0]} and {path}")
        result[doc_id] = (path, doc)
    return result


def read_manifest(path: Path) -> dict[str, dict]:
    manifest = read_json(path)
    if manifest.get("manifest_version") != "1.0.0" or not isinstance(manifest.get("documents"), list):
        raise DatasetError("manifest must have manifest_version='1.0.0' and a documents array")
    by_id: dict[str, dict] = {}
    group_splits: dict[str, str] = {}
    for idx, item in enumerate(manifest["documents"]):
        if not isinstance(item, dict):
            raise DatasetError(f"manifest documents[{idx}] must be an object")
        doc_id = item.get("document_id")
        split = item.get("split")
        group_id = item.get("group_id")
        annotation_path = item.get("annotation_path")
        if not isinstance(doc_id, str) or not doc_id:
            raise DatasetError(f"manifest documents[{idx}] has no document_id")
        if doc_id in by_id:
            raise DatasetError(f"duplicate document_id in manifest: {doc_id}")
        if split not in SPLITS | {"unassigned"}:
            raise DatasetError(f"manifest split for {doc_id} is invalid: {split!r}")
        if not isinstance(group_id, str) or not group_id.strip():
            raise DatasetError(f"manifest group_id is required for {doc_id}")
        if not isinstance(annotation_path, str) or not annotation_path.strip():
            raise DatasetError(f"manifest annotation_path is required for {doc_id}")
        if split != "unassigned":
            old = group_splits.get(group_id)
            if old is not None and old != split:
                raise DatasetError(f"group leakage: {group_id!r} assigned to both {old} and {split}")
            group_splits[group_id] = split
        by_id[doc_id] = item
    return by_id


def validate_approved_document(doc: dict, path: Path | str = "<memory>") -> None:
    where = str(path)
    if doc.get("schema_version") != "1.0.0":
        raise DatasetError(f"{where}: schema_version must be '1.0.0'")
    doc_id = doc.get("document_id")
    text = doc.get("text")
    if not isinstance(doc_id, str) or not doc_id:
        raise DatasetError(f"{where}: document_id is missing")
    if not isinstance(text, str) or not text:
        raise DatasetError(f"{where}: text must be a non-empty string")
    meta = doc.get("annotation_metadata")
    if not isinstance(meta, dict) or meta.get("status") not in APPROVED_STATUSES:
        status = meta.get("status") if isinstance(meta, dict) else None
        raise DatasetError(f"{where}: refusing to train from annotation_metadata.status={status!r}; expected reviewed/adjudicated")
    anns = doc.get("annotations")
    if not isinstance(anns, list):
        raise DatasetError(f"{where}: annotations must be an array")
    seen_ids: set[str] = set()
    by_layer: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for i, ann in enumerate(anns):
        p = f"{where}: annotations[{i}]"
        if not isinstance(ann, dict):
            raise DatasetError(f"{p} must be an object")
        aid = ann.get("annotation_id")
        layer = ann.get("layer")
        label = ann.get("label")
        start = ann.get("start_char")
        end = ann.get("end_char")
        status = ann.get("review_status")
        if not isinstance(aid, str) or not aid or aid in seen_ids:
            raise DatasetError(f"{p}: annotation_id must be unique and non-empty")
        seen_ids.add(aid)
        if layer not in LAYERS:
            raise DatasetError(f"{p}: invalid layer {layer!r}")
        allowed_labels = STRUCTURE_LABELS if layer == "structure" else SEMANTIC_LABELS
        if label not in allowed_labels:
            raise DatasetError(f"{p}: invalid {layer} label {label!r}")
        if status not in APPROVED_STATUSES:
            raise DatasetError(f"{p}: review_status={status!r}; all annotations must be reviewed/adjudicated")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start or end > len(text):
            raise DatasetError(f"{p}: invalid character range {start!r}:{end!r}")
        if ann.get("text") != text[start:end]:
            raise DatasetError(f"{p}: annotation text does not equal the exact canonical source slice")
        if layer == "semantic" and label == "requirement":
            attrs = ann.get("attributes")
            if not isinstance(attrs, dict):
                raise DatasetError(f"{p}: requirement attributes must be an object")
            if attrs.get("requirement_type") not in REQUIREMENT_TYPES:
                raise DatasetError(f"{p}: requirement_type missing or invalid")
            if attrs.get("atomicity") not in ATOMICITY:
                raise DatasetError(f"{p}: atomicity missing or invalid")
            if attrs.get("assessment_relevance") not in ASSESSMENT_RELEVANCE:
                raise DatasetError(f"{p}: assessment_relevance missing or invalid")
        by_layer[layer].append((start, end, aid))
    for layer, spans in by_layer.items():
        spans.sort()
        for a, b in zip(spans, spans[1:]):
            if b[0] < a[1]:
                raise DatasetError(f"{where}: overlapping annotations within {layer}: {a[2]} and {b[2]}")


def coverage_template(doc: dict, scope_doc: dict | None = None) -> dict:
    """Build a pending coverage review template; never promotes in-scope text to reviewed training coverage."""
    text = doc.get("text")
    doc_id = doc.get("document_id")
    if not isinstance(text, str) or not isinstance(doc_id, str):
        raise DatasetError("annotation document must have document_id and text")
    text_hash = sha256_text(text)
    segments: list[dict] = []
    if scope_doc is not None:
        if scope_doc.get("document_id") != doc_id:
            raise DatasetError("scope ledger document_id does not match annotation document")
        if scope_doc.get("source_text_sha256") != text_hash:
            raise DatasetError(f"{doc_id}: scope ledger source_text_sha256 mismatch")
        source_segments = scope_doc.get("segments")
        if not isinstance(source_segments, list) or not source_segments:
            raise DatasetError(f"{doc_id}: scope ledger has no segments")
        for i, src in enumerate(source_segments, 1):
            start, end = src.get("start_char"), src.get("end_char")
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start or end > len(text):
                raise DatasetError(f"{doc_id}: scope segment {src.get('segment_id')} has invalid range")
            actual = text[start:end]
            if actual != src.get("text") or sha256_text(actual) != src.get("text_sha256"):
                raise DatasetError(f"{doc_id}: scope segment {src.get('segment_id')} text/hash mismatch")
            scope_status = src.get("scope_status")
            status = {
                "in_scope": "pending",
                "out_of_scope": "out_of_scope",
                "pending": "pending",
                "separator": "separator",
            }.get(scope_status)
            if status is None:
                raise DatasetError(f"{doc_id}: unknown scope status {scope_status!r}")
            segments.append({
                "coverage_id": f"COV{i:05d}",
                "source_segment_id": src.get("segment_id"),
                "start_char": start,
                "end_char": end,
                "text_sha256": sha256_text(actual),
                "status": status,
                "reviewed_layers": [],
                "scope_status": scope_status,
                "question_id": src.get("question_id"),
                "page_numbers": src.get("page_numbers", []),
                "reviewer": "",
                "review_date": "",
                "notes": "Scope copied from existing ledger; in-scope is pending annotation-coverage review.",
            })
    else:
        # The page ranges alone are not a scope decision. Keep all page text pending.
        ranges = doc.get("page_ranges")
        if not isinstance(ranges, list) or not ranges:
            raise DatasetError(f"{doc_id}: page_ranges required when no scope ledger is supplied")
        cursor = 0
        i = 1
        for page in sorted(ranges, key=lambda x: x.get("start_char", -1)):
            start, end = page.get("start_char"), page.get("end_char")
            if not isinstance(start, int) or not isinstance(end, int) or start < cursor or end <= start or end > len(text):
                raise DatasetError(f"{doc_id}: invalid/overlapping page range")
            if start > cursor:
                gap_text = text[cursor:start]
                gap_status = "separator" if gap_text.isspace() else "pending"
                segments.append({
                    "coverage_id": f"COV{i:05d}", "source_segment_id": None,
                    "start_char": cursor, "end_char": start, "text_sha256": sha256_text(gap_text),
                    "status": gap_status, "reviewed_layers": [], "scope_status": None,
                    "question_id": None, "page_numbers": [],
                    "reviewer": "", "review_date": "", "notes": "Generated inter-page gap; no labels."
                })
                i += 1
            page_text = text[start:end]
            segments.append({
                "coverage_id": f"COV{i:05d}",
                "source_segment_id": page.get("source_segment_id"),
                "start_char": start, "end_char": end, "text_sha256": sha256_text(page_text),
                "status": "pending", "reviewed_layers": [], "scope_status": None,
                "question_id": None,
                "page_numbers": [page["page_number"]] if isinstance(page.get("page_number"), int) else [],
                "reviewer": "", "review_date": "", "notes": "Page text pending annotation-coverage review."
            })
            i += 1
            cursor = end
        if cursor < len(text):
            tail = text[cursor:]
            segments.append({
                "coverage_id": f"COV{i:05d}", "source_segment_id": None,
                "start_char": cursor, "end_char": len(text), "text_sha256": sha256_text(tail),
                "status": "separator" if tail.isspace() else "pending", "reviewed_layers": [],
                "scope_status": None, "question_id": None, "page_numbers": [],
                "reviewer": "", "review_date": "", "notes": "Trailing text pending unless confirmed as separator."
            })
    segments.sort(key=lambda s: (s["start_char"], s["end_char"]))
    # Require a non-overlapping partition from 0 to len(text), so no character status is inferred.
    cursor = 0
    for seg in segments:
        if seg["start_char"] != cursor:
            raise DatasetError(f"{doc_id}: coverage template has a gap/overlap at character {cursor}")
        cursor = seg["end_char"]
    if cursor != len(text):
        raise DatasetError(f"{doc_id}: coverage template stops at {cursor}, expected {len(text)}")
    return {
        "coverage_schema_version": "1.0.0",
        "document_id": doc_id,
        "source_text_sha256": text_hash,
        "text_length": len(text),
        "ledger_status": "draft",
        "source_scope_ledger_document_id": scope_doc.get("document_id") if scope_doc else None,
        "source_scope_ledger_policy_version": scope_doc.get("policy_version") if scope_doc else None,
        "review_contract": (
            "Template only. No in-scope range is approved for O labels until a reviewer sets status=reviewed "
            "and lists reviewed_layers. Out-of-scope, pending and separator text is always masked. "
            "Scope may be manually reviewed; this file does not infer scope."
        ),
        "segments": segments,
    }


def validate_coverage(coverage: dict, doc: dict, path: Path | str = "<coverage>") -> list[dict]:
    where = str(path)
    text = doc.get("text", "")
    doc_id = doc.get("document_id")
    if coverage.get("coverage_schema_version") != "1.0.0":
        raise DatasetError(f"{where}: unsupported coverage_schema_version")
    if coverage.get("document_id") != doc_id:
        raise DatasetError(f"{where}: document_id mismatch")
    if coverage.get("ledger_status") not in APPROVED_STATUSES:
        raise DatasetError(f"{where}: ledger_status must be reviewed/adjudicated before compiling training data")
    if coverage.get("source_text_sha256") != sha256_text(text) or coverage.get("text_length") != len(text):
        raise DatasetError(f"{where}: source text hash/length mismatch")
    segments = coverage.get("segments")
    if not isinstance(segments, list) or not segments:
        raise DatasetError(f"{where}: segments must be a non-empty array")
    ordered = sorted(segments, key=lambda x: (x.get("start_char", -1), x.get("end_char", -1)))
    cursor = 0
    valid = []
    for i, seg in enumerate(ordered):
        if not isinstance(seg, dict):
            raise DatasetError(f"{where}: segments[{i}] must be an object")
        start, end = seg.get("start_char"), seg.get("end_char")
        if not isinstance(start, int) or not isinstance(end, int) or start != cursor or end <= start or end > len(text):
            raise DatasetError(f"{where}: coverage ranges must form an exact, non-overlapping partition; issue at {start}:{end}")
        piece = text[start:end]
        if seg.get("text_sha256") != sha256_text(piece):
            raise DatasetError(f"{where}: text hash mismatch for {seg.get('coverage_id')}")
        status = seg.get("status")
        layers = seg.get("reviewed_layers", [])
        if status not in COVERAGE_STATUSES:
            raise DatasetError(f"{where}: invalid status {status!r}")
        if not isinstance(layers, list) or any(layer not in LAYERS for layer in layers) or len(set(layers)) != len(layers):
            raise DatasetError(f"{where}: invalid reviewed_layers for {seg.get('coverage_id')}")
        if status == "reviewed" and not layers:
            raise DatasetError(f"{where}: reviewed segment {seg.get('coverage_id')} must specify reviewed_layers")
        if status != "reviewed" and layers:
            raise DatasetError(f"{where}: non-reviewed segment {seg.get('coverage_id')} cannot have reviewed_layers")
        source_scope = seg.get("scope_status")
        if source_scope == "out_of_scope" and status != "out_of_scope":
            raise DatasetError(f"{where}: source out-of-scope region cannot be promoted")
        if source_scope == "separator" and status != "separator":
            raise DatasetError(f"{where}: source separator cannot be relabeled")
        if source_scope == "pending" and status != "pending":
            raise DatasetError(f"{where}: source scope-pending region cannot be promoted")
        if source_scope == "in_scope" and status not in {"pending", "reviewed"}:
            raise DatasetError(f"{where}: in-scope region must remain pending or reviewed")
        valid.append(seg)
        cursor = end
    if cursor != len(text):
        raise DatasetError(f"{where}: coverage partition ends at {cursor}, expected {len(text)}")
    return valid


def ranges_cover(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    """True only when the full half-open span is covered by the union of ranges with no gaps."""
    if end <= start:
        return False
    cursor = start
    for left, right in sorted(ranges):
        if right <= cursor:
            continue
        if left > cursor:
            return False
        cursor = max(cursor, right)
        if cursor >= end:
            return True
    return False


def label_maps() -> dict:
    def make(labels: Iterable[str]) -> dict:
        tags = ["O"]
        for label in sorted(labels):
            tags.extend([f"B-{label}", f"I-{label}"])
        return {"tag_to_id": {tag: idx for idx, tag in enumerate(tags)}, "id_to_tag": {str(idx): tag for idx, tag in enumerate(tags)}, "ignore_index": IGNORE_INDEX}
    return {"structure": make(STRUCTURE_LABELS), "semantic": make(SEMANTIC_LABELS)}


def _tokenize_without_specials(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    except Exception as exc:
        raise DatasetError(f"tokenizer must support fast offset mappings: {exc}") from exc
    ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise DatasetError("tokenizer returned unexpected batch shape for one document")
        ids, offsets = ids[0], offsets[0]
    if not isinstance(ids, (list, tuple)) or not isinstance(offsets, (list, tuple)) or len(ids) != len(offsets):
        raise DatasetError("tokenizer must return equal-length input_ids and offset_mapping")
    norm_offsets = []
    last_start = -1
    for i, pair in enumerate(offsets):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise DatasetError(f"invalid tokenizer offset at token {i}")
        start, end = int(pair[0]), int(pair[1])
        if start < 0 or end < start or end > len(text):
            raise DatasetError(f"token offset out of source bounds at token {i}: {start}:{end}")
        if end > start and start < last_start:
            raise DatasetError("token offsets are not monotonic")
        if end > start:
            last_start = start
        norm_offsets.append((start, end))
    return [int(x) for x in ids], norm_offsets


def _coverage_ranges(coverage_segments: list[dict], layer: str) -> list[tuple[int, int]]:
    return [
        (s["start_char"], s["end_char"])
        for s in coverage_segments
        if s["status"] == "reviewed" and layer in s.get("reviewed_layers", [])
    ]


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) < min(a[1], b[1])


def compile_document(
    doc: dict,
    coverage: dict,
    manifest_entry: dict,
    tokenizer: Any,
    max_length: int = 512,
    stride: int = 64,
) -> tuple[list[dict], dict]:
    """Compile a reviewed document into windows with independent structure/semantic BIO labels."""
    validate_approved_document(doc)
    text = doc["text"]
    segs = validate_coverage(coverage, doc)
    if manifest_entry.get("split") not in SPLITS:
        raise DatasetError("compile_document requires an explicit train/validation/test split")
    for ann in doc["annotations"]:
        layer = ann["layer"]
        ranges = _coverage_ranges(segs, layer)
        if not ranges_cover(ann["start_char"], ann["end_char"], ranges):
            raise DatasetError(f"annotation {ann['annotation_id']} is not fully inside reviewed coverage for {layer}")

    all_ids, offsets = _tokenize_without_specials(tokenizer, text)
    special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    payload_size = max_length - special_count
    if payload_size <= 0:
        raise DatasetError("max_length must exceed tokenizer special-token count")
    if stride < 0 or stride >= payload_size:
        raise DatasetError(f"stride must be in [0, {payload_size - 1}]")
    if not all_ids:
        raise DatasetError(f"{doc['document_id']}: tokenizer returned no tokens")

    maps = label_maps()
    layer_tags: dict[str, list[str | int]] = {}
    alignment_warnings: list[dict] = []
    annotation_token_indices: dict[str, list[int]] = {}

    for layer in ("structure", "semantic"):
        layer_map = maps[layer]["tag_to_id"]
        ranges = _coverage_ranges(segs, layer)
        tags: list[str | int] = [IGNORE_INDEX] * len(all_ids)
        # Explicit O labels only where the reviewer confirmed coverage for this layer.
        for ti, (start, end) in enumerate(offsets):
            if end > start and ranges_cover(start, end, ranges):
                tags[ti] = "O"
        anns = [a for a in doc["annotations"] if a["layer"] == layer]
        for ann in anns:
            start, end = ann["start_char"], ann["end_char"]
            overlaps = [i for i, (ts, te) in enumerate(offsets) if te > ts and _spans_overlap((ts, te), (start, end))]
            contained = [i for i in overlaps if start <= offsets[i][0] and offsets[i][1] <= end]
            if not contained:
                raise DatasetError(f"annotation {ann['annotation_id']} aligns to no complete tokenizer token")
            # Any tokenizer token straddling an annotation boundary is masked and reported, never guessed.
            for i in overlaps:
                ts, te = offsets[i]
                if not (start <= ts and te <= end):
                    tags[i] = IGNORE_INDEX
                    alignment_warnings.append({
                        "document_id": doc["document_id"], "annotation_id": ann["annotation_id"],
                        "layer": layer, "token_index": i, "token_start_char": ts,
                        "token_end_char": te, "issue": "token_crosses_annotation_boundary",
                    })
            for i in contained:
                if tags[i] == IGNORE_INDEX:
                    raise DatasetError(f"annotation {ann['annotation_id']} intersects a token outside reviewed coverage")
            label = ann["label"]
            if f"B-{label}" not in layer_map:
                raise DatasetError(f"missing BIO label {label!r} in {layer} label map")
            # The first fully-contained token receives B; subsequent tokens receive I.
            for pos, i in enumerate(contained):
                tags[i] = f"{'B' if pos == 0 else 'I'}-{label}"
            annotation_token_indices[ann["annotation_id"]] = contained
        layer_tags[layer] = tags

    # A positive span must appear completely in at least one payload window.
    windows: list[tuple[int, int]] = []
    window_start = 0
    while window_start < len(all_ids):
        window_end = min(len(all_ids), window_start + payload_size)
        windows.append((window_start, window_end))
        if window_end == len(all_ids):
            break
        next_start = window_end - stride
        if next_start <= window_start:
            raise DatasetError("windowing made no forward progress")
        window_start = next_start
    for ann in doc["annotations"]:
        indexes = annotation_token_indices[ann["annotation_id"]]
        first, last = min(indexes), max(indexes) + 1
        if not any(ws <= first and last <= we for ws, we in windows):
            raise DatasetError(
                f"annotation {ann['annotation_id']} token span {first}:{last} does not fit wholly in any window; "
                "increase max_length or define special handling"
            )

    windows_out = []
    token_counts = Counter()
    for wi, (ws, we) in enumerate(windows):
        payload_ids = all_ids[ws:we]
        prepared = tokenizer.prepare_for_model(
            payload_ids,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_token_type_ids=False,
        )
        final_ids = [int(x) for x in prepared["input_ids"]]
        attention = [int(x) for x in prepared.get("attention_mask", [1] * len(final_ids))]
        special_mask = prepared.get("special_tokens_mask")
        if special_mask is None:
            special_mask = tokenizer.get_special_tokens_mask(payload_ids, already_has_special_tokens=False)
        special_mask = [int(x) for x in special_mask]
        if len(final_ids) != len(special_mask) or len(final_ids) != len(attention):
            raise DatasetError("tokenizer's prepared window has inconsistent lengths")
        local_offsets = []
        local_global_idx: list[int | None] = []
        payload_cursor = 0
        for is_special in special_mask:
            if is_special:
                local_offsets.append((0, 0))
                local_global_idx.append(None)
            else:
                if payload_cursor >= len(payload_ids):
                    raise DatasetError("tokenizer special-token mask does not align with payload tokens")
                local_offsets.append(offsets[ws + payload_cursor])
                local_global_idx.append(ws + payload_cursor)
                payload_cursor += 1
        if payload_cursor != len(payload_ids):
            raise DatasetError("tokenizer special-token mask lost payload tokens")

        row = {
            "schema_version": "distilbert-bio-window-v1.0",
            "document_id": doc["document_id"],
            "source_filename": (doc.get("source") or {}).get("filename"),
            "split": manifest_entry["split"],
            "group_id": manifest_entry["group_id"],
            "subject": manifest_entry.get("subject"),
            "course": manifest_entry.get("course"),
            "window_index": wi,
            "document_token_start": ws,
            "document_token_end": we,
            "input_ids": final_ids,
            "attention_mask": attention,
            "special_tokens_mask": special_mask,
            "offset_mapping": [[int(a), int(b)] for a, b in local_offsets],
            "structure_labels": [],
            "semantic_labels": [],
            "structure_tags": [],
            "semantic_tags": [],
            "annotation_targets": [],
        }
        for layer in ("structure", "semantic"):
            layer_map = maps[layer]["tag_to_id"]
            layer_window: list[int] = []
            readable: list[str] = []
            for is_special, global_idx in zip(special_mask, local_global_idx):
                if is_special or global_idx is None:
                    tag: str | int = IGNORE_INDEX
                else:
                    tag = layer_tags[layer][global_idx]
                readable.append(str(tag))
                layer_window.append(IGNORE_INDEX if tag == IGNORE_INDEX else layer_map[str(tag)])
                if tag == IGNORE_INDEX:
                    token_counts[f"{layer}_masked_tokens"] += 1
                else:
                    token_counts[f"{layer}_{tag}"] += 1
            row[f"{layer}_labels"] = layer_window
            row[f"{layer}_tags"] = readable

        # Preserve reviewed span-level attributes for downstream auxiliary targets/review.
        for ann in doc["annotations"]:
            indexes = annotation_token_indices[ann["annotation_id"]]
            if ws <= min(indexes) and max(indexes) < we:
                local_positions = [
                    k for k, gi in enumerate(local_global_idx)
                    if gi is not None and min(indexes) <= gi <= max(indexes)
                ]
                if not local_positions:
                    continue
                row["annotation_targets"].append({
                    "annotation_id": ann["annotation_id"],
                    "layer": ann["layer"],
                    "label": ann["label"],
                    "start_char": ann["start_char"],
                    "end_char": ann["end_char"],
                    "local_token_start": min(local_positions),
                    "local_token_end": max(local_positions) + 1,
                    "attributes": ann.get("attributes", {}),
                })
        windows_out.append(row)

    stats = {
        "document_id": doc["document_id"],
        "split": manifest_entry["split"],
        "group_id": manifest_entry["group_id"],
        "token_count_before_specials": len(all_ids),
        "window_count": len(windows_out),
        "approved_annotations": len(doc["annotations"]),
        "annotations_by_layer": dict(Counter(a["layer"] for a in doc["annotations"])),
        "alignment_warning_count": len(alignment_warnings),
        "alignment_warnings": alignment_warnings,
        "label_token_counts": dict(token_counts),
        "masked_token_count_by_layer": {
            layer: sum(1 for t in layer_tags[layer] if t == IGNORE_INDEX)
            for layer in ("structure", "semantic")
        },
    }
    return windows_out, stats


def find_coverage_files(coverage_dir: Path, recursive: bool = True) -> dict[str, tuple[Path, dict]]:
    if not coverage_dir.is_dir():
        raise DatasetError(f"coverage directory does not exist: {coverage_dir}")
    files = sorted(coverage_dir.rglob("*.training-coverage.json") if recursive else coverage_dir.glob("*.training-coverage.json"))
    result = {}
    for path in files:
        coverage = read_json(path)
        doc_id = coverage.get("document_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise DatasetError(f"{path}: missing document_id")
        if doc_id in result:
            raise DatasetError(f"duplicate training coverage document_id {doc_id}")
        result[doc_id] = (path, coverage)
    return result
