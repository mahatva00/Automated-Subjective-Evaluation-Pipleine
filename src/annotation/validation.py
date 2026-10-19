"""Dependency-free validation for cross-domain annotation JSON records."""
from __future__ import annotations

STRUCTURE_LABELS = {
    "document_title", "section_heading", "question_heading", "subquestion_heading",
    "question_body", "subquestion_body", "instruction_heading", "list_marker", "list_item",
    "table_heading", "table", "table_row", "table_cell", "code_block", "equation_block",
    "figure_caption", "page_header", "page_footer", "document_metadata", "continuation_block", "other",
}
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
ATOMICITY = {"atomic", "compound", "uncertain"}
ASSESSMENT_RELEVANCE = {"yes", "no", "uncertain"}
RELATION_TYPES = {
    "parent_of", "belongs_to", "continues", "alternative_to", "option_for", "constrains",
    "depends_on", "precedes", "references", "applies_to", "other",
}
DOC_STATUSES = {"draft", "reviewed", "adjudicated", "example"}
ANNOTATION_STATUSES = {"unreviewed", "reviewed", "adjudicated"}


def validate_document(doc: dict) -> list[str]:
    """Return validation errors. Empty list means the document is structurally valid."""
    errors: list[str] = []
    if not isinstance(doc, dict):
        return ["document must be a JSON object"]
    required = {"schema_version", "document_id", "source", "text", "page_ranges", "annotations", "relations", "annotation_metadata"}
    missing = sorted(required - doc.keys())
    if missing:
        errors.append(f"missing top-level keys: {', '.join(missing)}")
        return errors
    if doc.get("schema_version") != "1.0.0":
        errors.append("schema_version must be '1.0.0'")
    if not isinstance(doc.get("document_id"), str) or not doc["document_id"].strip():
        errors.append("document_id must be a non-empty string")
    source = doc.get("source")
    if not isinstance(source, dict):
        errors.append("source must be an object")
    else:
        for field in ("filename", "format", "document_type"):
            if not isinstance(source.get(field), str) or not source[field].strip():
                errors.append(f"source.{field} must be a non-empty string")
        if source.get("document_type") != "question_paper":
            errors.append("source.document_type must be 'question_paper'")
    text = doc.get("text")
    if not isinstance(text, str) or not text.strip():
        errors.append("text must be a non-empty string")
        text = ""

    page_ranges = doc.get("page_ranges")
    if not isinstance(page_ranges, list):
        errors.append("page_ranges must be an array")
        page_ranges = []
    previous_page_start = -1
    seen_page_ids: set[str] = set()
    for idx, page in enumerate(page_ranges):
        where = f"page_ranges[{idx}]"
        if not isinstance(page, dict):
            errors.append(f"{where} must be an object")
            continue
        sid = page.get("source_segment_id")
        start, end = page.get("start_char"), page.get("end_char")
        if not isinstance(sid, str) or not sid:
            errors.append(f"{where}.source_segment_id must be a non-empty string")
        elif sid in seen_page_ids:
            errors.append(f"duplicate source_segment_id: {sid}")
        else:
            seen_page_ids.add(sid)
        if not isinstance(start, int) or not isinstance(end, int):
            errors.append(f"{where} start_char/end_char must be integers")
            continue
        if start < 0 or end < start or end > len(text):
            errors.append(f"{where} has invalid character range {start}:{end}")
        if start < previous_page_start:
            errors.append(f"{where} is not ordered by start_char")
        if idx > 0 and isinstance(page_ranges[idx - 1], dict):
            previous_end = page_ranges[idx - 1].get("end_char")
            if isinstance(previous_end, int) and start < previous_end:
                errors.append(f"{where} overlaps the previous page range")
        previous_page_start = start

    annotations = doc.get("annotations")
    if not isinstance(annotations, list):
        errors.append("annotations must be an array")
        annotations = []
    ids: set[str] = set()
    spans_by_layer: dict[str, list[tuple[int, int, str]]] = {"structure": [], "semantic": []}
    valid_annotation_ids: set[str] = set()
    for idx, ann in enumerate(annotations):
        where = f"annotations[{idx}]"
        if not isinstance(ann, dict):
            errors.append(f"{where} must be an object")
            continue
        aid = ann.get("annotation_id")
        if not isinstance(aid, str) or not aid:
            errors.append(f"{where}.annotation_id must be a non-empty string")
        elif aid in ids:
            errors.append(f"duplicate annotation_id: {aid}")
        else:
            ids.add(aid)
            valid_annotation_ids.add(aid)
        layer, label = ann.get("layer"), ann.get("label")
        if layer not in {"structure", "semantic"}:
            errors.append(f"{where}.layer must be 'structure' or 'semantic'")
        elif layer == "structure" and label not in STRUCTURE_LABELS:
            errors.append(f"{where}.label {label!r} is not a v1 structure label")
        elif layer == "semantic" and label not in SEMANTIC_LABELS:
            errors.append(f"{where}.label {label!r} is not a v1 semantic label")
        start, end, ann_text = ann.get("start_char"), ann.get("end_char"), ann.get("text")
        if not isinstance(start, int) or not isinstance(end, int):
            errors.append(f"{where}.start_char/end_char must be integers")
        elif start < 0 or end <= start or end > len(text):
            errors.append(f"{where} has invalid character range {start}:{end}")
        elif isinstance(ann_text, str) and text[start:end] != ann_text:
            errors.append(f"{where}.text does not equal document.text[{start}:{end}]")
        if not isinstance(ann_text, str) or not ann_text:
            errors.append(f"{where}.text must be a non-empty string")
        if layer in spans_by_layer and isinstance(start, int) and isinstance(end, int) and start >= 0 and end > start and end <= len(text):
            spans_by_layer[layer].append((start, end, str(aid)))
        review = ann.get("review_status")
        if review not in ANNOTATION_STATUSES:
            errors.append(f"{where}.review_status is invalid")
        attrs = ann.get("attributes", {})
        if not isinstance(attrs, dict):
            errors.append(f"{where}.attributes must be an object")
            attrs = {}
        if layer == "semantic" and label == "requirement":
            req_type = attrs.get("requirement_type")
            atomicity = attrs.get("atomicity")
            relevance = attrs.get("assessment_relevance")
            if req_type not in REQUIREMENT_TYPES:
                errors.append(f"{where}.attributes.requirement_type is invalid or missing")
            if atomicity not in ATOMICITY:
                errors.append(f"{where}.attributes.atomicity is invalid or missing")
            if relevance not in ASSESSMENT_RELEVANCE:
                errors.append(f"{where}.attributes.assessment_relevance is invalid or missing")
        if label == "other" and not attrs.get("label_note") and not ann.get("note"):
            errors.append(f"{where}: label 'other' requires attributes.label_note or note")

    for layer, spans in spans_by_layer.items():
        spans.sort()
        for previous, current in zip(spans, spans[1:]):
            if current[0] < previous[1]:
                errors.append(f"overlapping annotations in {layer} layer: {previous[2]} and {current[2]}")

    relations = doc.get("relations")
    if not isinstance(relations, list):
        errors.append("relations must be an array")
        relations = []
    relation_ids: set[str] = set()
    for idx, rel in enumerate(relations):
        where = f"relations[{idx}]"
        if not isinstance(rel, dict):
            errors.append(f"{where} must be an object")
            continue
        rid = rel.get("relation_id")
        if not isinstance(rid, str) or not rid:
            errors.append(f"{where}.relation_id must be a non-empty string")
        elif rid in relation_ids:
            errors.append(f"duplicate relation_id: {rid}")
        else:
            relation_ids.add(rid)
        src, tgt = rel.get("source_annotation_id"), rel.get("target_annotation_id")
        if src not in valid_annotation_ids:
            errors.append(f"{where} references missing source annotation {src!r}")
        if tgt not in valid_annotation_ids:
            errors.append(f"{where} references missing target annotation {tgt!r}")
        if src == tgt:
            errors.append(f"{where} cannot relate an annotation to itself")
        if rel.get("type") not in RELATION_TYPES:
            errors.append(f"{where}.type is invalid")
        if "review_status" in rel and rel["review_status"] not in ANNOTATION_STATUSES:
            errors.append(f"{where}.review_status is invalid")

    meta = doc.get("annotation_metadata")
    if not isinstance(meta, dict):
        errors.append("annotation_metadata must be an object")
    else:
        if meta.get("status") not in DOC_STATUSES:
            errors.append("annotation_metadata.status is invalid")
        if meta.get("guidelines_version") != "1.0.0":
            errors.append("annotation_metadata.guidelines_version must be '1.0.0'")
        if meta.get("status") in {"reviewed", "adjudicated"}:
            for idx, ann in enumerate(annotations):
                if isinstance(ann, dict) and ann.get("review_status") == "unreviewed":
                    errors.append(f"reviewed document contains unreviewed annotation at annotations[{idx}]")
                    break
            for idx, rel in enumerate(relations):
                if isinstance(rel, dict) and rel.get("review_status") == "unreviewed":
                    errors.append(f"reviewed document contains unreviewed relation at relations[{idx}]")
                    break
    return errors
