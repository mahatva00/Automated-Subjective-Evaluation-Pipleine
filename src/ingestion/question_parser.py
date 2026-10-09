import re
from collections import Counter


QUESTION_RE = re.compile(r"^\s*(\d{1,3})\s*[.)]\s+(.+?)\s*$")
SUBPART_RE = re.compile(r"^\s*\(([a-z])\)\s*(.*)$", re.I)
INLINE_SUBPART_RE = re.compile(r"^\s*(\d{1,3})\(([a-z])\)\s*(.*)$", re.I)
ROMAN_RE = re.compile(r"^\s*\(([ivxlcdm]+)\)\s*(.*)$", re.I)
INLINE_ROMAN_RE = re.compile(r"\((viii|vii|vi|v|iv|iii|ii|ix|x|i)\)\s*", re.I)
NOTE_RE = re.compile(r"^\s*(note|notes|important note|general note)\s*:\s*(.*)$", re.I)
OR_RE = re.compile(r"^\s*(?:OR|EITHER|ALTERNATIVELY)\s*[.:]?\s*$", re.I)
INLINE_MEASURE_CHOICE_RE = re.compile(
    r"\b(?:mean|median|mode)\b.{0,160}\bor\b.{0,100}\b(?:mean|median|mode)\b",
    re.I | re.S,
)


ADMIN_PATTERNS = [
    re.compile(r"\bafter completing the exam\b", re.I),
    re.compile(r"\bname your file as\b", re.I),
    re.compile(r"\bupload the single\b", re.I),
    re.compile(r"\byou may use ai tools\b", re.I),
    re.compile(r"\bfor each question, if you used any ai\b", re.I),
    re.compile(r"\bexample format\s*:", re.I),
]

TITLE_PATTERNS = [
    re.compile(r"^\s*(machine learning|practice makes you perfect)\s*$", re.I),
    re.compile(r"^\s*(lab practice sheet|periodic assessment test|final assessment test)\b", re.I),
    re.compile(r"^\s*(programme|program|course|faculty|slot/class no\.?|time|date of completion)\b", re.I),
    re.compile(r"^\s*max\.?\s*marks\b", re.I),
]


def _page_number(segment, fallback):
    location = segment.get("source_location", {})
    return location.get("page_number", fallback)


def _is_administrative(text):
    return any(pattern.search(text) for pattern in ADMIN_PATTERNS)


def _is_title_or_metadata(text):
    return any(pattern.search(text) for pattern in TITLE_PATTERNS)


def _is_roman(label):
    return label.lower() in {
        "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"
    }


def _new_question(number, text, page, occurrence):
    return {
        "question_id": f"q{number}_occurrence_{occurrence}",
        "number": str(number),
        "occurrence": occurrence,
        "text": text,
        "raw_lines": [text],
        "source_pages": [page],
        "subparts": [],
        "alternative_markers": [],
    }


def _append_unique_page(item, page):
    if page not in item["source_pages"]:
        item["source_pages"].append(page)


def _is_page_number(line):
    return bool(re.fullmatch(r"\s*\d{1,2}\s*", line))


def _is_inline_alternative(question, text):
    if re.search(r"\b(?:EITHER|ALTERNATIVELY)\b", text, re.I):
        return True

    # Only evaluate the measure-choice pattern on a line containing "or".
    # The surrounding context can contain the first option on a previous PDF
    # line, but later continuation lines must not create duplicate markers.
    if not re.search(r"\bor\b", text, re.I):
        return False

    recent_context = " ".join(question["raw_lines"][-5:] + [text])
    return bool(INLINE_MEASURE_CHOICE_RE.search(recent_context))


def parse_question_paper(document):
    """Parse extracted PDF text into a preliminary question map.

    This function does not execute student code, call an LLM, grade answers,
    or attempt to repair the source wording.
    """
    lines = []
    diagnostics = []
    global_instructions = []
    notes = []
    alternatives = []

    for index, segment in enumerate(document.get("segments", []), start=1):
        page = _page_number(segment, index)
        content = segment.get("content", "")

        if not isinstance(content, str):
            diagnostics.append({
                "type": "invalid_segment_content",
                "page": page,
                "message": "Segment content is not text.",
            })
            continue

        for raw_line in content.splitlines():
            text = raw_line.strip()
            if text:
                lines.append({"text": text, "page": page})

    # Identify candidate question headings while excluding recognizable
    # exam-administration lines.
    candidates = []
    for index, line in enumerate(lines):
        match = QUESTION_RE.match(line["text"])
        if match and not _is_administrative(line["text"]):
            candidates.append((index, match.group(1), match.group(2).strip()))

    # A paper without recognizable numbered headings is not silently treated
    # as successfully parsed.
    if not candidates:
        diagnostics.append({
            "type": "no_question_headings",
            "message": "No numbered question headings were recognized.",
        })

    first_candidate_index = candidates[0][0] if candidates else len(lines)

    # Preserve pre-question instructions, excluding obvious title/metadata
    # and recognizable administration instructions.
    for line in lines[:first_candidate_index]:
        text = line["text"]
        if _is_title_or_metadata(text) or _is_administrative(text):
            continue
        if re.match(r"^\s*instructions?\s*:\s*$", text, re.I):
            global_instructions.append({
                "text": text,
                "source_pages": [line["page"]],
            })
        elif text.startswith(("-", "–", "•", "*")) or re.search(
            r"\b(dataset|notebook|submit|minimum of|at least \d+)\b",
            text,
            re.I,
        ):
            global_instructions.append({
                "text": text,
                "source_pages": [line["page"]],
            })

    candidate_by_index = {
        index: (number, text) for index, number, text in candidates
    }
    questions = []
    question_counts = Counter()
    current_question = None
    current_subpart = None
    current_nested_part = None

    for index, line in enumerate(lines):
        text = line["text"]
        page = line["page"]

        # Skip administrative numbered lines even if they appear after
        # the initial title/instruction page.
        if _is_administrative(text):
            continue

        # PDF page numbers are standalone numeric lines; question headings
        # include punctuation and are therefore not removed here.
        if _is_page_number(text):
            continue

        if index in candidate_by_index:
            number, question_text = candidate_by_index[index]
            question_counts[number] += 1
            current_question = _new_question(
                number, text, page, question_counts[number]
            )
            questions.append(current_question)
            if _is_inline_alternative(current_question, text):
                marker = {
                    "text": text,
                    "source_pages": [page],
                    "question_id": current_question["question_id"],
                    "kind": "inline_marker",
                }
                alternatives.append(marker)
                current_question["alternative_markers"].append(marker)
            current_subpart = None
            current_nested_part = None
            continue

        if current_question is None:
            continue

        # Notes are separate entities, not part of the preceding question.
        note_match = NOTE_RE.match(text)
        if note_match:
            notes.append({
                "text": text,
                "source_pages": [page],
            })
            current_subpart = None
            current_nested_part = None
            continue

        # Explicit alternative marker on its own line.
        if OR_RE.match(text):
            marker = {
                "text": text,
                "source_pages": [page],
                "question_id": current_question["question_id"],
            }
            alternatives.append(marker)
            current_question["alternative_markers"].append(marker)
            current_question["raw_lines"].append(text)
            _append_unique_page(current_question, page)
            current_subpart = None
            current_nested_part = None
            continue

        # A subpart written as 4(c) belongs to question 4, even if it
        # starts on a later page.
        inline_match = INLINE_SUBPART_RE.match(text)
        if inline_match:
            parent_number, label, body = inline_match.groups()
            if parent_number == current_question["number"]:
                part = {
                    "label": label.lower(),
                    "text": text,
                    "raw_lines": [text],
                    "source_pages": [page],
                    "nested_subparts": [],
                }
                current_question["subparts"].append(part)
                current_question["raw_lines"].append(text)
                _append_unique_page(current_question, page)
                current_subpart = part
                current_nested_part = None
                continue

        # Extract multiple parenthesized Roman items appearing on one line,
        # including lists that have no lettered parent such as (i) Recall
        # (ii) Precision. The original full line remains in the question text.
        inline_roman_matches = list(INLINE_ROMAN_RE.finditer(text))
        if len(inline_roman_matches) >= 2:
            nested_mode = (
                current_subpart is not None
                and not _is_roman(current_subpart["label"])
            )
            created_parts = []

            for marker_index, marker_match in enumerate(inline_roman_matches):
                start = marker_match.start()
                end = (
                    inline_roman_matches[marker_index + 1].start()
                    if marker_index + 1 < len(inline_roman_matches)
                    else len(text)
                )
                part_text = text[start:end].strip()
                label = marker_match.group(1).lower()

                if nested_mode:
                    part = {
                        "label": label,
                        "text": part_text,
                        "source_pages": [page],
                    }
                    current_subpart["nested_subparts"].append(part)
                else:
                    part = {
                        "label": label,
                        "text": part_text,
                        "raw_lines": [part_text],
                        "source_pages": [page],
                        "nested_subparts": [],
                    }
                    current_question["subparts"].append(part)
                created_parts.append(part)

            current_question["raw_lines"].append(text)
            _append_unique_page(current_question, page)
            if nested_mode:
                current_nested_part = created_parts[-1]
            else:
                current_subpart = created_parts[-1]
                current_nested_part = None
            continue

        roman_match = ROMAN_RE.match(text)
        if roman_match:
            label, body = roman_match.groups()
            if _is_roman(label):
                if current_subpart is not None and not _is_roman(current_subpart["label"]):
                    nested = {
                        "label": label.lower(),
                        "text": text,
                        "source_pages": [page],
                    }
                    current_subpart["nested_subparts"].append(nested)
                    current_question["raw_lines"].append(text)
                    _append_unique_page(current_question, page)
                    current_nested_part = nested
                else:
                    part = {
                        "label": label.lower(),
                        "text": text,
                        "raw_lines": [text],
                        "source_pages": [page],
                        "nested_subparts": [],
                    }
                    current_question["subparts"].append(part)
                    current_question["raw_lines"].append(text)
                    _append_unique_page(current_question, page)
                    current_subpart = part
                    current_nested_part = None
                continue

        subpart_match = SUBPART_RE.match(text)
        if subpart_match:
            label, body = subpart_match.groups()

            # Roman numerals belong to the current lettered subpart when
            # that hierarchy is evident.
            if _is_roman(label) and current_subpart is not None:
                nested = {
                    "label": label.lower(),
                    "text": text,
                    "source_pages": [page],
                }
                current_subpart["nested_subparts"].append(nested)
                current_question["raw_lines"].append(text)
                _append_unique_page(current_question, page)
                current_nested_part = nested
                continue

            part = {
                "label": label.lower(),
                "text": text,
                "raw_lines": [text],
                "source_pages": [page],
                "nested_subparts": [],
            }
            current_question["subparts"].append(part)
            current_question["raw_lines"].append(text)
            _append_unique_page(current_question, page)
            current_subpart = part
            current_nested_part = None
            continue

        # Avoid treating every ordinary use of "or" as a question alternative.
        # Explicit cues and a narrowly recognized imputation-method choice are
        # retained; a standalone OR/EITHER/ALTERNATIVELY line is handled above.
        if _is_inline_alternative(current_question, text):
            marker = {
                "text": text,
                "source_pages": [page],
                "question_id": current_question["question_id"],
                "kind": "inline_marker",
            }
            alternatives.append(marker)
            current_question["alternative_markers"].append(marker)

        current_question["raw_lines"].append(text)
        _append_unique_page(current_question, page)

        if current_nested_part is not None:
            current_nested_part["text"] += "\n" + text
            _append_unique_page(current_nested_part, page)
        elif current_subpart is not None:
            current_subpart["text"] += "\n" + text
            current_subpart["raw_lines"].append(text)
            _append_unique_page(current_subpart, page)
        else:
            current_question["text"] += "\n" + text

    # Finalize source page ordering and detect uncertainty without rewriting text.
    for question in questions:
        question["source_pages"].sort()
        question["text"] = "\n".join(question["raw_lines"])

        if re.search(r"\bthat\s*\.\s*$", question["text"], re.I):
            diagnostics.append({
                "type": "possibly_incomplete_question",
                "question_id": question["question_id"],
                "source_pages": question["source_pages"],
                "message": "Question text may be incomplete; inspect the source paper.",
            })

        for part in question["subparts"]:
            part["source_pages"].sort()
            for nested in part["nested_subparts"]:
                nested["source_pages"].sort()

    # Repeated numbering is retained as separate occurrences and reported
    # for review; it is not automatically treated as an error.
    for number, count in question_counts.items():
        if count > 1:
            diagnostics.append({
                "type": "repeated_question_number",
                "number": number,
                "occurrences": count,
                "message": "Repeated numbering was preserved as separate occurrences.",
            })

    submission = document.get("submission", {})
    source_filename = (
        submission.get("source_filename")
        or submission.get("filename")
        or "unknown"
    )

    return {
        "schema_version": "1.0",
        "source_filename": source_filename,
        "global_instructions": global_instructions,
        "questions": questions,
        "notes": notes,
        "alternatives": alternatives,
        "diagnostics": diagnostics,
    }
