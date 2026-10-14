import re
from collections import Counter


QUESTION_RE = re.compile(r"^\s*(\d{1,3})\s*[.)]\s+(.+?)\s*$")
SUBPART_RE = re.compile(r"^\s*\(([a-z])\)\s*(.*)$", re.I)
INLINE_SUBPART_RE = re.compile(r"^\s*(\d{1,3})\(([a-z])\)\s*(.*)$", re.I)
ROMAN_RE = re.compile(r"^\s*\(([ivxlcdm]+)\)\s*(.*)$", re.I)
ROMAN_DOTTED_RE = re.compile(r"^\s*(viii|vii|vi|iv|iii|ii|ix|x|v|i)\.\s+(.+?)\s*$", re.I)
# Uppercase A-I labels used for experiment lists must not be normalized into
# Roman numerals (especially I, which otherwise collides with lowercase i).
ALPHA_DOTTED_RE = re.compile(r"^\s*([A-I])\.\s+(.+?)\s*$")
NUMBERED_SUBPART_RE = re.compile(r"^\s*(\d{1,3})\s*[.)]\s+(.+?)\s*$")
BULLET_RE = re.compile(r"^\s*[\u0088\u2022•▪◦‣\-–—*]\s+(.+?)\s*$")
STANDALONE_TASK_RE = re.compile(r"\bquestion\b.*\b(generate|create|construct|build|prepare|produce|design|develop|make)\b", re.I)
MARKS_RE = re.compile(r"\[\s*(\d+)\s*marks?\s*\]", re.I | re.S)
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
    re.compile(r"^\s*%?\s*my use\s*:", re.I),
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


def _marks_in(text):
    match = MARKS_RE.search(text or "")
    return int(match.group(1)) if match else None


def _is_heading_only(text):
    heading = QUESTION_RE.match(text)
    body = heading.group(2).strip() if heading else text.strip()
    body = MARKS_RE.sub("", body).strip()
    words = re.findall(r"[A-Za-z0-9]+", body)
    if not words or len(words) > 7:
        return False
    action_start = re.compile(
        r"^(?:calculate|compute|plot|display|show|create|build|train|fit|predict|evaluate|compare|explain|discuss|write|implement|derive|find|identify|clean|preprocess|split|load|read|import|visuali[sz]e|print|report|measure|test|apply|use|perform|generate|draw|state|justify|interpret|analy[sz]e|transform|encode|scale|normalize|standardize|remove|replace|handle|check|determine|estimate|classify|select|extract|construct|verify|retrieve|add|include|provide|save|choose|download|follow|answer|complete|submit|set|attach|describe|prepare|develop|obtain|list|specify|label|present|mention|discuss)\b",
        re.I,
    )
    return bool(MARKS_RE.search(heading.group(2) if heading else text)) and not action_start.search(body)


def _new_question(number, text, page, occurrence):
    heading = QUESTION_RE.match(text)
    heading_body = heading.group(2).strip() if heading else text
    title_body = MARKS_RE.sub("", heading_body).strip()
    return {
        "question_id": f"q{number}_occurrence_{occurrence}",
        "number": str(number),
        "occurrence": occurrence,
        "heading_text": text,
        "title_text": title_body,
        "title_only": _is_heading_only(text),
        "marks": _marks_in(heading_body),
        "text": text,
        "raw_lines": [text],
        "source_pages": [page],
        "subparts": [],
        "alternative_markers": [],
        "alternative_groups": [],
    }


def _partition_numbered_candidates(lines, raw_candidates):
    """Identify simple numbered lists nested under 'answer the following' prompts."""
    top_level = []
    nested = {}
    active_top = None
    nested_parent = None
    last_nested_number = None

    for candidate in raw_candidates:
        index, number, body = candidate
        numeric = int(number)
        if nested_parent is not None:
            if numeric == last_nested_number + 1:
                nested[index] = (str(number), body, nested_parent["index"])
                last_nested_number = numeric
                continue
            nested_parent = None
            last_nested_number = None

        if active_top is not None:
            interstitial = " ".join(line["text"] for line in lines[active_top["index"] + 1:index])
            if re.search(r"\banswer\s+the\s+following\b|\banswer\s+the\s+questions\s+below\b", interstitial, re.I):
                nested_parent = active_top
                last_nested_number = numeric
                nested[index] = (str(number), body, active_top["index"])
                continue

        top_level.append(candidate)
        active_top = {"index": index, "number": str(number), "body": body}

    return top_level, nested


def _append_unique_page(item, page):
    if page not in item["source_pages"]:
        item["source_pages"].append(page)


def _is_page_number(line):
    return bool(re.fullmatch(r"\s*\d{1,2}\s*", line))


def _is_wrapped_continuation(previous, current):
    left, right = previous.rstrip(), current.lstrip()
    if not left or not right or re.match(r"^[\-–—•*\u0088]", right):
        return False
    if re.search(r"[.!?:;][\"'’”)]?\s*$", left):
        return False
    return right[0].islower() or right[0].isdigit() or right[0] in "'\"([{“‘’”-–—" or left.endswith(("-", "‐", "‑"))


def _is_inline_alternative(question, text):
    if re.search(r"\b(?:EITHER|ALTERNATIVELY)\b", text, re.I):
        return True
    if not re.search(r"\bor\b", text, re.I):
        return False
    recent_context = " ".join(question["raw_lines"][-5:] + [text])
    return bool(INLINE_MEASURE_CHOICE_RE.search(recent_context))


def _record_inline_choice(question, text, page, node=None, alternatives=None):
    """Record an explicit in-line choice once and attach its stable group ID."""
    if not _is_inline_alternative(question, text):
        return
    for marker in question.get("alternative_markers", []):
        if marker.get("kind") == "inline_choice" and marker.get("text") == text:
            if node is not None:
                node["inline_choice_group_id"] = marker["alternative_group_id"]
            return
    index = sum(1 for group in question.get("alternative_groups", []) if group.get("kind") == "inline_choice") + 1
    group_id = f"{question['question_id']}_choice_{index}"
    context = " ".join(question.get("raw_lines", [])[-5:] + [text])
    options = []
    for item in re.findall(r"\b(mean|median|mode)\b", context, re.I):
        if item.lower() not in options:
            options.append(item.lower())
    marker = {
        "text": text, "source_pages": [page], "question_id": question["question_id"],
        "kind": "inline_choice", "alternative_group_id": group_id,
    }
    question.setdefault("alternative_markers", []).append(marker)
    question.setdefault("alternative_groups", []).append({
        "alternative_group_id": group_id, "question_id": question["question_id"],
        "kind": "inline_choice", "marker_text": text, "source_pages": [page],
        "options": options,
    })
    if node is not None:
        node["inline_choice_group_id"] = group_id
    if alternatives is not None:
        alternatives.append(marker)


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
    # exam-administration lines, then distinguish numbered inference lists
    # that explicitly follow an "answer the following" instruction.
    raw_candidates = []
    for index, line in enumerate(lines):
        match = QUESTION_RE.match(line["text"])
        if match and not _is_administrative(line["text"]):
            raw_candidates.append((index, match.group(1), match.group(2).strip()))

    candidates, nested_numbered_candidates = _partition_numbered_candidates(lines, raw_candidates)

    if not candidates:
        diagnostics.append({
            "type": "no_question_headings",
            "message": "No numbered question headings were recognized.",
        })

    first_candidate_index = candidates[0][0] if candidates else len(lines)

    # A question banner such as "Question — Generate a dataset ..." before
    # numbered questions is a standalone task, not a general instruction.
    standalone_tasks = []
    standalone_task_line_indices = set()
    task_start = next((i for i, line in enumerate(lines[:first_candidate_index])
                       if STANDALONE_TASK_RE.search(line["text"])), None)
    if task_start is not None:
        task_lines = []
        task_pages = []
        for i in range(task_start, first_candidate_index):
            text = lines[i]["text"]
            if _is_title_or_metadata(text) or _is_administrative(text):
                continue
            task_lines.append(text)
            task_pages.append(lines[i]["page"])
            standalone_task_line_indices.add(i)
        if task_lines:
            standalone_tasks.append({
                "task_id": "task_unnumbered_1",
                "text": "\n".join(task_lines),
                "raw_lines": task_lines,
                "source_pages": sorted(set(task_pages)),
                "marks": _marks_in("\n".join(task_lines)),
            })

    # Preserve pre-question instructions, excluding extracted standalone tasks,
    # title/metadata, and recognizable administration instructions.
    for line_index, line in enumerate(lines[:first_candidate_index]):
        if line_index in standalone_task_line_indices:
            continue
        text = line["text"]
        if _is_title_or_metadata(text) or _is_administrative(text):
            continue
        if re.match(r"^\s*instructions?\s*:\s*$", text, re.I):
            global_instructions.append({"text": text, "source_pages": [line["page"]]})
        elif text.startswith(("-", "–", "•", "*", "\u0088")) or re.search(
            r"\b(dataset|notebook|submit|minimum of|at least \d+)\b", text, re.I
        ):
            global_instructions.append({"text": text, "source_pages": [line["page"]]})

    candidate_by_index = {index: (number, text) for index, number, text in candidates}
    questions = []
    question_counts = Counter()
    current_question = None
    current_subpart = None
    current_nested_part = None
    current_dotted_roman_parent = None
    current_detail_part = None
    current_alternative_branch = None

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
            current_question = _new_question(number, text, page, question_counts[number])
            questions.append(current_question)
            current_subpart = None
            current_nested_part = None
            current_dotted_roman_parent = None
            current_detail_part = None
            current_alternative_branch = None
            continue

        if index in nested_numbered_candidates:
            number, question_text, parent_index = nested_numbered_candidates[index]
            # The nested list belongs to the most recent active top-level
            # question; it is not a new top-level question occurrence.
            if current_question is not None:
                part = {
                    "label": number,
                    "text": text,
                    "raw_lines": [text],
                    "source_pages": [page],
                    "nested_subparts": [],
                    "alternative_branches": [],
                    "marks": _marks_in(text),
                    "node_kind": "numbered_subquestion",
                }
                current_question["subparts"].append(part)
                current_question["raw_lines"].append(text)
                _append_unique_page(current_question, page)
                current_subpart = part
                current_nested_part = None
                current_dotted_roman_parent = None
                current_detail_part = None
                current_alternative_branch = None
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
            current_dotted_roman_parent = None
            current_detail_part = None
            current_alternative_branch = None
            continue

        # Explicit OR starts a second branch. Keep the preceding node as the
        # primary branch and attach the following text to an alternative node.
        if OR_RE.match(text):
            alternative_index = len(current_question.get("alternative_groups", [])) + 1
            group_id = f"{current_question['question_id']}_alternative_{alternative_index}"
            primary_node = current_nested_part or current_subpart
            branch = {
                "label": "alternative_2",
                "text": "",
                "raw_lines": [],
                "source_pages": [page],
                "nested_subparts": [],
                "alternative_branches": [],
                "alternative_group_id": group_id,
                "marks": None,
                "node_kind": "alternative_branch",
            }
            if primary_node is not None:
                primary_node["alternative_group_id"] = group_id
                primary_node.setdefault("alternative_branches", []).append(branch)
                primary_label = primary_node.get("label")
            else:
                current_question.setdefault("alternative_branches", []).append(branch)
                primary_label = None
            group = {
                "alternative_group_id": group_id,
                "question_id": current_question["question_id"],
                "kind": "or_branch",
                "marker_text": text,
                "source_pages": [page],
                "primary_node_label": primary_label,
                "alternative_node_label": "alternative_2",
            }
            current_question["alternative_groups"].append(group)
            marker = {
                "text": text,
                "source_pages": [page],
                "question_id": current_question["question_id"],
                "kind": "branch_separator",
                "alternative_group_id": group_id,
            }
            alternatives.append(marker)
            current_question["alternative_markers"].append(marker)
            current_question["raw_lines"].append(text)
            _append_unique_page(current_question, page)
            current_detail_part = None
            current_alternative_branch = branch
            continue

        # Content after an OR marker belongs to its alternative branch until
        # another top-level or nested numbered heading starts.
        if current_alternative_branch is not None:
            current_alternative_branch["raw_lines"].append(text)
            current_alternative_branch["text"] = "\n".join(current_alternative_branch["raw_lines"])
            _append_unique_page(current_alternative_branch, page)
            current_question["raw_lines"].append(text)
            _append_unique_page(current_question, page)
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
                    "alternative_branches": [],
                    "marks": _marks_in(text),
                }
                current_question["subparts"].append(part)
                current_question["raw_lines"].append(text)
                _append_unique_page(current_question, page)
                current_subpart = part
                current_nested_part = None
                current_dotted_roman_parent = None
                current_detail_part = None
                _record_inline_choice(current_question, text, page, part, alternatives)
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
                        "raw_lines": [part_text],
                        "source_pages": [page],
                        "nested_subparts": [],
                        "alternative_branches": [],
                        "marks": _marks_in(part_text),
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
            current_dotted_roman_parent = None
            if nested_mode:
                current_nested_part = created_parts[-1]
            else:
                current_subpart = created_parts[-1]
                current_nested_part = None
            continue

        # Uppercase A-I headings are a distinct alphabetic list when they
        # occur under an existing dotted-Roman section. In particular, an
        # uppercase "I. Experiment 09" after A-H must remain an alpha item,
        # not collide with its lowercase Roman parent "i.".
        alpha_dotted = ALPHA_DOTTED_RE.match(text)
        if alpha_dotted and current_dotted_roman_parent is not None:
            label, body = alpha_dotted.groups()
            existing_alpha_labels = {
                child.get("label")
                for child in current_dotted_roman_parent.get("nested_subparts", []) or []
                if isinstance(child, dict) and child.get("node_kind") == "alpha_item"
            }
            if label != "I" or existing_alpha_labels:
                part = {
                    "label": label,
                    "text": text,
                    "raw_lines": [text],
                    "source_pages": [page],
                    "nested_subparts": [],
                    "alternative_branches": [],
                    "marks": _marks_in(text),
                    "node_kind": "alpha_item",
                }
                current_dotted_roman_parent.setdefault("nested_subparts", []).append(part)
                current_nested_part = part
                current_detail_part = None
                current_question["raw_lines"].append(text)
                _append_unique_page(current_question, page)
                continue

        # Dotted Roman headings (i., ii., ...), common in experiment lists,
        # are parsed as nested items under a current lettered/numeric part.
        dotted_roman = ROMAN_DOTTED_RE.match(text)
        if dotted_roman:
            label, body = dotted_roman.groups()
            label = label.lower()
            part = {
                "label": label,
                "text": text,
                "raw_lines": [text],
                "source_pages": [page],
                "nested_subparts": [],
                "alternative_branches": [],
                "marks": _marks_in(text),
                "node_kind": "roman_item",
            }
            if current_subpart is not None and not _is_roman(current_subpart.get("label", "")):
                current_subpart.setdefault("nested_subparts", []).append(part)
            else:
                current_question["subparts"].append(part)
                current_subpart = part
            current_question["raw_lines"].append(text)
            _append_unique_page(current_question, page)
            current_nested_part = part
            current_dotted_roman_parent = part
            current_detail_part = None
            continue

        # Bullet items below a Roman experiment become independently traceable
        # child nodes; wrapped continuation lines are appended to that child.
        bullet_match = BULLET_RE.match(text)
        if bullet_match and current_nested_part is not None:
            child_index = len(current_nested_part.setdefault("nested_subparts", [])) + 1
            detail = {
                "label": f"item_{child_index}",
                "text": text,
                "raw_lines": [text],
                "source_pages": [page],
                "nested_subparts": [],
                "alternative_branches": [],
                "marks": _marks_in(text),
                "node_kind": "bullet_item",
            }
            current_nested_part["nested_subparts"].append(detail)
            current_detail_part = detail
            current_question["raw_lines"].append(text)
            _append_unique_page(current_question, page)
            continue

        # Mark annotations following an experiment heading are metadata, not
        # separate assessable requirements.
        if current_nested_part is not None and re.fullmatch(r"\[\s*\d+\s*marks?\s*\]", text, re.I):
            current_nested_part["marks"] = _marks_in(text)
            current_nested_part["marks_source_text"] = text
            current_nested_part.setdefault("source_pages", [])
            _append_unique_page(current_nested_part, page)
            current_question["raw_lines"].append(text)
            _append_unique_page(current_question, page)
            current_detail_part = None
            continue

        roman_match = ROMAN_RE.match(text)
        if roman_match:
            label, body = roman_match.groups()
            if _is_roman(label):
                if current_subpart is not None and not _is_roman(current_subpart["label"]):
                    nested = {
                        "label": label.lower(),
                        "text": text,
                        "raw_lines": [text],
                        "source_pages": [page],
                        "nested_subparts": [],
                        "alternative_branches": [],
                        "marks": _marks_in(text),
                    }
                    current_subpart["nested_subparts"].append(nested)
                    current_question["raw_lines"].append(text)
                    _append_unique_page(current_question, page)
                    current_dotted_roman_parent = None
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
                    current_dotted_roman_parent = None
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
                current_dotted_roman_parent = None
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
            current_dotted_roman_parent = None
            current_detail_part = None
            _record_inline_choice(current_question, text, page, part, alternatives)
            continue

        # Inline method alternatives are recorded on the relevant source node.
        choice_node = current_nested_part or current_subpart
        _record_inline_choice(current_question, text, page, choice_node, alternatives)

        current_question["raw_lines"].append(text)
        _append_unique_page(current_question, page)

        if current_detail_part is not None:
            previous = current_detail_part["raw_lines"][-1] if current_detail_part.get("raw_lines") else ""
            if _is_wrapped_continuation(previous, text):
                current_detail_part["raw_lines"].append(text)
                current_detail_part["text"] = "\n".join(current_detail_part["raw_lines"])
                _append_unique_page(current_detail_part, page)
            else:
                # A new unbulleted specification after a bullet is retained as
                # its own child rather than swallowed into the previous item.
                child_index = len(current_nested_part.setdefault("nested_subparts", [])) + 1
                detail = {
                    "label": f"context_{child_index}",
                    "text": text,
                    "raw_lines": [text],
                    "source_pages": [page],
                    "nested_subparts": [],
                    "alternative_branches": [],
                    "marks": _marks_in(text),
                    "node_kind": "context_item",
                }
                current_nested_part["nested_subparts"].append(detail)
                current_detail_part = detail
        elif current_nested_part is not None:
            current_nested_part["text"] += "\n" + text
            current_nested_part.setdefault("raw_lines", []).append(text)
            _append_unique_page(current_nested_part, page)
        elif current_subpart is not None:
            current_subpart["text"] += "\n" + text
            current_subpart.setdefault("raw_lines", []).append(text)
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

        def finalize_node(node):
            node["source_pages"] = sorted(set(node.get("source_pages", [])))
            if node.get("marks") is None:
                node["marks"] = _marks_in(node.get("text", ""))
            for child in node.get("nested_subparts", []) or []:
                finalize_node(child)
            for branch in node.get("alternative_branches", []) or []:
                finalize_node(branch)

        for part in question["subparts"]:
            finalize_node(part)
        for branch in question.get("alternative_branches", []) or []:
            finalize_node(branch)

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
        "schema_version": "1.1",
        "source_filename": source_filename,
        "global_instructions": global_instructions,
        "standalone_tasks": standalone_tasks,
        "questions": questions,
        "notes": notes,
        "alternatives": alternatives,
        "diagnostics": diagnostics,
    }
