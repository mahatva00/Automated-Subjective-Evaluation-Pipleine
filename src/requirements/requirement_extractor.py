"""Rule-assisted extraction of atomic requirements from parsed question maps.

Preserves original source wording and provenance. No student code is executed
and no LLM is called in this deterministic baseline.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

SCHEMA_VERSION = "1.1"
ACTION_VERBS = (
    "calculate", "compute", "plot", "display", "show", "create", "build",
    "train", "fit", "predict", "evaluate", "compare", "explain", "discuss",
    "write", "implement", "derive", "find", "identify", "clean", "preprocess",
    "split", "load", "read", "import", "visualize", "visualise", "print", "output",
    "report", "measure", "test", "apply", "use", "perform", "generate", "draw",
    "mention", "state", "justify", "interpret", "analyze", "analyse", "transform",
    "encode", "scale", "normalize", "normalise", "standardize", "standardise",
    "remove", "replace", "handle", "check", "investigate", "determine", "estimate",
    "classify", "cluster", "select", "extract", "demonstrate", "comment", "conclude",
    "summarize", "summarise", "construct", "verify", "retrieve", "confirm", "execute",
    "add", "include", "provide", "paste", "save", "choose", "attempt", "upload",
    "ensure", "download", "follow", "answer", "complete", "submit", "set", "attach",
    "keep", "describe", "present", "prepare", "consider", "develop", "obtain", "list",
    "specify", "label", "title", "return", "select", "investigate", "fit", "name",
)
ACTION_PATTERN = r"(?:" + "|".join(sorted(set(ACTION_VERBS), key=len, reverse=True)) + r")\b"
ACTION_RE = re.compile(r"\b" + ACTION_PATTERN, re.I)
CONSTRAINT_RE = re.compile(
    r"(?:\b(?:at least|at most|minimum of|maximum of|exactly|only|must|should|using|without|not use|do not|no more than|greater than|less than|between|minimum|maximum|random state|random_state|train(?:ing)?\s*(?:set|data|dataset)|test\s*(?:set|data|dataset)|validation\s*(?:set|data|dataset)|instances|rows|columns|features|records|samples|percent)\b|(?<=\d)%(?!\w))",
    re.I,
)
REASONING_RE = re.compile(
    r"\b(?:explain|explanation|discuss|justify|interpret|conclude|comment|reason|why|describe|compare|analysis|inference)\b", re.I
)
DELIVERABLE_RE = re.compile(
    r"\b(?:plot|figure|chart|graph|table|report|notebook|file|output|visuali[sz]e|display|print|draw|code|prompt(?:s)?|details of ai usage|question no\.?|question description|ai-tool.?s? response|explanation)\b",
    re.I,
)
SPLIT_CONNECTOR_RE = re.compile(r"\s*;\s*")
COMMA_ACTION_RE = re.compile(r",\s*(?=" + ACTION_PATTERN + r"\b)", re.I)
TEMPORAL_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=(?:after|then|next|finally|subsequently)\b)", re.I)
MARKS_AT_END_RE = re.compile(r"\s*\[\s*(\d+)\s*marks?\s*\]\s*$", re.I | re.S)
MARKS_ANY_RE = re.compile(r"\[\s*(\d+)\s*marks?\s*\]", re.I | re.S)
STRUCTURAL_PREFIXES = (
    r"^\s*\d{1,3}\s*[.)]\s*",
    r"^\s*\d{1,3}\s*\([a-z]\)\s*",
    r"^\s*\([a-z]\)\s*",
    r"^\s*\((?:viii|vii|vi|v|iv|iii|ii|ix|x|i)\)\s*",
    r"^\s*(?:viii|vii|vi|iv|iii|ii|ix|x|v|i)\.\s*",
    r"^\s*[—\-\s]*question[—\-\s]*:?\s*",
    r"^\s*[\u0088\u2022•▪◦‣\-–—*]+\s*",
)
ROMAN_LABELS = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}
CHOICE_TERMS_RE = re.compile(r"\b(?:mean|median|mode)\b.{0,160}\bor\b.{0,100}\b(?:mean|median|mode)\b", re.I | re.S)


def _string(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _pages(node: dict[str, Any]) -> list[int]:
    pages = node.get("source_pages", [])
    return sorted({p for p in pages if isinstance(p, int)}) if isinstance(pages, list) else []


def _marks(text: str) -> int | None:
    match = MARKS_AT_END_RE.search(text or "")
    return int(match.group(1)) if match else None


def _strip_structural_prefix(text: str) -> tuple[str, int]:
    # Uppercase A.-I. labels are structural alphabetic item markers. Handle
    # them case-sensitively before the general case-insensitive prefixes so a
    # label such as "B." is not emitted as a standalone requirement.
    match = re.match(r"^\s*[A-I]\.\s+", text)
    if match:
        return text[match.end():], match.end()
    for pattern in STRUCTURAL_PREFIXES:
        match = re.match(pattern, text, re.I)
        if match:
            return text[match.end():], match.end()
    return text, 0


def _without_marks(text: str) -> str:
    match = MARKS_AT_END_RE.search(text)
    return text[:match.start()].rstrip() if match else text


def _split_atomic_phrases(text: str) -> list[tuple[str, int, int]]:
    """Split obvious separate actions, preserving exact spans into source_text."""
    candidates: list[tuple[int, int]] = []
    candidates.extend((match.start(), match.end()) for match in SPLIT_CONNECTOR_RE.finditer(text))
    # A comma before a verb can separate two actions only when the left clause
    # already begins with an action; do not split "From dataset D, construct ..."
    # or "After executing ..., confirm ..." into misleading fragments.
    for match in COMMA_ACTION_RE.finditer(text):
        prior = text[:match.start()]
        last_boundary = max(prior.rfind("."), prior.rfind("!"), prior.rfind("?"), prior.rfind(";"), prior.rfind("\n"))
        left_clause = prior[last_boundary + 1:].strip()
        if re.match(r"^" + ACTION_PATTERN, left_clause, re.I):
            candidates.append((match.start(), match.end()))

    connector_pattern = re.compile(
        r"\b(?:and then|then|and|also)\s+(?="
        + ACTION_PATTERN
        + r"\s+(?!(?:the\s+)?same\b|(?:data|dataset|datasets|set)\b))",
        re.I,
    )
    candidates.extend((match.start(), match.end()) for match in connector_pattern.finditer(text))
    candidates.extend((match.start(), match.end()) for match in TEMPORAL_SENTENCE_RE.finditer(text))
    sentence_boundary_pattern = re.compile(r"(?<=[.!?])\s*(?=[A-Z0-9])")
    for match in sentence_boundary_pattern.finditer(text):
        # Do not interpret the last period in an ellipsis (e.g. L1, L2,...L12)
        # as the start of a new sentence.
        if text[max(0, match.start() - 3):match.start()] == "...":
            continue
        # For a match after a decimal point, match.start() is immediately
        # after the period (before any optional whitespace). The digits are
        # therefore at start-2 and start, not start-1 and start.
        if (
            match.start() >= 2
            and match.start() < len(text)
            and text[match.start() - 1] == "."
            and text[match.start() - 2].isdigit()
            and text[match.start()].isdigit()
        ):
            continue
        # A period after a single uppercase letter A.-I. is an item marker,
        # not a sentence boundary. This also handles labels inside a multi-line
        # block such as "i. ...\nA. First step ...\nB. Next step ...".
        if (
            match.start() >= 2
            and text[match.start() - 1] == "."
            and text[match.start() - 2] in "ABCDEFGHI"
            and (match.start() == 2 or text[match.start() - 3].isspace() or text[match.start() - 3] in "([")
        ):
            continue
        candidates.append((match.start(), match.end()))
    candidates.sort()

    ranges: list[tuple[int, int]] = []
    cursor = 0
    for start, end in candidates:
        if start < cursor:
            continue
        segment = text[cursor:start]
        left_trim = len(segment) - len(segment.lstrip())
        right_trim = len(segment) - len(segment.rstrip())
        if segment.strip():
            ranges.append((cursor + left_trim, start - right_trim))
        cursor = end
    tail = text[cursor:]
    left_trim = len(tail) - len(tail.lstrip())
    right_trim = len(tail) - len(tail.rstrip())
    if tail.strip():
        ranges.append((cursor + left_trim, len(text) - right_trim))

    result = []
    for start, end in ranges:
        phrase = text[start:end].strip()
        if not phrase or re.fullmatch(r"(?:and|then|also|and then)[., ]*", phrase, re.I):
            continue
        leading = len(text[start:end]) - len(text[start:end].lstrip())
        trailing = len(text[start:end]) - len(text[start:end].rstrip())
        result.append((phrase, start + leading, end - trailing))
    return result or ([(text.strip(), len(text) - len(text.lstrip()), len(text.rstrip()))] if text.strip() else [])


def _normal_for_classification(text: str) -> str:
    text = re.sub(r"(?<=\w)[\-‐‑]\s*\n\s*(?=[A-Za-z])", "", text)
    return re.sub(r"\s*\n\s*", " ", text)


def _kind(text: str) -> str:
    normalized = _normal_for_classification(text)
    if CONSTRAINT_RE.search(normalized):
        return "constraint"
    if REASONING_RE.search(normalized):
        return "reasoning"
    if DELIVERABLE_RE.search(normalized):
        return "deliverable"
    if ACTION_RE.search(normalized):
        return "action"
    return "needs_review"


def _is_structural_heading(text: str) -> bool:
    candidate = re.sub(r"^[\s\-–—•*\\\u0088]+", "", text).strip()
    return bool(re.fullmatch(
        r"(?:instructions?|general instructions?|notes?|requirements?|question|questions|answer|answers)\s*: ?",
        candidate, re.I,
    )) or not re.search(r"[A-Za-z0-9]", candidate)


def _looks_like_section_title(text: str) -> bool:
    body, _ = _strip_structural_prefix(text)
    body = _without_marks(body).strip()
    words = re.findall(r"[A-Za-z0-9]+", body)
    if not words or len(words) > 7:
        return False
    first = words[0].lower()
    # Short title fragments usually have a mark annotation and lack an action
    # verb at the start; actual tasks like "Generate a dataset" are retained.
    return bool(MARKS_ANY_RE.search(text)) and not re.match(r"^" + ACTION_PATTERN, body, re.I) and first not in {"with", "based", "from", "for", "using", "consider", "given"}


def _make_requirements(
    source_text: str,
    source_node_id: str,
    question_id: str | None,
    question_number: str | None,
    source_pages: list[int],
    category: str,
    alternative_group_id: str | None = None,
    inline_choice_group_id: str | None = None,
    parent_id: str | None = None,
) -> list[dict[str, Any]]:
    if _is_structural_heading(source_text) or _looks_like_section_title(source_text):
        return []
    working_text = _without_marks(source_text)
    wording, body_offset = _strip_structural_prefix(working_text)
    result = []
    source_marks = _marks(source_text)
    for index, (phrase, start, end) in enumerate(_split_atomic_phrases(wording), start=1):
        requirement_text = phrase.strip()
        if not requirement_text:
            continue
        absolute_start = body_offset + start
        absolute_end = body_offset + end
        # Inline method-choice IDs apply only to the phrase containing the
        # actual choice, not to a later verification/output step in the same node.
        phrase_choice_group = inline_choice_group_id if CHOICE_TERMS_RE.search(requirement_text) else None
        result.append({
            "requirement_id": f"{source_node_id}_r{index}",
            "parent_id": parent_id or source_node_id,
            "question_id": question_id,
            "question_number": question_number,
            "source_node_id": source_node_id,
            "requirement_type": _kind(requirement_text),
            "source_category": category,
            "atomic_text": requirement_text,
            "source_text": source_text,
            "source_span": {"start": absolute_start, "end": absolute_end},
            "source_pages": source_pages,
            "source_node_marks": source_marks,
            "alternative_group_id": alternative_group_id or phrase_choice_group,
            "dependencies": [],
            "status": "unassessed",
            "extraction_method": "rule_assisted_baseline",
            "needs_review": _kind(requirement_text) == "needs_review",
        })
    return result


def _raw_lines(node: dict[str, Any]) -> list[str]:
    lines = node.get("raw_lines", [])
    if isinstance(lines, list) and all(isinstance(line, str) for line in lines):
        return lines
    text = _string(node.get("text"))
    return text.splitlines() if text else []


def _all_descendant_nodes(node: dict[str, Any]):
    yield node
    for child in node.get("nested_subparts", []) or []:
        if isinstance(child, dict):
            yield from _all_descendant_nodes(child)
    for branch in node.get("alternative_branches", []) or []:
        if isinstance(branch, dict):
            yield from _all_descendant_nodes(branch)


def _question_level_blocks(question: dict[str, Any]) -> list[tuple[int, str]]:
    """Return parent prompt/context blocks, excluding all child-node lines."""
    raw_lines = question.get("raw_lines", [])
    if not isinstance(raw_lines, list) or not raw_lines:
        text = _string(question.get("text"))
        return [(1, text)] if text and not question.get("title_only") else []

    represented = set()
    roman_line_labels = set()
    for top in question.get("subparts", []) or []:
        if not isinstance(top, dict):
            continue
        for node in _all_descendant_nodes(top):
            for line in _raw_lines(node):
                represented.add(line.strip())
            node_text = _string(node.get("text"))
            represented.update(line.strip() for line in node_text.splitlines() if line.strip())
            label = _string(node.get("label")).lower()
            if label in ROMAN_LABELS:
                roman_line_labels.add(label)
    for branch in question.get("alternative_branches", []) or []:
        if isinstance(branch, dict):
            for node in _all_descendant_nodes(branch):
                represented.update(line.strip() for line in _raw_lines(node) if line.strip())

    result: list[tuple[int, str]] = []
    pending: list[str] = []
    pending_start = 0
    previous_index = 0

    def flush() -> None:
        nonlocal pending, pending_start, previous_index
        if pending:
            block = "\n".join(pending)
            if not _is_structural_heading(block) and not _looks_like_section_title(block):
                result.append((pending_start, block))
        pending = []
        pending_start = 0
        previous_index = 0

    for index, raw in enumerate(raw_lines, start=1):
        if not isinstance(raw, str) or not raw.strip():
            flush()
            continue
        line = raw.strip()
        roman_count = len(re.findall(r"\((?:viii|vii|vi|v|iv|iii|ii|ix|x|i)\)", line, re.I))
        if (index == 1 and question.get("title_only")) or line in represented or (roman_line_labels and roman_count >= 2):
            flush()
            continue
        if re.fullmatch(r"(?:OR|EITHER|ALTERNATIVELY)[.:]?", line, re.I) or _is_structural_heading(line):
            flush()
            continue
        if pending and index == previous_index + 1 and _is_wrapped_continuation(pending[-1], line):
            pending.append(line)
            previous_index = index
        else:
            flush()
            pending = [line]
            pending_start = index
            previous_index = index
    flush()
    return result


def _is_wrapped_continuation(previous: str, current: str) -> bool:
    left, right = previous.rstrip(), current.lstrip()
    if not left or not right or re.match(r"^[\-–—•*\u0088]", right):
        return False
    if re.search(r"[.!?:;][\"'’”)]?\s*$", left):
        return False
    return right[0].islower() or right[0].isdigit() or right[0] in "'\"([{“‘’”-–—" or left.endswith(("-", "‐", "‑"))


def _extract_node(
    node: dict[str, Any], node_id: str, category: str,
    question_id: str, question_number: str, question_pages: list[int],
    collected: list[dict[str, Any]], parent_node_id: str | None = None,
    inherited_alternative_group: str | None = None,
) -> dict[str, Any]:
    text = _string(node.get("text"))
    explicit_group = _string(node.get("alternative_group_id")) or inherited_alternative_group
    inline_group = _string(node.get("inline_choice_group_id")) or None
    reqs = _make_requirements(
        source_text=text, source_node_id=node_id, question_id=question_id,
        question_number=question_number, source_pages=_pages(node) or question_pages,
        category=category, alternative_group_id=explicit_group,
        inline_choice_group_id=inline_group,
        parent_id=parent_node_id or question_id,
    ) if text else []
    collected.extend(reqs)
    nested_outputs = []
    children = [child for child in (node.get("nested_subparts", []) or []) if isinstance(child, dict)]
    child_labels = [
        _string(child.get("label"), str(index))
        for index, child in enumerate(children, start=1)
    ]
    child_label_counts = Counter(child_labels)
    child_label_occurrences: Counter[str] = Counter()
    for child_index, child in enumerate(children, start=1):
        label = _string(child.get("label"), str(child_index))
        child_label_occurrences[label] += 1
        child_id = f"{node_id}_part_{label}"
        # Labels can repeat in OCR-affected or malformed source hierarchies.
        # Disambiguate only duplicated sibling labels, keeping ordinary IDs stable.
        if child_label_counts[label] > 1:
            child_id += f"_occurrence_{child_label_occurrences[label]}"
        child_category = "alternative_detail" if child.get("node_kind") == "alternative_branch" else "nested_subpart"
        child_group = _string(child.get("alternative_group_id")) or explicit_group
        child_out = _extract_node(
            child, child_id, child_category, question_id, question_number,
            question_pages, collected, node_id, child_group,
        )
        nested_outputs.append(child_out)
    alternative_outputs = []
    for branch_index, branch in enumerate(node.get("alternative_branches", []) or [], start=2):
        if not isinstance(branch, dict):
            continue
        label = _string(branch.get("label"), f"alternative_{branch_index}")
        branch_id = f"{node_id}_{label}"
        group = _string(branch.get("alternative_group_id")) or explicit_group
        alternative_outputs.append(_extract_node(
            branch, branch_id, "alternative_branch", question_id, question_number,
            question_pages, collected, node_id, group,
        ))
    return {
        "source_node_id": node_id,
        "parent_node_id": parent_node_id,
        "label": node.get("label"),
        "node_kind": node.get("node_kind"),
        "source_text": text,
        "source_pages": _pages(node) or question_pages,
        "marks": node.get("marks", _marks(text)),
        "alternative_group_id": explicit_group,
        "requirements": reqs,
        "nested_subparts": nested_outputs,
        "alternative_branches": alternative_outputs,
    }


def extract_atomic_requirements(question_map: dict[str, Any]) -> dict[str, Any]:
    """Create traceable requirement records from a parsed question map."""
    if not isinstance(question_map, dict):
        raise TypeError("question_map must be a dictionary returned by the question parser")
    source_filename = _string(question_map.get("source_filename"), "unknown")
    diagnostics = list(question_map.get("diagnostics", []) or [])
    global_requirements = []
    for index, item in enumerate(question_map.get("global_instructions", []) or [], start=1):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            diagnostics.append({"type": "invalid_global_instruction", "index": index})
            continue
        global_requirements.extend(_make_requirements(
            source_text=item["text"], source_node_id=f"global_{index}",
            question_id=None, question_number=None, source_pages=_pages(item),
            category="global_instruction",
        ))

    standalone_tasks_out = []
    all_requirements = list(global_requirements)
    for index, task in enumerate(question_map.get("standalone_tasks", []) or [], start=1):
        if not isinstance(task, dict) or not isinstance(task.get("text"), str):
            diagnostics.append({"type": "invalid_standalone_task", "index": index})
            continue
        task_id = _string(task.get("task_id"), f"task_unnumbered_{index}")
        task_reqs = _make_requirements(
            source_text=task["text"], source_node_id=task_id,
            question_id=None, question_number=None, source_pages=_pages(task),
            category="standalone_task",
        )
        standalone_tasks_out.append({
            "task_id": task_id, "source_text": task["text"],
            "source_pages": _pages(task), "marks": task.get("marks", _marks(task["text"])),
            "requirements": task_reqs,
        })
        all_requirements.extend(task_reqs)

    questions_out = []
    alternative_groups = []
    for question_index, question in enumerate(question_map.get("questions", []) or [], start=1):
        if not isinstance(question, dict):
            diagnostics.append({"type": "invalid_question_node", "index": question_index})
            continue
        question_id = _string(question.get("question_id"), f"q{question_index}_occurrence_1")
        question_number = _string(question.get("number"), str(question_index))
        question_pages = _pages(question)
        question_requirements = []
        subparts_out = []
        source_subparts = [part for part in (question.get("subparts", []) or []) if isinstance(part, dict)]
        subpart_labels = [
            _string(part.get("label"), str(index))
            for index, part in enumerate(source_subparts, start=1)
        ]
        subpart_label_counts = Counter(subpart_labels)
        subpart_label_occurrences: Counter[str] = Counter()
        for line_index, parent_line in _question_level_blocks(question):
            source_node_id = f"{question_id}_line_{line_index}"
            parent_reqs = _make_requirements(
                parent_line, source_node_id, question_id, question_number,
                question_pages, "question_prompt" if line_index == 1 else "question_context",
                parent_id=question_id,
            )
            question_requirements.extend(parent_reqs)

        for part_index, part in enumerate(source_subparts, start=1):
            label = _string(part.get("label"), str(part_index))
            subpart_label_occurrences[label] += 1
            node_id = f"{question_id}_part_{label}"
            if subpart_label_counts[label] > 1:
                node_id += f"_occurrence_{subpart_label_occurrences[label]}"
            part_out = _extract_node(
                part, node_id, "subpart", question_id, question_number,
                question_pages, question_requirements,
                parent_node_id=question_id,
            )
            subparts_out.append(part_out)

        top_level_alternatives = []
        for branch_index, branch in enumerate(question.get("alternative_branches", []) or [], start=1):
            if not isinstance(branch, dict):
                continue
            label = _string(branch.get("label"), f"alternative_{branch_index}")
            branch_id = f"{question_id}_{label}"
            top_level_alternatives.append(_extract_node(
                branch, branch_id, "alternative_branch", question_id,
                question_number, question_pages, question_requirements,
                parent_node_id=question_id,
                inherited_alternative_group=_string(branch.get("alternative_group_id")) or None,
            ))

        all_requirements.extend(question_requirements)
        q_groups = list(question.get("alternative_groups", []) or [])
        alternative_groups.extend(q_groups)
        questions_out.append({
            "question_id": question_id,
            "question_number": question_number,
            "occurrence": question.get("occurrence", 1),
            "title": question.get("title_text"),
            "title_only": bool(question.get("title_only", False)),
            "marks": question.get("marks", _marks(_string(question.get("heading_text")))),
            "source_text": _string(question.get("text")),
            "source_pages": question_pages,
            "requirements": question_requirements,
            "subparts": subparts_out,
            "alternative_branches": top_level_alternatives,
            "alternative_groups": q_groups,
            "alternative_markers": question.get("alternative_markers", []) or [],
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "source_filename": source_filename,
        "extraction": {
            "method": "rule_assisted_baseline",
            "model_name": None,
            "model_used": False,
            "student_code_executed": False,
        },
        "global_requirements": global_requirements,
        "standalone_tasks": standalone_tasks_out,
        "questions": questions_out,
        "alternative_groups": alternative_groups,
        "requirements": all_requirements,
        "diagnostics": diagnostics,
    }
