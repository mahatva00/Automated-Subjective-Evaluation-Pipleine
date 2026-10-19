import argparse
import hashlib
import json
import re
from pathlib import Path

ROLL_PATTERN = re.compile(r"^\d{2}[A-Za-z]{2,}\d{3,}$", re.IGNORECASE)
QUESTION_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:question(?:\s+no\.?)?|q)\s*[:#.\-]?\s*"
    r"(?P<number>\d{1,3})(?:\s*[:.)\-]\s*|\s+)(?P<title>.*\S)?\s*$",
    re.IGNORECASE,
)
NUMERIC_QUESTION_HEADING_RE = re.compile(
    r"^\s*#{1,6}\s*(?P<number>\d{1,3})\s*[.)\-:]\s*(?P<title>.*\S)\s*$"
)
AI_LOG_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:ai\s*(?:log|usage|assistance|tools?)|"
    r"ai[-\s]?generated|my\s+use|tools?\s+used|"
    r"(?:no|not|without)\s+(?:any\s+)?ai(?:\s+tools?)?\b|"
    r"ai\s+prompt\b)|"
    r"\b(?:i|we)\s+(?:used|did\s+not\s+use|didn't\s+use)\s+"
    r"(?:any\s+)?(?:ai|chatgpt|gemini|ollama|copilot)\b",
)
QUESTION_COPY_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:\([a-z]\)|[a-z][.)])\s+|"
    r"^\s*(?:#{1,6}\s*)?(?:[A-Z]\.)?\s*Experiment\s+\d+\b|"
    r"\[\s*\d+(?:\.\d+)?\s*marks?\s*\]",
)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def source_to_text(source):
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(x, str) for x in source):
        return "".join(source)
    return None


def _output_to_text(outputs):
    parts = []
    if not isinstance(outputs, list):
        return ""
    for output in outputs:
        if not isinstance(output, dict):
            continue
        for key in ("text", "data"):
            value = output.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value)
                break
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                text = "".join(value)
                if text.strip():
                    parts.append(text)
                    break
            if isinstance(value, dict):
                rendered_text = ""
                for mime in ("text/plain", "text/markdown", "text/html"):
                    rendered = value.get(mime)
                    if isinstance(rendered, str) and rendered.strip():
                        rendered_text = rendered
                        break
                    if isinstance(rendered, list) and all(isinstance(item, str) for item in rendered):
                        rendered_text = "".join(rendered)
                        if rendered_text.strip():
                            break
                if rendered_text:
                    parts.append(rendered_text)
                    break
    return "\n".join(parts).strip()


def _question_number(text):
    for line in text.splitlines():
        for pattern in (QUESTION_HEADING_RE, NUMERIC_QUESTION_HEADING_RE):
            match = pattern.match(line)
            if match:
                return str(int(match.group("number"))), (match.group("title") or "").strip()
    return None, ""


def _markdown_role(text):
    number, title = _question_number(text)
    if number is not None:
        return "question", number, title
    if AI_LOG_RE.search(text):
        return "ai_log", None, ""
    if QUESTION_COPY_RE.search(text):
        return "question_copy", None, ""
    return "explanation", None, ""

def extract_filename_metadata(path):
    match = re.search(
        r"(?<![A-Za-z0-9])(\d{2}[A-Za-z]{2,}\d{3,})(?![A-Za-z0-9])",
        path.stem,
        re.IGNORECASE
    )

    if not match:
        return {
            "student_roll_no": None,
            "course_name": None,
            "assessment_name": None
        }, ["Could not identify roll number using the configured filename pattern."]

    roll_no = match.group(1)
    course = path.stem[:match.start()].rstrip("_- ") or None
    assessment = path.stem[match.end():].lstrip("_- ") or None
    diagnostics = []

    if course is None:
        diagnostics.append("Course name is missing from the filename.")
    if assessment is None:
        diagnostics.append("Assessment name is missing from the filename.")

    return {
        "student_roll_no": roll_no,
        "course_name": course,
        "assessment_name": assessment
    }, diagnostics

def make_segment(cell, index, current_question_number=None):
    diagnostics = []
    cell_type = cell.get("cell_type", "unknown")
    source = source_to_text(cell.get("source", []))

    if source is None:
        diagnostics.append("Cell source has an unsupported structure.")
        source = ""

    outputs = cell.get("outputs", [])
    if not isinstance(outputs, list):
        diagnostics.append("Cell outputs are not represented as a list.")
        outputs = []

    if cell_type not in {"code", "markdown", "raw"}:
        diagnostics.append(f"Unsupported or unknown cell type: {cell_type}")

    if not source.strip():
        diagnostics.append("Cell source is empty.")

    execution_count = cell.get("execution_count")
    if cell_type == "code" and execution_count is not None:
        if not isinstance(execution_count, int) or isinstance(execution_count, bool):
            diagnostics.append("Execution count has an unexpected type.")

    if cell_type == "markdown":
        role, heading_number, heading_title = _markdown_role(source)
        question_number = heading_number or current_question_number
    elif cell_type == "code":
        role = "code"
        heading_number = None
        heading_title = ""
        question_number = current_question_number
    else:
        role = "explanation"
        heading_number = None
        heading_title = ""
        question_number = current_question_number

    blocks = []
    if source.strip() or role == "question":
        blocks.append({
            "block_id": f"cell_{index:04d}_{role}",
            "component": role,
            "question_number": question_number,
            "source_text": source,
            "output_text": "",
            "source_location": {"cell_index": index},
            "segment_id": f"cell_{index:04d}",
        })
    output_text = _output_to_text(outputs)
    if cell_type == "code" and output_text:
        blocks.append({
            "block_id": f"cell_{index:04d}_output",
            "component": "output",
            "question_number": question_number,
            "source_text": "",
            "output_text": output_text,
            "source_location": {"cell_index": index},
            "segment_id": f"cell_{index:04d}",
        })

    return {
        "segment_id": f"cell_{index:04d}",
        "position": index,
        "source_location": {"cell_index": index},
        "original_type": cell_type,
        "content": source,
        "metadata": cell.get("metadata", {}),
        "execution_count": execution_count,
        "outputs": outputs,
        "attachments": cell.get("attachments", {}),
        "semantic_role": role,
        "component_role": role,
        "question_number": question_number,
        "question_heading_title": heading_title,
        "blocks": blocks,
        "extraction_status": (
            "partial"
            if source_to_text(cell.get("source", [])) is None
            or not isinstance(cell.get("outputs", []), list)
            else "success"
        ),
        "diagnostics": diagnostics,
        "original_cell": cell
    }

def parse_notebook(path, input_root):
    diagnostics = []
    relative_path = path.relative_to(input_root)

    record = {
        "schema_version": "1.0",
        "submission": {
            "submission_id": sha256_file(path)[:16],
            "student_roll_no": None,
            "course_name": None,
            "assessment_name": None,
            "source_filename": path.name,
            "source_relative_path": relative_path.as_posix(),
            "source_format": "ipynb",
            "source_hash": None,
            "extraction_status": "failed"
        },
        "segments": [],
        "diagnostics": []
    }

    try:
        record["submission"]["source_hash"] = sha256_file(path)
        filename_metadata, filename_diagnostics = extract_filename_metadata(path)
        record["submission"].update(filename_metadata)
        diagnostics.extend(filename_diagnostics)

        with path.open("r", encoding="utf-8-sig") as file:
            notebook = json.load(file)

        if not isinstance(notebook, dict):
            raise ValueError("Notebook root must be a JSON object.")

        if not isinstance(notebook.get("cells"), list):
            raise ValueError("Notebook does not contain a valid cells list.")

        record["notebook_metadata"] = notebook.get("metadata", {})
        record["nbformat"] = notebook.get("nbformat")
        record["nbformat_minor"] = notebook.get("nbformat_minor")

        question_blocks = {}
        current_question_number = None
        for index, cell in enumerate(notebook["cells"]):
            if not isinstance(cell, dict):
                diagnostics.append(f"Cell {index} is not a JSON object.")
                record["segments"].append({
                    "segment_id": f"cell_{index:04d}",
                    "position": index,
                    "source_location": {"cell_index": index},
                    "original_type": "unknown",
                    "content": "",
                    "semantic_role": "unclassified",
                    "extraction_status": "failed",
                    "diagnostics": ["Cell is not a JSON object."],
                    "original_cell": cell
                })
                continue

            segment = make_segment(cell, index, current_question_number)
            if segment.get("component_role") == "question":
                current_question_number = segment.get("question_number")
                for block in segment.get("blocks", []):
                    block["question_number"] = current_question_number
            else:
                for block in segment.get("blocks", []):
                    block["question_number"] = current_question_number
            for block in segment.get("blocks", []):
                number = block.get("question_number")
                if number is not None:
                    question_blocks.setdefault(str(number), []).append(block)
            segment["question_number"] = current_question_number
            record["segments"].append(segment)

            for message in segment["diagnostics"]:
                diagnostics.append(f"Cell {index}: {message}")

        record["question_blocks"] = question_blocks
        record["component_contract"] = {
            "scored_components": ["code", "output", "explanation"],
            "optional_components": ["question", "question_copy", "ai_log"],
            "ai_log_scored": False,
        }
        record["question_component_summary"] = {
            number: {
                "present_components": sorted({block["component"] for block in blocks}),
                "missing_components": [
                    component for component in ("code", "output", "explanation")
                    if component not in {block["component"] for block in blocks}
                ],
                "ai_log_present": any(block["component"] == "ai_log" for block in blocks),
            }
            for number, blocks in question_blocks.items()
        }
        record["submission"]["extraction_status"] = (
            "partial"
            if any(
                segment["extraction_status"] in {"partial", "failed"}
                for segment in record["segments"]
            )
            else "success"
        )

    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        diagnostics.append(f"Notebook extraction failed: {type(error).__name__}: {error}")

    record["diagnostics"] = diagnostics
    return record

def process_directory(input_dir, output_dir):
    input_root = Path(input_dir).resolve()
    output_root = Path(output_dir).resolve()

    if not input_root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_root}")

    if input_root == output_root or input_root in output_root.parents:
        raise ValueError("Output directory must be outside the input directory.")

    files = sorted(input_root.rglob("*.ipynb"))
    if not files:
        print(f"No .ipynb files found in {input_root}")
        return

    counts = {"success": 0, "partial": 0, "failed": 0}

    for path in files:
        relative_path = path.relative_to(input_root)
        output_path = (output_root / relative_path).with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            record = parse_notebook(path, input_root)
            with output_path.open("w", encoding="utf-8") as file:
                json.dump(record, file, ensure_ascii=False, indent=2)

            status = record["submission"]["extraction_status"]
            counts[status] += 1
            print(f"[{status.upper()}] {relative_path} -> {output_path}")

        except OSError as error:
            counts["failed"] += 1
            print(f"[FAILED] {relative_path}: {error}")

    print("\nProcessing summary")
    print(f"Notebooks found: {len(files)}")
    print(f"Successful: {counts['success']}")
    print(f"Partial: {counts['partial']}")
    print(f"Failed: {counts['failed']}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract Jupyter notebook contents into JSON without executing student code."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()