
import argparse
import hashlib
import json
import re
from pathlib import Path

ROLL_PATTERN = re.compile(r"^\d{2}[A-Za-z]{2,}\d{3,}$", re.IGNORECASE)

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

def extract_filename_metadata(path):
    tokens = path.stem.split("_")
    roll_index = None
    roll_no = None

    for index, token in enumerate(tokens):
        if ROLL_PATTERN.fullmatch(token):
            roll_index = index
            roll_no = token
            break

    if roll_index is None:
        return {
            "student_roll_no": None,
            "course_name": None,
            "assessment_name": None
        }, ["Could not identify roll number using the configured filename pattern."]

    course = "_".join(tokens[:roll_index]) or None
    assessment = "_".join(tokens[roll_index + 1:]) or None
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

def make_segment(cell, index):
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
        "semantic_role": "unclassified",
        "extraction_status": "success" if not diagnostics else "warning",
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

            segment = make_segment(cell, index)
            record["segments"].append(segment)

            for message in segment["diagnostics"]:
                diagnostics.append(f"Cell {index}: {message}")

        record["submission"]["extraction_status"] = (
            "partial" if diagnostics else "success"
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
