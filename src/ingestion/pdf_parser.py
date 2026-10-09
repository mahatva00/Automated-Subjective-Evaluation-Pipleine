
import argparse
import hashlib
import json
import os
import re
import shutil
from io import BytesIO
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image

TESSERACT_PATH = (
    os.environ.get("TESSERACT_CMD")
    or shutil.which("tesseract")
    or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

ROLL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(\d{2}[A-Za-z]{2,}\d{3,})(?![A-Za-z0-9])",
    re.IGNORECASE
)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def extract_filename_metadata(path):
    match = ROLL_PATTERN.search(path.stem)

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

def extract_page_content(page):
    text = page.get_text("text", sort=True)

    if text.strip():
        return text, "pdf_text", "success", []

    diagnostics = [
        "Page contains no extractable PDF text; OCR fallback attempted."
    ]

    try:
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(3, 3),
            alpha=False
        )
        image_bytes = pixmap.tobytes("png")

        with Image.open(BytesIO(image_bytes)) as image:
            ocr_text = pytesseract.image_to_string(
                image,
                lang="eng",
                config="--oem 3 --psm 6"
            )

        if ocr_text.strip():
            diagnostics.append(
                "OCR text may contain recognition errors; verify against the original page."
            )
            return ocr_text, "ocr", "partial", diagnostics

        diagnostics.append("OCR did not recognize any text on this page.")
        return "", "none", "partial", diagnostics

    except Exception as error:
        diagnostics.append(
            f"OCR failed: {type(error).__name__}: {error}"
        )
        return "", "none", "partial", diagnostics

def parse_pdf(path, input_root):
    diagnostics = []
    segments = []

    submission = {
        "submission_id": None,
        "student_roll_no": None,
        "course_name": None,
        "assessment_name": None,
        "source_filename": path.name,
        "source_relative_path": path.relative_to(input_root).as_posix(),
        "source_format": "pdf",
        "source_hash": None,
        "extraction_status": "failed"
    }

    record = {
        "schema_version": "1.0",
        "submission": submission,
        "segments": segments,
        "diagnostics": diagnostics
    }

    try:
        submission["source_hash"] = sha256_file(path)
        submission["submission_id"] = submission["source_hash"][:16]

        filename_metadata, filename_diagnostics = extract_filename_metadata(path)
        submission.update(filename_metadata)
        diagnostics.extend(filename_diagnostics)

        with pymupdf.open(path) as document:
            if document.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported.")

            if len(document) == 0:
                raise ValueError("PDF contains no pages.")

            for page_index, page in enumerate(document):
                image_info = page.get_images(full=True)
                text, extraction_method, status, page_diagnostics = (
                    extract_page_content(page)
                )

                segment = {
                    "segment_id": f"page_{page_index + 1:04d}",
                    "position": page_index,
                    "source_location": {
                        "page_number": page_index + 1
                    },
                    "original_type": "pdf_page",
                    "content": text,
                    "extraction_method": extraction_method,
                    "metadata": {
                        "page_width": page.rect.width,
                        "page_height": page.rect.height
                    },
                    "execution_count": None,
                    "outputs": [],
                    "attachments": {},
                    "embedded_image_count": len(image_info),
                    "semantic_role": "unclassified",
                    "extraction_status": status,
                    "diagnostics": page_diagnostics
                }

                segments.append(segment)

                for message in page_diagnostics:
                    diagnostics.append(
                        f"Page {page_index + 1}: {message}"
                    )

        submission["extraction_status"] = (
            "partial"
            if any(s["extraction_status"] != "success" for s in segments)
            else "success"
        )

    except (OSError, ValueError, RuntimeError) as error:
        diagnostics.append(
            f"PDF extraction failed: {type(error).__name__}: {error}"
        )
        submission["extraction_status"] = "failed"

    return record

def process_directory(input_dir, output_dir):
    input_root = Path(input_dir).resolve()
    output_root = Path(output_dir).resolve()

    if not input_root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_root}")

    if input_root == output_root or input_root in output_root.parents:
        raise ValueError("Output directory must be outside the input directory.")

    files = sorted(input_root.rglob("*.pdf"))

    if not files:
        print(f"No PDF files found in {input_root}")
        return

    counts = {"success": 0, "partial": 0, "failed": 0}

    for path in files:
        relative_path = path.relative_to(input_root)
        output_path = (output_root / relative_path).with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            record = parse_pdf(path, input_root)

            with output_path.open("w", encoding="utf-8") as file:
                json.dump(record, file, ensure_ascii=False, indent=2)

            status = record["submission"]["extraction_status"]
            counts[status] += 1
            print(f"[{status.upper()}] {relative_path} -> {output_path}")

        except OSError as error:
            counts["failed"] += 1
            print(f"[FAILED] {relative_path}: {error}")

    print("\nProcessing summary")
    print(f"PDFs found: {len(files)}")
    print(f"Successful: {counts['success']}")
    print(f"Partial: {counts['partial']}")
    print(f"Failed: {counts['failed']}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract PDF pages into JSON without executing embedded code."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
