
import argparse
import hashlib
import json
import math
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


def _ocr_text(image):
    candidates = []
    for page_segmentation_mode in (6, 11):
        data = pytesseract.image_to_data(
            image,
            lang="eng",
            config=f"--oem 3 --psm {page_segmentation_mode}",
            output_type=pytesseract.Output.DICT
        )
        lines = {}
        confidences = []
        word_count = 0
        for index, value in enumerate(data.get("text", [])):
            word = str(value).strip()
            if not word:
                continue
            word_count += 1
            try:
                confidence = float(data["conf"][index])
            except (KeyError, IndexError, TypeError, ValueError):
                confidence = -1
            if confidence >= 0:
                confidences.append(confidence)
            line_key = tuple(
                data.get(key, [0] * len(data.get("text", [])))[index]
                for key in ("block_num", "par_num", "line_num")
            )
            lines.setdefault(line_key, []).append(word)

        text = "\n".join(" ".join(words) for words in lines.values()).strip()
        average_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        quality = average_confidence * math.log1p(word_count)
        candidates.append((quality, word_count, text, average_confidence, page_segmentation_mode))

    quality, word_count, text, confidence, psm = max(candidates, key=lambda row: (row[0], row[1]))
    return text, {
        "word_count": word_count,
        "average_confidence": round(confidence, 2),
        "page_segmentation_mode": psm,
    }


def _image_regions(page):
    page_area = max(page.rect.width * page.rect.height, 1)
    seen = set()
    regions = []
    for info in page.get_image_info(xrefs=True):
        bbox = info.get("bbox")
        if not bbox:
            continue
        rect = pymupdf.Rect(bbox)
        if rect.is_empty or rect.is_infinite:
            continue
        if rect.get_area() / page_area < 0.015:
            continue
        key = tuple(round(value, 1) for value in rect)
        if key not in seen:
            seen.add(key)
            regions.append(rect)
    return regions


def extract_page_content(page):
    text = page.get_text("text", sort=True)
    image_regions = _image_regions(page)

    if text.strip() and not image_regions:
        return text, "pdf_text", "success", [], []

    try:
        if text.strip():
            image_evidence = []
            diagnostics = []
            for region_index, region in enumerate(image_regions, start=1):
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(4, 4),
                    clip=region,
                    alpha=False
                )
                with Image.open(BytesIO(pixmap.tobytes("png"))) as image:
                    ocr_text, quality = _ocr_text(image)
                if ocr_text:
                    image_evidence.append({
                        "region_index": region_index,
                        "bbox": [round(value, 2) for value in region],
                        "ocr_text": ocr_text,
                        "ocr_confidence": quality["average_confidence"],
                        "is_authoritative": False,
                    })
                    diagnostics.append(
                        "OCR extracted text from embedded image region "
                        f"{region_index} (confidence {quality['average_confidence']:.2f}); "
                        "verify against the rendered page."
                    )
            if image_evidence:
                return text, "pdf_text+image_ocr", "partial", diagnostics, image_evidence
            if image_regions:
                return (
                    text,
                    "pdf_text",
                    "partial",
                    ["Embedded image regions were present but OCR found no readable text."],
                    [],
                )
            return text, "pdf_text", "success", [], []

        diagnostics = ["Page contains no extractable PDF text; OCR fallback attempted."]
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(4, 4), alpha=False)
        with Image.open(BytesIO(pixmap.tobytes("png"))) as image:
            ocr_text, quality = _ocr_text(image)
        if ocr_text:
            diagnostics.append(
                "OCR text may contain recognition errors; verify against the original page. "
                f"Selected page-segmentation mode {quality['page_segmentation_mode']} "
                f"with confidence {quality['average_confidence']:.2f}."
            )
            return ocr_text, "ocr", "partial", diagnostics, [{
                "region_index": None,
                "bbox": None,
                "ocr_text": ocr_text,
                "ocr_confidence": quality["average_confidence"],
                "is_authoritative": False,
                "full_page": True,
            }]

        diagnostics.append("OCR did not recognize any text on this page.")
        return "", "none", "partial", diagnostics, []
    except Exception as error:
        diagnostics = [
            f"OCR failed: {type(error).__name__}: {error}"
        ]
        return text, "pdf_text" if text.strip() else "none", "partial", diagnostics, []

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
                image_info = page.get_image_info(xrefs=True)
                text, extraction_method, status, page_diagnostics, image_evidence = (
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
                        "page_height": page.rect.height,
                        "extraction_method": extraction_method,
                        "embedded_image_count": len(image_info),
                        "image_evidence": image_evidence,
                        "canonical_text": extraction_method == "pdf_text",
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
