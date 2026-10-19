
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pymupdf
from PIL import Image

from src.ingestion.pdf_parser import parse_pdf


class TestPdfParser(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.pdf_path = self.root / "PAMCA504-25MCA1006-LPS2.pdf"

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_pdf(self, text=None, embedded_image=False):
        with pymupdf.open() as document:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
            if embedded_image:
                image_bytes = BytesIO()
                Image.new("RGB", (120, 80), "white").save(image_bytes, format="PNG")
                page.insert_image(
                    pymupdf.Rect(72, 120, 250, 220),
                    stream=image_bytes.getvalue()
                )
            document.save(self.pdf_path)

    @staticmethod
    def ocr_data(text, confidence="90"):
        words = text.split()
        return {
            "text": words,
            "conf": [confidence] * len(words),
            "block_num": [1] * len(words),
            "par_num": [1] * len(words),
            "line_num": [1] * len(words),
        }

    def test_extracts_native_pdf_text(self):
        self.create_pdf("Student answer: import pandas as pd")
        record = parse_pdf(self.pdf_path, self.root)
        segment = record["segments"][0]

        self.assertIn("import pandas as pd", segment["content"])
        self.assertEqual(segment["extraction_method"], "pdf_text")
        self.assertEqual(segment["extraction_status"], "success")

    @patch("src.ingestion.pdf_parser.pytesseract.image_to_data")
    def test_uses_ocr_when_page_has_no_text(self, mock_ocr):
        self.create_pdf()
        mock_ocr.return_value = self.ocr_data("class_sep = 1.0")

        record = parse_pdf(self.pdf_path, self.root)
        segment = record["segments"][0]

        self.assertEqual(segment["content"], "class_sep = 1.0")
        self.assertEqual(segment["extraction_method"], "ocr")
        self.assertEqual(segment["extraction_status"], "partial")
        self.assertEqual(mock_ocr.call_count, 2)
        self.assertTrue(any("psm 11" in call.kwargs["config"] for call in mock_ocr.call_args_list))

    @patch("src.ingestion.pdf_parser.pytesseract.image_to_data")
    def test_empty_ocr_result_is_partial(self, mock_ocr):
        self.create_pdf()
        mock_ocr.return_value = self.ocr_data("")

        record = parse_pdf(self.pdf_path, self.root)
        segment = record["segments"][0]

        self.assertEqual(segment["content"], "")
        self.assertEqual(segment["extraction_method"], "none")
        self.assertEqual(segment["extraction_status"], "partial")
        self.assertTrue(segment["diagnostics"])

    @patch("src.ingestion.pdf_parser.pytesseract.image_to_data")
    def test_ocr_is_kept_as_non_authoritative_image_evidence_on_mixed_pages(self, mock_ocr):
        self.create_pdf("Cell code: plt.show()", embedded_image=True)
        mock_ocr.return_value = self.ocr_data("DF1 Bar Plot P3 Count")

        record = parse_pdf(self.pdf_path, self.root)
        segment = record["segments"][0]

        self.assertIn("Cell code: plt.show()", segment["content"])
        self.assertNotIn("DF1 Bar Plot P3 Count", segment["content"])
        self.assertEqual(segment["extraction_method"], "pdf_text+image_ocr")
        self.assertEqual(segment["extraction_status"], "partial")
        self.assertEqual(segment["metadata"]["embedded_image_count"], 1)
        self.assertEqual(segment["metadata"]["canonical_text"], False)
        self.assertEqual(segment["metadata"]["image_evidence"][0]["ocr_text"], "DF1 Bar Plot P3 Count")
        self.assertFalse(segment["metadata"]["image_evidence"][0]["is_authoritative"])
        mock_ocr.assert_called()

    def test_preserves_page_number_and_filename_metadata(self):
        self.create_pdf("Sample answer")
        record = parse_pdf(self.pdf_path, self.root)
        submission = record["submission"]
        segment = record["segments"][0]

        self.assertEqual(submission["student_roll_no"], "25MCA1006")
        self.assertEqual(submission["course_name"], "PAMCA504")
        self.assertEqual(submission["assessment_name"], "LPS2")
        self.assertEqual(segment["source_location"]["page_number"], 1)

    def test_student_code_is_never_executed(self):
        self.create_pdf("import os\nos.remove('important_file.txt')")

        with patch("src.ingestion.pdf_parser.pytesseract.image_to_string") as mock_ocr:
            record = parse_pdf(self.pdf_path, self.root)

        self.assertIn("os.remove", record["segments"][0]["content"])
        mock_ocr.assert_not_called()


if __name__ == "__main__":
    unittest.main()
