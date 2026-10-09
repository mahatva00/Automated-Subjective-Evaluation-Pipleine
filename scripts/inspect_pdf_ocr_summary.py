
import json
from collections import Counter
from pathlib import Path

root = Path("data/processed/pdfs")
method_counts = Counter()
page_status_counts = Counter()
ocr_pages = 0
empty_ocr_pages = 0
ocr_characters = 0
json_count = 0

for path in sorted(root.rglob("*.json")):
    with path.open("r", encoding="utf-8") as file:
        record = json.load(file)

    json_count += 1

    for segment in record.get("segments", []):
        method = segment.get("extraction_method", "missing")
        method_counts[method] += 1
        page_status_counts[segment.get("extraction_status", "missing")] += 1

        if method == "ocr":
            ocr_pages += 1
            characters = len(segment.get("content", "").strip())
            ocr_characters += characters

            if characters == 0:
                empty_ocr_pages += 1

print(f"JSON files inspected: {json_count}")
print("\nExtraction methods:")
for method, count in sorted(method_counts.items()):
    print(f"  {method}: {count}")

print("\nPage statuses:")
for status, count in sorted(page_status_counts.items()):
    print(f"  {status}: {count}")

print(f"\nOCR pages: {ocr_pages}")
print(f"OCR pages with empty content: {empty_ocr_pages}")
print(f"Characters recovered by OCR: {ocr_characters}")
