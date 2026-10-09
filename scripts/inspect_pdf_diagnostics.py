
import json
from pathlib import Path

root = Path("data/processed/pdfs")
files = sorted(root.rglob("*.json"))
partial_count = 0

for path in files:
    record = json.loads(path.read_text(encoding="utf-8"))
    submission = record["submission"]

    if submission["extraction_status"] != "partial":
        continue

    partial_count += 1
    print(f"\nPDF: {submission['source_relative_path']}")

    for segment in record["segments"]:
        if segment["extraction_status"] != "success":
            location = segment["source_location"]
            print(
                f"  Page {location.get('page_number', '?')}: "
                f"characters={len(segment.get('content', ''))}, "
                f"embedded_images={segment.get('embedded_image_count', 0)}"
            )
            for diagnostic in segment.get("diagnostics", []):
                print(f"    Diagnostic: {diagnostic}")

print(f"\nPartial PDFs inspected: {partial_count}")
