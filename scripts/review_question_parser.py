
import json
import sys
from pathlib import Path
from pprint import pformat

from src.ingestion.question_parser import parse_question_paper

ROOT = Path("data/processed/pdfs")
OUTPUT_FILE = Path("question_parser_review.txt")
PAPERS_TO_REVIEW = {"FAT", "LPS3", "LPS4", "LPS7", "PAT2"}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    with OUTPUT_FILE.open("w", encoding="utf-8") as report:
        original_stdout = sys.stdout
        sys.stdout = report
        try:
            review()
        finally:
            sys.stdout = original_stdout

    print(f"Review completed. Report saved to: {OUTPUT_FILE.resolve()}")


def print_section(title, value):
    print(f"\n--- {title} ---")
    if not value:
        print("(none)")
    else:
        print(pformat(value, width=120, sort_dicts=False))


def review():
    files = sorted(ROOT.rglob("*-QP.json"))
    selected_files = [
        path for path in files
        if path.parent.name.upper() in PAPERS_TO_REVIEW
    ]
    found = {path.parent.name.upper() for path in selected_files}

    print("=" * 100)
    print("QUESTION PARSER CONTENT REVIEW")
    print(f"Input directory: {ROOT}")
    print(f"Requested papers: {sorted(PAPERS_TO_REVIEW)}")
    print(f"Found papers: {sorted(found)}")
    print("=" * 100)

    missing = PAPERS_TO_REVIEW - found
    if missing:
        print(f"WARNING: Papers not found: {sorted(missing)}")

    for path in selected_files:
        document = json.loads(path.read_text(encoding="utf-8"))
        result = parse_question_paper(document)

        print("\n" + "#" * 100)
        print(f"PAPER: {path.parent.name.upper()} | FILE: {path.name}")
        print("#" * 100)

        print_section("SOURCE FILENAME", result.get("source_filename"))
        print_section("GLOBAL INSTRUCTIONS", result.get("global_instructions"))

        questions = result.get("questions", [])
        print(f"\n--- QUESTIONS ({len(questions)}) ---")

        for index, question in enumerate(questions, start=1):
            print("\n" + "-" * 80)
            print(
                f"QUESTION RECORD {index}: "
                f"number={question.get('number')}, "
                f"occurrence={question.get('occurrence')}, "
                f"pages={question.get('source_pages')}"
            )
            print("-" * 80)

            print_section("QUESTION TEXT", question.get("text"))
            print_section("RAW LINES", question.get("raw_lines"))
            print_section("SUBPARTS", question.get("subparts"))
            print_section(
                "QUESTION ALTERNATIVE MARKERS",
                question.get("alternative_markers")
            )

        print_section("NOTES", result.get("notes"))
        print_section("ALTERNATIVES", result.get("alternatives"))
        print_section("DIAGNOSTICS", result.get("diagnostics"))

    print("\n" + "=" * 100)
    print("REVIEW COMPLETE")
    print("No student-submitted code was executed.")
    print("=" * 100)


if __name__ == "__main__":
    main()
