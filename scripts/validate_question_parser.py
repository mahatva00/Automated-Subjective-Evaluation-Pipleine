
import json
import re
from collections import Counter
from pathlib import Path

from src.ingestion.question_parser import parse_question_paper


ROOT = Path("data/processed/pdfs")
EXPECTED_PAPERS = {
    "FAT", "LPS1", "LPS2", "LPS3", "LPS4", "LPS5",
    "LPS6", "LPS7", "LPS8", "PAT1", "PAT2",
}

ADMIN_TEXT = re.compile(
    r"after completing the exam|name your file as|"
    r"upload the single pdf|example format|"
    r"you may use ai tools|for each question, if you used any ai",
    re.I,
)


def main():
    files = sorted(ROOT.rglob("*-QP.json"))
    found = {path.parent.name.upper() for path in files}

    print(f"Question papers found: {len(files)}")
    print("Missing papers:", sorted(EXPECTED_PAPERS - found))
    print("Unexpected papers:", sorted(found - EXPECTED_PAPERS))

    if found != EXPECTED_PAPERS:
        print("WARNING: The available question-paper set differs from the expected set.")

    total_questions = 0
    total_subparts = 0
    total_diagnostics = 0
    all_results = {}

    for path in files:
        document = json.loads(path.read_text(encoding="utf-8"))
        result = parse_question_paper(document)
        paper = path.parent.name.upper()
        questions = result["questions"]

        total_questions += len(questions)
        total_subparts += sum(len(q["subparts"]) for q in questions)
        total_diagnostics += len(result["diagnostics"])
        all_results[paper] = result

        print(f"\n{'=' * 90}")
        print(f"{paper} | {path.name}")
        print(
            f"Questions: {len(questions)} | "
            f"Subparts: {sum(len(q['subparts']) for q in questions)} | "
            f"Global instructions: {len(result['global_instructions'])} | "
            f"Notes: {len(result['notes'])} | "
            f"Alternatives flagged: {len(result['alternatives'])}"
        )

        for question in questions:
            first_line = question["text"].splitlines()[0][:160]
            subpart_labels = [part["label"] for part in question["subparts"]]
            print(
                f"  Q{question['number']} "
                f"(occurrence {question['occurrence']}, "
                f"pages {question['source_pages']}, "
                f"subparts {subpart_labels}): {first_line}"
            )

            if ADMIN_TEXT.search(first_line):
                print("    REVIEW: Possible administrative text parsed as a question.")

        if result["diagnostics"]:
            print("Diagnostics:")
            for diagnostic in result["diagnostics"]:
                print(f"  - {diagnostic}")

    print(f"\n{'=' * 90}")
    print("OVERALL SUMMARY")
    print(f"Papers processed: {len(all_results)}")
    print(f"Total question records: {total_questions}")
    print(f"Total subpart records: {total_subparts}")
    print(f"Total parser diagnostics: {total_diagnostics}")

    counts = Counter(
        len(result["questions"]) for result in all_results.values()
    )
    print("Question-count distribution across papers:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
