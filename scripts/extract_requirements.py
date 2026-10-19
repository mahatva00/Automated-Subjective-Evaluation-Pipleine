"""Run the baseline atomic-requirement extractor on a parsed question-map JSON."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.requirements.requirement_extractor import extract_atomic_requirements


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Parsed question-map JSON file")
    parser.add_argument("--output", required=True, help="Output requirements JSON file")
    args = parser.parse_args()
    source = Path(args.input)
    destination = Path(args.output)
    if not source.is_file():
        parser.error(f"Input file does not exist: {source}")
    try:
        question_map = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"Cannot read valid JSON from {source}: {exc}")
    result = extract_atomic_requirements(question_map)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Extracted {len(result['requirements'])} requirements from {source.name}")
    print(f"Output: {destination.resolve()}")
    print("Student code executed: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
