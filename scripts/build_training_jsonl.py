#!/usr/bin/env python3
"""Compatibility entry point; refuses the legacy unscoped training export.

Use build_training_jsonl_coverage_safe.py (directly or via this wrapper) with both
--coverage-dir and --scope-dir. This prevents out-of-scope/pending text from becoming O labels.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.build_training_jsonl_coverage_safe import main as safe_main


def main() -> int:
    argv = sys.argv[1:]
    if "--coverage-dir" not in argv or "--scope-dir" not in argv:
        print("ERROR: unscoped training export is disabled by question-scope policy v1.2.")
        print("Use scripts/build_training_jsonl_coverage_safe.py with --coverage-dir and --scope-dir.")
        print("Out-of-scope or unresolved tokens must not be converted into background/O training labels.")
        return 2
    return safe_main()


if __name__ == "__main__":
    raise SystemExit(main())
