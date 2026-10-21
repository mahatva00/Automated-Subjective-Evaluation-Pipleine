# Dependency Manifests

Keep core assessment dependencies separate from dependencies needed only for DOCX policies or DistilBERT workflows.

## What Happened
The root project had no dependency manifest despite supporting PDF parsing, OCR, multiple evaluator providers, DOCX policy parsing, and DistilBERT workflows. The active Windows environment established tested versions for the base and ML packages, while DOCX support was not installed or exercised. This was identified during the intelligent-notebook-evaluator setup/documentation follow-up (session 1bd4afd4-d15d-4ea5-a699-020f5d06be2c).

## Takeaway
Use `requirements.txt` for the captured assessment runtime, `requirements-ml.txt` for optional model work, and `requirements-docx.txt` for optional DOCX policies. Keep external executables such as Tesseract and provider services/credentials documented separately; Python package manifests do not install or configure them.

## History
- 2026-10-19 (intelligent-notebook-evaluator/session 1bd4afd4-d15d-4ea5-a699-020f5d06be2c): initial
