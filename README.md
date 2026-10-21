# Intelligent Notebook Evaluator

Python tooling for static, evidence-grounded assessment of student notebooks and PDFs, plus a separate human-reviewed question-paper annotation workflow.

## Safety and evaluation boundaries

- Student code is parsed as text and is never executed.
- Requirement extraction and evidence mapping are heuristic; review diagnostics and source provenance.
- LLM assessments are classifications of supplied evidence, not official grades.
- Numeric scoring requires explicit policy data and remains subject to teacher approval.
- Annotation and model-training proposals are not ground truth until reviewed; unreviewed, out-of-scope, and pending text is masked from training labels.

## Setup

Use Python 3.10 or newer. From PowerShell at the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` pins the direct Python dependencies captured from the working environment. OCR also requires the Tesseract executable to be installed and available on `PATH`; installing `pytesseract` alone does not install Tesseract.

Optional features:

- DOCX assessment policies: `python -m pip install -r requirements-docx.txt`
- Groq provider: `groq` is included in the base requirements; set the provider's API key using the expected environment configuration.
- DistilBERT classification/training: `python -m pip install -r requirements-ml.txt`. A trained checkpoint must be configured separately; predictions are advisory.
- The default Ollama evaluator requires a running Ollama service and the configured model. Gemini uses the existing HTTP client implementation and requires its API-key environment configuration.

## Run a teacher assessment

Place one question-paper PDF, student `.ipynb`/PDF submissions, and an optional assessment policy inside an input folder. Prefer clearly named subfolders such as `Question Paper`, `Student Submissions`, and `Assessment Policy`.

```powershell
.\.venv\Scripts\python.exe scripts\run_teacher_assessment.py --input-dir "D:\assessments\assignment_01"
```

See [docs/TEACHER_ASSESSMENT_PIPELINE_V1.md](docs/TEACHER_ASSESSMENT_PIPELINE_V1.md) for input, output, provider, scoring, and resume details.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test*.py"
```

## Project guides

- [SESSION_HANDOFF.md](SESSION_HANDOFF.md) — architecture, contracts, history, and next steps.
- [docs/annotation_guidelines.md](docs/annotation_guidelines.md) — annotation labels and review principles.
- [docs/question_scope_policy_v1_2.md](docs/question_scope_policy_v1_2.md) — question-scope boundary.
- [docs/review_coverage_contract_v1.md](docs/review_coverage_contract_v1.md) — review coverage and training-label safety.
- [docs/training_data_contract.md](docs/training_data_contract.md) — intended token-level training workflow and limits.

The annotation/training workflow is separate from routine teacher assessment. The current reviewed corpus is not sufficient to claim cross-domain model generalization.
