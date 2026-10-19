# Teacher Assessment Pipeline v1 — patch package

This patch adds a runnable one-folder assessment orchestrator on top of the existing project modules. It is meant to be extracted into the **project root** (`D:\intelligent-notebook-evaluator`), not into the `tools` directory. It does not include or overwrite the rest of your project, datasets, virtual environment, or model checkpoint.

## Files added

- `src/policy/policy_parser.py`
- `src/evaluation/grading_engine.py`
- `src/reporting/student_report.py`
- `src/orchestration/teacher_assessment.py`
- `scripts/run_teacher_assessment.py`
- `tests/test_teacher_assessment_pipeline.py`
- `docs/TEACHER_ASSESSMENT_PIPELINE_V1.md`
- `docs/PROJECT_MODULE_STATUS_AFTER_SOURCE_AUDIT.md`

## Install/test

```powershell
$ProjectRoot = "D:\intelligent-notebook-evaluator"
$PatchZip = "$env:USERPROFILE\Downloads\teacher_assessment_pipeline_v1_patch.zip"
Expand-Archive -LiteralPath $PatchZip -DestinationPath $ProjectRoot -Force
Set-Location $ProjectRoot
& "$ProjectRoot\.venv\Scripts\Activate.ps1"
python -m unittest discover -s tests -p "test_teacher_assessment_pipeline.py" -v
```

## Run

```powershell
python scripts\run_teacher_assessment.py --input-dir "D:\assessments\assignment_01"
```

The default provider is Ollama. The input folder should contain one QP PDF, one assessment policy, and at least one student `.ipynb`/PDF. Prefer clearly named folders: `Question Paper`, `Assessment Policy`, `Student Submissions`. Optional `--question-paper` and `--policy-file` overrides are available if automatic discovery is ambiguous.

To add advisory candidate-role predictions with a trained DistilBERT checkpoint, set `INE_DISTILBERT_MODEL_DIR` or pass `--model-dir`. Do not retrain per teacher assignment. The current classifier does not discover boundaries and most existing predictions are low confidence.

Numeric marks are generated only if a numeric credit map is explicit in policy and mark allocation is supported. Otherwise the report contains fulfillment classifications and explains why scoring was withheld. The pipeline never executes submitted code and never issues an official grade.
