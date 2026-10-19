# Teacher Assessment Pipeline v1 — integration status

**Status:** implemented as a development patch; not yet installed or exercised against a real teacher input folder on the user's Windows environment.

## What it runs

`python scripts/run_teacher_assessment.py --input-dir <teacher-folder>` orchestrates existing project modules and adds the currently missing policy/scoring/report adapters.

1. Discover one question-paper PDF, student `.ipynb`/PDF submissions, and an optional assessment policy inside a single input folder. It uses file/folder names and question/policy text cues. If discovery is ambiguous, use `--question-paper` or `--policy-file` as diagnostic overrides; these paths must still be inside the teacher folder.
2. Parse the question-paper PDF with `src.ingestion.pdf_parser.parse_pdf`.
3. Build a fresh question map with `src.ingestion.question_parser.parse_question_paper`.
4. Extract atomic requirement candidates with `src.requirements.requirement_extractor.extract_atomic_requirements`.
5. Optionally classify those candidates with the configured, already-trained DistilBERT role model. Predictions remain advisory metadata; they never delete or approve requirements and do not discover new boundaries.
6. Normalize the policy from JSON, TXT, Markdown, PDF, or DOCX.
7. Parse student notebooks/PDFs using the existing ingestion modules without executing student code. Notebook parsing and notebook-export PDF mapping preserve question-scoped component blocks: `question`, `code`, `output`, `explanation`, and `ai_log`.
8. Build evidence maps through `src.evaluation.evidence_mapper.build_evidence_map`. Component blocks are mapped by explicit question numbers, compact known question headings are accepted, and numbered prose is not treated as a question heading unless its title matches the question paper. Each question records missing scored components (`code`, `output`, or `explanation`). Atomic requirements receive code-only links based on their subquestion anchor and code concepts; output, explanation, question-copy, and AI-log blocks are retained for component-presence reporting but are not requirement evidence links. AI-log blocks are kept separate and are not scored.
9. Evaluate requirement evidence via the existing static answer evaluator and configured provider.
10. Audit evaluation IDs/statuses/exact evidence quotes with `scripts.audit_static_evaluations.audit`.
11. Calculate a conservative requirement-level score only when a numeric policy credit map and defensible mark allocation are available. Component-policy scoring is available through `score_component_evaluation` and uses the explicit per-question component allocations; missing components receive zero allocation for that component and `needs_review` remains provisional.

The semantic/structure proposal generators are a development-time annotation/training workflow, not a required step for each live student assessment. Their outputs remain reproducible through `tools/automated_corpus_completion_v1/scripts/run_full_batch.py`.

## Input convention

Recommended layout (names are examples; the scanner uses both cues and extracted content):

```text
assignment_01/
  Question Paper/
    assignment-QP.pdf
  Assessment Policy/
    marking-policy.json   # or .txt, .md, .pdf, .docx
  Student Submissions/
    23MCA1006_assignment.ipynb
    23MCA1015_assignment.pdf
```

The scanner currently expects a PDF question paper and at least one `.ipynb` or PDF student submission. If several files are equally plausible question papers or policies, it stops rather than silently selecting the wrong one.

## Notebook component contract

The notebook parser never executes cells. It emits the original cell segments for compatibility and additionally emits `question_blocks`:

- `question`: an explicit question heading such as `# Q1: ...`; context only.
- `code`: source from a code cell.
- `output`: stored notebook output from a code cell; absent when no stored output exists.
- `explanation`: non-AI markdown associated with the current question.
- `ai_log`: markdown explicitly marked as AI usage/log/prompt/tool content.

The scored-component baseline is `code`, `output`, and `explanation`. A missing component is reported on the question evidence map. The AI log is preserved separately and is currently non-scoring. Copied subquestion Markdown is classified as `question_copy` rather than `explanation`. Atomic requirement links contain `linked_code_block_ids`; non-code components are used only to determine presence/missing status. The parser uses explicit markers for AI-log classification and does not treat ordinary explanation text as AI-log content merely because it is markdown.

For notebook-export PDFs, `In [n]` and `Out [n]` markers are split into code and output evidence. Explanation and AI-log prose following an output is separated when explicit explanation/AI markers are present. Embedded-image OCR is stored in `metadata.image_evidence` with `is_authoritative: false`; it is not appended to native page text.

## Assessment policy and scores

For numeric scores, the policy must explicitly provide `complete`, `partial`, and `absent` credit factors in the range 0–1. JSON example:

```json
{
  "strictness": "balanced",
  "credit_map": {"complete": 1.0, "partial": 0.5, "absent": 0.0},
  "mark_allocation": {"mode": "per_source_node_equal_split"},
  "teacher_approval_required": true
}
```

Plain-text policies can use explicit numeric factors such as `complete: 100%; partial: 50%; absent: 0%`. The parser will not infer partial-credit fractions from a statement such as “be lenient”. If the mapping is missing, the pipeline still emits the static evaluation report, but numeric scoring remains disabled.

Markdown policies may also declare per-question component allocations, for example `Q1 ... [10 marks]` followed by `Split: Code 50%, Explanation 25%, Output 25%`. These are normalized under `question_component_policy`, including explicit sub-question marks when the policy provides them. Component scoring applies complete/partial/absent as 100/50/0 and does not treat AI-log evidence as explanation evidence.

When multiple atomic requirements share a marked source node, that node's marks are split evenly between them so the same marks are not duplicated. Alternative-answer groups and missing mark allocations are flagged; unsupported cases do not receive a purported final grade. `final_grade` remains `null`; a numeric result is provisional and requires teacher approval.

## DistilBERT

Configure a checkpoint already trained in development via `--model-dir` or environment variable `INE_DISTILBERT_MODEL_DIR`. The teacher assessment runner does not retrain the model for each assignment. The classifier annotates copies of the fresh requirements JSON; the base requirements are preserved. Current known classifier report from the 11-paper corpus has 909/912 candidates flagged low-confidence, so the classifier is not allowed to control inclusion, scoring, or approval.

The available model is a semantic-role candidate classifier—not the token-level BIO boundary detector. It can classify extracted candidates but does not guarantee discovery of omitted requirements.

## Output layout

Each run creates a timestamped directory outside the input folder, containing:

- `intermediate/parsed_question_paper.json`
- `intermediate/question_map.json`
- `intermediate/requirements_base/` and `requirements_for_mapping/`
- `intermediate/normalized_assessment_policy.json`
- optional `intermediate/requirements_classified/`
- `intermediate/parsed_submissions/`, `evidence_maps/`, `evaluations/`, `evaluation_audits/`, `scores/`
- `reports/*.report.json` and `reports/*.report.html`
- `RUN_REPORT.json`, `RUN_MANIFEST.json`, and `ASSESSMENT_SUMMARY.html`

Evaluation checkpoints are written after each completed batch. To resume, pass `--resume-run` with the same run directory; the pipeline verifies the input paths/hashes, provider/model and evidence-map hash before reusing checkpointed results. Changed inputs or a different provider/model refuse checkpoint reuse.

## Run

From the project root in PowerShell:

```powershell
python scripts\run_teacher_assessment.py --input-dir "D:\assessments\assignment_01"
```

The default evaluator is Ollama at `http://localhost:11434`. Provider and model can be configured with `INE_EVALUATOR_PROVIDER`, `INE_EVALUATOR_MODEL`, and, for Ollama, `OLLAMA_BASE_URL`. Groq/Gemini require their existing provider API-key environment configuration. Optional DistilBERT:

```powershell
$env:INE_DISTILBERT_MODEL_DIR = "D:\intelligent-notebook-evaluator\tools\automated_corpus_completion_v1\runs\batch_20261014_002235\models\distilbert_span_role_mvp_v1"
```

## Validation performed in the patch environment

- 6 new tests pass, including a synthetic teacher-folder run using a fake assessment client and resume/checkpoint reuse.
- Syntax compilation and CLI `--help` pass.
- The existing source-bundle test suite ran 202 tests: 201 passed; one test could not run because `data/annotations/approved_pilot/pilot_review.csv` was omitted from the uploaded source ZIP (the bundle-creation filter included only a limited set of file extensions). This is a missing test fixture in the uploaded bundle, not evidence that the test fails on the full Windows project.
- No live provider call or real student assessment was run by this patch test.

## Known limits

- The rule-assisted extractor and question parser can miss or mis-split requirements on unfamiliar QP formats; those diagnostics are retained.
- The static evaluator can return `needs_review`; current evaluation quality is not production-validated.
- Automatic policy discovery depends on name/folder/content cues; ambiguous layouts stop with an error.
- The patch adds a first runnable teacher-folder pipeline but does not make the candidate classifier a trained BIO boundary detector.
- Student code is never executed.
