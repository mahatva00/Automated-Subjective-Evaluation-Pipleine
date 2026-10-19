# Intelligent Notebook Evaluator — Session Handoff

This document is the complete technical handoff for continuing work on the evidence-grounded notebook assessment pipeline.

## 1. Project Purpose

The project evaluates student submissions against a teacher question paper and assessment policy. It supports:

- PDF student submissions;
- Jupyter Notebook (`.ipynb`) submissions;
- question-paper PDF ingestion;
- assessment-policy parsing;
- atomic requirement extraction;
- question/subquestion ownership;
- structured evidence extraction;
- code-to-requirement mapping;
- contextual output and explanation evidence;
- static LLM-based fulfillment assessment;
- diagnostic reporting;
- HTML and JSON reports.

The pipeline is intended to be domain-neutral. It must work for machine learning, SQL, cloud computing, and other subjective or code-supported domains without adding subject-specific correctness rules.

Student code must never be executed. The pipeline only parses, extracts, maps, and evaluates submitted evidence.

Numeric grading is currently intentionally disabled unless a complete, defensible policy credit map exists.

## 2. Repository and Runtime

Repository:

```text
D:\intelligent-notebook-evaluator
```

Python environment:

```text
D:\intelligent-notebook-evaluator\.venv
```

Recommended test command:

```powershell
D:\intelligent-notebook-evaluator\.venv\Scripts\python.exe -m unittest discover -s tests -p "test*.py"
```

`pytest` is not available in the virtual environment. Use `unittest`.

Default local evaluator:

```text
Provider: Ollama
Model: qwen2.5:3b-instruct
Base URL: http://localhost:11434
```

The current date in the last work session was 2026-10-19.

## 3. Important Repository Structure

### Root

- `README.md` — existing project documentation.
- `SESSION_HANDOFF.md` — this continuation document.
- `requirements.txt` / environment files — Python dependencies and runtime setup.
- `.gitignore` — repository exclusions.

### `src/`

Main production package.

- `src/ingestion/`
  - `notebook_parser.py`
    - Parses `.ipynb` files.
    - Detects notebook Markdown headings.
    - Preserves code, outputs, explanations, question copies, and AI logs.
  - `pdf_parser.py`
    - Extracts PDF text and embedded-image metadata.
    - Handles notebook-export PDFs with `In [n]` and `Out [n]` markers.
    - Uses OCR as auxiliary evidence when required.

- `src/requirements/`
  - `requirement_extractor.py`
    - Converts question-paper structures into atomic requirements.
    - Preserves dependent clauses that must be evaluated together.
    - Stores provenance and source text.

- `src/evaluation/`
  - `evidence_mapper.py`
    - Assigns submission blocks to question/subquestion owners.
    - Maps code blocks to atomic requirements.
    - Builds same-subquestion evidence bundles.
    - Produces mapping categories and mapping diagnostics.
  - `static_answer_evaluator.py`
    - Selects bounded evidence context.
    - Calls the LLM.
    - Normalizes and validates model responses.
    - Repairs missing assessments and invalid quotes.
    - Produces diagnostic classifications.
  - `grading_engine.py`
    - Contains scoring/report support.
    - Official numeric grading remains disabled unless policy data is complete.

- `src/orchestration/`
  - `teacher_assessment.py`
    - End-to-end orchestration:
      discovery → question-paper parsing → policy parsing → requirement extraction
      → submission parsing → evidence mapping → static evaluation → audit → report.
    - Supports checkpoint files and `--resume-run`.
    - Never executes student code.

- `src/reporting/`
  - `student_report.py`
    - Produces JSON and HTML student reports.
    - Propagates evidence maps, bundles, components, diagnostics, and warnings.

- `src/policy/`
  - `policy_parser.py`
    - Normalizes policy files.
    - Numeric scoring remains blocked when the policy does not provide a complete credit map.

### `scripts/`

- `run_teacher_assessment.py`
  - Main CLI for end-to-end assessment.
- `benchmark_frozen_evidence.py`
  - Re-evaluates a frozen evidence map without reparsing or remapping.
  - Used to compare batch sizes, repeatability, and model behavior.
- Other scripts support classification, inspection, and utilities.

### `tests/`

Important test files:

- `test_notebook_parser.py`
- `test_pdf_parser.py`
- `test_requirement_extractor.py`
- `test_evidence_mapper.py`
- `test_static_answer_evaluator.py`
- `test_teacher_assessment_pipeline.py`
- `test_format_parity.py`
- `test_evidence_auditor.py`
- `test_structural_evidence_fixtures.py`

The structural fixture suite was the most recent addition.

## 4. Core Data and Design Contracts

### 4.1 Evidence components

The intended component vocabulary is:

- `question`
  - Original question-paper context when retained.
- `question_copy`
  - Student-copied question text.
- `code`
  - Student-authored code.
- `output`
  - Notebook/PDF output associated with code.
- `explanation`
  - Student-authored prose or interpretation.
- `ai_log`
  - AI usage declarations, prompts, or AI-generated interaction logs.

Scored/presence components:

- `code`;
- `output`;
- `explanation`.

Non-scoring component:

- `ai_log`.

Optional context components:

- `question`;
- `question_copy`.

AI-log content must be retained for audit/presence but excluded from fulfillment scoring context.

### 4.2 Deterministic mapping contract

Only code is eligible for deterministic atomic requirement links.

Canonical field:

```text
linked_code_block_ids
```

Compatibility alias:

```text
linked_evidence_block_ids
```

The compatibility alias must contain code block IDs only. Output, explanation, and AI-log blocks must not be treated as deterministic implementation evidence.

Output and explanation can still be selected as contextual evidence inside the same question/subquestion.

### 4.3 Mapping categories

The mapper uses domain-neutral categories:

- `mapped_code`
  - A code block is deterministically linked to the requirement.
- `contextual_evidence`
  - Non-code evidence provides useful context within the same subquestion.
- `explanation_only`
  - The requirement is supported only by explanation.
- `output_only`
  - The requirement is supported only by output.
- `true_no_evidence`
  - No relevant evidence exists.

These categories must not be confused with model fulfillment status.

### 4.4 Model statuses

The evaluator can produce:

- `complete`
- `partial`
- `absent`
- `needs_review`

Important safety rule:

Incomplete context must not be converted into `absent`. A missing output, explanation, mapping, or truncated bundle should generally produce `needs_review` or a diagnostic rather than a definitive absence judgment.

### 4.5 Diagnostics

Current diagnostic categories include:

- `mapped_evidence`
- `mapping_gap`
- `missing_required_component`
- `incomplete_context`
- `invalid_model_quote`
- `invalid_model_status`
- `ambiguous_model_judgment`
- `no_evidence`
- `missing_model_assessment`

Context reason metadata includes:

- `evidence_truncated`;
- `bundle_blocks_omitted`;
- `missing_required_component`.

### 4.6 Evidence selection

The evaluator receives a bounded context. The current default is:

```text
max_evidence_blocks = 4
max_evidence_chars = 2800
```

Selection priorities:

1. linked code blocks;
2. same-cell output;
3. same-subquestion output;
4. same-subquestion explanation;
5. visual auxiliary metadata where relevant.

Selection must not cross question/subquestion boundaries.

The complete evidence bundle remains in the evidence map even when the evaluator context is truncated. Truncation must be explicit in diagnostics.

### 4.7 Quote validation

Model quotes are accepted only when they match stored evidence exactly or through approved line-wrap recovery.

Fabricated quotes must never be accepted.

Invalid quote rows are retried individually. If still invalid, the row becomes `needs_review`.

### 4.8 Missing assessment repair

The evaluator:

1. detects omitted requirement IDs;
2. retries omitted requirements individually;
3. filters unknown IDs;
4. suppresses duplicates;
5. leaves unrecovered omissions explicitly diagnosed.

This changed the frozen benchmark from many missing rows to complete requirement coverage.

## 5. Notebook Parser Work Completed

`src/ingestion/notebook_parser.py` was strengthened to:

1. Accept compact headings:

```text
### 1.DATA PREPROCESSING
### 1. DATA PREPROCESSING
### 3.Inferences:
### 3. Inferences:
```

2. Avoid treating ordinary numbered Markdown list items as question headings.

3. Preserve notebook output forms:

   - string values;
   - list values;
   - `text/plain`;
   - `text/html`;
   - stream outputs;
   - display-data outputs.

4. Classify explicit AI-log cells such as:

```text
No AI used
no AI tool used
AI prompt
```

as `ai_log`.

5. Rebuild question blocks after heading detection so ownership is correct.

The notebook parser must continue to be tested against output MIME diversity and heading/list ambiguity.

## 6. PDF Parser Work Completed

`src/ingestion/pdf_parser.py` was strengthened to:

- recognize notebook-export markers such as `In [n]` and `Out [n]`;
- split code, output, explanation, and AI-log evidence;
- retain embedded image/OCR metadata as auxiliary evidence;
- avoid using global question-paper instructions as student requirements;
- preserve canonical extracted text;
- emit parser diagnostics when OCR or low-confidence extraction is used.

Important limitation discovered in assignment 03:

The PDF parser currently handles explicit markers such as:

```text
Question 1:
```

but may fail when a student PDF begins with a compact answer marker such as:

```text
1 a
```

This caused one student submission to have evidence blocks but no question ownership.

## 7. Requirement Extraction Work Completed

`src/requirements/requirement_extractor.py` now:

- extracts atomic requirements from question/subquestion structures;
- groups dependent clauses that must remain one coherent requirement;
- preserves requirement provenance;
- uses atomic text for mapping rather than full sibling-question text;
- avoids turning question-paper preamble instructions into student requirements;
- remains domain-neutral.

Do not add ML-specific, SQL-specific, cloud-specific, plotting-specific, or API-specific correctness rules here.

## 8. Evidence Mapping Work Completed

`src/evaluation/evidence_mapper.py` now handles:

- question/component assignment;
- code-only deterministic linking;
- same-subquestion evidence bundles;
- mapping categories;
- notebook anchor inheritance from question-copy cells;
- requirement-level mapping diagnostics;
- visual auxiliary evidence metadata;
- code/output/explanation grouping;
- question and subquestion boundary protection.

The mapper is the central source of truth for structural evidence relationships.

## 9. Evaluation Work Completed

`src/evaluation/static_answer_evaluator.py` now handles:

- bounded evidence selection;
- same-cell output selection;
- typed fallback context;
- evidence bundle consumption;
- component-diverse context selection;
- response normalization;
- unknown/duplicate response filtering;
- missing-assessment retry;
- invalid-quote retry;
- exact quote validation;
- line-wrap quote recovery;
- context diagnostics;
- status validation;
- model-response classification.

The LLM is advisory and should not be allowed to remove baseline requirements or silently create unsupported evidence.

## 10. Reports and Auditing

`src/reporting/student_report.py` includes:

- component summaries;
- missing component lists;
- evidence bundles;
- mapping diagnostics;
- context reasons;
- omitted bundle IDs;
- evaluation statuses;
- warnings;
- audit results.

The pipeline runs an independent evaluation audit before any scoring step.

Official grades are not issued. Numeric scoring is blocked when the policy lacks a complete credit map.

## 11. Historical Runs and Findings

### Assignment 01

The original `0/1` run was traced to structural ingestion and mapping problems, not only to the LLM:

- PDF/notebook-export extraction;
- question ownership;
- code/output/explanation splitting;
- evidence mapping;
- insufficient evaluator context.

### Assignment 02

Assignment 02 compared the same student work in PDF and notebook formats.

The corrected notebook produced approximately:

```text
3 questions
28 requirements
74 evidence blocks
74 mapped blocks
0 unassigned blocks
```

The historical PDF artifact retained approximately:

```text
37 evidence blocks
36 mapped blocks
1 unassigned block
```

Semantic linked-code overlap was approximately `20/28`.

Important assignment 02 runs:

- `D:\assessments\assignment_02_assessment_output\run_20261018_165725`
- `D:\assessments\assignment_02_assessment_output\run_20261018_171011`
- `D:\assessments\assignment_02_assessment_output\run_20261018_171713`
- `D:\assessments\assignment_02_assessment_output\run_20261018_182104`

Latest assignment 02 full run:

```text
complete: 9
partial: 9
needs_review: 10
absent: 0
audit: passed
student code executed: no
```

### Frozen evaluator benchmarks

Artifacts:

- `D:\assessments\assignment_02_assessment_output\frozen_benchmark_qwen_baseline.json`
- `D:\assessments\assignment_02_assessment_output\frozen_benchmark_qwen_repair.json`
- `D:\assessments\assignment_02_assessment_output\frozen_benchmark_qwen_quote_repair.json`
- `D:\assessments\assignment_02_assessment_output\frozen_benchmark_qwen_context_diversity.json`

The latest frozen benchmark after repair and component-diverse context selection reached:

```text
28/28 validated
8 complete
9 partial
11 needs_review
```

The same evidence map was reused. Parsing and mapping did not rerun.

### Assignment 03

Input folder:

```text
D:\assessments\assignment_03
```

Contents:

- `assessment_policy.txt`
- `LPS4-QP.pdf`
- seven PDF submissions under `submissions\`

Submissions:

- `PAMCA504-25MCA1020-LPS4.pdf`
- `PAMCA504-25MCA1034-LPS4.pdf`
- `PAMCA504-25MCA1063-LPS4.pdf`
- `PAMCA504_25MCA1006_LPS4.pdf`
- `PAMCA504_25MCA1015_LPS4.pdf`
- `PAMCA504_25MCA1029_LPS4.pdf`
- `PAMCA504_25MCA1081_LPS4.pdf`

Final run directory:

```text
D:\assessments\assignment_03_assessment_output\run_20261018_201031
```

Important artifacts:

- `RUN_REPORT.json`
- `ASSESSMENT_SUMMARY.html`
- `intermediate\parsed_question_paper.json`
- `intermediate\question_map.json`
- `intermediate\requirements_base\`
- `intermediate\parsed_submissions\`
- `intermediate\evidence_maps\`
- `intermediate\evaluations\`
- `intermediate\evaluation_audits\`
- `reports\`
- `logs\submissions\`

Final assignment 03 result:

```text
students discovered: 7
students evaluated: 7
students failed: 0
student code executed: no
numeric scoring: disabled
```

Across 518 requirement assessments:

```text
complete: 38
partial: 144
needs_review: 336
absent: 0
```

The high `needs_review` count is not purely model uncertainty. It includes incomplete PDF evidence, unmapped content, missing required components, and model ambiguity.

## 12. Assignment 03 Batch-Size Finding

The first run used:

```powershell
D:\intelligent-notebook-evaluator\.venv\Scripts\python.exe D:\intelligent-notebook-evaluator\scripts\run_teacher_assessment.py `
  --input-dir "D:\assessments\assignment_03" `
  --batch-size 4 `
  --max-evidence-blocks 4 `
  --max-evidence-chars 2800
```

Initial result:

```text
5/7 students evaluated
```

Two students stopped after 8/74 requirements. Ollama reported that it stopped before completing the response, probably because of the `num_predict` limit.

The run was then resumed with:

```powershell
D:\intelligent-notebook-evaluator\.venv\Scripts\python.exe D:\intelligent-notebook-evaluator\scripts\run_teacher_assessment.py `
  --input-dir "D:\assessments\assignment_03" `
  --resume-run "D:\assessments\assignment_03_assessment_output\run_20261018_201031" `
  --batch-size 1 `
  --max-evidence-blocks 4 `
  --max-evidence-chars 2800
```

Checkpoint reuse recovered both students and produced `7/7`.

Conclusion:

- batch size `4` is unsafe for this model/evidence volume;
- batch size `1` is reliable but slow;
- batch size `2` or `3` should be tested;
- incomplete responses should automatically retry at batch size `1`;
- checkpoints must remain the recovery mechanism.

## 13. Assignment 03 Anomaly: Student 25MCA1063

Treat `PAMCA504-25MCA1063-LPS4.pdf` as an anomaly for evaluation purposes.

The student appears not to have followed the expected submission/export instructions. The first extracted page begins with compact content such as:

```text
1 a
import numpy as np
...
```

Unlike the other submissions, it does not use the expected explicit question markers.

Observed parser result:

```text
33 page segments
34 evidence blocks
0 mapped blocks
34 unassigned blocks
0 questions with evidence
10 questions without evidence
```

All 74 requirements consequently became `needs_review`.

This does not prove that the student supplied no answers. It proves that the current PDF question-ownership parser could not connect the extracted pages to the question map.

Decision:

- Do not add a domain-specific workaround solely for this student.
- It is acceptable to classify this file as an input anomaly.
- Still consider strengthening generic PDF anchor detection if the change is safe and domain-neutral.
- Add a regression fixture only if it tests a reusable structural pattern such as `1 a`, not the student’s subject matter.

## 14. Assignment 03 Per-Student Structural Results

### Students 25MCA1020 and 25MCA1034

```text
35 segments
114 evidence blocks
113 mapped blocks
1 unassigned block
10 questions with evidence
```

These produced slightly better results than the smaller evidence group but still had many mapping/context diagnostics.

### Students 25MCA1006, 25MCA1015, 25MCA1029, 25MCA1081

```text
23 segments
68 evidence blocks
67 mapped blocks
1 unassigned block
9 questions with evidence
1 question without evidence
```

Their model outputs were structurally consistent:

```text
complete: 8–10
partial: 18–19
needs_review: 46–47
```

This consistency suggests the PDF export structure strongly affects available evidence.

## 15. What Must Not Be Changed

Do not:

- execute student code;
- add ML-specific correctness rules;
- add SQL-specific correctness rules;
- add cloud-specific correctness rules;
- force output or explanation into deterministic code links;
- treat AI logs as scoring evidence;
- issue official numeric grades without a complete policy credit map;
- classify incomplete evidence as definitive `absent`;
- remove baseline requirements based on DistilBERT/advisory model predictions;
- rewrite unrelated pre-existing changes;
- reset or discard the dirty worktree.

## 16. Current Test Baseline

Before assignment 03 analysis, the complete test suite passed:

```text
254 tests passed
```

The structural fixture suite specifically passed:

```text
3 tests passed
```

The last validation command was:

```powershell
D:\intelligent-notebook-evaluator\.venv\Scripts\python.exe -m unittest discover -s tests -p "test*.py"
```

Also run:

```powershell
git --no-pager diff --check
```

after code changes.

## 17. Recommended Project Plan

### Phase 1 — Preserve the current baseline

1. Read this handoff document.
2. Inspect the current worktree before editing.
3. Run the 254-test baseline.
4. Confirm no student code execution path has been introduced.
5. Keep numeric scoring disabled.

### Phase 2 — Generic PDF anchor hardening

Only implement this if it improves a reusable structural case:

- recognize `Question 1`;
- recognize `Q1`;
- recognize `1.`;
- recognize `1)`;
- recognize `1 a`;
- recognize `1(a)`;
- avoid interpreting ordinary numbered prose/list items as question boundaries.

Required safeguards:

- distinguish question anchors from ordinary numbered lists;
- rebuild ownership after anchor detection;
- never let question-paper text become student evidence;
- preserve page-level fallback evidence when ownership is uncertain;
- emit a diagnostic when ownership is inferred with low confidence.

Add tests in:

- `tests/test_pdf_parser.py`;
- `tests/test_evidence_mapper.py`;
- `tests/test_structural_evidence_fixtures.py`.

Use a synthetic generic fixture rather than hard-coding the 1063 student or ML terminology.

### Phase 3 — Batch-size reliability

Test frozen or end-to-end evaluation with:

- batch size `1`;
- batch size `2`;
- batch size `3`;
- batch size `4`.

Measure:

- completed requirements;
- valid JSON rate;
- valid quote rate;
- missing-assessment rate;
- retry count;
- total time;
- status counts;
- repeatability.

Preferred design:

1. Use batch size `2` or `3` as the default candidate.
2. Detect truncated/incomplete provider responses.
3. Retry only the failed batch at size `1`.
4. Preserve checkpoints after every successful batch.
5. Never discard completed batches.

This is a reliability improvement, not a scoring rule.

### Phase 4 — Frozen-payload evaluator comparison

Before comparing models, freeze equivalent evidence payloads.

Do not compare model results if parsing or mapping changed between runs.

For each model/provider, measure:

- valid JSON rate;
- valid quote rate;
- missing-assessment rate;
- invalid-status rate;
- ambiguous-judgment rate;
- false-absence risk;
- needs-review rate;
- repeatability across repeated calls;
- agreement with manual review labels.

The `benchmark_frozen_evidence.py` utility is the starting point.

### Phase 5 — Cross-domain validation

After structural stability, collect real question papers and submissions from:

- SQL;
- cloud computing;
- another subjective/code-supported domain.

For each domain:

1. parse the question paper;
2. inspect atomic requirements;
3. parse submissions;
4. inspect question/subquestion ownership;
5. inspect component retention;
6. inspect deterministic code mappings;
7. inspect contextual bundles;
8. freeze evidence payloads;
9. compare evaluator behavior.

Do not introduce domain-specific logic unless a genuinely domain-neutral structural defect is proven.

### Phase 6 — Human agreement and scoring decision

Only after structural and evaluator validation:

1. manually label a representative sample;
2. compare LLM judgments with human labels;
3. quantify false positives and false negatives;
4. decide whether model output is suitable for assisted review;
5. separately validate policy credit maps;
6. only then consider enabling numeric scoring.

## 18. How to Run the Pipeline

Fresh run:

```powershell
D:\intelligent-notebook-evaluator\.venv\Scripts\python.exe `
  D:\intelligent-notebook-evaluator\scripts\run_teacher_assessment.py `
  --input-dir "D:\assessments\assignment_03" `
  --batch-size 2 `
  --max-evidence-blocks 4 `
  --max-evidence-chars 2800
```

Resume a run:

```powershell
D:\intelligent-notebook-evaluator\.venv\Scripts\python.exe `
  D:\intelligent-notebook-evaluator\scripts\run_teacher_assessment.py `
  --input-dir "D:\assessments\assignment_03" `
  --resume-run "D:\assessments\assignment_03_assessment_output\run_20261018_201031" `
  --batch-size 1
```

Frozen benchmark:

```powershell
D:\intelligent-notebook-evaluator\.venv\Scripts\python.exe `
  D:\intelligent-notebook-evaluator\scripts\benchmark_frozen_evidence.py `
  --evidence-map "<path-to-evidence-map.json>" `
  --model "qwen2.5:3b-instruct" `
  --repeats 3 `
  --batch-size 1 `
  --report "<output-report.json>"
```

## 19. How to Interpret Results

Always separate these layers:

1. **Parsing**
   - Was the source text extracted?
   - Were code/output/explanation/AI-log components retained?

2. **Ownership**
   - Was evidence assigned to the correct question/subquestion?

3. **Mapping**
   - Was code linked only to relevant atomic requirements?
   - Were output/explanation blocks retained as context without becoming code links?

4. **Context selection**
   - Did the evaluator receive the relevant bundle?
   - Was context truncated?
   - Were required components missing?

5. **Model response**
   - Was JSON valid?
   - Were all requirement IDs present?
   - Were quotes exact?
   - Was the judgment unambiguous?

6. **Policy/scoring**
   - Is a complete credit map available?
   - If not, report fulfillment summaries only.

Never interpret a high `needs_review` count as direct evidence that students failed. First inspect the structural diagnostics.

## 20. Final Handoff Summary

The structural pipeline is substantially stronger than at the start of the session:

- notebook headings and list handling are improved;
- notebook MIME outputs are preserved;
- AI logs are classified and excluded from scoring;
- PDF notebook exports are split into useful evidence types;
- question ownership and subquestion boundaries are enforced;
- only code receives deterministic atomic links;
- same-subquestion bundles preserve output and explanation context;
- visual metadata remains auxiliary;
- missing components and incomplete context are explicit;
- model omissions and invalid quotes are repaired safely;
- frozen evidence benchmarking is available;
- structural fixtures cover the major domain-neutral contracts;
- 254 tests pass.

The main remaining structural issue discovered in assignment 03 is generic PDF question-anchor detection for an anomalous submission format. The main remaining evaluator issue is Qwen reliability and high `needs_review` frequency under incomplete PDF evidence.

The next safest action is:

1. do not add subject-specific rules for student 1063;
2. test generic PDF anchor support with a synthetic fixture;
3. compare evaluator batch sizes on frozen payloads;
4. implement automatic fallback from batch size 2/3 to batch size 1 if needed;
5. then move to real cross-domain papers;
6. keep numeric scoring disabled until human-agreement and policy validation are complete.
