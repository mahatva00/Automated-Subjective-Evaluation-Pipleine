# Project Module Status — source bundle audit

**Audit basis:** `ine_source_bundle.zip` received 2026-10-14, plus the user's reported DistilBERT classification output. Status rows distinguish existing source modules from the new v1 patch (not yet installed into the user's Windows project as of this audit).

## Existing source modules and demonstrated capability

| Module / path | Status | Evidence and boundary |
|---|---|---|
| `src/ingestion/notebook_parser.py` | Implemented / tested previously | Extracts notebook segments and outputs. It reads serialized notebook data; it does not run cells. |
| `src/ingestion/pdf_parser.py` | Implemented / tested previously | PDF native-text extraction, OCR fallback, metadata and diagnostics. It also produces the generic `segments` payload for question papers and student PDFs. |
| `src/ingestion/question_parser.py` | Implemented / exercised on the 11-paper corpus | Creates a question map from extracted PDF page segments; parser is heuristic and may report diagnostic ambiguity. |
| `src/requirements/requirement_extractor.py` | Implemented / exercised on the 11-paper corpus | Rule-assisted atomic requirement candidates. Not a transformer boundary extractor; retains `needs_review` diagnostics. |
| `src/annotation/scope.py` + scope scripts | Implemented, limited | Scope review/ledger tools exist; 10 non-FAT corpus scopes were inferred automatically and remain provisional. FAT-QP had a prior manual scope ledger. |
| `src/annotation/structure_segmentation.py` + proposal script | Implemented, limited | Generates structure candidates; generated proposals are not approved automatically. |
| `src/annotation/semantic_proposals.py` + proposal/review scripts | Implemented, limited | Generates semantic proposals and TSV review workflow; proposals are not ground truth until reviewed/imported. |
| `src/annotation/coverage.py`, `src/annotation/token_dataset.py` | Tooling implemented; dataset readiness partial | Coverage, token windows, label masks and leakage checks exist; reviewed cross-domain coverage is incomplete. |
| `scripts/train_distilbert_mvp.py` or bundled equivalent | Implemented and ran on user's Windows | Reported 0.8333 accuracy and 0.8413 macro-F1 on 12 validation examples from a narrow ML domain. No test set; do not claim generalized performance. |
| `scripts/classify_requirement_candidates.py` and bundled equivalent | Implemented; standalone inference succeeded | User processed 11 requirement files / 912 candidates. Predicted roles: requirement 664, instruction/constraint 90, context/example 82, option/alternative 76. 909 rows were low-confidence at threshold 0.55; 3 rows needed candidate-role check. Confidence is uncalibrated. It does not discover new boundaries. |
| `src/evaluation/evidence_mapper.py` | Implemented, limited | Assigns notebook/PDF evidence blocks to questions using explicit headings/document order and attaches requirements to questions. It does not materialize a fully adjudicated per-atomic-requirement evidence map independently. |
| `src/evaluation/static_answer_evaluator.py` | Implemented, limited / quality work needed | Evaluates each requirement against selected candidate evidence with Ollama/Groq/Gemini adapters and supports checkpoints. Prior audit noted many `needs_review` and incomplete-context outcomes. No marks are assigned by this module. |
| `scripts/audit_static_evaluations.py` | Implemented | Checks requirement ID parity, statuses, evidence IDs/quotes, confidence, no-code-execution and no-grading flags. |
| `scripts/batch_annotation_workflow.py` and `tools/automated_corpus_completion_v1/scripts/run_full_batch.py` | Implemented development workflow | Generates scope, structure and semantic proposal artifacts and review queues for the existing corpus. This is not the teacher-facing assessment runner. |

## Gaps confirmed in the uploaded source

- No `src/policy/` package or verified policy parser was present.
- No `src/evaluation/grading_engine.py` was present.
- No `src/reporting/` package was present in the uploaded project source root.
- No `src/orchestration/` teacher-folder runner was present.
- Existing `scripts/evaluate_student_answers.py` accepts a prebuilt evidence map and outputs static classifications; it does not take a teacher input folder or issue marks.
- No verified single command was demonstrated for: teacher folder → parsing → requirements → student ingestion → mapping → evaluation → policy-aware report.

## New in `teacher_assessment_pipeline_v1_patch.zip` (not yet installed/Windows-verified)

| Added file(s) | Intended role | Validation so far |
|---|---|---|
| `src/policy/policy_parser.py` | Normalize JSON/TXT/MD/PDF/DOCX policies; only parse explicit numeric credit maps | Unit tests pass in patch environment |
| `src/evaluation/grading_engine.py` | Conservative marks allocation and status-to-credit scoring | Unit tests pass in patch environment; alternatives/unmarked items prevent final-grade claims |
| `src/reporting/student_report.py` | JSON + HTML student reports | Tested with synthetic report |
| `src/orchestration/teacher_assessment.py` | Discover one teacher folder and connect existing parsers, extractor, optional classifier, evidence mapper, evaluator, audit, scoring, and reports | Synthetic folder end-to-end test passes with fake provider; no real provider call made |
| `scripts/run_teacher_assessment.py` | CLI for the one-folder pipeline | Syntax and `--help` pass |
| `tests/test_teacher_assessment_pipeline.py` | Regression coverage for new modules | 6 tests pass; full uploaded bundle had one missing CSV fixture unrelated to this patch |

## Operating rules retained

1. Never execute student notebook/PDF code.
2. Keep original submissions, V5 requirements, and reviewed annotation artifacts immutable.
3. DistilBERT role predictions remain metadata only; never drop, approve, or score candidates based solely on this MVP.
4. Do not infer numerical partial credit from prose like “be lenient”; score only when explicit factors and mark allocation are available.
5. Never publish the automated output as an official grade; unresolved items and scores require teacher review.
6. Update this module register only after user-side installation or another observed test/run provides evidence.
