# FAT-QP review finalization v1

## What this package contains
- `data/annotations/training_coverage_reviewed/FAT-QP.annotation.training-coverage.json`: finalized coverage ledger.
- `data/annotations/training_coverage_reviewed/FAT-QP.annotation.training-coverage.tsv`: same decisions in the importer TSV format.
- `data/annotations/reviewed/FAT-QP.annotation.json`: a reviewed canonical annotation copy; original draft is not overwritten.
- `data/annotations/review_audits/FAT-QP.annotation.review-audit.json`: audit with decisions and coverage state.
- `FAT-QP_semantic_coverage_review_v1.reviewed.tsv`: coverage worksheet reflecting the user confirmation.
- `FINALIZATION_AUDIT.json`: programmatic validation summary.

## Finalized decisions
- 24 in-scope question-associated regions: coverage reviewed for both `structure` and `semantic`.
- `COV00105`: global introductory dataset instruction remains unassigned and `pending`; it remains masked from training labels.
- 55 regions remain `out_of_scope`; 72 separators remain masked.
- Existing 96 semantic annotation spans and labels were preserved; schema/span/overlap/review-status validation passed.
- No submitted student code was executed.

## Important next step
This finalizes one paper only. Do not train or report this as a robust cross-paper DistilBERT model yet. Review and approve the remaining question papers, then create an explicit paper-level train/validation/test manifest. The current reviewer identifier `reviewer-confirmed-chat-2026-10-13` is a pseudonymous audit alias; replace it with the project-approved stable ID before publication if desired.

## Installing into the project
Copy the included `data/annotations/training_coverage_reviewed`, `data/annotations/reviewed`, and `data/annotations/review_audits` directories into the project root, keeping existing `drafts`, `scope_reviews`, `semantic_review_imported_v1`, and `training_coverage` folders intact.
