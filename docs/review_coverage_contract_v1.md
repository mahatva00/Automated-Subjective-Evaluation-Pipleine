# Review coverage sidecar contract v1.1.0 — question-scope aware

## Purpose

Canonical annotation JSON contains positive spans, but it cannot by itself distinguish source text that has been reviewed and confirmed not to contain a span from text that nobody reviewed. Training unreviewed text as BIO background (`O`) silently creates false negatives.

The coverage sidecar tracks structure and semantic review per source-text range. The question-scope sidecar separately marks every source range as in-scope question text, out-of-scope text, pending, or whitespace separator. **Only text inside an actual question or subquestion belongs to question requirement extraction.**

## Binding policy

See `docs/question_scope_policy_v1_2.md`. That policy takes precedence over the general semantic label taxonomy in `annotation_decision_policy_v1_1.md`.

- Out-of-scope text must not be annotated for this task and must not become `O` background.
- Pending or mixed-scope text must not become positive or background training labels.
- Separator/special tokens and tokens with offsets crossing a scope boundary must receive `-100` / `IGNORE_INDEX` for token classification.
- Only human-reviewed in-scope tokens can receive a positive BIO label or background `O`.
- Legacy out-of-scope annotations may be filtered from the downstream exported copy. The canonical source file is never mutated by the exporter; the filter action is recorded in `scope_filter_audit`.
- The source text stays exact and keeps its offsets, even if out-of-scope text remains physically present in the input string.

## Workflow

1. Create a question-scope ledger and TSV with `scripts/prepare_question_scope_review.py`.
2. Review the TSV. Mark each nonblank segment `in_scope` or `out_of_scope`; use `pending` if uncertain. `in_scope` requires the exact question-map ID; `out_of_scope` must leave `question_id` blank. Supply reviewer and ISO date for each decision.
3. If one segment contains both actual question text and a preamble/other global instruction, split it with `scripts/split_question_scope_segment.py` before making scope decisions. Do not alter offsets, hashes, or the JSON-encoded preview by hand.
4. Import the reviewed TSV with `scripts/import_question_scope_review.py`. Audit using `scripts/audit_question_scope.py`. A structural audit may pass while decisions remain pending; use `--require-complete` when all segments have been decided.
5. Create/retain ordinary coverage sidecars with `scripts/prepare_annotation_coverage.py`. Do not overwrite a coverage sidecar that already contains human reviews.
6. Review `structure` and `semantic` layers only for in-scope question text within each coverage chunk. Global instructions within the same chunk are ignored and do not require annotations. Mark a coverage layer reviewed only after all in-scope text in that chunk has been checked for that layer.
7. Run `scripts/audit_annotation_coverage.py --scope-dir ...` to see scope-aware progress.
8. Only after every in-scope range is decided, every required coverage chunk is reviewed in both layers, and all retained annotations/relations are reviewed, may the document be marked `reviewed`/`adjudicated` and eligible for strict export.

## Training-label contract

`src.annotation.scope.project_token_labels` receives source offsets for tokens and returns:

- positive `B-label` / `I-label` values for approved annotation spans inside reviewed in-scope regions;
- `O` only for in-scope tokens whose corresponding layer coverage is reviewed and which are not covered by a positive annotation;
- `-100` for out-of-scope, pending, mixed-boundary, separator/special, or unreviewed-coverage tokens.

The existing coarse semantic role dataset is only a positive-span classifier dataset and is not the full requirement-extraction token dataset. Do not claim full extraction capability based on it.

## Audit limits

The scripts verify range partition, exact source text and hashes, reviewed metadata, and scope/coverage consistency. They cannot prove that a person assigned the correct semantic label. Human review and held-out document evaluation remain required.
