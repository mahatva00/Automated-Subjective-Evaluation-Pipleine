# Domain-Neutral Requirement Actions

Keep generic task verbs in the requirement extractor's shared action vocabulary so compound requests can split on them without subject-specific rules.

## What Happened
Synthetic cross-domain tests in intelligent-notebook-evaluator found that “Configure a virtual network and deploy an application...” remained one `needs_review` span because `configure` and `deploy` were absent from `ACTION_VERBS`. Adding both allowed exact, independently assessable action spans and preserved the original wording (session 1bd4afd4-d15d-4ea5-a699-020f5d06be2c).

## Takeaway
When a common action is missing, reproduce it with a minimal cross-domain fixture. Add the action to the shared vocabulary only if it is generic; assert both the exact source slices and classification, then run the full `unittest` suite.

## History
- 2026-10-19 (intelligent-notebook-evaluator/session 1bd4afd4-d15d-4ea5-a699-020f5d06be2c): initial
