# Semantic review data contract

## Inputs and trust boundary

The importer requires three matching, hash-checked inputs: canonical annotation JSON, semantic proposal JSON, and the scope ledger. The review sheet is treated as untrusted editor output. Proposal identity, immutable source spans, proposal attributes, question/scope IDs, confidence, rationale and flags are rechecked against the source proposal before any annotation can be emitted.

## Decision semantics

- `pending`: no decision; emits no annotation.
- `defer`: explicitly set aside; emits no annotation.
- `reject`: reviewed proposal was rejected; emits no annotation.
- `accept`: accept a non-empty proposed semantic label, exact proposed offsets and proposed requirement attributes.
- `revise`: reviewer supplies a final semantic label and may revise start/end offsets inside the same reviewed scope segment; requirement labels need valid requirement attributes.
- `split`: reviewer specifies two or more exact source spans, all within the original proposal. Split spans must be ordered/non-overlapping and each source slice becomes a separate annotation.

Candidate rows may not be deleted. Every original proposal must have exactly one review row. To ignore a candidate, use `reject` or `defer`.

## Annotation semantics

Each accepted/revised/split annotation records `review_status=reviewed` and the reviewer ID. The canonical document metadata remains `draft`; this import attests only to individual reviewed spans, not complete coverage for the whole semantic layer. Existing semantic annotations are checked for same-layer overlap. Structure and semantic layers may overlap each other.

## Source and scope constraints

All intervals are zero-based and end-exclusive. The exact text is derived from canonical document source using the final offsets; candidate text is never used to replace source. A candidate must be entirely inside a matching scope segment whose status is `in_scope`. Candidate rows may not be used to promote out-of-scope, pending, or separator text. Source text, scope and candidate hashes are validated before review export/import.

## Training gate

This workflow only creates reviewed span records. It does not certify that the full semantic layer is covered. BIO export must still require a reviewed/adjudicated document, explicit reviewed coverage for the semantic layer, and a document-level split manifest. Unreviewed regions remain masked, not `O`.
