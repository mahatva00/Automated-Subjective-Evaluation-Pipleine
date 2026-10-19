# FAT-QP semantic proposal pilot — v1

## Source and scope

- Document: `FAT-QP.pdf`
- Document ID: `fat-qp_5e15afa66bd0`
- Canonical source text: 14,695 characters
- Source-text SHA-256: `14f20cf60ef2a4f23637197c0b46a8d176216967a68e3518e9e7513f8d99a08b`
- Scope ledger: manually prepared FAT-QP pilot boundary; not generated automatically by the pipeline
- Recognized top-level question boundaries: Q1 and Q2

## Proposal counts

| Proposed label / status | Count |
|---|---:|
| `requirement` | 72 |
| `grading_instruction` | 17 |
| `example` | 2 |
| `context` | 1 |
| `explanatory_note` | 2 |
| `unclassified` (`label_proposal = null`) | 10 |
| **Total semantic candidates** | **104** |

Question reconstruction uses 9 reviewed in-scope fragments for Q1 and 15 for Q2. One in-scope introductory/global region before Q1 remains intentionally unassigned. No out-of-scope text is included in candidate spans or reconstructed question text.

## Review observations

- Requirements are proposals from generic action/constraint language, not approved annotations. Some may be compound or require a different `requirement_type` after contextual review.
- All 17 per-question marks annotations remain pending `grading_instruction` proposals. General/global grading-policy text is excluded by scope.
- Two explicit example spans are separated from surrounding task instructions where the phrase `For example` provides a boundary cue.
- Some misspelled/poorly extracted task phrases remain unclassified, such as `Caluclate ...`; source text is not silently corrected.
- A table-like block containing `ModelAccuracy / Train data / Validation data` remains unclassified instead of being mistaken for an imperative based on the word `Train`.
- A source continuation is proposed as a pending `continues` relation only where the structural layer supplies an explicit continuation group.

## Integrity checks

The pilot passed independent checks for:

- exact source slices and SHA-256 hashes;
- unique proposal IDs;
- non-overlapping same-layer semantic candidate spans;
- containment within reviewed in-scope source segments;
- assignment only to Q1/Q2 recognized question ranges;
- all candidates pending, with null annotation/training labels and `training_eligible = false`;
- no use of the unassigned global introduction as semantic candidate text.

The test suite contains 21 passing tests. These are contract/regression checks, not semantic accuracy metrics. A teacher/reviewer must confirm every semantic label and requirement atomicity before this output can feed a span-training dataset.
