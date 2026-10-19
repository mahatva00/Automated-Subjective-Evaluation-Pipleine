# FAT-QP structure proposal v2 — pilot summary

These are example outputs from rerunning the v2 generator against the FAT-QP draft and the manually reviewed FAT-QP scope ledger. The ledger is pilot input, not an automatically generated scope decision.

- Source document ID: `fat-qp_5e15afa66bd0`
- Source text length: 14,695 characters
- Whole-source SHA-256 matched the draft
- Question nodes: Q1 and Q2
- Total candidates: 104
- Candidate-type summary values sum to 104 and match the candidate records
- 64 header nodes, 35 nonempty header-excluded body spans, 5 fragments
- One unattached introductory/global-context fragment, retained without question membership or parent assignment by design
- Four fragments link to a continuation context across eligible reviewed segment boundaries
- Q2(b) retains question membership `Q2` across an excluded numeric page footer
- Two in-scope fragments remain unclassified; they are preserved for review rather than force-fitted to a heading
- All 104 candidates remain pending; no annotation labels or training labels were generated
- Automated pilot integrity check: 0 errors for exact source alignment, hashes, scope confinement, parent/pair references, and pending label state
- Regression tests: 17 passed

## Candidate type counts

```json
{
  "continuation_fragment": 2,
  "experiment_block": 9,
  "experiment_header": 9,
  "lettered_item_block": 3,
  "lettered_item_header": 4,
  "list_item_block": 17,
  "list_item_header": 37,
  "question_header": 2,
  "roman_item_block": 2,
  "roman_item_header": 3,
  "subquestion_block": 4,
  "subquestion_header": 9,
  "unattached_in_scope_fragment": 1,
  "unclassified_in_scope_fragment": 2
}
```

These counts describe proposal records, not accuracy or completeness. Body spans are emitted only when substantive text exists after a header within the same reviewed in-scope segment. Different paper formats can lead to missed headings, and a missed heading is acceptable at this stage. The important contract is exact source alignment, cautious hierarchy, explicit uncertainty, and no automatically approved labels.
