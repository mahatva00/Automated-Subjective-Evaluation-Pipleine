# Assessment Mapping Run Analysis

**Mode:** Analysis, implementation outline, and targeted follow-up fix.

**Run reviewed:** `run_20261014_213325`, from
`D:\assessments\assignment_01_postchange_assessment_output\run_20261014_213325.zip`

## Scope and limits

The ZIP is a run-output bundle. It contains parsed JSON, maps, evaluation and
checkpoint results, audit/scoring artifacts, logs, and HTML/JSON reports. It
does not contain the original question-paper PDF, the submitted PDF, or a
notebook. This review can verify what the pipeline extracted and assigned, but
cannot independently inspect the original page images or confirm visual plot
correctness from the ZIP alone.

The run manifest identifies one PDF submission, `PAMCA504-25MCA1081-LPS2.pdf`,
with `qwen2.5:3b-instruct`. It records student code as not executed.

## Findings

### 1. Question-paper global instructions are being evaluated as requirements

The parsed question map has six `global_instructions`. The requirement extractor
turns them into six records (`global_1_r1` through `global_6_r1`), and the
evaluation includes six `global_evaluations` with no question number. Their text
matches instructions from the question paper, including the notebook-format,
question-cell, title, dataset-size, and academic-integrity statements.

The requirement file contains 56 requirements: 6 derived from question-paper
global instructions and 50 attached to the ten numbered questions. The
`requirements_for_mapping` copy is byte-identical to `requirements_base`.
Meanwhile, the normalized policy artifact has policy metadata and `raw_text`,
but no structured policy-requirement list. Thus this run did not evaluate
global requirements sourced from the policy; it evaluated question-paper
globals instead. That conflicts with the stated rule that global instructions
outside a question in the question paper must be ignored.

The question paper also ends with a `Note:` outside the numbered questions.
The extracted question map records the note separately, but its final text
(`parameters not mentioned in the question.`) is also appended to Question 10
and becomes a Question 10 requirement. This is another scope leak from
question-paper end matter.

### 2. Coarse question-to-page assignment exists, but is not proof of solution matching

The evidence-map summary says 73 of 74 blocks were assigned and that all ten
questions have at least one block. The assigned PDF page ranges are mostly
plausible for this submission:

| Question | Assigned pages | Map status |
|---|---:|---|
| 1 | 1–3 | `evidence_found` |
| 2 | 3–4 | `evidence_found` |
| 3 | 4–20 | `evidence_found` |
| 4 | 21–48 | `evidence_found` |
| 5 | 49–52 | `evidence_found` |
| 6 | 53–61 | `evidence_found` |
| 7 | 62–63 | `evidence_found` |
| 8 | 64–65 | `ambiguous_repeated_number` |
| 9 | 65–66 | `evidence_found` |
| 10 | 67–69 | `evidence_found` |

The one unassigned block is the first page's student-identifying text and
imports before the first question heading, which is appropriate to leave
unassigned.

These results show that broad sectioning by question anchor and document order
is happening. They do **not** establish that each requirement is paired with
its corresponding code/output. `evidence_found` means blocks were attached to
a question; it is not a semantic correctness or requirement-coverage result.
For example, Question 4 receives 28 blocks for four requirements, while the
evaluator selects a limited subset of evidence blocks for each requirement.
The map does not expose a verified one-to-one requirement-to-cell/result
relationship.

### 3. Question 8 is falsely marked ambiguous by a prose mention

The student's page 64 begins with text saying `Q8 & Q9 not possible ...`, then
contains the actual heading `Q8) Genarate ...`. Both are treated as explicit
Question 8 anchors. The mapper therefore marks Question 8
`ambiguous_repeated_number`, even though the first occurrence is explanatory
prose and the second is the answer heading. The actual Question 8 solution
continues onto page 65.

This is an anchor-detection problem: a question number mentioned in explanatory
text should not be equivalent to a structural question heading.

### 4. Question-paper extraction contaminates requirement boundaries

The extracted Question 5 text includes OCR from an embedded chart:
`Advertisement effect on sales`, `Smartphone`, `medium`, and other chart
labels. These are unrelated to Question 5's regression task, but the extractor
turns some of them into Question 5 requirements. Question 5 has 16 extracted
requirements, including these spurious items and fragments split across lines
such as `If any column gets duplicated in` and `DF, remove the duplicate.`

Question 1's reference-figure OCR is likewise emitted as text such as chart
titles and axis labels, and one such OCR string becomes a requirement. The
visual references may be important to understanding the question, but OCR
labels alone do not express the expected pattern.

The parser also splits Question 1 into text/figure subparts, but part (b) has
only an empty label in the extracted question map. This means text extraction
cannot fully represent the two visual references as question content.

### 5. Evaluation and audit statuses overstate what has been verified

- The run completed with warnings and assessed one student.
- The evaluator processed 56 requirements: 13 `partial`, 43 `needs_review`,
  zero `complete`, and zero `absent`.
- The static audit passed with zero structural errors because all 56 input
  requirement IDs had corresponding assessment records. This is a structural
  completeness check, not a check that a requirement was mapped to the right
  solution or judged correctly.
- The evaluation records contain 43 exact-quotation validation failures,
  three failures because `absent` was returned without complete context, and
  one failure because a purported complete/partial result had no valid exact
  quotation. Model quotation problems remain a separate reason for unresolved
  results; they should not be mistaken for question-mapping errors.
- The final checkpoint contains all 56 assessment results. The log records
  progress through 56/56.
- The run/report `assessment_status: completed` means processing finished; it
  does not mean the work was successfully evaluated. The batch report also
  says zero students failed/partially assessed despite 43 requirements being
  unresolved.
- Numeric scoring is disabled because no explicit credit map is present. That
  is consistent with the current policy and is outside the immediate mapping
  work.

### 6. The student report obscures mapping quality

The student HTML and JSON reports contain 56 requirement rows, including six
question-paper global requirements labelled with an em dash and 50
question-scoped requirements. The HTML includes large evidence quotations,
but does not show the per-question mapping method or mapping status. In
particular, it does not surface Question 8's `ambiguous_repeated_number`
status. The high-level evidence-map summary counts all ten questions as
having evidence, which can look like successful mapping even when a mapping is
ambiguous or the selected blocks do not prove the associated requirement.

For this work, report redesign is not proposed. The internal map and run
diagnostics should be the source for comparing mappings.

## Follow-up review: PAT1 run `run_20261015_004620`

The input folder contained exactly three files: the assessment policy, the
question paper, and one student PDF. Their hashes matched the run manifest.
The parsed policy is retained as `raw_text`; it does not produce a structured
global-requirement list, and the static evaluator does not receive that policy
text as evaluation context.

The question map contains 38 question-paper `global_instructions`. The
requirement extractor split these into 39 `global_requirements`; together with
129 question requirements from all four variants, this produced the reported
168 requirements. All 39 question-paper globals were evaluated against
student evidence. The evaluation audit passed because every input requirement
had an output row, even though all 168 results were `needs_review`.

The student's first page explicitly says `QUESTION-B` and gives the matching
dataset equation. The question inventory nevertheless contains all four
variants. Evidence mapping assigned 9 of 15 blocks and left 6 unassigned:
the blocks on pages 1-5 and the first block on page 6, including preprocessing
work, were unassigned; the
training section was attached to the first occurrence (variant A); and the
numbered inference list on page 10 was interpreted as question headings and
assigned across the second occurrences. Only 4 of 12 question nodes received
evidence. This is a separate variant/heading-mapping defect, not a consequence
of global requirements alone.

Question-scoped evaluation also failed independently: 78 of 129 requirements
had no evidence blocks, with question results additionally rejected for
unknown block IDs, non-verbatim quotations, or insufficient context for an
`absent` judgment. The audit's structural pass therefore does not demonstrate
correct requirement-to-answer mapping. Numeric scoring was disabled because
the policy parser found no explicit numeric credit map; score/report design
remains outside this mapping fix.

### Implemented follow-up

The teacher-assessment pipeline now keeps question-paper preamble instructions
in the parsed question-map artifact for traceability, but removes them from
the requirement-extraction input. This prevents question-paper global
instructions from becoming evaluated `global_requirements`. A regression test
verifies both that the extracted global-requirement list is empty and that the
evaluator receives no global evaluations.

The change does not implement extraction or use of policy-level global
instructions, and does not select variant B or repair the PDF heading
assignments. Reprocessing this input after the fix would create 129
question-scoped requirements and no question-paper global requirements; the
variant and evidence-mapping problems still need a separate fix.

Validation passed: 7 teacher-pipeline tests, 15 requirement-extractor tests,
and 12 evidence-mapper tests. Replaying the requirement extraction against
this run's parsed question map produced 0 global requirements and 129 total
question requirements.

## Prioritized outline for the next phase

### P0 — Establish source scope and a fair comparison

1. **Enforce the global-instruction boundary.** (Partially completed: question-
   paper preamble requirements are excluded; policy-level global-rule extraction
   remains open.) Do not create evaluated
   requirements from question-paper material outside numbered questions.
   Later, derive global assessment requirements only from the assessment
   policy. Keep question-paper instructions/notes available as source metadata
   if useful, but do not let them enter the assessed requirement inventory.
2. **Separate input acceptance from deliverable compliance.** The product
   must accept both PDF and `.ipynb` inputs. The current policy text names an
   `.ipynb`; when implementing policy checks, treat an accepted PDF as an
   allowed representation of the submitted notebook rather than silently
   rejecting the input. Do not claim that PDF alone proves notebook metadata
   or interactivity.
3. **Run a controlled format comparison.** If possible, provide the same
   student's work in PDF and `.ipynb`, with the same question paper, policy,
   evaluator model, and settings. If the upcoming notebook is different work,
   record that content difference so it is not attributed to the format.
4. **Create a small human-verified mapping reference.** For each question,
   record the expected answer page range or notebook cell IDs and at least
   representative evidence for each requirement. Use that reference to judge
   assignment correctness instead of using `evidence_found` or audit pass as
   proxies.

### P1 — Improve structural question and answer mapping

1. **Use format-aware anchors.** For notebooks, prefer markdown/text question
   headings and preserve cell IDs, cell types, execution counts, and output
   association. For PDFs, require heading-shaped markers and distinguish them
   from prose mentions such as `Q8 & Q9 not possible`.
2. **Keep question scope and subparts explicit.** Treat numbered questions,
   lettered subparts, continuations across pages/cells, and notes as separate
   structural elements. Do not attach the final general note to Question 10.
3. **Treat reference figures as figures.** Preserve image location and
   association to the question. Do not turn chart axes/titles from an embedded
   question-paper image into standalone requirements; use the figure as
   visual context when assessing a visual task.
4. **Map requirements to answer evidence, not just a whole question section.**
   Select cells/blocks using the requirement text and its question/subpart
   context, retaining adjacent code/output where it is needed to interpret
   the result. Mark uncertain matches as uncertain rather than presenting
   them as verified.
5. **Prevent OCR fragments from becoming atomic criteria.** Join line-wrap
   continuations, preserve equations, filter unrelated page furniture/figures,
   and keep source page/cell provenance for every extracted criterion.

### P2 — Validate generalization and evaluator outcomes

1. Compare PDF and notebook runs by stable requirement ID and by manually
   verified question-to-answer assignments: correct assignment rate,
   unassigned answer content, cross-question leakage, and false/ambiguous
   anchors.
2. Track OCR/parse quality separately from mapping quality and evaluator
   status. Compare equivalent extracted source content before interpreting
   model status changes as a format effect.
3. Audit exact citations and `needs_review` causes separately from mapping.
   Preserve conservative review outcomes when evidence is genuinely
   insufficient.
4. Measure the desired accuracy target against labelled examples from
   multiple submissions. This single PDF run, with no originals in the ZIP
   and no verified answer labels, cannot establish a general accuracy rate.
5. Defer score generation and report presentation changes, as requested.

## Decision for the upcoming `.ipynb` input

For the cleanest comparison, keep the same question paper and policy and
include a PDF rendering of the notebook if available. Compare:

1. Extracted question headings and subparts.
2. Question-to-page/cell ranges and the cause of any ambiguity.
3. Requirement counts and provenance, especially whether question-paper
   globals or end notes appear as evaluated criteria.
4. Requirement-to-code/output evidence selections.
5. Quote-validation and `needs_review` reasons.

The question-paper-global leakage has since been addressed narrowly. Variant
selection and evidence-anchor improvements remain pending a human-verified
mapping reference and dedicated regression tests.
