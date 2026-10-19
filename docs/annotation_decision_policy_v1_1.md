> **Scope precedence — question_scope_policy_v1.2:** This taxonomy must only be applied to text inside an actual question or subquestion. Text outside question boundaries—including administrative, grading, submission, AI-use, motivational, and exam-logistics instructions—is out of scope for question requirement extraction and must not be labelled with any category below. See `docs/question_scope_policy_v1_2.md`. Assessment-policy interpretation is a separate downstream stage and must use the supplied assessment-policy file only.

# Annotation Decision Policy v1.1 — Whole-document DistilBERT Span Extraction

## Purpose and scope

This policy clarifies decisions for annotating the existing document schema v1.0.0. It does not change the current schema, source text, requirement-extraction outputs V1–V5, evidence maps, or existing model checkpoints. Labels describe what a span does in the document; labels must not be chosen from the subject name or programming language.

The immediate training objective is **span detection from canonical source text**. A future model may predict separate structural and semantic BIO tags, requirement-type attributes, and relations. V1–V5 candidate spans and previous model predictions are suggestions for review only, not ground truth.

## Non-negotiable source rules

1. Annotate the `text` string in the source annotation JSON. `start_char` is zero-based and `end_char` is exclusive.
2. For every annotation, `annotation.text` must equal `document.text[start_char:end_char]` exactly. Preserve OCR/PDF oddities, spelling, punctuation, URLs, formulas and line breaks. Do not silently repair, paraphrase, or complete truncated text.
3. If a page separator or extraction artifact is not itself meaningful, do not include it in a semantic annotation. Never alter `document.text` to make annotation easier.
4. Each document must be reviewed across **both layers** (structure and semantics). A blank area is a confirmed no-span/background only after it has actually been reviewed. An unreviewed area is unknown and must never be treated as a negative training example.
5. Do not execute student code or any code printed in a paper. Annotation is reading and labeling source text only.

## Layer 1: structure labels

Choose one primary structure label per span. Same-layer spans must not overlap; a semantic span may cover the same characters because it belongs to the other layer.

| Label | Use when |
|---|---|
| `document_title` | The document's title. |
| `section_heading` | A heading grouping a section or topic, such as “Submission Format”. |
| `question_heading` | The heading/number/title for a main question or standalone task. |
| `subquestion_heading` | A labeled subpart or experiment heading nested within a question. |
| `question_body` | Prose that belongs to a question and is not more specifically a heading, list item, table, etc. |
| `subquestion_body` | A distinct prose body for a subquestion where the distinction is useful. |
| `instruction_heading` | A heading that introduces document-level instructions. |
| `list_marker` | A bullet, number, or letter marker only when separately useful. |
| `list_item` | The item text without its separately annotated list marker. |
| `table_heading`, `table`, `table_row`, `table_cell` | Apply the most specific practical table structure. Do not annotate every cell if a table-level span is enough for the task. |
| `code_block`, `equation_block`, `figure_caption` | A code/configuration/query block, displayed equation, or figure caption included in the paper. |
| `page_header`, `page_footer` | Repeating layout text unrelated to the question content. |
| `document_metadata` | Course, date, exam code, duration, maximum marks, and similar metadata. |
| `continuation_block` | Text whose main role is continuing a block across a page/layout break, only when no more specific label fits. |
| `other` | Meaningful structure not covered above; explain `attributes.label_note`. |

Do not create an enclosing `question_body` over characters already used by a heading/list/table on the structure layer. Leave decorative whitespace unannotated. Structural annotations are not required for every line if the structure is clear, but each chunk must still be reviewed for missed structural spans.

## Layer 2: semantic labels

Choose one primary semantic label per span. Semantic spans must not overlap each other in v1.

| Label | Decision rule |
|---|---|
| `requirement` | Text asking for an assessable task-specific action, deliverable, constraint, criterion, input/output, method/tool, calculation, explanation, justification, or comparison. Add all three requirement attributes defined below. |
| `context` | Background/setup needed to understand the task, but not itself something the student is being asked to do. |
| `example` | Illustrative text introduced as an example or demonstration, unless the source explicitly requires the example itself. |
| `option` | A selectable method, value, path, or item within a set of choices. |
| `alternative_marker` | The actual connector/marker that expresses choice, such as “OR”, not the content of either option. |
| `grading_instruction` | Marks, point allocations, weights, or explicit criteria describing how work is graded. Do not absorb the task wording just because `[2 marks]` follows it. |
| `submission_instruction` | Notebook/document format, required output packaging, filename, export, upload, or other submission mechanics. |
| `administrative_instruction` | Exam logistics, eligibility, academic-integrity rules, or general conduct policy rather than the assessed task itself. |
| `definition` | A definitional statement or explanation of a term not itself phrased as an assessed task. |
| `explanatory_note` | Explanation/contextual note that does not meet a more specific label. |
| `non_assessable` | Use only when a reviewer explicitly decides the text is not assessable and that distinction is useful. Do not use as a shortcut for uncertainty. |
| `other` | Meaningful semantic content not covered above; explain `attributes.label_note`. |

### Requirement versus instructions

Ask: **Could a marker reasonably decide whether the student fulfilled this text from the submitted work?** If yes and it describes the subject task, annotate it as `requirement`. If it concerns how to package/upload the submission, use `submission_instruction`. If it concerns exam rules or academic conduct, use `administrative_instruction`. If it states marks or grading weights, use `grading_instruction`.

Source-grounded examples from FAT:

- “Check the dimension and display the first 15 records of the dataset.” contains two independently assessable actions. Prefer two adjacent, non-overlapping `requirement` spans: one for checking the dimension and one for displaying the first 15 records. Preserve every character in each span and do not repeat the shared source text.
- “After completing the exam: Save your notebook as a PDF...” is a submission instruction, not a machine-learning method requirement.
- “Full marks will be awarded only if both code and corresponding output are present.” is a grading instruction/criterion. Annotate the criterion as `grading_instruction` unless the project explicitly chooses to model scoring criteria as requirements in a future schema version; do not mix policies during this dataset.
- “For example, if ...” is an `example` unless the task explicitly requires that exact example.
- A methods list (e.g., kernel names) is not automatically a set of requirements. Use `option` when the text offers selectable choices; use `context` where it merely lists background examples. Use relationships to connect options to the requirement/choice group.

When a phrase could fit two categories, use the rule above, consult local context, add a note if genuinely uncertain, and request adjudication instead of inventing a new label or subject-specific rule.

## Requirement attributes and atomization

Every semantic `requirement` annotation must have:

- `requirement_type`: `action`, `constraint`, `deliverable`, `success_criterion`, `input_specification`, `output_specification`, `method_or_tool`, `evaluation_or_comparison`, `explanation_or_justification`, or `other`.
- `atomicity`: `atomic`, `compound`, or `uncertain`.
- `assessment_relevance`: `yes`, `no`, or `uncertain`.

Use `atomic` when the span expresses one independently assessable expectation. Split a sentence into separate adjacent requirement spans when it contains separable actions or constraints and the exact source text can be partitioned without overlap. Use `compound` when multiple expectations are inseparable in the source or splitting would make at least one span unintelligible; record a note if a plausible split needs adjudication. Use `uncertain` rather than guessing.

Do not copy a whole multi-sentence question into one requirement span merely because V5 did so. Conversely, do not split every phrase into tiny fragments: the span must retain enough words to preserve the actual expectation. When a heading gives task context and the sentence below states the action, annotate them separately and connect them with relations if needed.

Examples:

- “Train a model” and “Report its validation error” are distinct tasks when both occur in the text; each can be checked independently.
- “Train a model and compare the two evaluation measures” ordinarily contains two independently assessable expectations, so split if exact boundaries can be chosen without overlap.
- A compound constraint with multiple conditions that cannot be made meaningful alone can remain one `requirement` with `atomicity: compound`; do not silently rewrite it.

## Relations

Add relations only when the relationship is supported by the source or clear document hierarchy; adjacency alone is not evidence.

- `parent_of`: section/question/subquestion structurally contains the target.
- `belongs_to`: a requirement/context/option is associated with a question; keep direction consistent with the existing schema's examples.
- `continues`: a source block continues across layout/page breaks.
- `alternative_to` and `option_for`: connect mutually exclusive alternatives/options to the appropriate choice group.
- `constrains`: one instruction/requirement limits another requirement.
- `depends_on`, `precedes`, `references`, `applies_to`: use only when the dependency/order/reference/scope is explicit or genuinely clear.
- `other`: explain the relation in a note/attributes when necessary.

Do not invent a relation solely to fill a field. Missing/uncertain relationships should remain absent or flagged for review.

## Completeness and review states

A document's annotation status describes the **whole document**, not only the selected spans:

- `draft`: annotation is incomplete, candidate labels remain unreviewed, or any source region/layer is not fully reviewed.
- `reviewed`: a human has reviewed all source text in both layers, confirmed spans and labels, reviewed relations, and verified that any unlabeled text is true background for that layer.
- `adjudicated`: a designated reviewer has resolved disagreements and the document is suitable for a gold-standard evaluation set.
- `example`: schema demonstration only; never training data.

For now, use the coverage sidecar workflow to track review of the whole text independently for `structure` and `semantic`. A chunk is `reviewed` only after every relevant span in that layer has been identified—or the reviewer has confirmed that the chunk has no span for that layer. Merely reviewing candidate rows in `pilot_review.csv` does not make a document complete.

Do not mark a whole paper `reviewed` while any coverage chunk is pending. Do not call the three-document pilot a cross-domain dataset. Do not convert unannotated regions, V1–V5 candidate absence, or model suggestions into gold negatives.

## Annotation disagreements and dataset split

- Keep uncertainty explicit and add a short note when necessary.
- Double-annotate a subset if feasible; record disagreements and adjudicate them before using those documents as gold evaluation data.
- Split by complete document or related document family, never random spans from the same paper into train and validation/test.
- The existing drafts are all Machine Learning question papers. Cross-domain claims require reviewed documents from additional subjects and a held-out domain/paper family.
- Keep exact source text and offsets; model canonicalizations, if used, belong in separate attributes and never replace the source span.
