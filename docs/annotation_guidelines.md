# Cross-domain annotation guidelines v1

## Purpose

Create teacher-reviewed annotations that can train a model to understand question-paper structure and assessable meaning across subjects. Annotate what the text does, not which subject-specific pattern it resembles. Do not introduce a new label solely because a paper comes from Python, DBMS, cloud, networking, mathematics, or another subject.

## Source text and offsets

1. Annotate the `text` field in the document JSON, not the PDF viewer's visual coordinates.
2. `start_char` is zero-based and `end_char` is exclusive. `text` on an annotation must equal `document.text[start_char:end_char]` exactly.
3. Preserve spelling, punctuation, capitalization, line breaks, URLs, formulas, and awkward wording in the source. Do not silently repair text or fill missing phrases.
4. Page text ranges identify which portion came from each source page. Page separators are `\n\f\n`; do not include the separator in a text annotation unless the separator itself is explicitly relevant (normally it is not).
5. Mark only text supported by the source. If OCR or extraction is ambiguous, leave a note or use `other` with a note; do not guess.

## Two independent annotation layers

### Structure layer

Use exactly one structural label for each annotated structural span. Spans in the same layer must not overlap in v1. Use relations to connect spans across hierarchy rather than relying on nested/overlapping spans.

- `document_title`: title of the assessment/document.
- `section_heading`: heading that groups a section or topic.
- `question_heading`: label/title introducing a major question or task.
- `subquestion_heading`: label/title for a subquestion or nested task.
- `question_body`: prose belonging to a question but not separately classified as a heading or list item.
- `subquestion_body`: body text belonging to a subquestion when a distinct body block is useful.
- `instruction_heading`: heading introducing instructions/rules.
- `list_marker`: visible bullet/number/letter marker when separately useful.
- `list_item`: content of a list item, excluding its marker if separately annotated.
- `table_heading`: a heading/title associated with a table.
- `table`: an entire table block when individual rows/cells are not annotated.
- `table_row`: row text when row-level grouping matters.
- `table_cell`: an individual table cell when table structure matters.
- `code_block`: a code, command, query, configuration, or similar verbatim block included in the question paper; the language is not used to define a separate label.
- `equation_block`: a displayed equation or mathematical expression treated as a block.
- `figure_caption`: text identifying/explaining a figure or diagram.
- `page_header`: recurring header unrelated to the actual question content.
- `page_footer`: recurring footer/page number unrelated to the actual question content.
- `document_metadata`: course, date, exam code, duration, or other administrative metadata.
- `continuation_block`: text whose primary role is to continue a block across layout/page boundaries, when it cannot be assigned more specifically.
- `other`: structural span not covered above; explain in `attributes.label_note`.

Do not create a `question_body` annotation over the same characters as list items or headings on the structure layer. Leave unannotated prose unannotated where assigning a structural label adds no value. Semantic annotations can overlap structural annotations because they are a separate layer.

### Semantic layer

Use one primary semantic label per span. Spans in the same semantic layer must not overlap in v1.

- `requirement`: an assessable requested action, deliverable, constraint, criterion, input/output specification, explanation, comparison, configuration, query, calculation, or other task-specific expectation. Use `attributes.requirement_type` to record its broad role.
- `context`: background facts or setup needed to interpret a requirement but not itself something the student must do.
- `example`: illustrative example that explains a requirement without itself being a required output unless the source explicitly says so.
- `option`: one selectable option among alternatives.
- `alternative_marker`: wording/symbol linking mutually exclusive or substitutable paths, such as an explicit OR choice.
- `grading_instruction`: marks, points, weighting, or grading criteria stated by the source.
- `submission_instruction`: required format, filename, upload method, packaging, or submission behavior.
- `administrative_instruction`: exam logistics and administrative rules that are not part of the assessed subject task.
- `definition`: a definition or terminology explanation that is not itself an assessed action.
- `explanatory_note`: explanatory prose that is not itself a requirement, example, or definition.
- `non_assessable`: text explicitly determined by the annotator not to be assessable, when distinguishing it is useful.
- `other`: semantically meaningful text that does not fit; provide `attributes.label_note`.

A line like `Linear kernel` may be a selectable `option`, context, or part of a requirement depending on its context. Do not label it by wording alone. Use its surrounding question and relationships. A bullet is not automatically a requirement; its meaning determines the semantic label.

## Requirement attributes

Every semantic annotation with label `requirement` must have:

- `requirement_type`: one of `action`, `constraint`, `deliverable`, `success_criterion`, `input_specification`, `output_specification`, `method_or_tool`, `evaluation_or_comparison`, `explanation_or_justification`, `other`.
- `atomicity`: `atomic`, `compound`, or `uncertain`.
- `assessment_relevance`: `yes`, `no`, or `uncertain`.

Atomicity guidance:

- `atomic`: one independently assessable expectation.
- `compound`: multiple expectations are combined into the same source span and cannot be split without changing the source annotation; flag it for adjudication if it can reasonably be split.
- `uncertain`: not confident whether this should be one requirement or multiple requirements.

Prefer separate adjacent requirement spans for distinct actions or constraints, while preserving exact source wording. Do not rewrite or summarize source text in the span. An optional `canonical_text` in attributes may hold a teacher-approved concise interpretation, but it is not the source span and must never replace it.

## Relations

Relations are directed. Use `source_annotation_id` and `target_annotation_id` from existing annotations.

- `parent_of`: a structural node/section directly groups the target node.
- `belongs_to`: target or source is explicitly associated with a question/subquestion; use a consistent direction and document it in the annotation notes if needed.
- `continues`: one span continues the text/meaning of another across a line or page break.
- `alternative_to`: two alternatives are mutually exclusive or substitutable.
- `option_for`: an option is available for a requirement or choice group.
- `constrains`: one span places a constraint on another requirement.
- `depends_on`: one task depends on another.
- `precedes`: order matters and the source implies sequence.
- `references`: one span points to another span or an external artefact mentioned in the text.
- `applies_to`: a general instruction applies to the target question(s).
- `other`: a meaningful relation not listed; describe it in relation attributes.

Do not infer relationships merely because items are adjacent. Use `other` with a note when the relationship is meaningful but cannot be described confidently.

## Review and adjudication

- `draft`: not yet teacher-reviewed; excluded from training.
- `reviewed`: reviewed by an annotator using these guidelines.
- `adjudicated`: disagreements have been reconciled or reviewed by the designated adjudicator; best candidate for gold evaluation data.
- `example`: synthetic schema demonstration; always excluded from training export.

Keep uncertainty explicit. Do not turn V1–V5 parser output or model suggestions into gold labels without a human checking the exact source. When possible, double-annotate a subset, record disagreements, and adjudicate it before using it for evaluation.
