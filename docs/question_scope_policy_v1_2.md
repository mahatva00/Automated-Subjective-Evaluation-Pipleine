# Question Scope Policy v1.2 — Requirement Extraction Only

## Binding scope rule

Only the content that belongs to an actual question or one of its nested subquestions/parts is in scope for question-paper requirement extraction. Text outside a question must be ignored by this stage, even if it sounds like a possible student obligation.

Examples of out-of-scope text include document title and course/exam metadata; general preamble/instructions before, between, or after questions; submission/export/upload/filename directions; global notebook-format directions; academic-integrity or AI-use remarks; motivational comments such as “Use of intelligence is highly appreciated”; and other exam logistics. Do not annotate such text as `administrative_instruction`, `submission_instruction`, `context`, `other`, or a requirement. It is not part of this extraction task.

Do not derive assessment or grading rules from out-of-scope question-paper text. Those rules belong to the separate assessment-policy stage and may be applied only from the supplied assessment-policy file. If a relevant rule is absent from that file, do not invent it from the question-paper preamble.

## What is in scope

Within an actual question/subquestion, annotate the task, requested deliverables, and explicit assessable requirements. Retain the original wording and exact source offsets. Question headings may be retained as structure/context to associate requirements with the correct question, but a heading by itself is not automatically an assessable requirement.

A method restriction that appears within a question is in scope. For example, “calculate the mean without using a built-in mean function” contains a task and a stated method restriction. Represent these as separate requirements when each is independently assessable and can be spanned exactly without overlap or loss of meaning. Do not invent method restrictions that the question does not state.

Split a question into atomic requirements when its separately checkable expectations can be represented faithfully as non-overlapping source spans. Preserve inseparable compound requirements as compound/uncertain with a note; do not rewrite or complete the source text. This policy does not require turning every phrase into a separate requirement.

Marks, grading criteria, general submission instructions, or global AI-use directions are not converted into requirements by this stage. Any scoring effect is deferred to the assessment-policy stage and requires support from the assessment-policy file. If such text occurs in a question's source range, it may remain in the source text for offset stability, but must not be reinterpreted as an independent task requirement unless the question explicitly asks the student to do something with it.

## Handling text the system cannot exclude automatically

If extraction code cannot remove out-of-scope text from its intermediate representation, downstream components must filter it out before candidate creation, training, model inference, evidence mapping, or grading. In particular, out-of-scope tokens must not be converted into negative/background (`O`) training labels. Do not classify them into an ordinary semantic class as a workaround.

Scope decisions are stored separately from the canonical annotation file. Out-of-scope text receives no structural or semantic annotation spans for this task. A human-reviewed scope decision is not an annotation and is not evidence of a negative training label.

## Review checklist

1. Is this text inside a clearly identified question or subquestion? If no, mark it out of scope.
2. If in scope, does it state an assessable task, deliverable, explicit constraint/method, input/output condition, comparison, or explanation requirement?
3. Is a stated method restriction explicit in the question text? If yes, represent it as a separate requirement only when independently assessable.
4. Are exact source text, zero-based start offsets, and end-exclusive offsets preserved?
5. Has any policy or grading rule been inferred from global question-paper instructions rather than the separate assessment-policy file? If yes, remove that inference.
