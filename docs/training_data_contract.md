# Training-data contract and DistilBERT plan

## Canonical source of truth

One `*.annotation.json` file represents one complete source question paper. It stores the canonical text, source-page character ranges, human span annotations, relations, and review status. It is independent of tokenizer choice, max sequence length, or model checkpoint. `model_dataset.jsonl` contains one reviewed/adjudicated document record per line plus its split/group metadata.

## First learned tasks

1. **Structure span detection:** token/span prediction for the structure layer (headings, question/subquestion blocks, list structures, metadata and other generic layout roles).
2. **Semantic span detection:** a separate prediction layer for requirement, context, example, option, alternative marker, grading/submission/administrative instruction, definition, explanatory note, and other roles.
3. **Requirement-type classification:** classify a detected requirement span using the broad `requirement_type` attribute.
4. **Relation prediction:** predict relationships such as `parent_of`, `continues`, `alternative_to`, `constrains`, and `depends_on` between candidate spans. This may require a separate relation classifier or a deterministic graph assembly step after predictions.

Do not expect a base DistilBERT encoder to generate a free-form list of requirements. In the first implementation it predicts spans/classes; hierarchy is reconstructed from predicted spans and relations. A later generative model may help with paraphrasing or semantic verification, but that is a distinct stage and does not alter the source annotations.

## Tokenization/windows

Keep canonical annotations in character offsets. During model-data compilation, use the selected Hugging Face tokenizer with offset mappings to align character spans to token spans. Do not hand-split documents using question-number regexes. Split long papers into overlapping token windows (initial experiment: at most 512 tokens with a 64-token overlap), mapping every local token/span back to the global document and source-page ranges. Tune the window size/overlap empirically and do not silently drop annotations crossing a window boundary; either make a window that contains the span or flag the record for special handling.

Produce BIO or BIOES labels independently for the structure and semantic layers. The v1 annotation validator disallows same-layer overlap so each token has a single gold label per layer. Cross-layer overlap is allowed. Requirement-type and relation labels remain separate targets rather than being forced into the BIO label inventory.

## Split strategy

The split manifest is assigned at the document/group level, never at the line or span level. Every paper or near-duplicate version in the same `group_id` must stay in the same split. Prefer a grouped validation/test set and, as the dataset grows, reserve whole subjects or course families to test cross-domain generalization. If all available annotated papers cover Machine Learning, a good ML test score must not be described as proof of DBMS/cloud generalization.

Do not include synthetic examples, drafts, unreviewed model predictions, or generated V1–V5 requirements as ground-truth labels. Only `reviewed` or `adjudicated` annotations with explicit `train`, `validation`, or `test` assignments are exported.

## Evaluation

Report metrics separately by task and by paper/subject:

- exact-span precision/recall/F1 and relaxed span F1 for structure and semantic spans;
- requirement-type macro-F1;
- relation F1 and hierarchy reconstruction accuracy;
- requirement-level exact match / coverage for source requirements;
- error counts for dropped text, invented spans, invalid offsets, duplicate IDs, and unresolved cases;
- cross-domain per-subject results and abstention/human-review rate.

Never use only aggregate accuracy; class imbalance can hide failures on less frequent roles. Keep a manually adjudicated test set untouched while labels and model choices are tuned.

## Not included yet

This package defines and validates the label data and exports canonical reviewed JSONL. It does **not** fine-tune a model, automatically label papers, create paraphrases, assess students, assign marks, parse assessment policy, or execute student code. Those are later phases after we have enough reviewed examples and an agreed split.
