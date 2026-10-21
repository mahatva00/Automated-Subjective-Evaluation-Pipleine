# PDF Numbered Subpart Anchors

Treat compact `<question number> <subpart letter>` PDF lines as question anchors only when the question map confirms that exact subpart.

## What Happened
Assignment 03's anomalous extracted pages used markers such as `1 a` and `3 b`, which the PDF evidence mapper did not recognize. Accepting every number-plus-letter line would risk promoting ordinary lists into question boundaries. The mapper now validates the part label against the unique question's known subparts, and repeated parts of one question do not trigger repeated-question ambiguity. Replaying saved parsed artifacts improved assignment, but unanchored pages still inherit document-order ownership and several questions remain empty (intelligent-notebook-evaluator, session 1bd4afd4-d15d-4ea5-a699-020f5d06be2c).

## Takeaway
Use source question-map structure to validate compact PDF anchors. Test both valid and unknown part labels. Report mapping improvement as partial only when later pages lack explicit anchors; never infer that a student response or grade is correct from block assignment alone.

## History
- 2026-10-19 (intelligent-notebook-evaluator/session 1bd4afd4-d15d-4ea5-a699-020f5d06be2c): initial
