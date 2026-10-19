"""Conservative deterministic scoring over static assessment statuses.

Never infers credit factors that the supplied policy did not state. The score
is marked provisional while any requirement is unresolved or approval is needed.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from typing import Any

VALID_STATUSES = {"complete", "partial", "absent", "needs_review"}
COMPONENTS = ("code", "explanation", "output")
COMPONENT_CREDITS = {"complete": 1.0, "partial": 0.5, "absent": 0.0}


def _all_evaluations(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(evaluation.get("global_evaluations", []) or [])
    for question in evaluation.get("questions", []) or []:
        rows.extend(question.get("evaluations", []) or [])
    return [x for x in rows if isinstance(x, dict)]


def _requirements_by_id(requirements: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in requirements.get("requirements", []) or []:
        if isinstance(row, dict) and row.get("requirement_id") is not None:
            result[str(row["requirement_id"])] = row
    return result


def _allocate_marks(requirements: dict[str, Any], policy: dict[str, Any]) -> tuple[dict[str, float], list[str], list[str]]:
    rows = _requirements_by_id(requirements)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for req in rows.values():
        groups[str(req.get("source_node_id") or req.get("requirement_id"))].append(req)
    weights: dict[str, float] = {}
    missing: list[str] = []
    for node_id, members in groups.items():
        values = [m.get("source_node_marks") for m in members]
        numeric = [float(x) for x in values if isinstance(x, (int, float)) and not isinstance(x, bool) and float(x) >= 0]
        if numeric:
            node_marks = max(numeric)
            share = node_marks / len(members) if members else 0.0
            for req in members:
                weights[str(req["requirement_id"])] = share
        else:
            missing.extend(str(m["requirement_id"]) for m in members)

    unallocated_questions = []
    mode = (policy.get("mark_allocation") or {}).get("mode", "per_source_node_equal_split")
    if mode == "equal_within_question":
        question_rows = {str(q.get("question_id")): q for q in requirements.get("questions", []) or [] if isinstance(q, dict)}
        by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for req in rows.values():
            qid = req.get("question_id")
            if qid is not None:
                by_qid[str(qid)].append(req)
        for qid, members in by_qid.items():
            # A question-level split is used only when every item in that question lacks explicit marks.
            if not members or any(str(m["requirement_id"]) in weights for m in members):
                continue
            q = question_rows.get(qid, {})
            marks = q.get("marks")
            if isinstance(marks, (int, float)) and not isinstance(marks, bool) and marks >= 0:
                share = float(marks) / len(members)
                for req in members:
                    weights[str(req["requirement_id"])] = share
                    if str(req["requirement_id"]) in missing:
                        missing.remove(str(req["requirement_id"]))
            else:
                unallocated_questions.append(qid)
    return weights, sorted(set(missing)), sorted(set(unallocated_questions))


def score_evaluation(evaluation: dict[str, Any], requirements: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    evaluations = _all_evaluations(evaluation)
    status_counts = Counter(str(row.get("status") or "needs_review") for row in evaluations)
    if any(s not in VALID_STATUSES for s in status_counts):
        invalid = sorted(s for s in status_counts if s not in VALID_STATUSES)
    else:
        invalid = []
    total = len(evaluations)
    resolved = status_counts.get("complete", 0) + status_counts.get("partial", 0) + status_counts.get("absent", 0)
    result: dict[str, Any] = {
        "schema_version": "1.0", "artifact_type": "policy_scoring_result",
        "numeric_scoring_enabled": False, "score_status": "not_scored",
        "score": None, "max_marks": None, "percentage": None,
        "status_counts": {s: status_counts.get(s, 0) for s in sorted(VALID_STATUSES)},
        "total_requirements": total, "resolved_requirements": resolved,
        "unresolved_requirements": status_counts.get("needs_review", 0),
        "teacher_approval_required": True, "final_grade": None,
        "notes": [],
    }
    if invalid:
        result["notes"].append(f"Evaluation includes unknown statuses: {invalid}.")
    if evaluation.get("student_code_executed") is not False:
        result["notes"].append("Static-only guardrail failed; numeric scoring disabled.")
        return result
    credit_map = policy.get("credit_map")
    if not isinstance(credit_map, dict) or any(s not in credit_map for s in ("complete", "partial", "absent")):
        result["notes"].append("Assessment policy does not provide explicit numeric credit for complete, partial, and absent.")
        result["notes"].append("No mark or percentage score was inferred from prose-only strictness terms.")
        return result
    for s in ("complete", "partial", "absent"):
        value = credit_map[s]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
            result["notes"].append(f"Invalid credit factor for {s}; numeric scoring disabled.")
            return result

    requirements_by_id = _requirements_by_id(requirements)
    weights, missing_marks, unallocated_questions = _allocate_marks(requirements, policy)
    eval_by_id = {str(row.get("requirement_id")): row for row in evaluations if row.get("requirement_id") is not None}
    alternatives = sorted({str(req.get("alternative_group_id")) for req in requirements_by_id.values() if req.get("alternative_group_id")})
    extra_evaluations = sorted(set(eval_by_id) - set(requirements_by_id))
    absent_evaluations = sorted(set(requirements_by_id) - set(eval_by_id))
    if not weights:
        result["notes"].append("No marks could be allocated from requirement provenance.")
        return result
    max_marks = sum(weights.values())
    known_awarded = 0.0
    pending_marks = 0.0
    unweighted_evaluations = []
    for rid, ev in eval_by_id.items():
        weight = weights.get(rid)
        if weight is None:
            unweighted_evaluations.append(rid)
            continue
        status = ev.get("status")
        if status in credit_map:
            known_awarded += weight * float(credit_map[status])
        elif status == "needs_review":
            pending_marks += weight
        else:
            unweighted_evaluations.append(rid)
    all_req_ids = set(requirements_by_id)
    all_have_weights = all_req_ids.issubset(weights)
    full_evaluation = not missing_marks and not unallocated_questions and not unweighted_evaluations and not absent_evaluations and not extra_evaluations
    no_alternatives = not alternatives
    no_unresolved = status_counts.get("needs_review", 0) == 0
    numeric_ready = max_marks > 0 and all_have_weights
    score_ready = numeric_ready and full_evaluation and no_alternatives and no_unresolved
    percentage_lower_bound = round((known_awarded / max_marks) * 100, 2) if max_marks > 0 else None
    approval_required = bool(policy.get("teacher_approval_required", True))
    result.update({
        "numeric_scoring_enabled": numeric_ready,
        "score_status": "ready_for_teacher_approval" if score_ready else ("provisional" if numeric_ready else "incomplete_mark_allocation"),
        "score": round(known_awarded, 3) if numeric_ready else None,
        "max_marks": round(max_marks, 3) if numeric_ready else None,
        "percentage": percentage_lower_bound if numeric_ready else None,
        "percentage_interpretation": "lower_bound_until_unresolved_items_are_reviewed" if status_counts.get("needs_review", 0) else "provisional_until_teacher_approval",
        "pending_review_marks": round(pending_marks, 3),
        "resolved_requirements": resolved,
        "unresolved_requirements": status_counts.get("needs_review", 0),
        "teacher_approval_required": approval_required,
        "final_grade": None,
        "mark_allocation": (policy.get("mark_allocation") or {}).get("mode", "per_source_node_equal_split"),
        "unweighted_requirement_ids": sorted(set(unweighted_evaluations + missing_marks)),
        "alternative_group_ids": alternatives,
        "unmatched_evaluation_ids": extra_evaluations,
        "requirements_missing_evaluation": absent_evaluations,
        "notes": result["notes"] + [
            "Scores are provisional; the tool does not issue an official grade or letter grade.",
            "Requirement marks are shared equally within a source node to prevent duplicated parent marks.",
        ],
    })
    if alternatives:
        result["notes"].append("Alternative-answer groups exist; they require dedicated alternative scoring logic before finalization.")
    if unallocated_questions:
        result["notes"].append(f"Question-level marks unavailable for {unallocated_questions}.")
    if status_counts.get("needs_review", 0):
        result["notes"].append("Unresolved requirements contribute no awarded points; percentage is a lower bound, not a final grade.")
    if approval_required:
        result["notes"].append("Teacher approval remains required by policy.")
    return result


def score_component_evaluation(
    evaluation: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Score question-level component statuses from explicit policy weights."""
    component_policy = policy.get("question_component_policy")
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "component_policy_scoring_result",
        "numeric_scoring_enabled": False,
        "score_status": "not_scored",
        "score": None,
        "max_marks": None,
        "percentage": None,
        "component_rows": [],
        "notes": [],
    }
    if not isinstance(component_policy, dict) or not component_policy:
        result["notes"].append("Assessment policy does not provide per-question component weights.")
        return result
    if evaluation.get("student_code_executed") is not False:
        result["notes"].append("Static-only guardrail failed; numeric scoring disabled.")
        return result

    question_evaluations = {
        str(question.get("question_number")): question
        for question in evaluation.get("questions", []) or []
        if isinstance(question, dict) and question.get("question_number") is not None
    }
    rows: list[dict[str, Any]] = []
    pending = False
    invalid = False
    score = 0.0
    max_marks = 0.0
    for question_number, allocation in component_policy.items():
        if not isinstance(allocation, dict):
            invalid = True
            result["notes"].append(f"Invalid component policy for question Q{question_number}.")
            continue
        marks = allocation.get("marks")
        weights = allocation.get("component_weights")
        if isinstance(marks, bool) or not isinstance(marks, (int, float)) or marks < 0:
            invalid = True
            result["notes"].append(f"Missing or invalid marks for question Q{question_number}.")
            continue
        if not isinstance(weights, dict) or any(
            component not in weights
            or isinstance(weights[component], bool)
            or not isinstance(weights[component], (int, float))
            or not 0 <= float(weights[component]) <= 1
            for component in COMPONENTS
        ):
            invalid = True
            result["notes"].append(f"Missing or invalid component weights for question Q{question_number}.")
            continue
        question = question_evaluations.get(str(question_number), {})
        raw_statuses = question.get("component_evaluations", {})
        if isinstance(raw_statuses, list):
            statuses = {
                str(item.get("component")): item.get("status")
                for item in raw_statuses
                if isinstance(item, dict)
            }
        elif isinstance(raw_statuses, dict):
            statuses = {
                component: value.get("status") if isinstance(value, dict) else value
                for component, value in raw_statuses.items()
            }
        else:
            statuses = {}
        for component in COMPONENTS:
            allocation_marks = float(marks) * float(weights[component])
            status = statuses.get(component, "absent")
            if status not in COMPONENT_CREDITS and status != "needs_review":
                invalid = True
                result["notes"].append(
                    f"Unknown status {status!r} for Q{question_number} {component}."
                )
                continue
            awarded = None if status == "needs_review" else allocation_marks * COMPONENT_CREDITS[status]
            if status == "needs_review":
                pending = True
            else:
                score += float(awarded)
            max_marks += allocation_marks
            rows.append({
                "question_number": str(question_number),
                "component": component,
                "status": status,
                "weight": float(weights[component]),
                "allocated_marks": round(allocation_marks, 3),
                "awarded_marks": round(awarded, 3) if awarded is not None else None,
            })

    if invalid or not rows:
        if not rows:
            result["notes"].append("No component marks could be allocated.")
        result["component_rows"] = rows
        return result
    result.update({
        "numeric_scoring_enabled": True,
        "score_status": "provisional" if pending else "ready_for_teacher_approval",
        "score": round(score, 3),
        "max_marks": round(max_marks, 3),
        "percentage": round((score / max_marks) * 100, 2) if max_marks else None,
        "component_rows": rows,
        "pending_review_components": sum(row["status"] == "needs_review" for row in rows),
        "teacher_approval_required": bool(policy.get("teacher_approval_required", True)),
    })
    return result
