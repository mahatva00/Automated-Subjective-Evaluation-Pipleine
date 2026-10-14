"""Produce auditable JSON and HTML student assessment reports."""
from __future__ import annotations
import html
import json
from pathlib import Path
from typing import Any
from collections import Counter


def evaluation_rows(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(evaluation.get("global_evaluations", []) or [])
    for q in evaluation.get("questions", []) or []:
        for row in q.get("evaluations", []) or []:
            copy = dict(row)
            copy.setdefault("question_number", q.get("question_number"))
            copy.setdefault("question_title", q.get("question_title"))
            rows.append(copy)
    return rows


def build_student_report(student: dict[str, Any], evaluation: dict[str, Any] | None,
                         scoring: dict[str, Any] | None, policy: dict[str, Any],
                         evidence_map: dict[str, Any] | None, warnings: list[str] | None = None) -> dict[str, Any]:
    rows = evaluation_rows(evaluation or {})
    statuses = Counter(str(x.get("status") or "needs_review") for x in rows)
    return {
        "schema_version": "1.0", "artifact_type": "student_assessment_report",
        "student": student, "student_code_executed": False,
        "assessment_status": "completed" if evaluation is not None else "evaluation_failed",
        "question_count": len((evidence_map or {}).get("questions", []) or []),
        "evidence_map_summary": (evidence_map or {}).get("summary"),
        "evaluation_summary": (evaluation or {}).get("summary"),
        "status_counts": {s: statuses.get(s, 0) for s in ("complete", "partial", "absent", "needs_review")},
        "scoring": scoring or {"score_status": "not_scored", "score": None, "final_grade": None,
                               "notes": ["No numeric score generated."]},
        "policy": {k: policy.get(k) for k in ("source_filename", "source_sha256", "strictness", "numeric_scoring_enabled", "credit_map", "warnings")},
        "requirements": rows,
        "warnings": list(warnings or []),
        "audit": {"grading_policy_used_for_evaluation": False, "policy_scoring_stage_ran": scoring is not None,
                  "official_grade_issued": False, "student_code_executed": False},
    }


def render_student_report(report: dict[str, Any]) -> str:
    student = report.get("student", {}) or {}
    scoring = report.get("scoring", {}) or {}
    name = html.escape(str(student.get("student_roll_no") or student.get("display_name") or student.get("source_filename") or "Student"))
    title = html.escape(str(student.get("source_filename") or name))
    counts = report.get("status_counts", {}) or {}
    status_rows = "".join(f"<tr><th>{html.escape(s.replace('_',' ').title())}</th><td>{int(counts.get(s, 0) or 0)}</td></tr>" for s in ("complete", "partial", "absent", "needs_review"))
    if scoring.get("numeric_scoring_enabled"):
        score_html = (f"<p><strong>Provisional points:</strong> {html.escape(str(scoring.get('score')))} / "
                      f"{html.escape(str(scoring.get('max_marks')))}"
                      f" ({html.escape(str(scoring.get('percentage')))}%)</p>")
    else:
        score_html = "<p><strong>Numeric marks:</strong> Not calculated because the supplied policy or mark allocation did not support a defensible score.</p>"
    req_rows = []
    for row in report.get("requirements", []) or []:
        qnum = row.get("question_number") or "—"
        req_text = row.get("requirement_text") or row.get("atomic_text") or row.get("source_text") or row.get("requirement_id")
        evid = "<br/>".join(html.escape(str(ev.get("quote") or "")) for ev in row.get("evidence", []) or [])
        req_rows.append("<tr>" + "".join([
            f"<td>{html.escape(str(qnum))}</td>",
            f"<td>{html.escape(str(row.get('requirement_id') or ''))}</td>",
            f"<td>{html.escape(str(req_text or ''))}</td>",
            f"<td>{html.escape(str(row.get('status') or 'needs_review'))}</td>",
            f"<td>{html.escape(str(row.get('confidence') if row.get('confidence') is not None else ''))}</td>",
            f"<td>{evid}</td>",
            f"<td>{html.escape(str(row.get('rationale') or ''))}</td>",
        ]) + "</tr>")
    warnings = "".join(f"<li>{html.escape(str(x))}</li>" for x in report.get("warnings", []) or [])
    score_notes = "".join(f"<li>{html.escape(str(x))}</li>" for x in scoring.get("notes", []) or [])
    return f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>Assessment report — {title}</title>
<style>body{{font-family:Arial,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem;color:#202124}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #bbb;padding:.5rem;text-align:left;vertical-align:top}}th{{background:#f2f2f2}}td{{font-size:.94rem}}.note{{background:#fff8df;padding:.8rem;border-left:4px solid #a67c00}}code{{overflow-wrap:anywhere}}</style></head><body>
<h1>Student assessment report</h1><p><strong>Student:</strong> {name}</p><p><strong>Submission:</strong> {title}</p>
<p><strong>Assessment status:</strong> {html.escape(str(report.get('assessment_status')))}</p>
<div class=\"note\"><strong>Static evaluation:</strong> student-submitted code was not executed. Model findings and marks are provisional and require teacher review. No official grade has been issued.</div>
<h2>Marks</h2>{score_html}<ul>{score_notes}</ul>
<h2>Requirement summary</h2><table><thead><tr><th>Status</th><th>Count</th></tr></thead><tbody>{status_rows}</tbody></table>
<h2>Requirement-level findings</h2><table><thead><tr><th>Question</th><th>Requirement ID</th><th>Requirement</th><th>Status</th><th>Confidence</th><th>Evidence quote(s)</th><th>Rationale</th></tr></thead><tbody>{''.join(req_rows)}</tbody></table>
<h2>Warnings</h2><ul>{warnings or '<li>None recorded.</li>'}</ul>
<p>Assessment policy scoring enabled: {html.escape(str((report.get('policy') or {}).get('numeric_scoring_enabled')))}. Official grade: not issued.</p>
</body></html>"""


def write_student_report(output_dir: str | Path, report: dict[str, Any]) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stem = str((report.get("student", {}) or {}).get("report_id") or "student")
    json_path = target / f"{stem}.report.json"
    html_path = target / f"{stem}.report.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(render_student_report(report), encoding="utf-8")
    return json_path, html_path
