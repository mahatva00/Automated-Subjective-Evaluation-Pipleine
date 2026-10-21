"""Produce auditable JSON and HTML student assessment reports."""
from __future__ import annotations
import html
import json
from pathlib import Path
from typing import Any
from collections import Counter


def evaluation_rows(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in evaluation.get("global_evaluations", []) or [] if isinstance(row, dict)]
    for q in evaluation.get("questions", []) or []:
        for row in q.get("evaluations", []) or []:
            if not isinstance(row, dict):
                continue
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
    for row in rows:
        raw_status = row.get("status") or "needs_review"
        if raw_status in {"complete", "partial", "absent"}:
            row["final_status"] = raw_status
            row["final_status_source"] = "model_status"
        elif raw_status == "needs_review":
            row["final_status"] = "partial"
            row["final_status_source"] = "default_partial_pending_teacher_review"
        else:
            row["final_status"] = None
            row["final_status_source"] = "invalid_model_status"
    question_components = []
    for question in (evidence_map or {}).get("questions", []) or []:
        component_blocks = question.get("component_blocks") or {}
        question_components.append({
            "question_id": question.get("question_id"),
            "question_number": question.get("question_number"),
            "question_title": question.get("question_title"),
            "mapping_status": question.get("mapping_status"),
            "present_components": [
                component for component in ("question", "question_copy", "code", "output", "explanation", "ai_log")
                if component_blocks.get(component)
            ],
            "component_block_counts": {
                component: len(component_blocks.get(component) or [])
                for component in ("question", "question_copy", "code", "output", "explanation", "ai_log")
            },
            "missing_components": list(question.get("missing_components") or []),
            "ai_log_present": bool(question.get("ai_log_present")),
            "requirements": [
                {
                    "requirement_id": requirement.get("requirement_id"),
                    "linked_evidence_block_ids": list(requirement.get("linked_evidence_block_ids") or []),
                    "evidence_link_status": requirement.get("evidence_link_status"),
                    "evidence_link_confidence": requirement.get("evidence_link_confidence"),
                    "evidence_bundle": requirement.get("evidence_bundle"),
                    "mapping_diagnostics": requirement.get("mapping_diagnostics"),
                }
                for requirement in question.get("requirements", []) or []
                if isinstance(requirement, dict)
            ],
            "evidence_bundle": question.get("evidence_bundle"),
        })
    return {
        "schema_version": "1.0", "artifact_type": "student_assessment_report",
        "student": student, "student_code_executed": False,
        "assessment_status": "completed" if evaluation is not None else "evaluation_failed",
        "question_count": len((evidence_map or {}).get("questions", []) or []),
        "evidence_map_summary": (evidence_map or {}).get("summary"),
        "unassigned_evidence_blocks": list(
            (evidence_map or {}).get("unassigned_evidence_blocks", []) or []
        ),
        "question_components": question_components,
        "evaluation_summary": (evaluation or {}).get("summary"),
        "status_counts": {s: statuses.get(s, 0) for s in ("complete", "partial", "absent", "needs_review")},
        "scoring": scoring or {"score_status": "not_scored", "score": None, "final_grade": None,
                               "notes": ["No numeric score generated."]},
        "policy": {
            k: policy.get(k) for k in (
                "source_filename", "source_sha256", "strictness", "numeric_scoring_enabled",
                "credit_map", "component_scoring_enabled", "question_component_policy",
                "best_question_count", "warnings",
            )
        },
        "requirements": rows,
        "evidence_diagnostics": {
            "by_category": dict(Counter(
                str(row.get("evidence_diagnostic", {}).get("category") or "unknown")
                for row in rows
            )),
            "requirements": [
                {
                    "requirement_id": row.get("requirement_id"),
                    "question_number": row.get("question_number"),
                    "category": (row.get("evidence_diagnostic") or {}).get("category"),
                    "context_reasons": (row.get("evidence_diagnostic") or {}).get("context_reasons", []),
                    "omitted_bundle_block_ids": (row.get("evidence_diagnostic") or {}).get("omitted_bundle_block_ids", []),
                    "selected_block_ids": (row.get("evidence_diagnostic") or {}).get("selected_block_ids", []),
                    "selected_component_counts": (row.get("evidence_diagnostic") or {}).get("selected_component_counts", {}),
                }
                for row in rows
            ],
        },
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
        score_html = (f"<p><strong>Provisional points:</strong> <span id=\"score-total\">"
                      f"{html.escape(str(scoring.get('score')))} / {html.escape(str(scoring.get('max_marks')))}</span>"
                      f" (<span id=\"score-percentage\">{html.escape(str(scoring.get('percentage')))}</span>%)</p>")
    else:
        score_html = "<p><strong>Numeric marks:</strong> <span id=\"score-display\">Not calculated because the supplied policy or mark allocation did not support a defensible score.</span></p>"
    req_rows = []
    for index, row in enumerate(report.get("requirements", []) or []):
        qnum = row.get("question_number") or "—"
        req_text = row.get("requirement_text") or row.get("atomic_text") or row.get("source_text") or row.get("requirement_id")
        evid = "<br/>".join(html.escape(str(ev.get("quote") or "")) for ev in row.get("evidence", []) or [])
        final_status = row.get("final_status")
        if row.get("status") == "needs_review":
            options = "".join(
                f'<option value="{status}"{" selected" if final_status == status else ""}>{status.title()}</option>'
                for status in ("complete", "partial", "absent")
            )
            final_status_html = f'<select class="final-status" data-row-index="{index}">{options}</select>'
        else:
            final_status_html = html.escape(str(final_status or "Invalid status"))
        req_rows.append("<tr>" + "".join([
            f"<td>{html.escape(str(qnum))}</td>",
            f"<td>{html.escape(str(row.get('requirement_id') or ''))}</td>",
            f"<td>{html.escape(str(req_text or ''))}</td>",
            f"<td>{html.escape(str(row.get('status') or 'needs_review'))}</td>",
            f"<td>{final_status_html}</td>",
            f"<td>{html.escape(str(row.get('confidence') if row.get('confidence') is not None else ''))}</td>",
            f"<td>{evid}</td>",
            f"<td>{html.escape(str(row.get('rationale') or ''))}</td>",
        ]) + "</tr>")
    component_rows = []
    for question in report.get("question_components", []) or []:
        counts = question.get("component_block_counts", {}) or {}
        component_rows.append("<tr>" + "".join([
            f"<td>{html.escape(str(question.get('question_number') or '—'))}</td>",
            f"<td>{html.escape(str(question.get('question_title') or ''))}</td>",
            f"<td>{html.escape(str(question.get('mapping_status') or ''))}</td>",
            f"<td>{html.escape(', '.join(str(x) for x in question.get('present_components', []) or []))}</td>",
            f"<td>{html.escape(', '.join(str(x) for x in question.get('missing_components', []) or []) or 'None')}</td>",
            f"<td>{html.escape(str({key: counts.get(key, 0) for key in ('code', 'output', 'explanation', 'ai_log')}))}</td>",
        ]) + "</tr>")
    warnings = "".join(f"<li>{html.escape(str(x))}</li>" for x in report.get("warnings", []) or [])
    score_notes = "".join(f"<li>{html.escape(str(x))}</li>" for x in scoring.get("notes", []) or [])
    component_table = "".join(component_rows) or '<tr><td colspan="6">No component extraction data.</td></tr>'
    question_score_rows = []
    for question_score in scoring.get("per_question", []) or []:
        component_scores = {
            item.get("component"): f"{item.get('score')} / {item.get('max_marks')}"
            for item in question_score.get("component_scores", []) or []
        }
        question_score_rows.append("<tr>" + "".join([
            f"<td>{html.escape(str(question_score.get('question_number') or ''))}</td>",
            f"<td>{html.escape(str(question_score.get('score')))} / {html.escape(str(question_score.get('max_marks')))}</td>",
            f"<td>{html.escape(str(component_scores.get('code', '—')))}</td>",
            f"<td>{html.escape(str(component_scores.get('explanation', '—')))}</td>",
            f"<td>{html.escape(str(component_scores.get('output', '—')))}</td>",
            f"<td>{'Yes' if question_score.get('counted_in_final') else 'No'}</td>",
        ]) + "</tr>")
    question_score_table = "".join(question_score_rows) or '<tr><td colspan="6">No question scores calculated.</td></tr>'
    unassigned_rows = []
    for block in report.get("unassigned_evidence_blocks", []) or []:
        metadata = block.get("metadata") or {}
        page_number = block.get("page_number") or metadata.get("page_number")
        source_text = str(block.get("source_text") or "")
        output_text = str(block.get("output_text") or "")
        content = []
        if source_text:
            content.append(f"<pre>{html.escape(source_text)}</pre>")
        if output_text:
            content.append(
                f"<strong>Extracted output</strong><pre>{html.escape(output_text)}</pre>"
            )
        unassigned_rows.append("<tr>" + "".join([
            f"<td>{html.escape(str(page_number if page_number is not None else '—'))}</td>",
            f"<td>{html.escape(str(block.get('mapping_method') or 'unassigned'))}</td>",
            f"<td>{''.join(content) or 'No text content.'}</td>",
        ]) + "</tr>")
    unassigned_table_rows = "".join(unassigned_rows) or '<tr><td colspan="3">No unassigned evidence.</td></tr>'
    unassigned_footer = (
        "<footer><h2>Unassigned evidence</h2>"
        "<p>These blocks could not be tied to a question explicitly and were not used as question-specific evidence.</p>"
        "<table><thead><tr><th>Page</th><th>Reason</th><th>Extracted content</th></tr></thead>"
        f"<tbody>{unassigned_table_rows}</tbody></table></footer>"
    )
    report_json = json.dumps(report, ensure_ascii=False).replace("<", "\\u003c")
    return f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>Assessment report — {title}</title>
<style>body{{font-family:Arial,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem;color:#202124}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #bbb;padding:.5rem;text-align:left;vertical-align:top}}th{{background:#f2f2f2}}td{{font-size:.94rem}}.note{{background:#fff8df;padding:.8rem;border-left:4px solid #a67c00}}code{{overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;margin:.25rem 0}}</style></head><body>
<h1>Student assessment report</h1><p><strong>Student:</strong> {name}</p><p><strong>Submission:</strong> {title}</p>
<p><strong>Assessment status:</strong> {html.escape(str(report.get('assessment_status')))}</p>
<div class=\"note\"><strong>Static evaluation:</strong> student-submitted code was not executed. Model findings and marks are provisional and require teacher review. No official grade has been issued.</div>
<h2>Marks</h2>{score_html}<button id="download-review-json" type="button">Download reviewed JSON</button>
<p id="review-state" aria-live="polite">Unresolved model findings default to partial credit until reviewed.</p>
<ul>{score_notes}</ul>
<h2>Requirement summary</h2><table><thead><tr><th>Status</th><th>Count</th></tr></thead><tbody>{status_rows}</tbody></table>
<h2>Question scoring</h2><table id="question-scores"><thead><tr><th>Question</th><th>Question score</th><th>Code</th><th>Explanation</th><th>Output</th><th>Counted</th></tr></thead><tbody>{question_score_table}</tbody></table>
<h2>Answer component extraction</h2><table><thead><tr><th>Question</th><th>Title</th><th>Mapping</th><th>Found</th><th>Missing scored components</th><th>Block counts</th></tr></thead><tbody>{component_table}</tbody></table>
<h2>Requirement-level findings</h2><table id="requirement-findings"><thead><tr><th>Question</th><th>Requirement ID</th><th>Requirement</th><th>Raw model status</th><th>Final status</th><th>Confidence</th><th>Evidence quote(s)</th><th>Rationale</th></tr></thead><tbody>{''.join(req_rows)}</tbody></table>
<h2>Warnings</h2><ul>{warnings or '<li>None recorded.</li>'}</ul>
<p>Assessment policy scoring enabled: {html.escape(str((report.get('policy') or {}).get('numeric_scoring_enabled')))}. Official grade: not issued.</p>
{unassigned_footer}
<script id="report-data" type="application/json">{report_json}</script>
<script>
(() => {{
  const report = JSON.parse(document.getElementById("report-data").textContent);
  const credits = report.scoring.status_credit_factors || {{complete: 1, partial: 0.7, absent: 0}};
  const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[char]));
  const recalculate = () => {{
    const grouped = new Map();
    let pending = 0;
    report.requirements.forEach((row) => {{
      if (row.status === "needs_review" && row.final_status_source !== "teacher_selected") pending += 1;
      if (!row.question_number || !credits.hasOwnProperty(row.final_status)) return;
      const key = String(row.question_number);
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key).push(row);
    }});
    const allocations = (report.policy || {{}}).question_component_policy || {{}};
    const policyKeys = Object.keys(allocations);
    const allQuestionsAvailable = policyKeys.length > 0
      && policyKeys.every((key) => grouped.has(String(key)))
      && Array.from(grouped.keys()).every((key) => Object.prototype.hasOwnProperty.call(allocations, key));
    if (allQuestionsAvailable) {{
      const questions = policyKeys.map((questionNumber) => {{
        const allocation = allocations[questionNumber];
        const rows = grouped.get(String(questionNumber));
        const weights = allocation.component_weights || {{}};
        if (!Number.isFinite(Number(allocation.marks)) || Number(allocation.marks) <= 0
          || ["code", "explanation", "output"].some((component) => !Number.isFinite(Number(weights[component])))
          || Math.abs(["code", "explanation", "output"].reduce((sum, component) => sum + Number(weights[component]), 0) - 1) > 0.001) return null;
        const factor = rows.reduce((sum, row) => sum + credits[row.final_status], 0) / rows.length;
        const componentScores = ["code", "explanation", "output"].map((component) => {{
          const maxMarks = Number(allocation.marks) * Number(weights[component]);
          return {{component, max_marks: maxMarks, score: maxMarks * factor}};
        }});
        return {{
          question_number: String(questionNumber),
          requirement_count: rows.length,
          normalized_status_factor: factor,
          score: componentScores.reduce((sum, item) => sum + item.score, 0),
          max_marks: Number(allocation.marks),
          component_scores: componentScores,
          counted_in_final: false
        }};
      }});
      if (questions.some((question) => question === null)) return;
      const bestCount = Number(report.policy.best_question_count) || questions.length;
      const ranked = [...questions].sort((a, b) => b.score - a.score || Number(a.question_number) - Number(b.question_number));
      const counted = new Set(ranked.slice(0, bestCount).map((row) => row.question_number));
      questions.forEach((row) => {{ row.counted_in_final = counted.has(row.question_number); }});
      const selected = questions.filter((row) => row.counted_in_final);
      const score = selected.reduce((sum, row) => sum + row.score, 0);
      const maxMarks = selected.reduce((sum, row) => sum + row.max_marks, 0);
      report.scoring = {{
        ...report.scoring,
        numeric_scoring_enabled: maxMarks > 0,
        score_status: pending ? "provisional" : "ready_for_teacher_approval",
        score: Number(score.toFixed(3)),
        max_marks: Number(maxMarks.toFixed(3)),
        percentage: maxMarks ? Number((score / maxMarks * 100).toFixed(2)) : null,
        per_question: questions,
        defaulted_needs_review_count: pending,
        teacher_approval_required: true,
        final_grade: null
      }};
      const total = document.getElementById("score-total");
      if (total) total.textContent = `${{score.toFixed(3)}} / ${{maxMarks.toFixed(3)}}`;
      document.getElementById("score-display")?.replaceChildren(document.createTextNode(`${{score.toFixed(3)}} / ${{maxMarks.toFixed(3)}}`));
      const percentage = document.getElementById("score-percentage");
      if (percentage) percentage.textContent = maxMarks ? (score / maxMarks * 100).toFixed(2) : "0.00";
      document.querySelector("#question-scores tbody").innerHTML = questions.map((question) => {{
        const byComponent = Object.fromEntries(question.component_scores.map((item) => [item.component, item]));
        const cell = (component) => `${{byComponent[component].score.toFixed(3)}} / ${{byComponent[component].max_marks.toFixed(3)}}`;
        return `<tr><td>${{escapeHtml(question.question_number)}}</td><td>${{question.score.toFixed(3)}} / ${{question.max_marks.toFixed(3)}}</td><td>${{cell("code")}}</td><td>${{cell("explanation")}}</td><td>${{cell("output")}}</td><td>${{question.counted_in_final ? "Yes" : "No"}}</td></tr>`;
      }}).join("");
    }}
    const defaults = report.requirements.filter((row) => row.status === "needs_review" && row.final_status_source !== "teacher_selected").length;
    document.getElementById("review-state").textContent = defaults
      ? `${{defaults}} needs-review item(s) still use provisional partial credit.`
      : "All needs-review items have a teacher-selected final status.";
  }};
  document.querySelectorAll(".final-status").forEach((select) => {{
    select.addEventListener("change", (event) => {{
      const row = report.requirements[Number(event.target.dataset.rowIndex)];
      row.final_status = event.target.value;
      row.final_status_source = "teacher_selected";
      recalculate();
    }});
  }});
  document.getElementById("download-review-json").addEventListener("click", () => {{
    const blob = new Blob([JSON.stringify(report, null, 2) + "\\n"], {{type: "application/json"}});
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${{(report.student || {{}}).report_id || "student"}}.reviewed.report.json`;
    link.click();
    URL.revokeObjectURL(url);
  }});
  recalculate();
}})();
</script>
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
