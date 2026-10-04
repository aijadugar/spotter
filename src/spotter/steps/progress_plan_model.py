from __future__ import annotations

import json
import re

from spotter.contracts import ProgressPlan, SessionRecord, Verification
from spotter.knowledge_cards import known_issue_labels
from spotter.steps.progress_plan import (
    SessionFindings,
    build_fallback_plan,
)

FORBIDDEN_PLAN_PATTERNS = (
    "diagnos",
    "injury",
    "tear",
    "impingement",
    "pathology",
    "fall risk",
    "risk of falling",
    "medical assessment",
)


def build_progress_plan_prompt(
    history: list[SessionRecord],
    findings: SessionFindings,
) -> str:
    history_lines = [
        (
            f"- {record.timestamp}: {record.rep_count} reps, "
            f"form {record.form_score:.2f}, issues {record.issue_counts}"
        )
        for record in history
    ] or ["- no previous sessions"]
    return (
        "You are a strength coach. Return ONLY a JSON object with keys "
        "focus, targets, next_session_cues, encouragement, confidence_notes.\n"
        f"Exercise: {findings.exercise}\n"
        f"Reps this session: {findings.rep_count}\n"
        f"Aggregate metrics: {json.dumps(findings.aggregate_metrics)}\n"
        f"Detected issues: {json.dumps(findings.issue_labels)}\n"
        "Previous sessions:\n"
        + "\n".join(history_lines)
        + "\nOnly mention issues from the detected issues list."
    )


def extract_plan_payload(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("Progress plan output must be a JSON object.")
    return payload


def _plan_from_payload(payload: dict) -> ProgressPlan:
    def as_list(value) -> list[str]:
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, list):
            return [str(item) for item in value]
        return []

    return ProgressPlan(
        focus=str(payload.get("focus", "")).strip() or "Keep building consistency.",
        targets=as_list(payload.get("targets")),
        next_session_cues=as_list(payload.get("next_session_cues")),
        encouragement=str(payload.get("encouragement", "")).strip() or "Keep going.",
        confidence_notes=as_list(payload.get("confidence_notes")),
    )


class ModelProgressPlanner:
    def __init__(self, model=None) -> None:
        if model is None:
            from spotter.slm.providers import get_coach_summary_model

            model = get_coach_summary_model()
        self.model = model

    def plan(self, history: list[SessionRecord], findings: SessionFindings) -> ProgressPlan:
        allowed_issues = allowed_issues_for(history, findings)
        try:
            generation = self.model.generate_summary(
                build_progress_plan_prompt(history, findings)
            )
            plan = _plan_from_payload(extract_plan_payload(generation.text))
        except Exception as exc:
            return build_fallback_plan(history, findings, failure_reason=f"model_failed:{exc}")
        verification = verify_plan(plan, allowed_issues=allowed_issues)
        if not verification.passed:
            return build_fallback_plan(
                history,
                findings,
                failure_reason="; ".join(verification.notes),
            )
        return plan


def allowed_issues_for(
    history: list[SessionRecord], findings: SessionFindings
) -> set[str]:
    allowed = set(findings.issue_labels)
    for record in history:
        allowed.update(record.issue_counts.keys())
    return allowed


def verify_plan(plan: ProgressPlan, allowed_issues: set[str]) -> Verification:
    text = " ".join(
        [
            plan.focus,
            *plan.targets,
            *plan.next_session_cues,
            plan.encouragement,
            *plan.confidence_notes,
        ]
    )
    lowered = text.lower()
    known = known_issue_labels()
    mentioned = {label for label in known if label in lowered}
    no_issue_outside_json = mentioned <= allowed_issues
    no_medical = all(pattern not in lowered for pattern in FORBIDDEN_PLAN_PATTERNS)
    checks = {
        "no_issue_outside_json": no_issue_outside_json,
        "no_medical_language": no_medical,
    }
    notes = []
    if not no_issue_outside_json:
        notes.append(f"Plan mentioned undetected issue labels: {sorted(mentioned - allowed_issues)}.")
    if not no_medical:
        notes.append("Plan used medical or injury language.")
    return Verification(passed=all(checks.values()), checks=checks, notes=notes)
