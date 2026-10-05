from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from typing import Any, Protocol

from spotter.contracts import ProgressPlan, SessionRecord
from spotter.steps.progress_insight import compare_sessions, compute_trend_over_sessions

PROGRESS_PLAN_PROVIDER_ENV = "SPOTTER_PROGRESS_PLAN_PROVIDER"


@dataclass(frozen=True)
class SessionFindings:
    exercise: str
    rep_count: int
    aggregate_metrics: dict[str, Any]
    issue_labels: list[str]


class ProgressPlanner(Protocol):
    def plan(self, history: list[SessionRecord], findings: SessionFindings) -> ProgressPlan:
        ...


def _metric(findings: SessionFindings, key: str, default: float = 0.0) -> float:
    value = findings.aggregate_metrics.get(key, default)
    return float(value) if isinstance(value, int | float) else default


def _trend(history: list[SessionRecord], key: str) -> float | None:
    if not history:
        return None
    first = history[0].aggregate_metrics.get(key)
    last = history[-1].aggregate_metrics.get(key)
    if not isinstance(first, int | float) or not isinstance(last, int | float):
        return None
    return float(last) - float(first)


def _top_issue(history: list[SessionRecord], findings: SessionFindings) -> str | None:
    counter: Counter[str] = Counter(findings.issue_labels)
    for record in history:
        counter.update(record.issue_counts.keys())
    if not counter:
        return None
    return counter.most_common(1)[0][0]


class FallbackProgressPlanner:
    def plan(self, history: list[SessionRecord], findings: SessionFindings) -> ProgressPlan:
        return build_fallback_plan(history, findings)


def build_fallback_plan(
    history: list[SessionRecord],
    findings: SessionFindings,
    failure_reason: str | None = None,
) -> ProgressPlan:
    rom = _metric(findings, "avg_rom_score", 0.7)
    stability = _metric(findings, "avg_stability_score", 0.7)
    symmetry = _metric(findings, "avg_symmetry_score", 0.7)

    top_issue = _top_issue(history, findings)
    if top_issue:
        focus = f"Keep working on `{top_issue}` this session."
    else:
        focus = "Keep the same controlled tempo and build consistency."

    targets: list[str] = []
    if rom < 0.8:
        targets.append("Aim for a little more range of motion on every rep.")
    if stability < 0.8:
        targets.append("Keep the core braced so the movement stays steady.")
    if symmetry < 0.8:
        targets.append("Check left and right sides stay even.")
    if not targets:
        targets.append("Repeat the set and hold this quality.")

    cues = ["Move at a slow, controlled tempo."]
    if top_issue:
        cues.append(f"Watch for `{top_issue}`, especially in the last reps.")

    # Add deterministic insight from last session
    if history:
        latest_record = SessionRecord(
            session_id=findings.exercise,  # dummy
            timestamp="",
            profile_key="",
            exercise=findings.exercise,
            rep_count=findings.rep_count,
            aggregate_metrics=findings.aggregate_metrics,
            issue_counts={label: 1 for label in findings.issue_labels},
            form_score=sum(findings.aggregate_metrics.values()) / len(findings.aggregate_metrics),
            source_run_id="",
        )
        prev_record = history[0]
        comp = compare_sessions(latest_record, prev_record)
        cues.append(format_comparison_for_plan(comp))

    # Add trend over last N sessions
    trend_data = compute_trend_over_sessions(history, n=5)
    if trend_data.get("form_trend") == "improving":
        encouragement = "Your form is trending up across sessions. Keep it going."
    elif trend_data.get("form_trend") == "declining":
        encouragement = "Form has dipped recently — focus on quality over quantity."
    elif history:
        encouragement = "Progress is not always linear. Consistency is what counts."
    else:
        encouragement = "Good first session. The next one builds on it."

    notes = [f"Based on {len(history)} previous session(s)."]
    if failure_reason:
        notes.append(f"Deterministic plan used because: {failure_reason}.")
    if trend_data.get("persistent_issues"):
        notes.append(f"Persistent issues: {', '.join(trend_data['persistent_issues'])}")
    return ProgressPlan(
        focus=focus,
        targets=targets,
        next_session_cues=cues,
        encouragement=encouragement,
        confidence_notes=notes,
    )


def get_progress_planner() -> ProgressPlanner:
    provider = os.getenv(PROGRESS_PLAN_PROVIDER_ENV, "fallback").strip().lower()
    if provider in {"fallback", "deterministic", ""}:
        return FallbackProgressPlanner()
    from spotter.steps.progress_plan_model import ModelProgressPlanner

    return ModelProgressPlanner()
