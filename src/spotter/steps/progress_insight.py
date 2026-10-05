from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from spotter.contracts import SessionRecord


@dataclass
class SessionComparison:
    """Deterministic comparison between current and previous session."""
    session_id: str
    exercise: str
    rep_delta: int  # current - previous
    issues_fixed: list[str]  # in previous but not current
    issues_new: list[str]    # in current but not previous
    issues_persistent: list[str]  # in both
    trend: str  # "improved", "regressed", "stable", "first_session"
    trend_reason: str


def compare_sessions(
    current: SessionRecord,
    previous: SessionRecord | None,
) -> SessionComparison:
    """Compare current session to previous session deterministically."""
    if previous is None:
        return SessionComparison(
            session_id=current.session_id,
            exercise=current.exercise,
            rep_delta=0,
            issues_fixed=[],
            issues_new=list(current.issue_counts.keys()),
            issues_persistent=[],
            trend="first_session",
            trend_reason="No previous session to compare",
        )

    # Rep delta
    rep_delta = current.rep_count - previous.rep_count

    # Issue comparison
    prev_issues = set(previous.issue_counts.keys())
    curr_issues = set(current.issue_counts.keys())

    issues_fixed = sorted(prev_issues - curr_issues)
    issues_new = sorted(curr_issues - prev_issues)
    issues_persistent = sorted(prev_issues & curr_issues)

    # Determine trend
    if current.form_score > previous.form_score + 0.05:
        trend = "improved"
        reason = f"Form score improved ({previous.form_score:.2f} → {current.form_score:.2f})"
    elif current.form_score < previous.form_score - 0.05:
        trend = "regressed"
        reason = f"Form score regressed ({previous.form_score:.2f} → {current.form_score:.2f})"
    elif issues_fixed and not issues_new:
        trend = "improved"
        reason = f"Fixed issues: {', '.join(issues_fixed)}"
    elif issues_new and not issues_fixed:
        trend = "regressed"
        reason = f"New issues: {', '.join(issues_new)}"
    else:
        trend = "stable"
        reason = f"Form score stable ({previous.form_score:.2f} → {current.form_score:.2f})"

    return SessionComparison(
        session_id=current.session_id,
        exercise=current.exercise,
        rep_delta=rep_delta,
        issues_fixed=issues_fixed,
        issues_new=issues_new,
        issues_persistent=issues_persistent,
        trend=trend,
        trend_reason=reason,
    )


def compute_trend_over_sessions(records: list[SessionRecord], n: int = 5) -> dict[str, Any]:
    """Compute trend over last N sessions."""
    if len(records) < 2:
        return {"trend": "insufficient_data", "sessions_analyzed": len(records)}

    recent = records[:n]
    form_scores = [r.form_score for r in recent]
    rep_counts = [r.rep_count for r in recent]

    # Overall form trend
    first_score = form_scores[-1]
    last_score = form_scores[0]
    score_delta = last_score - first_score

    if score_delta > 0.05:
        form_trend = "improving"
    elif score_delta < -0.05:
        form_trend = "declining"
    else:
        form_trend = "stable"

    # Rep count trend
    first_reps = rep_counts[-1]
    last_reps = rep_counts[0]
    rep_delta = last_reps - first_reps

    if rep_delta > 1:
        rep_trend = "increasing"
    elif rep_delta < -1:
        rep_trend = "decreasing"
    else:
        rep_trend = "stable"

    # Issue frequency trend
    all_issues: Counter[str] = Counter()
    for r in recent:
        all_issues.update(r.issue_counts.keys())

    # Issues that appear in >50% of sessions
    persistent_issues = [
        issue for issue, count in all_issues.items()
        if count > len(recent) / 2
    ]

    return {
        "form_trend": form_trend,
        "form_delta": score_delta,
        "rep_trend": rep_trend,
        "rep_delta": rep_delta,
        "persistent_issues": persistent_issues,
        "sessions_analyzed": len(recent),
        "latest_form": last_score,
        "latest_reps": last_reps,
    }


def generate_trend_svg(trend_data: dict[str, Any], width: int = 120, height: int = 40) -> str:
    """Generate tiny SVG trend sparkline for form scores."""
    # This is a placeholder - actual implementation would render recent form scores
    # For now, return a simple indicator
    trend = trend_data.get("form_trend", "stable")

    color = {
        "improving": "#22c55e",  # green
        "declining": "#ef4444",  # red
        "stable": "#f59e0b",     # amber
        "insufficient_data": "#6b7280",  # gray
    }.get(trend, "#6b7280")

    direction = {
        "improving": "▲",
        "declining": "▼",
        "stable": "●",
        "insufficient_data": "—",
    }.get(trend, "—")

    return f"""
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
    <rect width="{width}" height="{height}" fill="transparent"/>
    <text x="{width//2}" y="{height//2 + 5}" text-anchor="middle"
          font-family="system-ui, sans-serif" font-size="14" fill="{color}">
        {direction}
    </text>
    <text x="{width//2}" y="{height//2 + 20}" text-anchor="middle"
          font-family="system-ui, sans-serif" font-size="9" fill="#6b7280">
        {trend}
    </text>
</svg>
""".strip()


def format_comparison_for_plan(comp: SessionComparison) -> str:
    """Format comparison as text for progress plan context."""
    parts = []
    if comp.rep_delta > 0:
        parts.append(f"+{comp.rep_delta} reps vs last session")
    elif comp.rep_delta < 0:
        parts.append(f"{comp.rep_delta} reps vs last session")

    if comp.issues_fixed:
        parts.append(f"Fixed: {', '.join(comp.issues_fixed)}")
    if comp.issues_new:
        parts.append(f"New: {', '.join(comp.issues_new)}")
    if comp.issues_persistent:
        parts.append(f"Persistent: {', '.join(comp.issues_persistent)}")

    parts.append(comp.trend_reason)
    return "; ".join(parts)