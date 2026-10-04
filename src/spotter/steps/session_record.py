from __future__ import annotations

from collections import Counter
from typing import Any

from spotter.contracts import (
    ExerciseClassification,
    IssueMarkers,
    RepAnalysis,
    Reps,
    SessionRecord,
)

METRIC_KEYS = ("avg_rom_score", "avg_stability_score", "avg_symmetry_score")


def form_score_from_metrics(metrics: dict[str, Any]) -> float:
    values = [
        float(metrics[key])
        for key in METRIC_KEYS
        if isinstance(metrics.get(key), int | float)
    ]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def build_session_record(
    *,
    run_id: str,
    profile_key: str,
    timestamp: str,
    classification: ExerciseClassification,
    reps: Reps,
    analysis: RepAnalysis,
    issues: IssueMarkers,
) -> SessionRecord:
    issue_counts = Counter(issue.issue for issue in issues.issues)
    return SessionRecord(
        session_id=run_id,
        timestamp=timestamp,
        profile_key=profile_key,
        exercise=classification.exercise,
        rep_count=len(reps.reps),
        aggregate_metrics=dict(analysis.aggregate_metrics),
        issue_counts=dict(issue_counts),
        form_score=form_score_from_metrics(analysis.aggregate_metrics),
        source_run_id=run_id,
    )
