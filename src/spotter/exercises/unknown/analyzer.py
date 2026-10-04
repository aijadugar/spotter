from __future__ import annotations

from spotter.contracts import PoseFrame
from spotter.exercises.shared.analyzer import (
    ExerciseMetricResult,
    mean_visibility,
    range_optional,
    score,
    value_series,
)
from spotter.steps.rep_signals import average_axis


class UnknownAnalyzer:
    def metrics(self, frames: list[PoseFrame]) -> ExerciseMetricResult:
        movement_signal = value_series(
            frames,
            lambda frame: average_axis(frame, ("left_hip", "right_hip"), "y"),
        )
        movement = range_optional(movement_signal) or 0.0
        confidence = mean_visibility(frames)
        metrics = {
            "movement_consistency_score": score(1.0 - movement * 4.0),
            "landmark_confidence": confidence,
        }
        return metrics, 0.0, metrics["movement_consistency_score"], confidence, []
