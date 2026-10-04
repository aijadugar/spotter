from __future__ import annotations

from typing import Any

from spotter.contracts import RepAnalysis, Reps
from spotter.exercise_catalog import get_exercise_spec
from spotter.exercises.base import ExerciseStrategy
from spotter.exercises.unknown.analyzer import UnknownAnalyzer
from spotter.steps.rep_signals import SignalSample


class UnknownExercise(UnknownAnalyzer, ExerciseStrategy):
    exercise = "unknown"

    def _empty_debug(self) -> dict[str, Any]:
        return {
            "selected_signal": "none",
            "thresholds": {},
            "extrema": [],
            "accepted_reps": [],
            "body_line_mean": 0.0,
            "raw_signal_range": 0.0,
            "usable_signal_samples": 0,
        }

    def build_signal(self) -> tuple[list[SignalSample], dict[str, Any]]:
        return [], self._empty_debug()

    def count(self) -> tuple[Reps, dict[str, Any]]:
        reps = Reps(exercise=self.exercise, reps=[], partial_reps=[{"reason": "unknown_exercise"}])
        return reps, self._empty_debug()

    def detect_variation(self, analysis: RepAnalysis) -> tuple[str, float, list[str]]:
        exercise_spec = get_exercise_spec(analysis.exercise)
        return (
            exercise_spec.default_variation,
            exercise_spec.default_variation_confidence,
            list(exercise_spec.default_variation_not_issues),
        )
