from __future__ import annotations

from typing import Any

from spotter.contracts import RepAnalysis
from spotter.exercises.base import ExerciseStrategy
from spotter.exercises.squat.analyzer import SquatAnalyzer
from spotter.exercises.shared.issue_marker import IssueRule
from spotter.exercises.squat.issue_markers import RULES as ISSUE_RULES
from spotter.exercises.shared.rep_counter import combine, mean_optional, normalized_samples
from spotter.steps.rep_signals import SignalSample, angle_deg, average_axis, body_line_score


class SquatExercise(SquatAnalyzer, ExerciseStrategy):
    exercise = "squat"

    def issue_rules(self) -> tuple[IssueRule, ...]:
        return ISSUE_RULES

    def build_signal(self) -> tuple[list[SignalSample], dict[str, Any]]:
        sequence = self.pose_sequence
        hip_y = [average_axis(frame, ("left_hip", "right_hip"), "y") for frame in sequence.frames]
        knee_bend = [
            mean_optional(
                [
                    None if angle is None else max(0.0, 180.0 - angle)
                    for angle in (
                        angle_deg(frame, "left_hip", "left_knee", "left_ankle"),
                        angle_deg(frame, "right_hip", "right_knee", "right_ankle"),
                    )
                ]
            )
            for frame in sequence.frames
        ]
        body_line = [body_line_score(frame) for frame in sequence.frames]
        samples, signal_range = normalized_samples(sequence, combine(hip_y, knee_bend, weight=0.35))
        return samples, {
            "selected_signal": "hip_y_plus_knee_bend",
            "raw_signal_range": signal_range,
            "usable_signal_samples": len(samples),
            "body_line_mean": round(mean_optional(body_line) or 0.0, 4),
        }

    def detect_variation(self, analysis: RepAnalysis) -> tuple[str, float, list[str]]:
        stance_width = self.metric(analysis, "avg_stance_width_ratio")
        bottom_pause = self.metric(analysis, "avg_bottom_pause_sec")

        if bottom_pause is not None and bottom_pause >= 0.4:
            return "pause_squat", self.confidence(0.76, analysis, bottom_pause), ["bottom_pause"]
        if stance_width is not None and stance_width >= 1.35:
            return "wide_squat_stance", self.confidence(0.72, analysis, stance_width), ["wide_stance"]
        if stance_width is not None and stance_width <= 0.85:
            return (
                "narrow_squat_stance",
                self.confidence(0.72, analysis, 1.0 - stance_width),
                ["narrow_stance"],
            )

        not_issues = ["stance_width_unverified"] if stance_width is None else []
        return "normal_squat_stance", self.confidence(0.62, analysis, stance_width), not_issues

    def profile_not_issues(self, variation: str) -> list[str]:
        mapping = {
            "wide_squat_stance": ["wide_stance"],
            "narrow_squat_stance": ["narrow_stance"],
            "pause_squat": ["bottom_pause"],
        }
        return list(mapping.get(variation, []))
