from __future__ import annotations

from typing import Any

from spotter.contracts import RepAnalysis
from spotter.exercises.base import ExerciseStrategy
from spotter.exercises.shoulder_press.analyzer import ShoulderPressAnalyzer
from spotter.exercises.shared.issue_marker import IssueRule
from spotter.exercises.shoulder_press.issue_markers import RULES as ISSUE_RULES
from spotter.exercises.shared.rep_counter import combine, mean_optional, normalized_samples
from spotter.steps.rep_signals import SignalSample, angle_deg, average_axis, body_line_score


class ShoulderPressExercise(ShoulderPressAnalyzer, ExerciseStrategy):
    exercise = "shoulder_press"

    def issue_rules(self) -> tuple[IssueRule, ...]:
        return ISSUE_RULES

    def build_signal(self) -> tuple[list[SignalSample], dict[str, Any]]:
        sequence = self.pose_sequence
        wrist_y = [average_axis(frame, ("left_wrist", "right_wrist"), "y") for frame in sequence.frames]
        elbow_bend = [
            mean_optional(
                [
                    None if angle is None else max(0.0, 180.0 - angle)
                    for angle in (
                        angle_deg(frame, "left_shoulder", "left_elbow", "left_wrist"),
                        angle_deg(frame, "right_shoulder", "right_elbow", "right_wrist"),
                    )
                ]
            )
            for frame in sequence.frames
        ]
        body_line = [body_line_score(frame) for frame in sequence.frames]
        inverted_wrist = [None if value is None else -value for value in wrist_y]
        inverted_elbow_bend = [None if value is None else -value for value in elbow_bend]
        raw_signal = combine(inverted_wrist, inverted_elbow_bend, weight=0.2)
        samples, signal_range = normalized_samples(sequence, raw_signal)
        return samples, {
            "selected_signal": "negative_wrist_y_plus_elbow_extension_proxy",
            "raw_signal_range": signal_range,
            "usable_signal_samples": len(samples),
            "body_line_mean": round(mean_optional(body_line) or 0.0, 4),
        }

    def detect_variation(self, analysis: RepAnalysis) -> tuple[str, float, list[str]]:
        lockout_quality = self.metric(analysis, "avg_lockout_quality")
        wrist_travel = self.metric(analysis, "avg_wrist_travel")
        wrist_asymmetry = self.metric(analysis, "avg_wrist_height_asymmetry")

        if wrist_asymmetry is not None and wrist_asymmetry >= 0.12:
            return (
                "asymmetric_press",
                self.confidence(0.76, analysis, wrist_asymmetry),
                ["intentional_asymmetry_check"],
            )
        if (lockout_quality is not None and lockout_quality <= 0.65) or (
            wrist_travel is not None and wrist_travel < 0.24
        ):
            support = 1.0 - lockout_quality if lockout_quality is not None else wrist_travel
            return "partial_press", self.confidence(0.72, analysis, support), ["partial_range_of_motion"]

        not_issues = ["lockout_unverified"] if lockout_quality is None else []
        return "standing_shoulder_press", self.confidence(0.62, analysis, lockout_quality), not_issues

    def profile_not_issues(self, variation: str) -> list[str]:
        mapping = {
            "partial_press": ["partial_range_of_motion"],
            "asymmetric_press": ["intentional_asymmetry_check"],
        }
        return list(mapping.get(variation, []))
