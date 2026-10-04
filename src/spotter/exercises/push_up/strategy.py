from __future__ import annotations

from typing import Any

from spotter.contracts import RepAnalysis
from spotter.exercises.base import ExerciseStrategy
from spotter.exercises.push_up.analyzer import PushUpAnalyzer
from spotter.exercises.shared.issue_marker import IssueRule
from spotter.exercises.push_up.issue_markers import RULES as ISSUE_RULES
from spotter.exercises.shared.rep_counter import combine, mean_optional, normalized_samples
from spotter.steps.rep_signals import SignalSample, angle_deg, average_axis, body_line_score


class PushUpExercise(PushUpAnalyzer, ExerciseStrategy):
    exercise = "push_up"

    def issue_rules(self) -> tuple[IssueRule, ...]:
        return ISSUE_RULES

    def build_signal(self) -> tuple[list[SignalSample], dict[str, Any]]:
        sequence = self.pose_sequence
        hip_y = [average_axis(frame, ("left_hip", "right_hip"), "y") for frame in sequence.frames]
        shoulder_y = [average_axis(frame, ("left_shoulder", "right_shoulder"), "y") for frame in sequence.frames]
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
        chest_proxy = [
            mean_optional([shoulder_value, hip_value])
            for shoulder_value, hip_value in zip(shoulder_y, hip_y, strict=False)
        ]
        samples, signal_range = normalized_samples(sequence, combine(chest_proxy, elbow_bend, weight=0.25))
        return samples, {
            "selected_signal": "chest_y_plus_elbow_bend",
            "raw_signal_range": signal_range,
            "usable_signal_samples": len(samples),
            "body_line_mean": round(mean_optional(body_line) or 0.0, 4),
        }

    def detect_variation(self, analysis: RepAnalysis) -> tuple[str, float, list[str]]:
        hand_width = self.metric(analysis, "avg_hand_width_ratio")
        knee_support = self.metric(analysis, "avg_knee_support_score")
        not_issues: list[str] = []

        if knee_support is not None and knee_support >= 0.8:
            return "knee_push_up", self.confidence(0.74, analysis, knee_support), ["knee_contact"]
        if hand_width is not None and hand_width >= 1.45:
            return "wide_grip_push_up", self.confidence(0.72, analysis, hand_width), ["wide_hand_placement"]
        if hand_width is not None and hand_width <= 0.95:
            return (
                "close_grip_push_up",
                self.confidence(0.72, analysis, 1.0 - hand_width),
                ["close_hand_placement"],
            )

        if hand_width is None:
            not_issues.append("hand_width_unverified")
        return "standard_push_up", self.confidence(0.62, analysis, hand_width), not_issues

    def profile_not_issues(self, variation: str) -> list[str]:
        mapping = {
            "wide_grip_push_up": ["wide_hand_placement"],
            "close_grip_push_up": ["close_hand_placement"],
            "knee_push_up": ["knee_contact"],
        }
        return list(mapping.get(variation, []))
