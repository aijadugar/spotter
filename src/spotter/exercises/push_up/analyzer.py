from __future__ import annotations

from spotter.contracts import PoseFrame
from spotter.exercises.shared.analyzer import (
    ExerciseMetricResult,
    max_optional,
    mean_optional,
    mean_pair,
    min_optional,
    range_optional,
    round_optional,
    safe_ratio,
    score,
    side_delta,
    std_optional,
    value_series,
    width,
)
from spotter.steps.rep_signals import average_axis, body_line_score


class PushUpAnalyzer:
    def metrics(self, frames: list[PoseFrame]) -> ExerciseMetricResult:
        elbow_angles = value_series(
            frames,
            lambda frame: mean_pair(
                frame,
                ("left_shoulder", "left_elbow", "left_wrist"),
                ("right_shoulder", "right_elbow", "right_wrist"),
            ),
        )
        elbow_deltas = value_series(
            frames,
            lambda frame: side_delta(
                frame,
                ("left_shoulder", "left_elbow", "left_wrist"),
                ("right_shoulder", "right_elbow", "right_wrist"),
            ),
        )
        body_line = value_series(frames, body_line_score)
        shoulder_y = value_series(
            frames,
            lambda frame: average_axis(frame, ("left_shoulder", "right_shoulder"), "y"),
        )
        hip_y = value_series(frames, lambda frame: average_axis(frame, ("left_hip", "right_hip"), "y"))
        ankle_y = value_series(
            frames,
            lambda frame: average_axis(frame, ("left_ankle", "right_ankle"), "y"),
        )
        hand_width = value_series(frames, lambda frame: width(frame, "left_wrist", "right_wrist"))
        shoulder_width = value_series(
            frames,
            lambda frame: width(frame, "left_shoulder", "right_shoulder"),
        )
        elbow_width = value_series(frames, lambda frame: width(frame, "left_elbow", "right_elbow"))
        knee_y = value_series(
            frames,
            lambda frame: average_axis(frame, ("left_knee", "right_knee"), "y"),
        )

        min_elbow = min_optional(elbow_angles)
        max_elbow = max_optional(elbow_angles)
        elbow_rom = 0.0 if min_elbow is None or max_elbow is None else max_elbow - min_elbow
        chest_depth = range_optional(shoulder_y) or 0.0
        hand_width_ratio = safe_ratio(mean_optional(hand_width), mean_optional(shoulder_width))
        elbow_flare = safe_ratio(mean_optional(elbow_width), mean_optional(shoulder_width))
        body_line_mean = mean_optional(body_line)
        hip_sag_score = None
        if body_line_mean is not None:
            hip_sag_score = max(0.0, 1.0 - body_line_mean)
        knee_support_score = self._knee_support_score(hip_y, knee_y, ankle_y)

        rom_score = score((elbow_rom / 80.0) * 0.65 + (chest_depth / 0.16) * 0.35)
        hip_stability = 1.0 - min(1.0, (std_optional(hip_y) or 0.0) * 4.0)
        stability_score = score(((body_line_mean or 0.5) * 0.75) + hip_stability * 0.25)
        symmetry_score = score(1.0 - ((mean_optional(elbow_deltas) or 0.0) / 45.0))

        metrics = {
            "min_elbow_angle_deg": round_optional(min_elbow),
            "max_elbow_angle_deg": round_optional(max_elbow),
            "body_line_score": round_optional(body_line_mean),
            "hip_sag_score": round_optional(hip_sag_score),
            "hip_pike_score": round_optional(max(0.0, ((body_line_mean or 1.0) - 1.0) * -1.0), 4),
            "chest_depth_proxy": round_optional(chest_depth, 4),
            "hand_width_ratio": round_optional(hand_width_ratio, 3),
            "elbow_flare_ratio": round_optional(elbow_flare, 3),
            "lockout_quality": score(((max_elbow or 120.0) - 120.0) / 55.0),
            "knee_support_score": knee_support_score,
        }
        hints = []
        if hand_width_ratio is not None and hand_width_ratio > 1.45:
            hints.append("wide_grip_push_up")
        elif hand_width_ratio is not None and hand_width_ratio < 0.95:
            hints.append("close_grip_push_up")
        if knee_support_score >= 0.8:
            hints.append("knee_push_up")
        return metrics, rom_score, stability_score, symmetry_score, hints

    def _knee_support_score(
        self,
        hip_y: list[float | None],
        knee_y: list[float | None],
        ankle_y: list[float | None],
    ) -> float:
        scores: list[float] = []
        for hip, knee, ankle in zip(hip_y, knee_y, ankle_y, strict=False):
            if hip is None or knee is None or ankle is None:
                continue

            hip_to_ankle = abs(ankle - hip)
            if hip_to_ankle <= 1e-6:
                continue
            knee_to_hip = abs(knee - hip)
            bent_leg_score = score(1.0 - knee_to_hip / max(hip_to_ankle * 0.55, 0.01))
            scores.append(bent_leg_score)

        return score(mean_optional(scores) or 0.0)
