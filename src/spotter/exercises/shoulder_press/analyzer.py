from __future__ import annotations

from statistics import mean

from spotter.contracts import PoseFrame
from spotter.exercises.shared.analyzer import (
    ExerciseMetricResult,
    max_optional,
    mean_optional,
    mean_pair,
    min_optional,
    range_optional,
    round_optional,
    score,
    side_delta,
    value_series,
)
from spotter.steps.rep_signals import average_axis, landmark_axis


class ShoulderPressAnalyzer:
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
        wrist_y = value_series(
            frames,
            lambda frame: average_axis(frame, ("left_wrist", "right_wrist"), "y"),
        )
        wrist_x = value_series(
            frames,
            lambda frame: average_axis(frame, ("left_wrist", "right_wrist"), "x"),
        )
        left_wrist_y = value_series(frames, lambda frame: landmark_axis(frame, "left_wrist", "y"))
        right_wrist_y = value_series(frames, lambda frame: landmark_axis(frame, "right_wrist", "y"))
        wrist_asymmetry = [
            abs(left - right)
            for left, right in zip(left_wrist_y, right_wrist_y, strict=False)
            if left is not None and right is not None
        ]
        shoulder_y = value_series(
            frames,
            lambda frame: average_axis(frame, ("left_shoulder", "right_shoulder"), "y"),
        )
        hip_y = value_series(frames, lambda frame: average_axis(frame, ("left_hip", "right_hip"), "y"))

        min_elbow = min_optional(elbow_angles)
        max_elbow = max_optional(elbow_angles)
        wrist_travel = range_optional(wrist_y) or 0.0
        wrist_lateral_drift = range_optional(wrist_x) or 0.0
        lockout_quality = score(((max_elbow or 120.0) - 120.0) / 55.0)
        verticality = score(1.0 - wrist_lateral_drift / max(0.01, wrist_travel))
        asymmetry = mean(wrist_asymmetry) if wrist_asymmetry else 0.0
        back_arch_proxy = abs((range_optional(hip_y) or 0.0) - (range_optional(shoulder_y) or 0.0))

        rom_score = score((wrist_travel / 0.28) * 0.55 + lockout_quality * 0.45)
        stability_score = score(verticality * 0.65 + (1.0 - min(1.0, back_arch_proxy * 4.0)) * 0.35)
        symmetry_score = score(1.0 - max(asymmetry * 5.0, (mean_optional(elbow_deltas) or 0.0) / 45.0))

        metrics = {
            "min_elbow_angle_deg": round_optional(min_elbow),
            "max_elbow_angle_deg": round_optional(max_elbow),
            "wrist_path_verticality": verticality,
            "lockout_quality": lockout_quality,
            "wrist_height_asymmetry": round_optional(asymmetry, 4),
            "left_right_wrist_delta": round_optional(asymmetry, 4),
            "back_arch_proxy": round_optional(back_arch_proxy, 4),
            "overhead_stability_score": stability_score,
            "wrist_travel": round_optional(wrist_travel, 4),
        }
        hints = []
        if lockout_quality < 0.65:
            hints.append("partial_press")
        if asymmetry > 0.12:
            hints.append("asymmetric_press")
        return metrics, rom_score, stability_score, symmetry_score, hints
