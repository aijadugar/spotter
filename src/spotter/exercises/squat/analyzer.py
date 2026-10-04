from __future__ import annotations

from spotter.contracts import PoseFrame
from spotter.exercises.shared.analyzer import (
    ExerciseMetricResult,
    max_optional,
    mean_optional,
    mean_pair,
    min_optional,
    round_optional,
    safe_ratio,
    score,
    side_delta,
    std_optional,
    torso_lean_deg,
    value_series,
    width,
)
from spotter.steps.rep_signals import average_axis


class SquatAnalyzer:
    def metrics(self, frames: list[PoseFrame]) -> ExerciseMetricResult:
        knee_angles = value_series(
            frames,
            lambda frame: mean_pair(
                frame,
                ("left_hip", "left_knee", "left_ankle"),
                ("right_hip", "right_knee", "right_ankle"),
            ),
        )
        hip_angles = value_series(
            frames,
            lambda frame: mean_pair(
                frame,
                ("left_shoulder", "left_hip", "left_knee"),
                ("right_shoulder", "right_hip", "right_knee"),
            ),
        )
        knee_deltas = value_series(
            frames,
            lambda frame: side_delta(
                frame,
                ("left_hip", "left_knee", "left_ankle"),
                ("right_hip", "right_knee", "right_ankle"),
            ),
        )
        hip_deltas = value_series(
            frames,
            lambda frame: side_delta(
                frame,
                ("left_shoulder", "left_hip", "left_knee"),
                ("right_shoulder", "right_hip", "right_knee"),
            ),
        )
        hip_y = value_series(frames, lambda frame: average_axis(frame, ("left_hip", "right_hip"), "y"))
        knee_y = value_series(
            frames,
            lambda frame: average_axis(frame, ("left_knee", "right_knee"), "y"),
        )
        ankle_width = value_series(frames, lambda frame: width(frame, "left_ankle", "right_ankle"))
        shoulder_width = value_series(
            frames,
            lambda frame: width(frame, "left_shoulder", "right_shoulder"),
        )
        knee_width = value_series(frames, lambda frame: width(frame, "left_knee", "right_knee"))
        torso_lean = value_series(
            frames,
            lambda frame: mean_optional(
                [torso_lean_deg(frame, "left"), torso_lean_deg(frame, "right")]
            ),
        )

        min_knee = min_optional(knee_angles)
        max_knee = max_optional(knee_angles)
        min_hip = min_optional(hip_angles)
        max_hip = max_optional(hip_angles)
        hip_depth_delta = None
        max_hip_y = max_optional(hip_y)
        mean_knee_y = mean_optional(knee_y)
        if max_hip_y is not None and mean_knee_y is not None:
            hip_depth_delta = max_hip_y - mean_knee_y

        stance_ratio = safe_ratio(mean_optional(ankle_width), mean_optional(shoulder_width))
        knee_tracking_ratio = safe_ratio(mean_optional(knee_width), mean_optional(ankle_width))
        valgus_proxy = None if knee_tracking_ratio is None else max(0.0, 1.0 - knee_tracking_ratio)
        symmetry_delta = mean_optional(knee_deltas + hip_deltas) or 0.0
        stability_noise = (std_optional(hip_y) or 0.0) + (std_optional(knee_width) or 0.0)

        knee_rom = 0.0 if min_knee is None or max_knee is None else max_knee - min_knee
        depth_score = score((hip_depth_delta + 0.08) / 0.18) if hip_depth_delta is not None else 0.5
        angle_score = score(knee_rom / 65.0)
        rom_score = score(angle_score * 0.55 + depth_score * 0.45)
        stability_score = score(1.0 - stability_noise * 5.0)
        symmetry_score = score(1.0 - symmetry_delta / 45.0)

        hip_x = value_series(
            frames,
            lambda frame: average_axis(frame, ("left_hip", "right_hip"), "x"),
        )
        metrics = {
            "min_knee_angle_deg": round_optional(min_knee),
            "max_knee_angle_deg": round_optional(max_knee),
            "min_hip_angle_deg": round_optional(min_hip),
            "max_hip_angle_deg": round_optional(max_hip),
            "hip_depth_delta": round_optional(hip_depth_delta, 4),
            "hip_depth_relative_to_knee": (
                "below_parallel"
                if hip_depth_delta is not None and hip_depth_delta >= 0.03
                else "parallel"
                if hip_depth_delta is not None and hip_depth_delta >= -0.03
                else "above_parallel"
            ),
            "max_torso_lean_deg": round_optional(max_optional(torso_lean)),
            "knee_valgus_proxy": round_optional(valgus_proxy, 4),
            "knee_tracking_score": score(1.0 - (valgus_proxy or 0.0)),
            "stance_width_ratio": round_optional(stance_ratio, 3),
            "hip_shift": round_optional(std_optional(hip_x), 4),
            "bottom_stability_score": stability_score,
        }
        hints = []
        if stance_ratio is not None and stance_ratio > 1.35:
            hints.append("wide_squat_stance")
        elif stance_ratio is not None and stance_ratio < 0.85:
            hints.append("narrow_squat_stance")
        return metrics, rom_score, stability_score, symmetry_score, hints
