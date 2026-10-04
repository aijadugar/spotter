from __future__ import annotations

from spotter.contracts import PoseFrame
from spotter.exercises.shared.analyzer import mean_optional, safe_ratio, torso_lean_deg, width
from spotter.exercises.shared.issue_marker import IssueRule
from spotter.steps.rep_signals import average_axis


def _depth_delta(frame: PoseFrame) -> float | None:
    hip_y = average_axis(frame, ("left_hip", "right_hip"), "y")
    knee_y = average_axis(frame, ("left_knee", "right_knee"), "y")
    if hip_y is None or knee_y is None:
        return None
    return hip_y - knee_y


def _knee_valgus_proxy(frame: PoseFrame) -> float | None:
    knee_width = width(frame, "left_knee", "right_knee")
    ankle_width = width(frame, "left_ankle", "right_ankle")
    tracking_ratio = safe_ratio(knee_width, ankle_width)
    if tracking_ratio is None:
        return None
    return max(0.0, 1.0 - tracking_ratio)


def _torso_lean(frame: PoseFrame) -> float | None:
    return mean_optional([torso_lean_deg(frame, "left"), torso_lean_deg(frame, "right")])


RULES: tuple[IssueRule, ...] = (
    IssueRule(
        issue="shallow_depth",
        metric_name="hip_depth_delta",
        threshold=-0.03,
        comparison="lt",
        affected_joints=("left_hip", "right_hip", "left_knee", "right_knee"),
        getter=_depth_delta,
        phase="bottom",
    ),
    IssueRule(
        issue="knee_valgus",
        metric_name="knee_valgus_proxy",
        threshold=0.25,
        comparison="gt",
        affected_joints=("left_knee", "right_knee", "left_ankle", "right_ankle"),
        getter=_knee_valgus_proxy,
        phase="bottom",
    ),
    IssueRule(
        issue="excessive_torso_lean",
        metric_name="torso_lean_deg",
        threshold=35.0,
        comparison="gt",
        affected_joints=("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
        getter=_torso_lean,
        phase="bottom",
    ),
)
