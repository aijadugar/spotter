from __future__ import annotations

from spotter.contracts import PoseFrame
from spotter.exercises.shared.analyzer import mean_optional
from spotter.exercises.shared.issue_marker import IssueRule
from spotter.steps.rep_signals import angle_deg, landmark_axis


def _elbow_angle(frame: PoseFrame) -> float | None:
    return mean_optional(
        [
            angle_deg(frame, "left_shoulder", "left_elbow", "left_wrist"),
            angle_deg(frame, "right_shoulder", "right_elbow", "right_wrist"),
        ]
    )


def _wrist_height_asymmetry(frame: PoseFrame) -> float | None:
    left = landmark_axis(frame, "left_wrist", "y")
    right = landmark_axis(frame, "right_wrist", "y")
    if left is None or right is None:
        return None
    return abs(left - right)


RULES: tuple[IssueRule, ...] = (
    IssueRule(
        issue="incomplete_lockout",
        metric_name="elbow_angle_deg",
        threshold=155.0,
        comparison="lt",
        affected_joints=("left_elbow", "right_elbow", "left_wrist", "right_wrist"),
        getter=_elbow_angle,
        phase="top",
    ),
    IssueRule(
        issue="asymmetry",
        metric_name="wrist_height_asymmetry",
        threshold=0.1,
        comparison="gt",
        affected_joints=("left_wrist", "right_wrist", "left_elbow", "right_elbow"),
        getter=_wrist_height_asymmetry,
    ),
)
