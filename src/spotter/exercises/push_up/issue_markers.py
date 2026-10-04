from __future__ import annotations

from spotter.contracts import PoseFrame
from spotter.exercises.shared.analyzer import mean_optional
from spotter.exercises.shared.issue_marker import IssueRule
from spotter.steps.rep_signals import angle_deg, body_line_score


def _elbow_angle(frame: PoseFrame) -> float | None:
    return mean_optional(
        [
            angle_deg(frame, "left_shoulder", "left_elbow", "left_wrist"),
            angle_deg(frame, "right_shoulder", "right_elbow", "right_wrist"),
        ]
    )


RULES: tuple[IssueRule, ...] = (
    IssueRule(
        issue="hip_sag",
        metric_name="body_line_score",
        threshold=0.65,
        comparison="lt",
        affected_joints=(
            "left_hip",
            "right_hip",
            "left_shoulder",
            "right_shoulder",
            "left_ankle",
            "right_ankle",
        ),
        getter=body_line_score,
    ),
    IssueRule(
        issue="incomplete_depth",
        metric_name="elbow_angle_deg",
        threshold=115.0,
        comparison="gt",
        affected_joints=(
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
        ),
        getter=_elbow_angle,
        phase="bottom",
    ),
)
