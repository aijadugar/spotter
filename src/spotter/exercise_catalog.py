from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


MetricFactory = Callable[[int, float], dict[str, Any]]


@dataclass(frozen=True)
class MockIssueSpec:
    issue: str
    affected_joints: tuple[str, ...]
    evidence_metric: str
    threshold: float = 0.8


@dataclass(frozen=True)
class ExerciseSpec:
    key: str
    display_name: str
    default_variation: str
    default_variation_confidence: float
    metric_factory: MetricFactory
    variation_hints: tuple[str, ...] = ()
    default_variation_not_issues: tuple[str, ...] = ()
    mock_issue: MockIssueSpec | None = None
    user_selectable: bool = True


def _push_up_metrics(rep_id: int, fatigue_penalty: float) -> dict[str, Any]:
    return {
        "min_elbow_angle_deg": 88 + rep_id,
        "body_line_score": round(0.9 - fatigue_penalty, 2),
        "hip_sag_score": round(0.18 + fatigue_penalty, 2),
        "hand_width_ratio": 1.42,
    }


def _squat_metrics(rep_id: int, fatigue_penalty: float) -> dict[str, Any]:
    return {
        "min_knee_angle_deg": 92 - rep_id,
        "hip_depth_relative_to_knee": "slightly_above_parallel" if rep_id >= 4 else "parallel",
        "max_torso_lean_deg": 28 + rep_id,
        "knee_tracking_score": round(0.84 - fatigue_penalty, 2),
    }


def _shoulder_press_metrics(rep_id: int, fatigue_penalty: float) -> dict[str, Any]:
    return {
        "min_elbow_angle_deg": 74 + rep_id,
        "lockout_quality": round(0.9 - fatigue_penalty, 2),
        "wrist_path_verticality": round(0.86 - fatigue_penalty, 2),
        "left_right_wrist_delta": round(0.02 + fatigue_penalty, 2),
    }


def _unknown_metrics(rep_id: int, fatigue_penalty: float) -> dict[str, Any]:
    return {
        "movement_consistency_score": round(0.72 - fatigue_penalty, 2),
        "mock_rep_id": rep_id,
    }


EXERCISE_CATALOG: dict[str, ExerciseSpec] = {
    "push_up": ExerciseSpec(
        key="push_up",
        display_name="Push-up",
        default_variation="wide_grip_push_up",
        default_variation_confidence=0.84,
        metric_factory=_push_up_metrics,
        variation_hints=("wide_grip_push_up",),
        default_variation_not_issues=("wide_hand_placement",),
        mock_issue=MockIssueSpec(
            issue="hip_sag",
            affected_joints=("left_hip", "right_hip", "left_shoulder", "right_shoulder"),
            evidence_metric="body_line_score",
        ),
    ),
    "squat": ExerciseSpec(
        key="squat",
        display_name="Squat",
        default_variation="bodyweight_squat",
        default_variation_confidence=0.82,
        metric_factory=_squat_metrics,
        mock_issue=MockIssueSpec(
            issue="shallow_depth",
            affected_joints=("left_hip", "right_hip", "left_knee", "right_knee"),
            evidence_metric="range_of_motion_score",
        ),
    ),
    "shoulder_press": ExerciseSpec(
        key="shoulder_press",
        display_name="Shoulder press",
        default_variation="standing_shoulder_press",
        default_variation_confidence=0.8,
        metric_factory=_shoulder_press_metrics,
        mock_issue=MockIssueSpec(
            issue="incomplete_lockout",
            affected_joints=("left_elbow", "right_elbow", "left_wrist", "right_wrist"),
            evidence_metric="lockout_quality",
        ),
    ),
    "unknown": ExerciseSpec(
        key="unknown",
        display_name="Unknown",
        default_variation="unknown",
        default_variation_confidence=0.3,
        metric_factory=_unknown_metrics,
        user_selectable=False,
    ),
}

DEFAULT_AUTO_EXERCISE = "squat"
EXERCISES = frozenset(EXERCISE_CATALOG)
USER_SELECTABLE_EXERCISES = tuple(
    key for key, spec in EXERCISE_CATALOG.items() if spec.user_selectable
)
INTENDED_EXERCISES = frozenset({"auto", *USER_SELECTABLE_EXERCISES, "unknown"})


def get_exercise_spec(exercise: str) -> ExerciseSpec:
    return EXERCISE_CATALOG.get(exercise, EXERCISE_CATALOG["unknown"])
