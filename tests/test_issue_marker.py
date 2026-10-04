from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Callable
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.contracts import (
    IssueMarkers,
    PoseFrame,
    PoseSequence,
    Rep,
    Reps,
    UserProfile,
    VideoManifest,
)
from spotter.exercises import create_exercise_strategy


def _frame(frame_index: int, landmarks: dict[str, dict[str, float]]) -> PoseFrame:
    return PoseFrame(
        frame_index=frame_index,
        timestamp_sec=round(frame_index / 30.0, 3),
        landmarks=landmarks,
        world_landmarks={},
        pose_quality={"mean_visibility": 0.92, "normalized": True},
    )


def _wave(frame_index: int, cycle_frames: int = 30) -> float:
    return (1.0 - math.cos(2.0 * math.pi * (frame_index / cycle_frames))) / 2.0


def _reps(exercise: str, end_frame: int = 29) -> Reps:
    return Reps(
        exercise=exercise,
        reps=[Rep(1, 0, end_frame // 2, end_frame, 0.0, round((end_frame // 2) / 30.0, 3), round(end_frame / 30.0, 3))],
        partial_reps=[],
    )


def _profile(exercise: str = "auto") -> UserProfile:
    return UserProfile(
        goal="beginner_practice",
        experience_level="beginner",
        intended_exercise=exercise,
        intended_variation=None,
        known_limitations=[],
        equipment="bodyweight",
    )


def _video_manifest(sequence: PoseSequence) -> VideoManifest:
    return VideoManifest(
        video_path=None,
        fps=30.0,
        duration_sec=round(len(sequence.frames) / 30.0, 3),
        total_frames=len(sequence.frames),
        sampled_frames=len(sequence.frames),
        width=720,
        height=1280,
        codec=None,
        container=None,
        brightness_mean=None,
        blur_laplacian_var=None,
        quality_warnings=[],
        analysis_allowed=True,
    )


def _sequence(factory: Callable[[int, float], dict[str, dict[str, float]]], count: int = 30) -> PoseSequence:
    return PoseSequence(
        frames=[_frame(frame_index, factory(frame_index, _wave(frame_index, count))) for frame_index in range(count)],
        normalized=True,
        smoothing_method="none",
        pose_valid_ratio=1.0,
    )


def _push_up_landmarks(frame_index: int, depth: float, *, noise_only: bool = False) -> dict[str, dict[str, float]]:
    shoulder_y = 0.38 + depth * 0.16
    hip_y = 0.46 + depth * 0.16
    ankle_y = 0.54 + depth * 0.16
    if 10 <= frame_index <= 18 and not noise_only:
        hip_y += 0.45
    if frame_index == 15 and noise_only:
        hip_y += 0.45

    shoulder_width = 0.4
    hand_width = shoulder_width * 1.7
    return {
        "left_shoulder": {"x": 0.3, "y": shoulder_y},
        "right_shoulder": {"x": 0.7, "y": shoulder_y},
        "left_elbow": {"x": 0.36 + depth * 0.02, "y": 0.46 + depth * 0.1},
        "right_elbow": {"x": 0.64 - depth * 0.02, "y": 0.46 + depth * 0.1},
        "left_wrist": {"x": 0.5 - hand_width / 2, "y": 0.5 + depth * 0.04},
        "right_wrist": {"x": 0.5 + hand_width / 2, "y": 0.5 + depth * 0.04},
        "left_hip": {"x": 0.42, "y": hip_y},
        "right_hip": {"x": 0.58, "y": hip_y},
        "left_knee": {"x": 0.44, "y": hip_y + 0.2},
        "right_knee": {"x": 0.56, "y": hip_y + 0.2},
        "left_ankle": {"x": 0.44, "y": ankle_y},
        "right_ankle": {"x": 0.56, "y": ankle_y},
    }


def _squat_landmarks(frame_index: int, depth: float) -> dict[str, dict[str, float]]:
    del frame_index
    hip_y = 0.54 + depth * 0.08
    knee_y = 0.74
    knee_half_width = 0.04 + (1.0 - depth) * 0.08
    return {
        "left_shoulder": {"x": 0.74, "y": 0.26 + depth * 0.02},
        "right_shoulder": {"x": 0.86, "y": 0.26 + depth * 0.02},
        "left_hip": {"x": 0.43, "y": hip_y},
        "right_hip": {"x": 0.57, "y": hip_y},
        "left_knee": {"x": 0.5 - knee_half_width, "y": knee_y},
        "right_knee": {"x": 0.5 + knee_half_width, "y": knee_y},
        "left_ankle": {"x": 0.3, "y": 0.94},
        "right_ankle": {"x": 0.7, "y": 0.94},
    }


def _shoulder_press_landmarks(frame_index: int, lift: float) -> dict[str, dict[str, float]]:
    del frame_index
    wrist_y = 0.74 - lift * 0.12
    return {
        "left_shoulder": {"x": 0.42, "y": 0.42},
        "right_shoulder": {"x": 0.58, "y": 0.42},
        "left_elbow": {"x": 0.35, "y": 0.56 - lift * 0.04},
        "right_elbow": {"x": 0.65, "y": 0.56 - lift * 0.04},
        "left_wrist": {"x": 0.18, "y": wrist_y + 0.08},
        "right_wrist": {"x": 0.82, "y": wrist_y - 0.08},
        "left_hip": {"x": 0.44, "y": 0.74},
        "right_hip": {"x": 0.56, "y": 0.74},
        "left_ankle": {"x": 0.45, "y": 0.96},
        "right_ankle": {"x": 0.55, "y": 0.96},
    }


def _run_markers(exercise: str, sequence: PoseSequence) -> IssueMarkers:
    exercise_strategy = create_exercise_strategy(
        exercise,
        video_manifest=_video_manifest(sequence),
        pose_sequence=sequence,
        profile=_profile(exercise),
    )
    reps = _reps(exercise, len(sequence.frames) - 1)
    analysis = exercise_strategy.analyze_reps(reps)
    variation = exercise_strategy.resolve_variation(analysis)
    return exercise_strategy.mark_issues(reps, analysis, variation)


class IssueMarkerTests(unittest.TestCase):
    def test_push_up_hip_sag_interval_includes_evidence_and_variation_context(self) -> None:
        markers = _run_markers("push_up", _sequence(_push_up_landmarks))
        hip_sag = next(issue for issue in markers.issues if issue.issue == "hip_sag")

        self.assertGreaterEqual(hip_sag.end_frame - hip_sag.start_frame, 2)
        self.assertLess(hip_sag.evidence["body_line_score"], hip_sag.evidence["threshold"])
        self.assertIn("confidence", hip_sag.evidence)
        self.assertIn("peak_frame", hip_sag.evidence)
        self.assertGreaterEqual(hip_sag.evidence["peak_frame"], hip_sag.start_frame)
        self.assertLessEqual(hip_sag.evidence["peak_frame"], hip_sag.end_frame)
        self.assertEqual(
            hip_sag.evidence["variation_context"]["detected_variation"],
            "wide_grip_push_up",
        )
        self.assertIn("wide_hand_placement", hip_sag.evidence["variation_context"]["not_issues"])
        self.assertIn("left_hip", hip_sag.affected_joints)

    def test_single_frame_push_up_sag_noise_is_filtered_out(self) -> None:
        markers = _run_markers(
            "push_up",
            _sequence(lambda frame_index, depth: _push_up_landmarks(frame_index, depth, noise_only=True)),
        )

        self.assertNotIn("hip_sag", {issue.issue for issue in markers.issues})

    def test_squat_rules_emit_depth_valgus_and_torso_lean_intervals(self) -> None:
        markers = _run_markers("squat", _sequence(_squat_landmarks))
        labels = {issue.issue for issue in markers.issues}

        self.assertIn("shallow_depth", labels)
        self.assertIn("knee_valgus", labels)
        self.assertIn("excessive_torso_lean", labels)

    def test_shoulder_press_rules_emit_lockout_and_asymmetry_intervals(self) -> None:
        markers = _run_markers("shoulder_press", _sequence(_shoulder_press_landmarks))
        labels = {issue.issue for issue in markers.issues}

        self.assertIn("incomplete_lockout", labels)
        self.assertIn("asymmetry", labels)


if __name__ == "__main__":
    unittest.main()
