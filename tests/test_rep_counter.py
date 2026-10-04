from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.contracts import PoseFrame, PoseSequence, UserProfile, VideoManifest, validate_contract
from spotter.exercises import create_exercise_strategy
from spotter.exercises.push_up import PushUpExercise
from spotter.exercises.shoulder_press import ShoulderPressExercise
from spotter.exercises.squat import SquatExercise
from spotter.steps.rep_signals import angle_deg


def _frame(frame_index: int, landmarks: dict[str, dict[str, float]]) -> PoseFrame:
    return PoseFrame(
        frame_index=frame_index,
        timestamp_sec=round(frame_index / 30.0, 3),
        landmarks=landmarks,
        world_landmarks={},
        pose_quality={"mean_visibility": 0.95, "normalized": True},
    )


def _wave(frame_index: int, cycle_frames: int, *, phase_shift: float = 0.0) -> float:
    return (1.0 - math.cos(2.0 * math.pi * ((frame_index + phase_shift) / cycle_frames))) / 2.0


def _squat_landmarks(depth: float) -> dict[str, dict[str, float]]:
    hip_y = 0.52 + depth * 0.18
    shoulder_y = 0.28 + depth * 0.06
    knee_y = 0.7
    ankle_y = 0.92
    hip_x_left, hip_x_right = 0.43, 0.57
    knee_x_left = hip_x_left + depth * 0.06
    knee_x_right = hip_x_right - depth * 0.06
    return {
        "left_shoulder": {"x": 0.4, "y": shoulder_y, "smoothed_x": 0.4, "smoothed_y": shoulder_y},
        "right_shoulder": {"x": 0.6, "y": shoulder_y, "smoothed_x": 0.6, "smoothed_y": shoulder_y},
        "left_hip": {"x": hip_x_left, "y": hip_y, "smoothed_x": hip_x_left, "smoothed_y": hip_y},
        "right_hip": {"x": hip_x_right, "y": hip_y, "smoothed_x": hip_x_right, "smoothed_y": hip_y},
        "left_knee": {"x": knee_x_left, "y": knee_y, "smoothed_x": knee_x_left, "smoothed_y": knee_y},
        "right_knee": {"x": knee_x_right, "y": knee_y, "smoothed_x": knee_x_right, "smoothed_y": knee_y},
        "left_ankle": {"x": 0.42, "y": ankle_y, "smoothed_x": 0.42, "smoothed_y": ankle_y},
        "right_ankle": {"x": 0.58, "y": ankle_y, "smoothed_x": 0.58, "smoothed_y": ankle_y},
    }


def _push_up_landmarks(depth: float) -> dict[str, dict[str, float]]:
    shoulder_y = 0.38 + depth * 0.16
    hip_y = 0.46 + depth * 0.16
    ankle_y = 0.54 + depth * 0.16
    elbow_y = 0.46 + depth * 0.1
    wrist_y = 0.5 + depth * 0.04
    elbow_x_left = 0.36 + depth * 0.02
    elbow_x_right = 0.64 - depth * 0.02
    return {
        "left_shoulder": {"x": 0.3, "y": shoulder_y, "smoothed_x": 0.3, "smoothed_y": shoulder_y},
        "right_shoulder": {"x": 0.7, "y": shoulder_y, "smoothed_x": 0.7, "smoothed_y": shoulder_y},
        "left_elbow": {"x": elbow_x_left, "y": elbow_y, "smoothed_x": elbow_x_left, "smoothed_y": elbow_y},
        "right_elbow": {"x": elbow_x_right, "y": elbow_y, "smoothed_x": elbow_x_right, "smoothed_y": elbow_y},
        "left_wrist": {"x": 0.34, "y": wrist_y, "smoothed_x": 0.34, "smoothed_y": wrist_y},
        "right_wrist": {"x": 0.66, "y": wrist_y, "smoothed_x": 0.66, "smoothed_y": wrist_y},
        "left_hip": {"x": 0.42, "y": hip_y, "smoothed_x": 0.42, "smoothed_y": hip_y},
        "right_hip": {"x": 0.58, "y": hip_y, "smoothed_x": 0.58, "smoothed_y": hip_y},
        "left_ankle": {"x": 0.44, "y": ankle_y, "smoothed_x": 0.44, "smoothed_y": ankle_y},
        "right_ankle": {"x": 0.56, "y": ankle_y, "smoothed_x": 0.56, "smoothed_y": ankle_y},
    }


def _shoulder_press_landmarks(lift: float) -> dict[str, dict[str, float]]:
    wrist_y = 0.76 - lift * 0.34
    elbow_y = 0.6 - lift * 0.1
    elbow_x_left = 0.4 - lift * 0.05
    elbow_x_right = 0.6 + lift * 0.05
    return {
        "left_shoulder": {"x": 0.42, "y": 0.42, "smoothed_x": 0.42, "smoothed_y": 0.42},
        "right_shoulder": {"x": 0.58, "y": 0.42, "smoothed_x": 0.58, "smoothed_y": 0.42},
        "left_elbow": {"x": elbow_x_left, "y": elbow_y, "smoothed_x": elbow_x_left, "smoothed_y": elbow_y},
        "right_elbow": {"x": elbow_x_right, "y": elbow_y, "smoothed_x": elbow_x_right, "smoothed_y": elbow_y},
        "left_wrist": {"x": 0.4, "y": wrist_y, "smoothed_x": 0.4, "smoothed_y": wrist_y},
        "right_wrist": {"x": 0.6, "y": wrist_y, "smoothed_x": 0.6, "smoothed_y": wrist_y},
    }


def _sequence_for_exercise(exercise: str, cycles: int, *, partial_tail: bool = False) -> PoseSequence:
    cycle_frames = 24
    total_frames = cycles * cycle_frames + (cycle_frames // 2 if partial_tail else 0)
    frames: list[PoseFrame] = []
    for frame_index in range(total_frames):
        wave = _wave(frame_index, cycle_frames)
        if exercise == "squat":
            landmarks = _squat_landmarks(wave)
        elif exercise == "shoulder_press":
            landmarks = _shoulder_press_landmarks(wave)
        else:
            landmarks = _push_up_landmarks(wave)
        frames.append(_frame(frame_index, landmarks))
    return PoseSequence(
        frames=frames,
        normalized=True,
        smoothing_method="exponential_smoothing",
        pose_valid_ratio=1.0,
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


def _profile(exercise: str = "auto") -> UserProfile:
    return UserProfile(
        goal="beginner_practice",
        experience_level="beginner",
        intended_exercise=exercise,
        intended_variation=None,
        known_limitations=[],
        equipment="bodyweight",
    )


def _exercise_strategy(exercise: str, sequence: PoseSequence):
    return create_exercise_strategy(
        exercise,
        video_manifest=_video_manifest(sequence),
        pose_sequence=sequence,
        profile=_profile(exercise),
    )


class RepCounterTests(unittest.TestCase):
    def test_angle_deg_uses_3d_world_landmarks(self) -> None:
        frame = PoseFrame(
            frame_index=0,
            timestamp_sec=0.0,
            landmarks={},
            world_landmarks={
                "left_shoulder": {"x": 1.0, "y": 0.0, "z": 0.0},
                "left_elbow": {"x": 0.0, "y": 0.0, "z": 0.0},
                "left_wrist": {"x": 0.0, "y": 0.0, "z": 1.0},
            },
            pose_quality={"mean_visibility": 1.0},
        )

        self.assertEqual(round(angle_deg(frame, "left_shoulder", "left_elbow", "left_wrist") or 0.0), 90)

    def test_segments_squat_reps(self) -> None:
        sequence = _sequence_for_exercise("squat", 3)
        reps, debug = _exercise_strategy("squat", sequence).count()
        self.assertEqual(len(reps.reps), 3)
        self.assertEqual(reps.reps[0].start_frame, 0)
        self.assertTrue(debug["accepted_reps"])

    def test_segments_push_up_reps(self) -> None:
        sequence = _sequence_for_exercise("push_up", 3)
        reps, _debug = _exercise_strategy("push_up", sequence).count()
        self.assertEqual(len(reps.reps), 3)
        self.assertEqual(reps.partial_reps, [])

    def test_segments_shoulder_press_reps(self) -> None:
        sequence = _sequence_for_exercise("shoulder_press", 3)
        reps, _debug = _exercise_strategy("shoulder_press", sequence).count()
        self.assertEqual(len(reps.reps), 3)
        self.assertEqual(reps.reps[0].mid_frame, 12)

    def test_reports_partial_last_rep(self) -> None:
        sequence = _sequence_for_exercise("push_up", 2, partial_tail=True)
        reps, _debug = _exercise_strategy("push_up", sequence).count()
        self.assertEqual(len(reps.reps), 2)
        self.assertIn("ends_mid_rep", {item["reason"] for item in reps.partial_reps})

    def test_unknown_exercise_is_not_segmented(self) -> None:
        sequence = _sequence_for_exercise("push_up", 1)
        reps, debug = _exercise_strategy("unknown", sequence).count()
        self.assertEqual(reps.reps, [])
        self.assertEqual(reps.partial_reps, [{"reason": "unknown_exercise"}])
        self.assertEqual(debug["selected_signal"], "none")
        validate_contract("reps.json", reps)
        validate_contract("rep_debug.json", debug)

    def test_exercises_resolve_to_specific_strategies(self) -> None:
        sequence = _sequence_for_exercise("push_up", 1)
        push_up = _exercise_strategy("push_up", sequence)
        self.assertIsInstance(push_up, PushUpExercise)
        self.assertIsInstance(_exercise_strategy("shoulder_press", sequence), ShoulderPressExercise)
        self.assertIsInstance(_exercise_strategy("squat", sequence), SquatExercise)
        self.assertIsNot(push_up, _exercise_strategy("push_up", sequence))


if __name__ == "__main__":
    unittest.main()
