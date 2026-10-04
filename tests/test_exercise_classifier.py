from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.contracts import PoseFrame, PoseSequence, UserProfile, validate_contract
from spotter.ml.exercise_router_evaluation import (
    evaluate_router_predictions,
    select_router_candidate,
)
from spotter.ml.exercise_router_features import (
    FEATURE_SCHEMA,
    ROUTER_LABELS,
    ROUTER_LANDMARK_SCHEMA,
    extract_router_windows,
    window_tensor_feature_names,
    window_vector_feature_names,
)
from spotter.ml.exercise_router_inference import (
    DEFAULT_HF_REPO_ID,
    HF_DISABLE_ENV,
    RouterModelBundle,
    WindowRouterPrediction,
    aggregate_window_predictions,
    load_router_model,
    load_router_model_from_hf,
)
from spotter.steps import exercise_classifier
from spotter.steps.pose_backends.landmarks import LANDMARK_NAMES


def _profile(intended_exercise: str = "auto") -> UserProfile:
    return UserProfile(
        goal="beginner_practice",
        experience_level="beginner",
        intended_exercise=intended_exercise,
        intended_variation=None,
        known_limitations=[],
        equipment="bodyweight",
    )


def _base_landmarks(visibility: float) -> dict[str, dict[str, float]]:
    return {
        name: {
            "x": 0.5,
            "y": 0.5,
            "z": 0.0,
            "visibility": visibility,
            "normalized_x": 0.0,
            "normalized_y": 0.0,
            "normalized_z": 0.0,
        }
        for name in LANDMARK_NAMES
    }


def _set_landmark(
    landmarks: dict[str, dict[str, float]],
    name: str,
    x: float,
    y: float,
    visibility: float,
) -> None:
    landmarks[name].update(
        {
            "x": x,
            "y": y,
            "z": 0.0,
            "visibility": visibility,
            "normalized_x": x - 0.5,
            "normalized_y": y - 0.5,
            "normalized_z": 0.0,
        }
    )


def _landmarks_for_exercise(
    exercise: str, phase: float, visibility: float
) -> dict[str, dict[str, float]]:
    landmarks = _base_landmarks(visibility)
    wave = (1.0 - math.cos(2.0 * math.pi * phase)) / 2.0
    if exercise == "squat":
        _set_landmark(landmarks, "left_shoulder", 0.42, 0.3 + wave * 0.04, visibility)
        _set_landmark(landmarks, "right_shoulder", 0.58, 0.3 + wave * 0.04, visibility)
        _set_landmark(landmarks, "left_hip", 0.43, 0.52 + wave * 0.16, visibility)
        _set_landmark(landmarks, "right_hip", 0.57, 0.52 + wave * 0.16, visibility)
        _set_landmark(landmarks, "left_knee", 0.42 + wave * 0.05, 0.72, visibility)
        _set_landmark(landmarks, "right_knee", 0.58 - wave * 0.05, 0.72, visibility)
        _set_landmark(landmarks, "left_ankle", 0.41, 0.92, visibility)
        _set_landmark(landmarks, "right_ankle", 0.59, 0.92, visibility)
    elif exercise == "shoulder_press":
        _set_landmark(landmarks, "left_shoulder", 0.42, 0.42, visibility)
        _set_landmark(landmarks, "right_shoulder", 0.58, 0.42, visibility)
        _set_landmark(landmarks, "left_hip", 0.43, 0.7, visibility)
        _set_landmark(landmarks, "right_hip", 0.57, 0.7, visibility)
        _set_landmark(landmarks, "left_elbow", 0.4 - wave * 0.05, 0.62 - wave * 0.12, visibility)
        _set_landmark(landmarks, "right_elbow", 0.6 + wave * 0.05, 0.62 - wave * 0.12, visibility)
        _set_landmark(landmarks, "left_wrist", 0.4, 0.78 - wave * 0.34, visibility)
        _set_landmark(landmarks, "right_wrist", 0.6, 0.78 - wave * 0.34, visibility)
    else:
        _set_landmark(landmarks, "left_shoulder", 0.3, 0.38 + wave * 0.14, visibility)
        _set_landmark(landmarks, "right_shoulder", 0.7, 0.38 + wave * 0.14, visibility)
        _set_landmark(landmarks, "left_elbow", 0.36, 0.47 + wave * 0.08, visibility)
        _set_landmark(landmarks, "right_elbow", 0.64, 0.47 + wave * 0.08, visibility)
        _set_landmark(landmarks, "left_wrist", 0.34, 0.52, visibility)
        _set_landmark(landmarks, "right_wrist", 0.66, 0.52, visibility)
        _set_landmark(landmarks, "left_hip", 0.42, 0.48 + wave * 0.14, visibility)
        _set_landmark(landmarks, "right_hip", 0.58, 0.48 + wave * 0.14, visibility)
        _set_landmark(landmarks, "left_ankle", 0.44, 0.56 + wave * 0.14, visibility)
        _set_landmark(landmarks, "right_ankle", 0.56, 0.56 + wave * 0.14, visibility)
    return landmarks


def _sequence(
    exercise: str = "push_up", frame_count: int = 45, visibility: float = 0.95
) -> PoseSequence:
    frames = [
        PoseFrame(
            frame_index=index,
            timestamp_sec=round(index / 30.0, 3),
            landmarks=_landmarks_for_exercise(exercise, index / 24.0, visibility),
            world_landmarks={},
            pose_quality={"mean_visibility": visibility, "normalized": True},
        )
        for index in range(frame_count)
    ]
    return PoseSequence(
        frames=frames,
        normalized=True,
        smoothing_method="exponential_smoothing",
        pose_valid_ratio=1.0 if visibility >= 0.2 else 0.4,
    )


def _static_sequence(frame_count: int = 45, visibility: float = 0.95) -> PoseSequence:
    frames = [
        PoseFrame(
            frame_index=index,
            timestamp_sec=round(index / 30.0, 3),
            landmarks=_landmarks_for_exercise("push_up", 0.0, visibility),
            world_landmarks={},
            pose_quality={"mean_visibility": visibility, "normalized": True},
        )
        for index in range(frame_count)
    ]
    return PoseSequence(
        frames=frames,
        normalized=True,
        smoothing_method="exponential_smoothing",
        pose_valid_ratio=1.0 if visibility >= 0.2 else 0.4,
    )


class _FakePushUpModel:
    classes_ = np.asarray(ROUTER_LABELS)

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return np.tile(np.asarray([[0.03, 0.91, 0.02, 0.04]]), (values.shape[0], 1))


def _fake_router_artifact() -> dict[str, object]:
    return {
        "model": _FakePushUpModel(),
        "labels": list(ROUTER_LABELS),
        "model_kind": "baseline",
        "feature_schema": FEATURE_SCHEMA,
        "landmark_schema": ROUTER_LANDMARK_SCHEMA,
        "input_size": len(window_vector_feature_names()),
    }


class ExerciseRouterFeatureTests(unittest.TestCase):
    def test_extracts_windows_for_supported_exercise_motion(self) -> None:
        for exercise in ("squat", "push_up", "shoulder_press"):
            with self.subTest(exercise=exercise):
                windows = extract_router_windows(_sequence(exercise))

                self.assertGreaterEqual(len(windows), 2)
                self.assertEqual(windows[0].tensor.shape, (30, len(window_tensor_feature_names())))
                self.assertEqual(windows[0].vector.shape, (len(window_vector_feature_names()),))
                self.assertGreater(windows[0].mean_visibility, 0.9)

    def test_empty_and_low_visibility_sequences_do_not_produce_windows(self) -> None:
        empty = PoseSequence(
            frames=[],
            normalized=True,
            smoothing_method="none",
            pose_valid_ratio=0.0,
        )

        self.assertEqual(extract_router_windows(empty), [])
        self.assertEqual(extract_router_windows(_sequence(visibility=0.05)), [])

    def test_missing_landmark_is_zero_in_router_features(self) -> None:
        sequence = _sequence("push_up")
        frame = sequence.frames[0]
        frame.landmarks.pop("nose")

        windows = extract_router_windows(sequence, min_mean_visibility=0.0)

        self.assertGreater(len(windows), 0)
        nose_visibility_index = window_tensor_feature_names().index("nose_visibility")
        self.assertEqual(windows[0].tensor[0, nose_visibility_index], 0.0)

    def test_landmarks_without_visibility_still_produce_router_windows(self) -> None:
        sequence = _sequence("push_up")
        frames = [
            PoseFrame(
                frame.frame_index,
                frame.timestamp_sec,
                {
                    name: {key: value for key, value in landmark.items() if key != "visibility"}
                    for name, landmark in frame.landmarks.items()
                },
                frame.world_landmarks,
                frame.pose_quality,
            )
            for frame in sequence.frames
        ]
        no_visibility_sequence = PoseSequence(
            frames=frames,
            normalized=sequence.normalized,
            smoothing_method=sequence.smoothing_method,
            pose_valid_ratio=sequence.pose_valid_ratio,
        )

        windows = extract_router_windows(no_visibility_sequence)

        self.assertGreater(len(windows), 0)


class ExerciseRouterAggregationTests(unittest.TestCase):
    def test_aggregates_confident_windows(self) -> None:
        predictions = [
            WindowRouterPrediction(
                0.0,
                1.0,
                "push_up",
                0.91,
                {"squat": 0.03, "push_up": 0.91, "shoulder_press": 0.02, "unknown": 0.04},
            ),
            WindowRouterPrediction(
                0.5,
                1.5,
                "push_up",
                0.88,
                {"squat": 0.05, "push_up": 0.88, "shoulder_press": 0.02, "unknown": 0.05},
            ),
        ]

        aggregated = aggregate_window_predictions(predictions)

        self.assertEqual(aggregated.label, "push_up")
        self.assertFalse(aggregated.fallback_required)
        self.assertGreaterEqual(aggregated.confidence, 0.85)

    def test_low_confidence_and_inconsistent_windows_fallback_to_unknown(self) -> None:
        low_confidence = [
            WindowRouterPrediction(
                0.0,
                1.0,
                "push_up",
                0.45,
                {"squat": 0.25, "push_up": 0.45, "shoulder_press": 0.15, "unknown": 0.15},
            ),
            WindowRouterPrediction(
                0.5,
                1.5,
                "push_up",
                0.48,
                {"squat": 0.22, "push_up": 0.48, "shoulder_press": 0.15, "unknown": 0.15},
            ),
        ]
        inconsistent = [
            WindowRouterPrediction(
                0.0,
                1.0,
                "push_up",
                0.8,
                {"squat": 0.05, "push_up": 0.8, "shoulder_press": 0.1, "unknown": 0.05},
            ),
            WindowRouterPrediction(
                0.5,
                1.5,
                "squat",
                0.8,
                {"squat": 0.8, "push_up": 0.05, "shoulder_press": 0.1, "unknown": 0.05},
            ),
            WindowRouterPrediction(
                1.0,
                2.0,
                "shoulder_press",
                0.8,
                {"squat": 0.1, "push_up": 0.05, "shoulder_press": 0.8, "unknown": 0.05},
            ),
        ]

        self.assertTrue(aggregate_window_predictions(low_confidence).fallback_required)
        self.assertEqual(aggregate_window_predictions(low_confidence).label, "unknown")
        self.assertTrue(aggregate_window_predictions(inconsistent).fallback_required)
        self.assertEqual(aggregate_window_predictions(inconsistent).label, "unknown")


class ExerciseRouterModelLoadingTests(unittest.TestCase):
    def test_selection_file_controls_active_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            artifact_path = model_dir / "selected.joblib"
            joblib.dump(_fake_router_artifact(), artifact_path)
            (model_dir / "router_selection.json").write_text(
                json.dumps({"selected_artifact": artifact_path.name}),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {HF_DISABLE_ENV: "1"}):
                bundle = load_router_model(model_dir)

        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertEqual(bundle.labels, tuple(ROUTER_LABELS))
        result = exercise_classifier.run(
            _sequence("push_up"),
            _profile("auto"),
            mock=False,
            model_bundle=bundle,
        )
        self.assertEqual(result.exercise, "push_up")
        self.assertFalse(result.fallback_required)

    def test_selection_file_accepts_legacy_selected_model_field(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            artifact_path = model_dir / "selected.joblib"
            joblib.dump(_fake_router_artifact(), artifact_path)
            (model_dir / "router_selection.json").write_text(
                json.dumps({"selected_model": artifact_path.name}),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {HF_DISABLE_ENV: "1"}):
                bundle = load_router_model(model_dir)

        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertEqual(bundle.labels, tuple(ROUTER_LABELS))

    def test_hf_loader_honors_selection_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            artifact_path = model_dir / "router.joblib"
            selection_path = model_dir / "router_selection.json"
            joblib.dump(_fake_router_artifact(), artifact_path)
            selection_path.write_text(
                json.dumps({"selected_artifact": artifact_path.name}),
                encoding="utf-8",
            )

            def fake_download(*, repo_id: str, filename: str, revision: str | None) -> Path:
                self.assertEqual(repo_id, "owner/spotter-router")
                self.assertEqual(revision, "main")
                return {"router_selection.json": selection_path, "router.joblib": artifact_path}[
                    filename
                ]

            with patch(
                "spotter.ml.exercise_router_inference._hf_hub_download",
                side_effect=fake_download,
            ):
                bundle = load_router_model_from_hf("owner/spotter-router", revision="main")

        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertEqual(bundle.model_kind, "baseline")
        self.assertEqual(bundle.labels, tuple(ROUTER_LABELS))

    def test_hf_loader_defaults_to_spotter_router_repo(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            artifact_path = model_dir / "router.joblib"
            selection_path = model_dir / "router_selection.json"
            joblib.dump(_fake_router_artifact(), artifact_path)
            selection_path.write_text(
                json.dumps({"selected_artifact": artifact_path.name}),
                encoding="utf-8",
            )

            def fake_download(*, repo_id: str, filename: str, revision: str | None) -> Path:
                self.assertEqual(repo_id, DEFAULT_HF_REPO_ID)
                self.assertIsNone(revision)
                return {"router_selection.json": selection_path, "router.joblib": artifact_path}[
                    filename
                ]

            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "spotter.ml.exercise_router_inference._hf_hub_download",
                    side_effect=fake_download,
                ),
            ):
                bundle = load_router_model_from_hf()

        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertEqual(bundle.model_kind, "baseline")
        self.assertEqual(bundle.labels, tuple(ROUTER_LABELS))

    def test_hf_loader_reads_local_env_before_selecting_repo(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            model_dir.mkdir()
            artifact_path = model_dir / "router.joblib"
            selection_path = model_dir / "router_selection.json"
            joblib.dump(_fake_router_artifact(), artifact_path)
            selection_path.write_text(
                json.dumps({"selected_artifact": artifact_path.name}),
                encoding="utf-8",
            )
            (root / ".env").write_text(
                "SPOTTER_ROUTER_HF_REPO_ID=owner/env-router\n",
                encoding="utf-8",
            )
            original_cwd = Path.cwd()

            def fake_download(*, repo_id: str, filename: str, revision: str | None) -> Path:
                self.assertEqual(repo_id, "owner/env-router")
                self.assertIsNone(revision)
                return {"router_selection.json": selection_path, "router.joblib": artifact_path}[
                    filename
                ]

            try:
                os.chdir(root)
                with (
                    patch.dict(os.environ, {}, clear=True),
                    patch(
                        "spotter.ml.exercise_router_inference._hf_hub_download",
                        side_effect=fake_download,
                    ),
                ):
                    bundle = load_router_model_from_hf()
            finally:
                os.chdir(original_cwd)

        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertEqual(bundle.labels, tuple(ROUTER_LABELS))

    def test_old_router_artifact_without_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            joblib.dump(
                {
                    "model": _FakePushUpModel(),
                    "labels": list(ROUTER_LABELS),
                    "model_kind": "baseline",
                },
                model_dir / "router.joblib",
            )

            with patch.dict(os.environ, {HF_DISABLE_ENV: "1"}):
                with self.assertRaises(ValueError):
                    load_router_model(model_dir)


class ExerciseClassifierStepTests(unittest.TestCase):
    def test_manual_override_bypasses_model_and_validates_contract(self) -> None:
        result = exercise_classifier.run(
            _sequence("squat"),
            _profile("push_up"),
            mock=False,
        )

        self.assertEqual(result.exercise, "push_up")
        self.assertFalse(result.fallback_required)
        self.assertEqual(result.confidence, 0.98)
        validate_contract("exercise_classification.json", result)

    def test_missing_model_uses_pose_heuristic_for_clear_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {HF_DISABLE_ENV: "1"}):
                result = exercise_classifier.run(
                    _sequence("push_up"),
                    _profile("auto"),
                    mock=False,
                    model_dir=Path(temp_dir),
                )

        self.assertEqual(result.exercise, "push_up")
        self.assertFalse(result.fallback_required)
        self.assertGreaterEqual(result.confidence, 0.65)
        validate_contract("exercise_classification.json", result)

    def test_missing_model_keeps_static_motion_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {HF_DISABLE_ENV: "1"}):
                result = exercise_classifier.run(
                    _static_sequence(),
                    _profile("auto"),
                    mock=False,
                    model_dir=Path(temp_dir),
                )

        self.assertEqual(result.exercise, "unknown")
        self.assertTrue(result.fallback_required)
        validate_contract("exercise_classification.json", result)

    def test_fake_model_routes_and_persists_window_predictions(self) -> None:
        result = exercise_classifier.run(
            _sequence("push_up"),
            _profile("auto"),
            mock=False,
            model_bundle=RouterModelBundle(model=_FakePushUpModel(), labels=ROUTER_LABELS),
        )

        self.assertEqual(result.exercise, "push_up")
        self.assertFalse(result.fallback_required)
        self.assertGreater(result.confidence, 0.9)
        self.assertGreater(len(result.window_predictions), 0)
        self.assertEqual(
            sorted(result.window_predictions[0]),
            ["confidence", "end_sec", "label", "start_sec"],
        )
        validate_contract("exercise_classification.json", result)

    def test_low_pose_valid_ratio_falls_back_before_model_inference(self) -> None:
        sequence = _sequence("push_up", visibility=0.95)
        sequence = PoseSequence(
            frames=sequence.frames,
            normalized=sequence.normalized,
            smoothing_method=sequence.smoothing_method,
            pose_valid_ratio=0.4,
        )

        result = exercise_classifier.run(
            sequence,
            _profile("auto"),
            mock=False,
            model_bundle=RouterModelBundle(model=_FakePushUpModel(), labels=ROUTER_LABELS),
        )

        self.assertEqual(result.exercise, "unknown")
        self.assertTrue(result.fallback_required)


class ExerciseRouterEvaluationTests(unittest.TestCase):
    def test_evaluation_reports_accuracy_and_confusion_matrix(self) -> None:
        evaluation = evaluate_router_predictions(
            ["squat", "push_up", "shoulder_press", "unknown"],
            ["squat", "push_up", "unknown", "unknown"],
        )

        self.assertEqual(evaluation.accuracy, 0.75)
        self.assertEqual(evaluation.confusion_matrix["shoulder_press"]["unknown"], 1)
        self.assertEqual(evaluation.unknown_rejection_rate, 1.0)

    def test_prefers_temporal_when_available(self) -> None:
        baseline = {"name": "baseline", "accuracy": 0.91, "unknown_rejection_rate": 0.8}
        temporal = {"name": "temporal", "accuracy": 0.90, "unknown_rejection_rate": 0.7}

        self.assertEqual(select_router_candidate([baseline, temporal]), temporal)

    def test_selects_baseline_when_temporal_is_missing(self) -> None:
        baseline = {"name": "baseline", "accuracy": 0.91, "unknown_rejection_rate": 0.8}

        self.assertEqual(select_router_candidate([baseline]), baseline)


if __name__ == "__main__":
    unittest.main()
