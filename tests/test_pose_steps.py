from __future__ import annotations

# ruff: noqa: E402

import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.contracts import PoseFrame, PoseSequence, VideoManifest
from spotter.steps import pose_cleaning, pose_landmarker
from spotter.steps.pose_backends import (
    PoseBackendUnavailableError,
    PoseDetection,
    landmark_list_to_dict,
)
from spotter.steps.pose_backends.mediapipe import MediaPipePoseBackend, _MediaPipeTasksPoseAdapter


def _landmark(x: float, y: float, z: float = 0.0, visibility: float = 0.9) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def _landmark_result(offset: float = 0.0) -> SimpleNamespace:
    landmarks = [
        _landmark(0.2 + offset + index * 0.001, 0.1 + index * 0.01, -0.01, 0.91)
        for index in range(33)
    ]
    return SimpleNamespace(
        pose_landmarks=SimpleNamespace(landmark=landmarks),
        pose_world_landmarks=SimpleNamespace(landmark=landmarks),
    )


class _FakePose:
    source = "fake_pose"

    def __init__(self) -> None:
        self.calls = 0

    def __enter__(self) -> "_FakePose":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def detect(self, _frame: object, *, frame_index: int) -> PoseDetection:
        self.calls += 1
        result = _landmark_result(offset=self.calls * 0.01)
        return PoseDetection(
            landmarks=landmark_list_to_dict(result.pose_landmarks),
            world_landmarks=landmark_list_to_dict(result.pose_world_landmarks),
            source=self.source,
        )


class PoseStepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_video(self, frame_count: int = 4) -> Path:
        path = Path(self.temp_dir.name) / "pose.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
        self.assertTrue(writer.isOpened())
        for frame_index in range(frame_count):
            frame = np.full((480, 640, 3), 120 + frame_index, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return path

    def test_pose_landmarker_maps_to_coco17_landmarks_and_quality(self) -> None:
        path = self._write_video()
        manifest = VideoManifest(
            video_path=str(path),
            fps=30.0,
            duration_sec=0.133,
            total_frames=4,
            sampled_frames=4,
            width=640,
            height=480,
            codec="mp4v",
            container="mp4",
            brightness_mean=120.0,
            blur_laplacian_var=100.0,
            quality_warnings=[],
            analysis_allowed=True,
        )
        sequence = pose_landmarker.run(manifest, backend=_FakePose())

        self.assertEqual(len(sequence.frames), 4)
        self.assertEqual(len(sequence.frames[0].landmarks), 17)
        self.assertIn("left_ankle", sequence.frames[0].landmarks)
        self.assertNotIn("left_foot_index", sequence.frames[0].landmarks)
        self.assertEqual(len(sequence.frames[0].world_landmarks), 17)
        self.assertEqual(sequence.frames[0].pose_quality["source"], "fake_pose")
        self.assertEqual(sequence.frames[0].pose_quality["landmark_schema"], "coco17")
        self.assertGreater(sequence.frames[0].pose_quality["mean_visibility"], 0.9)
        self.assertTrue(sequence.frames[0].pose_quality["critical_landmarks_visible"])
        self.assertEqual(sequence.pose_valid_ratio, 1.0)

    def test_pose_landmarker_uses_dense_frames_for_real_backend(self) -> None:
        path = self._write_video(frame_count=130)
        manifest = VideoManifest(
            video_path=str(path),
            fps=30.0,
            duration_sec=4.333,
            total_frames=130,
            sampled_frames=12,
            width=640,
            height=480,
            codec="mp4v",
            container="mp4",
            brightness_mean=120.0,
            blur_laplacian_var=100.0,
            quality_warnings=[],
            analysis_allowed=True,
        )

        sequence = pose_landmarker.run(manifest, backend=_FakePose())

        self.assertEqual(len(sequence.frames), 130)

    def test_pose_landmarker_returns_unavailable_sequence_for_missing_backend_libs(
        self,
    ) -> None:
        path = self._write_video()
        manifest = VideoManifest(
            video_path=str(path),
            fps=30.0,
            duration_sec=0.133,
            total_frames=4,
            sampled_frames=4,
            width=640,
            height=480,
            codec="mp4v",
            container="mp4",
            brightness_mean=120.0,
            blur_laplacian_var=100.0,
            quality_warnings=[],
            analysis_allowed=True,
        )

        with patch(
            "spotter.steps.pose_landmarker.create_pose_backend",
            side_effect=PoseBackendUnavailableError("missing libGLESv2.so.2"),
        ):
            sequence = pose_landmarker.run(manifest, backend_name="mediapipe")

        self.assertEqual(sequence.pose_valid_ratio, 0.0)
        self.assertEqual(len(sequence.frames), 1)
        self.assertEqual(
            sequence.frames[0].pose_quality["pose_warning"],
            "pose_backend_unavailable",
        )
        self.assertIn("libGLESv2", sequence.frames[0].pose_quality["reason"])

    def test_dense_video_iteration_reads_sequentially_without_reseeking(self) -> None:
        class FakeCapture:
            def __init__(self) -> None:
                self.set_calls: list[tuple[int, int]] = []
                self.read_calls = 0

            def isOpened(self) -> bool:
                return True

            def set(self, prop: int, value: int) -> None:
                self.set_calls.append((prop, value))

            def read(self) -> tuple[bool, object | None]:
                if self.read_calls >= 3:
                    return False, None
                self.read_calls += 1
                return True, object()

            def release(self) -> None:
                return None

        capture = FakeCapture()
        manifest = VideoManifest(
            video_path="fake.mp4",
            fps=30.0,
            duration_sec=0.1,
            total_frames=3,
            sampled_frames=3,
            width=640,
            height=480,
            codec="mp4v",
            container="mp4",
            brightness_mean=120.0,
            blur_laplacian_var=100.0,
            quality_warnings=[],
            analysis_allowed=True,
        )

        with patch("spotter.steps.pose_landmarker.cv2.VideoCapture", return_value=capture):
            frames = list(pose_landmarker._iter_video_frames(manifest, sample_count=None))

        self.assertEqual([frame_index for frame_index, _ in frames], [0, 1, 2])
        seek_calls = [call for call in capture.set_calls if call[0] == cv2.CAP_PROP_POS_FRAMES]
        self.assertEqual(seek_calls, [])
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            self.assertIn((cv2.CAP_PROP_ORIENTATION_AUTO, 1), capture.set_calls)

    def test_pose_cleaning_interpolates_smooths_and_adds_normalized_fields(
        self,
    ) -> None:
        first_landmarks = _landmark_result(offset=0.0).pose_landmarks
        last_landmarks = _landmark_result(offset=0.2).pose_landmarks
        first = landmark_list_to_dict(first_landmarks)
        last = landmark_list_to_dict(last_landmarks)
        sequence = PoseSequence(
            frames=[
                PoseFrame(0, 0.0, first, first, {"mean_visibility": 0.9}),
                PoseFrame(1, 0.033, {}, {}, {"mean_visibility": 0.0}),
                PoseFrame(2, 0.067, last, last, {"mean_visibility": 0.9}),
            ],
            normalized=False,
            smoothing_method="none",
            pose_valid_ratio=0.667,
        )

        cleaned = pose_cleaning.run(sequence)

        self.assertTrue(cleaned.normalized)
        self.assertEqual(cleaned.smoothing_method, "exponential_smoothing")
        self.assertEqual(cleaned.pose_valid_ratio, 1.0)
        self.assertTrue(cleaned.frames[1].pose_quality["interpolated"])
        shoulder = cleaned.frames[1].landmarks["left_shoulder"]
        self.assertIn("x", shoulder)
        self.assertIn("smoothed_x", shoulder)
        self.assertIn("normalized_x", shoulder)
        self.assertTrue(cleaned.frames[1].pose_quality["normalized"])
        self.assertIn("smoothed_x", cleaned.frames[1].world_landmarks["left_shoulder"])
        self.assertIn("normalized_z", cleaned.frames[1].world_landmarks["left_shoulder"])

    @unittest.skipUnless(
        os.getenv("SPOTTER_RUN_REAL_POSE_TESTS") == "1",
        "set SPOTTER_RUN_REAL_POSE_TESTS=1 to run the real MediaPipe fixture smoke test",
    )
    def test_real_sample_mov_extracts_pose_landmarks(self) -> None:
        path = FIXTURES_DIR / "sample.MOV"
        self.assertTrue(path.exists(), path)

        from spotter.steps import video_qc

        manifest = video_qc.run(str(path))
        sequence = pose_landmarker.run(manifest, mock=False, backend_name="mediapipe")
        cleaned = pose_cleaning.run(sequence)

        self.assertTrue(manifest.analysis_allowed)
        self.assertGreater(len(sequence.frames), 0)
        self.assertGreater(sequence.pose_valid_ratio, 0.0)
        self.assertEqual(len(sequence.frames[0].landmarks), 17)
        self.assertEqual(sequence.frames[0].pose_quality["source"], "mediapipe_pose")
        self.assertTrue(cleaned.normalized)


class MediaPipeTasksDelegateTests(unittest.TestCase):
    def _fake_mediapipe(self, *, fail_gpu: bool = False) -> SimpleNamespace:
        class Delegate:
            CPU = "cpu"
            GPU = "gpu"

        class BaseOptions:
            def __init__(self, *, model_asset_path: str, delegate: str) -> None:
                self.model_asset_path = model_asset_path
                self.delegate = delegate

        BaseOptions.Delegate = Delegate

        class PoseLandmarkerOptions:
            def __init__(self, *, base_options: BaseOptions, **_kwargs: object) -> None:
                self.base_options = base_options

        class PoseLandmarker:
            @staticmethod
            def create_from_options(options: PoseLandmarkerOptions) -> object:
                if fail_gpu and options.base_options.delegate == "gpu":
                    raise RuntimeError("gpu delegate unavailable")
                return SimpleNamespace(delegate=options.base_options.delegate)

        return SimpleNamespace(
            tasks=SimpleNamespace(
                BaseOptions=BaseOptions,
                vision=SimpleNamespace(
                    PoseLandmarkerOptions=PoseLandmarkerOptions,
                    PoseLandmarker=PoseLandmarker,
                    RunningMode=SimpleNamespace(IMAGE="image"),
                ),
            )
        )

    def _adapter(self, fake_mediapipe: SimpleNamespace) -> _MediaPipeTasksPoseAdapter:
        adapter = object.__new__(_MediaPipeTasksPoseAdapter)
        adapter._mp = fake_mediapipe
        return adapter

    def test_mediapipe_tasks_prefers_cpu_outside_zero_gpu(self) -> None:
        adapter = self._adapter(self._fake_mediapipe())

        torch_without_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict(sys.modules, {"torch": torch_without_cuda}),
        ):
            self.assertEqual(adapter._preferred_delegate(), "cpu")

    def test_mediapipe_tasks_defaults_to_cpu_inside_zero_gpu(self) -> None:
        adapter = self._adapter(self._fake_mediapipe())

        with patch.dict(os.environ, {"SPACES_ZERO_GPU": "1"}, clear=True):
            self.assertEqual(adapter._preferred_delegate(), "cpu")

    def test_mediapipe_tasks_auto_prefers_gpu_inside_zero_gpu(self) -> None:
        adapter = self._adapter(self._fake_mediapipe())

        with patch.dict(
            os.environ,
            {"SPACES_ZERO_GPU": "1", "SPOTTER_MEDIAPIPE_DELEGATE": "auto"},
            clear=True,
        ):
            self.assertEqual(adapter._preferred_delegate(), "gpu")

    def test_mediapipe_delegate_env_can_force_gpu(self) -> None:
        adapter = self._adapter(self._fake_mediapipe())

        with patch.dict(os.environ, {"SPOTTER_MEDIAPIPE_DELEGATE": "gpu"}, clear=True):
            self.assertEqual(adapter._preferred_delegate(), "gpu")

    def test_mediapipe_tasks_falls_back_to_cpu_when_gpu_delegate_fails(self) -> None:
        adapter = self._adapter(self._fake_mediapipe(fail_gpu=True))

        with patch.dict(os.environ, {"SPACES_ZERO_GPU": "1"}, clear=True):
            landmarker = adapter._create_landmarker(Path("pose.task"))

        self.assertEqual(landmarker.delegate, "cpu")

    def test_mediapipe_backend_prefers_tasks_when_legacy_solution_exists(self) -> None:
        fake_mediapipe = self._fake_mediapipe()

        class LegacyPose:
            def __init__(self, **_kwargs: object) -> None:
                raise AssertionError("legacy CPU-only pose solution should not be used")

        fake_mediapipe.solutions = SimpleNamespace(
            pose=SimpleNamespace(Pose=LegacyPose),
        )
        backend = object.__new__(MediaPipePoseBackend)

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict(sys.modules, {"mediapipe": fake_mediapipe}),
            patch(
                "spotter.steps.pose_backends.mediapipe._ensure_pose_task_model",
                return_value=Path("pose.task"),
            ),
        ):
            pose = backend._create_pose()

        self.assertIsInstance(pose, _MediaPipeTasksPoseAdapter)
        self.assertEqual(pose._landmarker.delegate, "cpu")


if __name__ == "__main__":
    unittest.main()
