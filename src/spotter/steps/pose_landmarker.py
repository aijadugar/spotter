from __future__ import annotations

from collections.abc import Iterator
import os
from typing import Any

import cv2

from spotter.contracts import PoseFrame, PoseSequence, VideoManifest
from spotter.hf_spaces import default_spaces_gpu_duration, spaces_gpu
from spotter.steps.pose_backends import (
    LANDMARK_SCHEMA,
    MockPoseBackend,
    PoseBackend,
    PoseBackendUnavailableError,
    create_pose_backend,
)
from spotter.steps.video_qc import enable_capture_orientation, sample_frame_indices


CRITICAL_LANDMARKS = {
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
}
FULL_BODY_LANDMARKS = CRITICAL_LANDMARKS | {"left_wrist", "right_wrist"}
DEFAULT_POSE_SAMPLE_COUNT = 120


def _env_pose_backend() -> str:
    return os.getenv("SPOTTER_POSE_BACKEND", "mediapipe")


def _iter_video_frames(
    manifest: VideoManifest,
    *,
    sample_count: int | None,
) -> Iterator[tuple[int, Any]]:
    if not manifest.video_path or manifest.total_frames <= 0:
        return

    capture = cv2.VideoCapture(manifest.video_path)
    try:
        if not capture.isOpened():
            return
        enable_capture_orientation(capture)
        if sample_count is None:
            for frame_index in range(manifest.total_frames):
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                yield frame_index, frame
            return

        frame_indices = sample_frame_indices(
            manifest.total_frames, min(sample_count, manifest.total_frames)
        )
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if ok and frame is not None:
                yield frame_index, frame
    finally:
        capture.release()


def _pose_quality(landmarks: dict[str, dict[str, float]]) -> dict[str, Any]:
    if not landmarks:
        return {
            "mean_visibility": 0.0,
            "critical_landmarks_visible": False,
            "full_body_visibility_proxy": 0.0,
            "landmark_count": 0,
            "pose_warning": "pose_not_detected",
        }

    visibility_values = [landmark.get("visibility", 0.0) for landmark in landmarks.values()]
    critical_values = [
        landmarks[name].get("visibility", 0.0) for name in CRITICAL_LANDMARKS if name in landmarks
    ]
    full_body_values = [
        landmarks[name].get("visibility", 0.0) for name in FULL_BODY_LANDMARKS if name in landmarks
    ]
    critical_visible = (
        len(critical_values) == len(CRITICAL_LANDMARKS) and min(critical_values, default=0.0) >= 0.5
    )
    return {
        "mean_visibility": round(sum(visibility_values) / len(visibility_values), 4),
        "critical_landmarks_visible": critical_visible,
        "full_body_visibility_proxy": (
            round(sum(full_body_values) / len(full_body_values), 4) if full_body_values else 0.0
        ),
        "landmark_count": len(landmarks),
    }


def _coordinate_source(detection_source: str, world_landmarks: dict[str, dict[str, float]]) -> str:
    if not world_landmarks:
        return "image_landmarks"
    if detection_source.startswith("mediapipe"):
        return "mediapipe_world_landmarks"
    return "world_landmarks"


def _empty_sequence() -> PoseSequence:
    return PoseSequence(frames=[], normalized=False, smoothing_method="none", pose_valid_ratio=0.0)


def _unavailable_sequence(reason: str) -> PoseSequence:
    return PoseSequence(
        frames=[
            PoseFrame(
                frame_index=0,
                timestamp_sec=0.0,
                landmarks={},
                world_landmarks={},
                pose_quality={
                    "mean_visibility": 0.0,
                    "critical_landmarks_visible": False,
                    "full_body_visibility_proxy": 0.0,
                    "landmark_count": 0,
                    "pose_warning": "pose_backend_unavailable",
                    "source": "none",
                    "landmark_schema": LANDMARK_SCHEMA,
                    "coordinate_source": "none",
                    "reason": reason,
                },
            )
        ],
        normalized=False,
        smoothing_method="none",
        pose_valid_ratio=0.0,
    )


def _run_with_backend(manifest: VideoManifest, backend: PoseBackend) -> PoseSequence:
    if not manifest.analysis_allowed or not manifest.video_path:
        return _empty_sequence()

    frames: list[PoseFrame] = []
    valid_frames = 0

    with backend as extractor:
        for frame_index, frame in _iter_video_frames(manifest, sample_count=None):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection = extractor.detect(rgb_frame, frame_index=frame_index)
            if detection.landmarks:
                valid_frames += 1
            frames.append(
                PoseFrame(
                    frame_index=frame_index,
                    timestamp_sec=(round(frame_index / manifest.fps, 3) if manifest.fps else 0.0),
                    landmarks=detection.landmarks,
                    world_landmarks=detection.world_landmarks,
                    pose_quality={
                        **_pose_quality(detection.landmarks),
                        "source": detection.source,
                        "landmark_schema": LANDMARK_SCHEMA,
                        "coordinate_source": _coordinate_source(
                            detection.source, detection.world_landmarks
                        ),
                    },
                )
            )

    pose_valid_ratio = round(valid_frames / len(frames), 4) if frames else 0.0
    return PoseSequence(
        frames=frames,
        normalized=False,
        smoothing_method="none",
        pose_valid_ratio=pose_valid_ratio,
    )


def _gpu_duration(*_args: Any, **_kwargs: Any) -> int:
    return default_spaces_gpu_duration()


@spaces_gpu(duration=_gpu_duration)
def _run_named_backend(manifest: VideoManifest, backend_name: str) -> PoseSequence:
    try:
        selected_backend = create_pose_backend(backend_name)
    except PoseBackendUnavailableError as exc:
        return _unavailable_sequence(str(exc))
    return _run_with_backend(manifest, selected_backend)


def _run_mock(manifest: VideoManifest) -> PoseSequence:
    frames: list[PoseFrame] = []
    backend = MockPoseBackend()
    for frame_index in sample_frame_indices(manifest.total_frames, DEFAULT_POSE_SAMPLE_COUNT):
        detection = backend.detect(None, frame_index=frame_index)
        frames.append(
            PoseFrame(
                frame_index=frame_index,
                timestamp_sec=(round(frame_index / manifest.fps, 3) if manifest.fps else 0.0),
                landmarks=detection.landmarks,
                world_landmarks=detection.world_landmarks,
                pose_quality={
                    **_pose_quality(detection.landmarks),
                    "source": detection.source,
                    "mock": True,
                    "landmark_schema": LANDMARK_SCHEMA,
                    "coordinate_source": "mock_landmarks",
                },
            )
        )

    return PoseSequence(
        frames=frames,
        normalized=False,
        smoothing_method="none",
        pose_valid_ratio=1.0 if frames else 0.0,
    )


def run(
    manifest: VideoManifest,
    *,
    mock: bool = False,
    backend_name: str | None = None,
    backend: PoseBackend | None = None,
) -> PoseSequence:
    if mock:
        return _run_mock(manifest)
    if backend is not None:
        return _run_with_backend(manifest, backend)
    return _run_named_backend(manifest, backend_name or _env_pose_backend())
