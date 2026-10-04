from __future__ import annotations

import threading

from spotter.steps.pose_backends.base import PoseBackend
from spotter.steps.pose_backends.mediapipe import MediaPipePoseBackend
from spotter.steps.pose_backends.mmpose import MMPoseBackend
from spotter.steps.pose_backends.mock import MockPoseBackend


_MEDIAPIPE_SINGLETON: MediaPipePoseBackend | None = None
_MEDIAPIPE_LOCK = threading.Lock()


def create_pose_backend(name: str) -> PoseBackend:
    normalized_name = name.strip().lower().replace("-", "_")
    if normalized_name == "mock":
        return MockPoseBackend()
    if normalized_name == "mediapipe":
        global _MEDIAPIPE_SINGLETON
        if _MEDIAPIPE_SINGLETON is None:
            with _MEDIAPIPE_LOCK:
                if _MEDIAPIPE_SINGLETON is None:
                    _MEDIAPIPE_SINGLETON = MediaPipePoseBackend()
        return _MEDIAPIPE_SINGLETON
    if normalized_name == "mmpose":
        return MMPoseBackend()
    raise ValueError(f"Unknown pose backend: {name!r}")


def get_mediapipe_singleton() -> MediaPipePoseBackend | None:
    """Return the singleton MediaPipe backend if already created, without creating it."""
    return _MEDIAPIPE_SINGLETON


def warmup_mediapipe() -> MediaPipePoseBackend:
    """Ensure MediaPipe backend is created and ready. Returns the singleton."""
    global _MEDIAPIPE_SINGLETON
    if _MEDIAPIPE_SINGLETON is None:
        with _MEDIAPIPE_LOCK:
            if _MEDIAPIPE_SINGLETON is None:
                _MEDIAPIPE_SINGLETON = MediaPipePoseBackend()
    return _MEDIAPIPE_SINGLETON
