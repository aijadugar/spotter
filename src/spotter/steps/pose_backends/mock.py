from __future__ import annotations

from typing import Any

from spotter.steps.pose_backends.base import PoseDetection
from spotter.steps.pose_backends.landmarks import LANDMARK_NAMES


def _mock_landmarks(frame_index: int) -> dict[str, dict[str, float]]:
    phase = (frame_index % 90) / 90
    vertical_offset = 0.08 if phase < 0.5 else 0.0
    landmarks: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(LANDMARK_NAMES):
        side_offset = -0.06 if "left" in name else 0.06 if "right" in name else 0.0
        base_y = 0.2 + min(idx, 11) * 0.045
        landmarks[name] = {
            "x": round(0.5 + side_offset, 4),
            "y": round(base_y + vertical_offset, 4),
            "z": round(-0.02 * side_offset, 4),
            "visibility": 0.95,
        }
    return landmarks


class MockPoseBackend:
    source = "mock_pose"

    def __enter__(self) -> "MockPoseBackend":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def detect(self, rgb_frame: Any | None, *, frame_index: int) -> PoseDetection:
        return PoseDetection(
            landmarks=_mock_landmarks(frame_index),
            world_landmarks={},
            source=self.source,
        )
