from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class PoseDetection:
    landmarks: dict[str, dict[str, float]]
    world_landmarks: dict[str, dict[str, float]]
    source: str


class PoseBackendUnavailableError(RuntimeError):
    pass


class PoseBackend(Protocol):
    source: str

    def __enter__(self) -> "PoseBackend":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def detect(self, rgb_frame: Any | None, *, frame_index: int) -> PoseDetection:
        raise NotImplementedError
