from __future__ import annotations

from dataclasses import dataclass
import math

from spotter.contracts import PoseFrame, PoseSequence


@dataclass(frozen=True)
class SignalSample:
    frame_index: int
    timestamp_sec: float
    value: float


def _landmark_values(frame: PoseFrame, name: str) -> dict[str, float] | None:
    return frame.world_landmarks.get(name) or frame.landmarks.get(name)


def landmark_axis(frame: PoseFrame, name: str, axis: str) -> float | None:
    values = _landmark_values(frame, name)
    if values is None:
        return None
    fallback = 0.0 if axis == "z" else None
    return values.get(
        f"normalized_{axis}",
        values.get(f"smoothed_{axis}", values.get(axis, fallback)),
    )


def average_axis(frame: PoseFrame, names: tuple[str, ...], axis: str) -> float | None:
    values = [landmark_axis(frame, name, axis) for name in names]
    usable_values = [value for value in values if value is not None]
    if not usable_values:
        return None
    return sum(usable_values) / len(usable_values)


def angle_deg(frame: PoseFrame, first: str, middle: str, last: str) -> float | None:
    ax = landmark_axis(frame, first, "x")
    ay = landmark_axis(frame, first, "y")
    az = landmark_axis(frame, first, "z")
    bx = landmark_axis(frame, middle, "x")
    by = landmark_axis(frame, middle, "y")
    bz = landmark_axis(frame, middle, "z")
    cx = landmark_axis(frame, last, "x")
    cy = landmark_axis(frame, last, "y")
    cz = landmark_axis(frame, last, "z")
    if None in {ax, ay, az, bx, by, bz, cx, cy, cz}:
        return None

    abx = ax - bx
    aby = ay - by
    abz = az - bz
    cbx = cx - bx
    cby = cy - by
    cbz = cz - bz
    ab_length = math.sqrt(abx * abx + aby * aby + abz * abz)
    cb_length = math.sqrt(cbx * cbx + cby * cby + cbz * cbz)
    denom = ab_length * cb_length
    if denom <= 1e-6:
        return None
    cosine = max(-1.0, min(1.0, (abx * cbx + aby * cby + abz * cbz) / denom))
    return math.degrees(math.acos(cosine))


def body_line_score(frame: PoseFrame) -> float | None:
    shoulder = average_point(frame, ("left_shoulder", "right_shoulder"))
    hip = average_point(frame, ("left_hip", "right_hip"))
    ankle = average_point(frame, ("left_ankle", "right_ankle"))
    if shoulder is None or hip is None or ankle is None:
        return None
    midline = tuple((shoulder[index] + ankle[index]) / 2.0 for index in range(3))
    deviation = math.sqrt(sum((midline[index] - hip[index]) ** 2 for index in range(3)))
    return 1.0 - deviation


def average_point(frame: PoseFrame, names: tuple[str, ...]) -> tuple[float, float, float] | None:
    points: list[tuple[float, float, float]] = []
    for name in names:
        x = landmark_axis(frame, name, "x")
        y = landmark_axis(frame, name, "y")
        z = landmark_axis(frame, name, "z")
        if None in {x, y, z}:
            continue
        points.append((float(x), float(y), float(z)))
    if not points:
        return None
    count = len(points)
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
        sum(point[2] for point in points) / count,
    )


def distance(frame: PoseFrame, first: str, second: str) -> float | None:
    first_point = average_point(frame, (first,))
    second_point = average_point(frame, (second,))
    if first_point is None or second_point is None:
        return None
    return math.sqrt(
        sum((first_point[index] - second_point[index]) ** 2 for index in range(3))
    )


def smooth_signal(values: list[float | None], window_radius: int = 2) -> list[float | None]:
    smoothed: list[float | None] = []
    for index, value in enumerate(values):
        if value is None:
            smoothed.append(None)
            continue
        window = values[max(0, index - window_radius) : index + window_radius + 1]
        usable_window = [item for item in window if item is not None]
        smoothed.append(sum(usable_window) / len(usable_window) if usable_window else None)
    return smoothed


def normalize_optional(values: list[float | None]) -> list[float | None]:
    usable_values = [value for value in values if value is not None]
    if not usable_values:
        return [None for _ in values]
    min_value = min(usable_values)
    max_value = max(usable_values)
    value_range = max_value - min_value
    if value_range <= 1e-6:
        return [0.0 if value is not None else None for value in values]
    return [
        None if value is None else (value - min_value) / value_range
        for value in values
    ]


def samples_from_values(sequence: PoseSequence, values: list[float | None]) -> list[SignalSample]:
    return [
        SignalSample(
            frame_index=frame.frame_index,
            timestamp_sec=frame.timestamp_sec,
            value=value,
        )
        for frame, value in zip(sequence.frames, values, strict=False)
        if value is not None
    ]
