from __future__ import annotations

from statistics import pstdev
from typing import Any, Callable, Protocol

from spotter.contracts import PoseFrame
from spotter.steps.rep_signals import angle_deg, average_axis, distance, landmark_axis


NumberGetter = Callable[[PoseFrame], float | None]
ExerciseMetricResult = tuple[dict[str, Any], float, float, float, list[str]]


class ExerciseAnalyzer(Protocol):
    def metrics(self, frames: list[PoseFrame]) -> ExerciseMetricResult:
        ...


def round_optional(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def score(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 2)


def usable(values: list[float | None]) -> list[float]:
    return [value for value in values if value is not None]


def mean_optional(values: list[float | None]) -> float | None:
    usable_values = usable(values)
    if not usable_values:
        return None
    return sum(usable_values) / len(usable_values)


def min_optional(values: list[float | None]) -> float | None:
    usable_values = usable(values)
    return min(usable_values) if usable_values else None


def max_optional(values: list[float | None]) -> float | None:
    usable_values = usable(values)
    return max(usable_values) if usable_values else None


def range_optional(values: list[float | None]) -> float | None:
    usable_values = usable(values)
    if not usable_values:
        return None
    return max(usable_values) - min(usable_values)


def std_optional(values: list[float | None]) -> float | None:
    usable_values = usable(values)
    if len(usable_values) < 2:
        return 0.0 if usable_values else None
    return pstdev(usable_values)


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) <= 1e-6:
        return None
    return numerator / denominator


def width(frame: PoseFrame, left: str, right: str) -> float | None:
    return distance(frame, left, right)


def mean_pair(
    frame: PoseFrame,
    first: tuple[str, str, str],
    second: tuple[str, str, str],
) -> float | None:
    values = [angle_deg(frame, *first), angle_deg(frame, *second)]
    return mean_optional(values)


def side_delta(
    frame: PoseFrame,
    first: tuple[str, str, str],
    second: tuple[str, str, str],
) -> float | None:
    first_value = angle_deg(frame, *first)
    second_value = angle_deg(frame, *second)
    if first_value is None or second_value is None:
        return None
    return abs(first_value - second_value)


def torso_lean_deg(frame: PoseFrame, side: str) -> float | None:
    shoulder_x = landmark_axis(frame, f"{side}_shoulder", "x")
    shoulder_y = landmark_axis(frame, f"{side}_shoulder", "y")
    shoulder_z = landmark_axis(frame, f"{side}_shoulder", "z")
    hip_x = landmark_axis(frame, f"{side}_hip", "x")
    hip_y = landmark_axis(frame, f"{side}_hip", "y")
    hip_z = landmark_axis(frame, f"{side}_hip", "z")
    if None in {shoulder_x, shoulder_y, shoulder_z, hip_x, hip_y, hip_z}:
        return None
    horizontal_offset = (
        (float(shoulder_x) - float(hip_x)) ** 2
        + (float(shoulder_z) - float(hip_z)) ** 2
    ) ** 0.5
    vertical_offset = abs(float(shoulder_y) - float(hip_y))
    if vertical_offset <= 1e-6:
        return None
    from math import atan2, degrees

    return degrees(atan2(horizontal_offset, vertical_offset))


def value_series(frames: list[PoseFrame], getter: NumberGetter) -> list[float | None]:
    return [getter(frame) for frame in frames]


def mean_visibility(frames: list[PoseFrame]) -> float:
    values: list[float | None] = []
    for frame in frames:
        if "mean_visibility" in frame.pose_quality:
            values.append(float(frame.pose_quality["mean_visibility"]))
            continue
        landmark_values = [
            landmark.get("visibility")
            for landmark in frame.landmarks.values()
            if landmark.get("visibility") is not None
        ]
        values.extend(float(value) for value in landmark_values)
    return score(mean_optional(values) if values else 0.0)


def average_y(frame: PoseFrame, names: tuple[str, ...]) -> float | None:
    return average_axis(frame, names, "y")
