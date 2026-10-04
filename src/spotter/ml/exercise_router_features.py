from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from spotter.contracts import PoseFrame, PoseSequence
from spotter.steps.pose_backends.landmarks import LANDMARK_NAMES, LANDMARK_SCHEMA
from spotter.steps.rep_signals import landmark_axis


FEATURE_SCHEMA = "coco17_3d_v1"
ROUTER_LANDMARK_SCHEMA = LANDMARK_SCHEMA
ROUTER_INPUT_SIZE = 3 * (len(LANDMARK_NAMES) * 4 + 8 + 3)


ROUTER_LABELS = ("squat", "push_up", "shoulder_press", "unknown")
WINDOW_SIZE_FRAMES = 30
WINDOW_STRIDE_FRAMES = 15
MIN_WINDOW_MEAN_VISIBILITY = 0.2

ANGLE_TRIPLES = (
    ("left_knee_angle", "left_hip", "left_knee", "left_ankle"),
    ("right_knee_angle", "right_hip", "right_knee", "right_ankle"),
    ("left_hip_angle", "left_shoulder", "left_hip", "left_knee"),
    ("right_hip_angle", "right_shoulder", "right_hip", "right_knee"),
    ("left_elbow_angle", "left_shoulder", "left_elbow", "left_wrist"),
    ("right_elbow_angle", "right_shoulder", "right_elbow", "right_wrist"),
    ("left_shoulder_angle", "left_hip", "left_shoulder", "left_elbow"),
    ("right_shoulder_angle", "right_hip", "right_shoulder", "right_elbow"),
)

RELATIVE_DISTANCE_FEATURES = (
    "hand_width_over_shoulder_width",
    "stance_width_over_shoulder_width",
    "shoulder_width_over_hip_width",
)

LABEL_ALIASES = {
    "squat": "squat",
    "squats": "squat",
    "pushup": "push_up",
    "pushups": "push_up",
    "push-up": "push_up",
    "push-ups": "push_up",
    "push_up": "push_up",
    "push_ups": "push_up",
    "shoulderpress": "shoulder_press",
    "shoulder-press": "shoulder_press",
    "shoulder_press": "shoulder_press",
    "shoulder_presses": "shoulder_press",
    "overhead_press": "shoulder_press",
    "bicep_curl": "unknown",
    "biceps_curl": "unknown",
    "barbell_bicep_curl": "unknown",
    "barbell_biceps_curl": "unknown",
    "curl": "unknown",
    "unknown": "unknown",
}


@dataclass(frozen=True)
class RouterWindow:
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    mean_visibility: float
    tensor: np.ndarray
    vector: np.ndarray


def normalize_router_label(value: str | None) -> str:
    if value is None:
        return "unknown"
    normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
    return LABEL_ALIASES.get(normalized, "unknown")


def frame_feature_names() -> list[str]:
    base_names: list[str] = []
    for landmark in LANDMARK_NAMES:
        base_names.extend(
            [
                f"{landmark}_pose3d_x",
                f"{landmark}_pose3d_y",
                f"{landmark}_pose3d_z",
                f"{landmark}_visibility",
            ]
        )
    base_names.extend(name for name, *_ in ANGLE_TRIPLES)
    base_names.extend(RELATIVE_DISTANCE_FEATURES)
    return base_names


def window_tensor_feature_names() -> list[str]:
    base_names = frame_feature_names()
    return [
        *base_names,
        *(f"delta_{name}" for name in base_names),
        *(f"velocity_{name}" for name in base_names),
    ]


def window_vector_feature_names() -> list[str]:
    tensor_names = window_tensor_feature_names()
    return [
        *(f"mean_{name}" for name in tensor_names),
        *(f"std_{name}" for name in tensor_names),
        *(f"min_{name}" for name in tensor_names),
        *(f"max_{name}" for name in tensor_names),
        *(f"range_{name}" for name in tensor_names),
        *(f"trend_{name}" for name in tensor_names),
    ]


def _axis(values: dict[str, float] | None, axis: str) -> float:
    if values is None:
        return 0.0
    return float(
        values.get(
            f"normalized_{axis}",
            values.get(f"smoothed_{axis}", values.get(axis, 0.0)),
        )
    )


def _visibility(frame: PoseFrame, values: dict[str, float] | None) -> float:
    if values is None:
        return 0.0
    visibility = values.get("visibility", values.get("presence", 1.0))
    return max(0.0, min(1.0, float(visibility)))


def _point(frame: PoseFrame, name: str) -> tuple[float, float, float] | None:
    x = landmark_axis(frame, name, "x")
    y = landmark_axis(frame, name, "y")
    z = landmark_axis(frame, name, "z")
    if None in {x, y, z}:
        return None
    return float(x), float(y), float(z)


def _distance(frame: PoseFrame, first: str, second: str) -> float | None:
    first_point = _point(frame, first)
    second_point = _point(frame, second)
    if first_point is None or second_point is None:
        return None
    return math.sqrt(sum((first_point[index] - second_point[index]) ** 2 for index in range(3)))


def _safe_ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None or denominator <= 1e-6:
        return 0.0
    return float(numerator / denominator)


def _angle_deg(frame: PoseFrame, first: str, middle: str, last: str) -> float:
    first_point = _point(frame, first)
    middle_point = _point(frame, middle)
    last_point = _point(frame, last)
    if first_point is None or middle_point is None or last_point is None:
        return 0.0

    abx = first_point[0] - middle_point[0]
    aby = first_point[1] - middle_point[1]
    abz = first_point[2] - middle_point[2]
    cbx = last_point[0] - middle_point[0]
    cby = last_point[1] - middle_point[1]
    cbz = last_point[2] - middle_point[2]
    denom = math.sqrt(abx * abx + aby * aby + abz * abz) * math.sqrt(
        cbx * cbx + cby * cby + cbz * cbz
    )
    if denom <= 1e-6:
        return 0.0
    cosine = max(-1.0, min(1.0, (abx * cbx + aby * cby + abz * cbz) / denom))
    return math.degrees(math.acos(cosine))


def _frame_mean_visibility(frame: PoseFrame) -> float:
    if not frame.landmarks:
        return 0.0
    return sum(_visibility(frame, values) for values in frame.landmarks.values()) / len(frame.landmarks)


def frame_feature_vector(frame: PoseFrame) -> np.ndarray:
    values: list[float] = []
    for landmark in LANDMARK_NAMES:
        landmark_values = frame.world_landmarks.get(landmark) or frame.landmarks.get(landmark)
        point = _point(frame, landmark)
        values.extend(
            [
                point[0] if point is not None else 0.0,
                point[1] if point is not None else 0.0,
                point[2] if point is not None else 0.0,
                _visibility(frame, landmark_values),
            ]
        )

    for _, first, middle, last in ANGLE_TRIPLES:
        values.append(_angle_deg(frame, first, middle, last) / 180.0)

    shoulder_width = _distance(frame, "left_shoulder", "right_shoulder")
    hip_width = _distance(frame, "left_hip", "right_hip")
    values.extend(
        [
            _safe_ratio(_distance(frame, "left_wrist", "right_wrist"), shoulder_width),
            _safe_ratio(_distance(frame, "left_ankle", "right_ankle"), shoulder_width),
            _safe_ratio(shoulder_width, hip_width),
        ]
    )
    return np.asarray(values, dtype=np.float32)


def _window_tensor(frames: list[PoseFrame]) -> np.ndarray:
    base = np.vstack([frame_feature_vector(frame) for frame in frames]).astype(np.float32)
    deltas = np.zeros_like(base)
    deltas[1:] = base[1:] - base[:-1]

    velocities = np.zeros_like(base)
    for index in range(1, len(frames)):
        elapsed = frames[index].timestamp_sec - frames[index - 1].timestamp_sec
        if elapsed <= 1e-6:
            elapsed = 1.0
        velocities[index] = deltas[index] / elapsed

    return np.concatenate([base, deltas, velocities], axis=1).astype(np.float32)


def _window_vector(tensor: np.ndarray) -> np.ndarray:
    feature_range = np.max(tensor, axis=0) - np.min(tensor, axis=0)
    trend = tensor[-1] - tensor[0]
    return np.concatenate(
        [
            np.mean(tensor, axis=0),
            np.std(tensor, axis=0),
            np.min(tensor, axis=0),
            np.max(tensor, axis=0),
            feature_range,
            trend,
        ]
    ).astype(np.float32)


def _window_starts(frame_count: int, window_size: int, stride: int) -> list[int]:
    if frame_count < window_size:
        return []
    starts = list(range(0, frame_count - window_size + 1, stride))
    final_start = frame_count - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def extract_router_windows(
    sequence: PoseSequence,
    *,
    window_size: int = WINDOW_SIZE_FRAMES,
    stride: int = WINDOW_STRIDE_FRAMES,
    min_mean_visibility: float = MIN_WINDOW_MEAN_VISIBILITY,
) -> list[RouterWindow]:
    windows: list[RouterWindow] = []
    frames = sequence.frames
    for start in _window_starts(len(frames), window_size, stride):
        window_frames = frames[start : start + window_size]
        mean_visibility = sum(_frame_mean_visibility(frame) for frame in window_frames) / len(
            window_frames
        )
        if mean_visibility < min_mean_visibility:
            continue

        tensor = _window_tensor(window_frames)
        windows.append(
            RouterWindow(
                start_frame=window_frames[0].frame_index,
                end_frame=window_frames[-1].frame_index,
                start_sec=round(window_frames[0].timestamp_sec, 3),
                end_sec=round(window_frames[-1].timestamp_sec, 3),
                mean_visibility=round(float(mean_visibility), 4),
                tensor=tensor,
                vector=_window_vector(tensor),
            )
        )
    return windows
