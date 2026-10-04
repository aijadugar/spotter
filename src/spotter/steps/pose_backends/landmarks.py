from __future__ import annotations

from typing import Any


LANDMARK_SCHEMA = "coco17"

LANDMARK_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

MEDIAPIPE_COCO17_INDICES = [
    0,  # nose
    2,  # left_eye
    5,  # right_eye
    7,  # left_ear
    8,  # right_ear
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    15,  # left_wrist
    16,  # right_wrist
    23,  # left_hip
    24,  # right_hip
    25,  # left_knee
    26,  # right_knee
    27,  # left_ankle
    28,  # right_ankle
]


def landmark_to_dict(landmark: Any) -> dict[str, float]:
    presence = getattr(landmark, "presence", None)
    visibility = getattr(landmark, "visibility", presence if presence is not None else 1.0)
    payload = {
        "x": round(float(landmark.x), 6),
        "y": round(float(landmark.y), 6),
        "z": round(float(landmark.z), 6),
        "visibility": round(float(visibility), 6),
    }
    if presence is not None:
        payload["presence"] = round(float(presence), 6)
    return payload


def landmark_list_to_dict(landmarks: Any | None) -> dict[str, dict[str, float]]:
    if landmarks is None:
        return {}
    landmark_values = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
    landmark_values = list(landmark_values)
    if len(landmark_values) >= 33:
        landmark_values = [landmark_values[index] for index in MEDIAPIPE_COCO17_INDICES]
    return {
        name: landmark_to_dict(landmark)
        for name, landmark in zip(LANDMARK_NAMES, landmark_values, strict=False)
    }
