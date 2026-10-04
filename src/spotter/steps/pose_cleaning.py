from __future__ import annotations

from spotter.contracts import PoseFrame, PoseSequence


MAX_INTERPOLATION_GAP = 3
SMOOTHING_ALPHA = 0.45
SMOOTHED_FIELDS = ("x", "y", "z")


def _copy_landmarks(
    landmarks: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {name: dict(values) for name, values in landmarks.items()}


def _copy_frame(
    frame: PoseFrame,
    landmarks: dict[str, dict[str, float]] | None = None,
    world_landmarks: dict[str, dict[str, float]] | None = None,
) -> PoseFrame:
    return PoseFrame(
        frame_index=frame.frame_index,
        timestamp_sec=frame.timestamp_sec,
        landmarks=_copy_landmarks(landmarks if landmarks is not None else frame.landmarks),
        world_landmarks=_copy_landmarks(
            world_landmarks if world_landmarks is not None else frame.world_landmarks
        ),
        pose_quality=dict(frame.pose_quality),
    )


def _interpolate_landmarks(
    start: dict[str, dict[str, float]],
    end: dict[str, dict[str, float]],
    fraction: float,
) -> dict[str, dict[str, float]]:
    interpolated: dict[str, dict[str, float]] = {}
    for name in sorted(start.keys() & end.keys()):
        start_values = start[name]
        end_values = end[name]
        values: dict[str, float] = {}
        for field in ("x", "y", "z", "visibility", "presence"):
            if field in start_values and field in end_values:
                values[field] = round(
                    float(start_values[field])
                    + (float(end_values[field]) - float(start_values[field])) * fraction,
                    6,
                )
        if values:
            interpolated[name] = values
    return interpolated


def _interpolate_short_gaps(frames: list[PoseFrame]) -> list[PoseFrame]:
    cleaned = [_copy_frame(frame) for frame in frames]
    index = 0
    while index < len(cleaned):
        if cleaned[index].landmarks:
            index += 1
            continue

        gap_start = index
        while index < len(cleaned) and not cleaned[index].landmarks:
            index += 1
        gap_end = index - 1
        previous_index = gap_start - 1
        next_index = index
        gap_size = gap_end - gap_start + 1

        if (
            previous_index < 0
            or next_index >= len(cleaned)
            or gap_size > MAX_INTERPOLATION_GAP
            or not cleaned[previous_index].landmarks
            or not cleaned[next_index].landmarks
        ):
            continue

        for offset, frame_index in enumerate(range(gap_start, gap_end + 1), start=1):
            fraction = offset / (gap_size + 1)
            landmarks = _interpolate_landmarks(
                cleaned[previous_index].landmarks,
                cleaned[next_index].landmarks,
                fraction,
            )
            world_landmarks = _interpolate_landmarks(
                cleaned[previous_index].world_landmarks,
                cleaned[next_index].world_landmarks,
                fraction,
            )
            cleaned[frame_index] = PoseFrame(
                frame_index=cleaned[frame_index].frame_index,
                timestamp_sec=cleaned[frame_index].timestamp_sec,
                landmarks=landmarks,
                world_landmarks=world_landmarks,
                pose_quality={
                    **cleaned[frame_index].pose_quality,
                    "interpolated": bool(landmarks),
                    "interpolation_gap_frames": gap_size,
                },
            )
    return cleaned


def _add_smoothed_fields(frames: list[PoseFrame]) -> list[PoseFrame]:
    previous_landmarks: dict[str, dict[str, float]] = {}
    previous_world_landmarks: dict[str, dict[str, float]] = {}
    smoothed_frames: list[PoseFrame] = []

    for frame in frames:
        landmarks = _copy_landmarks(frame.landmarks)
        world_landmarks = _copy_landmarks(frame.world_landmarks)
        _smooth_landmarks(landmarks, previous_landmarks)
        _smooth_landmarks(world_landmarks, previous_world_landmarks)

        smoothed_frames.append(
            PoseFrame(
                frame_index=frame.frame_index,
                timestamp_sec=frame.timestamp_sec,
                landmarks=landmarks,
                world_landmarks=world_landmarks,
                pose_quality=frame.pose_quality,
            )
        )
    return smoothed_frames


def _smooth_landmarks(
    landmarks: dict[str, dict[str, float]],
    previous: dict[str, dict[str, float]],
) -> None:
    for name, values in landmarks.items():
        previous_values = previous.get(name, {})
        current_smoothed: dict[str, float] = {}
        for field in SMOOTHED_FIELDS:
            if field not in values:
                continue
            previous_value = previous_values.get(f"smoothed_{field}", values[field])
            smoothed = previous_value * (1.0 - SMOOTHING_ALPHA) + values[field] * SMOOTHING_ALPHA
            values[f"smoothed_{field}"] = round(smoothed, 6)
            current_smoothed[f"smoothed_{field}"] = values[f"smoothed_{field}"]
        previous[name] = current_smoothed


def _normalization_origin_and_scale(
    landmarks: dict[str, dict[str, float]],
) -> tuple[float, float, float, float, bool]:
    required = ("left_hip", "right_hip", "left_shoulder", "right_shoulder")
    if not all(name in landmarks for name in required):
        return 0.0, 0.0, 0.0, 1.0, False

    left_hip = landmarks["left_hip"]
    right_hip = landmarks["right_hip"]
    left_shoulder = landmarks["left_shoulder"]
    right_shoulder = landmarks["right_shoulder"]
    origin_x = (left_hip["x"] + right_hip["x"]) / 2.0
    origin_y = (left_hip["y"] + right_hip["y"]) / 2.0
    origin_z = (left_hip.get("z", 0.0) + right_hip.get("z", 0.0)) / 2.0
    mid_shoulder_x = (left_shoulder["x"] + right_shoulder["x"]) / 2.0
    mid_shoulder_y = (left_shoulder["y"] + right_shoulder["y"]) / 2.0
    mid_shoulder_z = (left_shoulder.get("z", 0.0) + right_shoulder.get("z", 0.0)) / 2.0
    torso_length = (
        (mid_shoulder_x - origin_x) ** 2
        + (mid_shoulder_y - origin_y) ** 2
        + (mid_shoulder_z - origin_z) ** 2
    ) ** 0.5
    if torso_length <= 1e-6:
        return origin_x, origin_y, origin_z, 1.0, False
    return origin_x, origin_y, origin_z, torso_length, True


def _vertical_sign(frames: list[PoseFrame], *, use_world_landmarks: bool) -> float:
    for frame in frames:
        landmarks = frame.world_landmarks if use_world_landmarks else frame.landmarks
        required = ("left_hip", "right_hip", "left_shoulder", "right_shoulder")
        if not all(name in landmarks for name in required):
            continue
        origin_y = (landmarks["left_hip"]["y"] + landmarks["right_hip"]["y"]) / 2.0
        shoulder_y = (
            landmarks["left_shoulder"]["y"] + landmarks["right_shoulder"]["y"]
        ) / 2.0
        return -1.0 if shoulder_y > origin_y else 1.0
    return 1.0


def _add_normalized_landmarks(
    landmarks: dict[str, dict[str, float]],
    *,
    vertical_sign: float,
) -> tuple[dict[str, dict[str, float]], bool]:
    normalized_landmarks = _copy_landmarks(landmarks)
    origin_x, origin_y, origin_z, scale, normalized = _normalization_origin_and_scale(
        normalized_landmarks
    )
    for values in normalized_landmarks.values():
        source_x = values.get("smoothed_x", values.get("x"))
        source_y = values.get("smoothed_y", values.get("y"))
        source_z = values.get("smoothed_z", values.get("z"))
        if source_x is None or source_y is None or source_z is None:
            continue
        values["normalized_x"] = round((source_x - origin_x) / scale, 6)
        values["normalized_y"] = round(((source_y - origin_y) * vertical_sign) / scale, 6)
        values["normalized_z"] = round((source_z - origin_z) / scale, 6)
    return normalized_landmarks, normalized


def _add_normalized_fields(frames: list[PoseFrame]) -> list[PoseFrame]:
    normalized_frames: list[PoseFrame] = []
    landmark_vertical_sign = _vertical_sign(frames, use_world_landmarks=False)
    world_vertical_sign = _vertical_sign(frames, use_world_landmarks=True)
    for frame in frames:
        landmarks, landmarks_normalized = _add_normalized_landmarks(
            frame.landmarks,
            vertical_sign=landmark_vertical_sign,
        )
        world_landmarks, world_normalized = _add_normalized_landmarks(
            frame.world_landmarks,
            vertical_sign=world_vertical_sign,
        )
        normalized = world_normalized if world_landmarks else landmarks_normalized

        normalized_frames.append(
            PoseFrame(
                frame_index=frame.frame_index,
                timestamp_sec=frame.timestamp_sec,
                landmarks=landmarks,
                world_landmarks=world_landmarks,
                pose_quality={
                    **frame.pose_quality,
                    "cleaned": True,
                    "normalized": normalized,
                    "normalization_origin": "mid_hip",
                    "normalization_scale": "mid_shoulder_to_mid_hip",
                    "world_landmarks_normalized": world_normalized,
                },
            )
        )
    return normalized_frames


def run(sequence: PoseSequence) -> PoseSequence:
    interpolated_frames = _interpolate_short_gaps(sequence.frames)
    smoothed_frames = _add_smoothed_fields(interpolated_frames)
    cleaned_frames = _add_normalized_fields(smoothed_frames)
    valid_frames = sum(1 for frame in cleaned_frames if frame.landmarks)

    return PoseSequence(
        frames=cleaned_frames,
        normalized=True,
        smoothing_method="exponential_smoothing",
        pose_valid_ratio=round(valid_frames / len(cleaned_frames), 4) if cleaned_frames else 0.0,
    )
