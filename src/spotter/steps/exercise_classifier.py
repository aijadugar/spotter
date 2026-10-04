from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from spotter.contracts import ExerciseClassification, PoseFrame, PoseSequence, UserProfile
from spotter.exercise_catalog import DEFAULT_AUTO_EXERCISE
from spotter.hf_spaces import default_spaces_gpu_duration, spaces_gpu
from spotter.ml.exercise_router_features import RouterWindow, extract_router_windows
from spotter.ml.exercise_router_inference import (
    MIN_POSE_VALID_RATIO,
    RouterModelBundle,
    aggregate_window_predictions,
    contract_window_predictions,
    load_router_model,
    predict_window_probabilities,
    window_predictions_from_scores,
)
from spotter.steps.rep_signals import angle_deg, smooth_signal


MIN_HEURISTIC_CONFIDENCE = 0.58
MIN_HEURISTIC_MARGIN = 0.14


def _sample_prediction_frames(sequence: PoseSequence, count: int = 4) -> list[object]:
    if not sequence.frames:
        return []
    if len(sequence.frames) <= count:
        return sequence.frames
    last_index = len(sequence.frames) - 1
    positions = sorted({round(index * last_index / (count - 1)) for index in range(count)})
    return [sequence.frames[position] for position in positions]


def _fixed_classification(
    sequence: PoseSequence,
    *,
    exercise: str,
    confidence: float,
    fallback_required: bool,
) -> ExerciseClassification:
    return ExerciseClassification(
        exercise=exercise,  # type: ignore[arg-type]
        confidence=confidence,
        window_predictions=[
            {
                "start_sec": frame.timestamp_sec,
                "end_sec": round(frame.timestamp_sec + 1.0, 3),
                "label": exercise,
                "confidence": confidence,
            }
            for frame in _sample_prediction_frames(sequence)
        ],
        fallback_required=fallback_required,
    )


def _manual_classification(sequence: PoseSequence, exercise: str) -> ExerciseClassification:
    windows = extract_router_windows(sequence, min_mean_visibility=0.0)
    if not windows:
        return _fixed_classification(
            sequence,
            exercise=exercise,
            confidence=0.98,
            fallback_required=False,
        )
    return ExerciseClassification(
        exercise=exercise,  # type: ignore[arg-type]
        confidence=0.98,
        window_predictions=[
            {
                "start_sec": window.start_sec,
                "end_sec": window.end_sec,
                "label": exercise,
                "confidence": 0.98,
            }
            for window in windows
        ],
        fallback_required=False,
    )


def _unknown_fallback(
    window_predictions: list[dict[str, float | str]] | None = None,
    *,
    confidence: float = 0.0,
) -> ExerciseClassification:
    return ExerciseClassification(
        exercise="unknown",
        confidence=round(confidence, 4),
        window_predictions=window_predictions or [],
        fallback_required=True,
    )


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _usable(values: Iterable[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None]


def _range(values: Iterable[float | None]) -> float:
    usable_values = _usable(smooth_signal(list(values)))
    if not usable_values:
        return 0.0
    return max(usable_values) - min(usable_values)


def _raw_axis(frame: PoseFrame, name: str, axis: str) -> float | None:
    values = frame.landmarks.get(name) or frame.world_landmarks.get(name)
    if values is None:
        return None
    value = values.get(
        f"smoothed_{axis}",
        values.get(axis, values.get(f"normalized_{axis}")),
    )
    return None if value is None else float(value)


def _mean_raw_axis(frame: PoseFrame, names: tuple[str, ...], axis: str) -> float | None:
    values = _usable(_raw_axis(frame, name, axis) for name in names)
    if not values:
        return None
    return sum(values) / len(values)


def _mean_abs_y_gap(
    frames: list[PoseFrame],
    first: tuple[str, ...],
    second: tuple[str, ...],
) -> float:
    gaps: list[float] = []
    for frame in frames:
        first_y = _mean_raw_axis(frame, first, "y")
        second_y = _mean_raw_axis(frame, second, "y")
        if first_y is None or second_y is None:
            continue
        gaps.append(abs(first_y - second_y))
    return sum(gaps) / len(gaps) if gaps else 0.0


def _joint_bend_deg(frame: PoseFrame, triples: tuple[tuple[str, str, str], ...]) -> float | None:
    values: list[float] = []
    for first, middle, last in triples:
        angle = angle_deg(frame, first, middle, last)
        if angle is not None:
            values.append(max(0.0, 180.0 - angle))
    return sum(values) / len(values) if values else None


def _heuristic_score_rows(sequence: PoseSequence) -> dict[str, float]:
    frames = [frame for frame in sequence.frames if frame.landmarks or frame.world_landmarks]
    if len(frames) < 9:
        return {"squat": 0.0, "push_up": 0.0, "shoulder_press": 0.0}

    shoulder_y = [
        _mean_raw_axis(frame, ("left_shoulder", "right_shoulder"), "y") for frame in frames
    ]
    hip_y = [_mean_raw_axis(frame, ("left_hip", "right_hip"), "y") for frame in frames]
    wrist_y = [_mean_raw_axis(frame, ("left_wrist", "right_wrist"), "y") for frame in frames]
    knee_bend = [
        _joint_bend_deg(
            frame,
            (
                ("left_hip", "left_knee", "left_ankle"),
                ("right_hip", "right_knee", "right_ankle"),
            ),
        )
        for frame in frames
    ]
    hip_bend = [
        _joint_bend_deg(
            frame,
            (
                ("left_shoulder", "left_hip", "left_knee"),
                ("right_shoulder", "right_hip", "right_knee"),
            ),
        )
        for frame in frames
    ]
    elbow_bend = [
        _joint_bend_deg(
            frame,
            (
                ("left_shoulder", "left_elbow", "left_wrist"),
                ("right_shoulder", "right_elbow", "right_wrist"),
            ),
        )
        for frame in frames
    ]

    shoulder_range = _range(shoulder_y)
    hip_range = _range(hip_y)
    wrist_range = _range(wrist_y)
    chest_range = (shoulder_range + hip_range) / 2.0
    knee_bend_range = _range(knee_bend)
    hip_bend_range = _range(hip_bend)
    elbow_bend_range = _range(elbow_bend)
    shoulder_hip_gap = _mean_abs_y_gap(
        frames,
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
    )
    hip_ankle_gap = _mean_abs_y_gap(
        frames,
        ("left_hip", "right_hip"),
        ("left_ankle", "right_ankle"),
    )

    standing_score = _clip01((shoulder_hip_gap + hip_ankle_gap) / 0.45)
    plank_score = (
        _clip01((0.28 - shoulder_hip_gap) / 0.28) + _clip01((0.24 - hip_ankle_gap) / 0.24)
    ) / 2.0
    wrist_not_dominant = _clip01(1.0 - wrist_range / max(hip_range + 0.05, 0.05))
    wrist_stable_for_push = _clip01(1.0 - wrist_range / max(chest_range + 0.04, 0.04))
    body_still_for_press = _clip01(1.0 - max(shoulder_range, hip_range) / max(wrist_range, 0.05))
    lower_body_bend = max(_clip01(knee_bend_range / 45.0), _clip01(hip_bend_range / 45.0))
    push_up_geometry = 0.55 + 0.45 * plank_score
    press_lower_body_gate = 1.0 - 0.55 * lower_body_bend

    squat_score = standing_score * (
        0.42 * _clip01(knee_bend_range / 45.0)
        + 0.24 * _clip01(hip_bend_range / 45.0)
        + 0.22 * _clip01(hip_range / 0.10)
        + 0.12 * wrist_not_dominant
    )
    push_up_score = (
        0.45 * _clip01(elbow_bend_range / 55.0)
        + 0.25 * _clip01(chest_range / 0.08)
        + 0.20 * plank_score
        + 0.10 * wrist_stable_for_push
    ) * push_up_geometry
    shoulder_press_score = press_lower_body_gate * (
        0.40 * _clip01(wrist_range / 0.16)
        + 0.25 * _clip01(elbow_bend_range / 55.0)
        + 0.20 * body_still_for_press
        + 0.15 * standing_score
    )
    return {
        "squat": round(_clip01(squat_score), 4),
        "push_up": round(_clip01(push_up_score), 4),
        "shoulder_press": round(_clip01(shoulder_press_score), 4),
    }


def _heuristic_classification(
    sequence: PoseSequence,
    windows: list[RouterWindow],
) -> ExerciseClassification:
    scores = _heuristic_score_rows(sequence)
    ranked = sorted(scores, key=lambda label: scores[label], reverse=True)
    winning_label = ranked[0]
    score_margin = scores[winning_label] - scores[ranked[1]]
    if scores[winning_label] < MIN_HEURISTIC_CONFIDENCE or score_margin < MIN_HEURISTIC_MARGIN:
        return _unknown_fallback(confidence=scores[winning_label])

    confidence = round(
        min(0.88, 0.55 + scores[winning_label] * 0.25 + min(score_margin, 0.4) * 0.20),
        4,
    )
    if not windows:
        return _fixed_classification(
            sequence,
            exercise=winning_label,
            confidence=confidence,
            fallback_required=False,
        )
    return ExerciseClassification(
        exercise=winning_label,  # type: ignore[arg-type]
        confidence=confidence,
        window_predictions=[
            {
                "start_sec": window.start_sec,
                "end_sec": window.end_sec,
                "label": winning_label,
                "confidence": confidence,
            }
            for window in windows
        ],
        fallback_required=False,
    )


def _gpu_duration(*_args: object, **_kwargs: object) -> int:
    return default_spaces_gpu_duration()


@spaces_gpu(duration=_gpu_duration)
def _predict_router_scores(
    windows: list[RouterWindow],
    model_dir: str,
) -> list[dict[str, float]]:
    # load_router_model is now a singleton loader; model_dir is kept for compat
    bundle = load_router_model()
    if bundle is None:
        return []
    return predict_window_probabilities(bundle, windows)


def run(
    sequence: PoseSequence,
    profile: UserProfile,
    *,
    mock: bool = False,
    model_bundle: RouterModelBundle | None = None,
    model_dir: Path | None = None,
) -> ExerciseClassification:
    if profile.intended_exercise != "auto":
        return _manual_classification(sequence, profile.intended_exercise)

    if mock:
        return _fixed_classification(
            sequence,
            exercise=DEFAULT_AUTO_EXERCISE,
            confidence=0.92,
            fallback_required=False,
        )

    windows = extract_router_windows(sequence)
    if not windows or sequence.pose_valid_ratio < MIN_POSE_VALID_RATIO:
        return _unknown_fallback()

    try:
        if model_bundle is not None:
            score_rows = predict_window_probabilities(model_bundle, windows)
        else:
            score_rows = _predict_router_scores(
                windows,
                str(model_dir or Path("models/exercise_router/active")),
            )
            if not score_rows:
                return _heuristic_classification(sequence, windows)
    except Exception:
        return _heuristic_classification(sequence, windows)

    predictions = window_predictions_from_scores(windows, score_rows)
    window_payload = contract_window_predictions(predictions)
    aggregated = aggregate_window_predictions(predictions)
    return ExerciseClassification(
        exercise=aggregated.label,  # type: ignore[arg-type]
        confidence=aggregated.confidence,
        window_predictions=window_payload,
        fallback_required=aggregated.fallback_required,
    )
