from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2

from spotter.contracts import VideoManifest


MIN_DURATION_SEC = 10.0
MAX_DURATION_SEC = 60.0
MIN_FPS = 15.0
MIN_WIDTH = 480
MIN_HEIGHT = 360
MIN_BRIGHTNESS = 45.0
MIN_BLUR_LAPLACIAN_VAR = 50.0
DEFAULT_SAMPLE_COUNT = 12

HARD_FAILURE_WARNINGS = {"video_decode_failed", "too_short", "too_long"}


def enable_capture_orientation(capture: cv2.VideoCapture) -> None:
    orientation_auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if orientation_auto is not None:
        capture.set(orientation_auto, 1)


def _decode_fourcc(value: float) -> str | None:
    code = int(value)
    if code <= 0:
        return None
    chars = [chr((code >> 8 * index) & 0xFF) for index in range(4)]
    decoded = "".join(chars).strip()
    return decoded or None


def _container_from_path(video_path: str | None) -> str | None:
    if video_path is None:
        return None
    suffix = Path(video_path).suffix.lower().lstrip(".")
    return suffix or None


def sample_frame_indices(total_frames: int, sample_count: int = DEFAULT_SAMPLE_COUNT) -> list[int]:
    if total_frames <= 0 or sample_count <= 0:
        return []
    if total_frames <= sample_count:
        return list(range(total_frames))

    last_index = total_frames - 1
    return sorted({round(index * last_index / (sample_count - 1)) for index in range(sample_count)})


def sample_video_frames(
    video_path: str,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
) -> Iterator[tuple[int, object]]:
    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            return

        enable_capture_orientation(capture)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for frame_index in sample_frame_indices(total_frames, sample_count):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if ok and frame is not None:
                yield frame_index, frame
    finally:
        capture.release()


def _empty_manifest(video_path: str | None, warnings: list[str]) -> VideoManifest:
    return VideoManifest(
        video_path=video_path,
        fps=0.0,
        duration_sec=0.0,
        total_frames=0,
        sampled_frames=0,
        width=0,
        height=0,
        codec=None,
        container=_container_from_path(video_path),
        brightness_mean=None,
        blur_laplacian_var=None,
        quality_warnings=warnings,
        analysis_allowed=False,
    )


def _brightness_and_blur(frame: object) -> tuple[float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brightness, blur


def run(video_path: str | None) -> VideoManifest:
    if video_path is None or not Path(video_path).exists():
        return _empty_manifest(video_path, ["video_decode_failed"])

    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            return _empty_manifest(video_path, ["video_decode_failed"])

        enable_capture_orientation(capture)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        codec = _decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC) or 0.0)
    finally:
        capture.release()

    if total_frames <= 0 or fps <= 0:
        return _empty_manifest(video_path, ["video_decode_failed"])

    sampled_metrics = [_brightness_and_blur(frame) for _, frame in sample_video_frames(video_path)]
    if not sampled_metrics:
        return _empty_manifest(video_path, ["video_decode_failed"])

    brightness_values = [brightness for brightness, _ in sampled_metrics]
    blur_values = [blur for _, blur in sampled_metrics]
    brightness_mean = round(sum(brightness_values) / len(brightness_values), 2)
    blur_laplacian_var = round(sum(blur_values) / len(blur_values), 2)
    duration_sec = round(total_frames / fps, 3)

    warnings: list[str] = []
    if duration_sec < MIN_DURATION_SEC:
        warnings.append("too_short")
    if duration_sec > MAX_DURATION_SEC:
        warnings.append("too_long")
    if brightness_mean < MIN_BRIGHTNESS:
        warnings.append("too_dark")
    if blur_laplacian_var < MIN_BLUR_LAPLACIAN_VAR:
        warnings.append("too_blurry")
    if fps < MIN_FPS:
        warnings.append("fps_too_low")
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        warnings.append("resolution_too_low")

    return VideoManifest(
        video_path=video_path,
        fps=round(fps, 3),
        duration_sec=duration_sec,
        total_frames=total_frames,
        sampled_frames=len(sampled_metrics),
        width=width,
        height=height,
        codec=codec,
        container=_container_from_path(video_path),
        brightness_mean=brightness_mean,
        blur_laplacian_var=blur_laplacian_var,
        quality_warnings=warnings,
        analysis_allowed=not any(warning in HARD_FAILURE_WARNINGS for warning in warnings),
    )
