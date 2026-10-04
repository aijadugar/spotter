from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import cv2

from spotter.contracts import IssueMarker, IssueMarkers, PoseSequence, Reps, VideoManifest


SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

PREFERRED_VIDEO_CODECS = ("mp4v", "avc1", "H264")
HDR_TRANSFERS = {"arib-std-b67", "smpte2084"}
HDR_PRIMARIES = {"bt2020"}
BT709_COLOR_ARGS = (
    "-color_primaries",
    "bt709",
    "-color_trc",
    "bt709",
    "-colorspace",
    "bt709",
)

NORMAL_EDGE_COLOR = (90, 220, 90)
NORMAL_JOINT_COLOR = (255, 240, 40)
ISSUE_EDGE_COLOR = (0, 130, 255)
ISSUE_JOINT_COLOR = (0, 40, 255)


@dataclass(frozen=True)
class RenderArtifacts:
    annotated_video_path: str | None
    issue_thumbnail_paths: list[dict[str, Any]]
    issue_clip_paths: list[dict[str, Any]]


def _tool_path(name: str) -> str | None:
    return shutil.which(name)


def _video_color_metadata(video_path: str) -> dict[str, str]:
    ffprobe = _tool_path("ffprobe")
    if ffprobe is None:
        return {}

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=color_space,color_transfer,color_primaries,color_range",
        "-of",
        "json",
        video_path,
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        payload = json.loads(result.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return {}

    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        return {}
    stream = streams[0]
    if not isinstance(stream, dict):
        return {}
    metadata: dict[str, str] = {}
    for key, value in stream.items():
        if value is None or isinstance(value, (dict, list)):
            continue
        normalized_value = str(value).lower()
        if normalized_value == "unknown":
            continue
        metadata[key] = normalized_value
    return metadata


def _needs_sdr_conversion(color_metadata: dict[str, str]) -> bool:
    transfer = color_metadata.get("color_transfer", "")
    primaries = color_metadata.get("color_primaries", "")
    return transfer in HDR_TRANSFERS or primaries in HDR_PRIMARIES


def _sdr_filter(color_metadata: dict[str, str]) -> str:
    transfer = color_metadata.get("color_transfer", "arib-std-b67")
    primaries = color_metadata.get("color_primaries", "bt2020")
    matrix = color_metadata.get("color_space", "bt2020nc")
    if transfer not in HDR_TRANSFERS:
        transfer = "arib-std-b67"
    if primaries not in HDR_PRIMARIES:
        primaries = "bt2020"
    if matrix not in {"bt2020nc", "bt2020c"}:
        matrix = "bt2020nc"

    return (
        f"zscale=transfer=linear:transferin={transfer}:"
        f"primariesin={primaries}:matrixin={matrix}:npl=100,"
        "tonemap=tonemap=hable:desat=0,"
        "zscale=transfer=bt709:primaries=bt709:matrix=bt709:range=tv,"
        "format=yuv420p"
    )


def _transcode_hdr_to_sdr(
    input_path: Path,
    output_path: Path,
    color_metadata: dict[str, str],
) -> bool:
    ffmpeg = _tool_path("ffmpeg")
    if ffmpeg is None:
        return False

    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-vf",
        _sdr_filter(color_metadata),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        *BT709_COLOR_ARGS,
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=120)
    except (subprocess.SubprocessError, OSError):
        return False
    return output_path.exists() and output_path.stat().st_size > 0


def _encode_bt709_output(
    raw_video_path: Path,
    output_path: Path,
    audio_source_path: Path | None,
) -> bool:
    ffmpeg = _tool_path("ffmpeg")
    if ffmpeg is None:
        return False

    command = _encode_bt709_command(ffmpeg, raw_video_path, output_path, audio_source_path)

    try:
        subprocess.run(command, check=True, capture_output=True, timeout=120)
    except (subprocess.SubprocessError, OSError):
        return False
    return output_path.exists() and output_path.stat().st_size > 0


def _encode_bt709_command(
    ffmpeg: str,
    raw_video_path: Path,
    output_path: Path,
    audio_source_path: Path | None,
) -> list[str]:
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(raw_video_path),
    ]
    if audio_source_path is not None:
        command.extend(["-i", str(audio_source_path)])

    command.extend(
        [
            "-map",
            "0:v:0",
        ]
    )
    if audio_source_path is not None:
        command.extend(["-map", "1:a:0?"])

    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-vf",
            "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709,format=yuv420p",
            "-pix_fmt",
            "yuv420p",
            *BT709_COLOR_ARGS,
        ]
    )
    if audio_source_path is not None:
        command.extend(["-c:a", "aac", "-b:a", "128k", "-shortest"])
    else:
        command.append("-an")
    command.append(str(output_path))
    return command


def _frame_landmark_points(
    frame_landmarks: dict[str, dict[str, float]],
    width: int,
    height: int,
) -> dict[str, tuple[int, int]]:
    points: dict[str, tuple[int, int]] = {}
    for name, values in frame_landmarks.items():
        x = values.get("x")
        y = values.get("y")
        if x is None or y is None:
            continue
        points[name] = (int(round(x * width)), int(round(y * height)))
    return points


def _draw_pose(
    frame: Any,
    points: dict[str, tuple[int, int]],
    highlighted_joints: set[str] | None = None,
) -> None:
    highlighted_joints = highlighted_joints or set()
    for start_name, end_name in SKELETON_EDGES:
        start = points.get(start_name)
        end = points.get(end_name)
        if start is None or end is None:
            continue
        cv2.line(frame, start, end, NORMAL_EDGE_COLOR, 2)

    for point in points.values():
        cv2.circle(frame, point, 3, NORMAL_JOINT_COLOR, -1)

    if not highlighted_joints:
        return

    for start_name, end_name in SKELETON_EDGES:
        if start_name not in highlighted_joints and end_name not in highlighted_joints:
            continue
        start = points.get(start_name)
        end = points.get(end_name)
        if start is None or end is None:
            continue
        cv2.line(frame, start, end, ISSUE_EDGE_COLOR, 4)

    for name in highlighted_joints:
        point = points.get(name)
        if point is not None:
            cv2.circle(frame, point, 6, ISSUE_JOINT_COLOR, -1)


def _issue_angle_label(issue: IssueMarker) -> str | None:
    for key, value in issue.evidence.items():
        if (
            not key.endswith("_deg")
            or isinstance(value, bool)
            or not isinstance(value, int | float)
        ):
            continue
        return f"{key.removesuffix('_deg').replace('_', ' ')} {round(float(value))} deg"
    return None


def _issue_label_anchor(issue: IssueMarker, points: dict[str, tuple[int, int]]) -> tuple[int, int]:
    anchors = [points[name] for name in issue.affected_joints if name in points]
    if not anchors:
        return 16, 132
    x = round(sum(point[0] for point in anchors) / len(anchors))
    y = round(sum(point[1] for point in anchors) / len(anchors))
    return x + 10, max(24, y - 10)


def _draw_angle_labels(
    frame: Any,
    points: dict[str, tuple[int, int]],
    active_issues: list[IssueMarker],
) -> None:
    for offset, issue in enumerate(active_issues):
        label = _issue_angle_label(issue)
        if label is None:
            continue
        anchor = _issue_label_anchor(issue, points)
        text_anchor = (anchor[0], anchor[1] + offset * 28)
        cv2.putText(
            frame,
            label,
            text_anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            ISSUE_EDGE_COLOR,
            2,
            cv2.LINE_AA,
        )


def _rep_boundaries(reps: Reps) -> dict[int, list[str]]:
    boundaries: dict[int, list[str]] = {}
    for rep in reps.reps:
        boundaries.setdefault(rep.start_frame, []).append(f"rep {rep.rep_id} start")
        boundaries.setdefault(rep.mid_frame, []).append(f"rep {rep.rep_id} mid")
        boundaries.setdefault(rep.end_frame, []).append(f"rep {rep.rep_id} end")
    return boundaries


def _rep_phase(frame_index: int, reps: Reps) -> str | None:
    for rep in reps.reps:
        if not rep.start_frame <= frame_index <= rep.end_frame:
            continue
        if frame_index == rep.start_frame:
            phase = "start"
        elif frame_index == rep.mid_frame:
            phase = "mid"
        elif frame_index == rep.end_frame:
            phase = "end"
        elif frame_index < rep.mid_frame:
            phase = "lowering"
        else:
            phase = "rising"
        return f"rep {rep.rep_id} {phase}"
    return None


def _active_issues(frame_index: int, issues: IssueMarkers) -> list[IssueMarker]:
    return [issue for issue in issues.issues if issue.start_frame <= frame_index <= issue.end_frame]


def _highlighted_joints(active_issues: list[IssueMarker]) -> set[str]:
    joints: set[str] = set()
    for issue in active_issues:
        joints.update(issue.affected_joints)
    return joints


def _primary_issue(active_issues: list[IssueMarker]) -> IssueMarker | None:
    if not active_issues:
        return None
    return max(active_issues, key=lambda issue: issue.severity)


def _confidence_warning(active_issues: list[IssueMarker], warnings: list[str]) -> str | None:
    confidences = [
        float(issue.evidence["confidence"])
        for issue in active_issues
        if isinstance(issue.evidence.get("confidence"), int | float)
    ]
    if confidences and min(confidences) < 0.55:
        return "Low confidence issue evidence"
    if warnings:
        return "Camera warning: " + ", ".join(warnings[:2])
    return None


def _thumbnail_frame(issue: IssueMarker) -> int:
    peak_frame = issue.evidence.get("peak_frame")
    if isinstance(peak_frame, int) and peak_frame >= 0:
        return peak_frame
    return round((issue.start_frame + issue.end_frame) / 2)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "issue"


def _thumbnail_targets(issues: IssueMarkers, run_dir: Path) -> dict[int, list[dict[str, Any]]]:
    targets: dict[int, list[dict[str, Any]]] = {}
    for index, issue in enumerate(issues.issues, start=1):
        frame = _thumbnail_frame(issue)
        filename = f"issue_thumbnail_{index}_{_slug(issue.issue)}.jpg"
        targets.setdefault(frame, []).append(
            {
                "issue": issue.issue,
                "rep_id": issue.rep_id,
                "frame": frame,
                "path": str(run_dir / filename),
            }
        )
    return targets


def _clip_metadata(issue: IssueMarker, index: int, run_dir: Path) -> dict[str, Any]:
    filename = f"issue_clip_{index}_{_slug(issue.issue)}.mp4"
    clip_start_sec = max(0.0, float(issue.start_sec) - 1.0)
    clip_end_sec = max(clip_start_sec + 0.1, float(issue.end_sec) + 1.0)
    return {
        "issue": issue.issue,
        "rep_id": issue.rep_id,
        "start_sec": issue.start_sec,
        "end_sec": issue.end_sec,
        "clip_start_sec": round(clip_start_sec, 3),
        "clip_end_sec": round(clip_end_sec, 3),
        "path": str(run_dir / filename),
    }


def _issue_clip_paths(
    source_path: Path,
    issues: IssueMarkers,
    run_dir: Path,
    fps: float,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = []
    for index, issue in enumerate(issues.issues, start=1):
        clip = _clip_metadata(issue, index, run_dir)
        output_path = Path(clip["path"])
        start_sec = float(clip["clip_start_sec"])
        end_sec = float(clip["clip_end_sec"])
        written = _write_issue_clip_ffmpeg(source_path, output_path, start_sec, end_sec)
        if not written:
            written = _write_issue_clip_cv2(
                source_path,
                output_path,
                start_sec,
                end_sec,
                fps,
                width,
                height,
            )
        if written:
            clips.append(clip)
    return clips


def _write_issue_clip_ffmpeg(
    source_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
) -> bool:
    ffmpeg = _tool_path("ffmpeg")
    if ffmpeg is None:
        return False

    duration = max(0.1, end_sec - start_sec)
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{duration:.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        *BT709_COLOR_ARGS,
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=60)
    except (subprocess.SubprocessError, OSError):
        return False
    return output_path.exists() and output_path.stat().st_size > 0


def _write_issue_clip_cv2(
    source_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    fps: float,
    width: int,
    height: int,
) -> bool:
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        capture.release()
        return False

    source_fps = fps if fps > 0 else capture.get(cv2.CAP_PROP_FPS) or 30.0
    source_width = width or int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = height or int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if source_width <= 0 or source_height <= 0:
        capture.release()
        return False

    start_frame = max(0, round(start_sec * source_fps))
    end_frame = max(start_frame + 1, round(end_sec * source_fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer, _codec = _open_video_writer(output_path, source_fps, source_width, source_height)
    if writer is None:
        capture.release()
        return False

    try:
        frame_index = start_frame
        while frame_index <= end_frame:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            writer.write(frame)
            frame_index += 1
    finally:
        writer.release()
        capture.release()

    return output_path.exists() and output_path.stat().st_size > 0


def _draw_overlays(
    frame: Any,
    frame_index: int,
    rep_count: int,
    issue_count: int,
    boundary_labels: dict[int, list[str]],
    phase_label: str | None,
    active_issues: list[IssueMarker],
    quality_warnings: list[str],
) -> None:
    cv2.putText(
        frame,
        f"Reps detected: {rep_count}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Issues: {issue_count}",
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 230, 255),
        2,
        cv2.LINE_AA,
    )
    labels = list(boundary_labels.get(frame_index, []))
    if phase_label:
        labels.append(phase_label)
    primary_issue = _primary_issue(active_issues)
    if primary_issue is not None:
        labels.append(f"{primary_issue.issue} severity {round(primary_issue.severity * 100)}%")
    warning = _confidence_warning(active_issues, quality_warnings)
    if warning is not None:
        labels.append(warning)

    for offset, label in enumerate(labels):
        color = (80, 180, 255)
        if primary_issue is not None and label.startswith(primary_issue.issue):
            color = ISSUE_EDGE_COLOR
        elif label.startswith("Low confidence") or label.startswith("Camera warning"):
            color = (0, 210, 255)
        cv2.putText(
            frame,
            label,
            (16, 96 + offset * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )


def _open_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> tuple[cv2.VideoWriter | None, str | None]:
    for codec in PREFERRED_VIDEO_CODECS:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer, codec
        writer.release()
    return None, None


def run(
    manifest: VideoManifest,
    pose_sequence: PoseSequence,
    reps: Reps,
    issues: IssueMarkers,
    run_dir: Path,
) -> RenderArtifacts:
    if not manifest.analysis_allowed or not manifest.video_path:
        return RenderArtifacts(
            annotated_video_path=manifest.video_path,
            issue_thumbnail_paths=[],
            issue_clip_paths=[],
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(manifest.video_path)
    color_metadata = _video_color_metadata(manifest.video_path)
    render_input_path = source_path
    temporary_paths: list[Path] = []
    if _needs_sdr_conversion(color_metadata):
        sdr_input_path = run_dir / "renderer_sdr_input.mp4"
        if _transcode_hdr_to_sdr(source_path, sdr_input_path, color_metadata):
            render_input_path = sdr_input_path
            temporary_paths.append(sdr_input_path)

    capture = cv2.VideoCapture(str(render_input_path))
    if not capture.isOpened():
        capture.release()
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        return RenderArtifacts(
            annotated_video_path=manifest.video_path,
            issue_thumbnail_paths=[],
            issue_clip_paths=[],
        )

    fps = manifest.fps if manifest.fps > 0 else 30.0
    width = manifest.width or int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = manifest.height or int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        return RenderArtifacts(
            annotated_video_path=manifest.video_path,
            issue_thumbnail_paths=[],
            issue_clip_paths=[],
        )

    output_path = run_dir / "annotated_video.mp4"
    raw_output_path = run_dir / "annotated_video_raw.mp4" if _tool_path("ffmpeg") else output_path
    writer, _codec = _open_video_writer(raw_output_path, fps, width, height)
    if writer is None:
        capture.release()
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        return RenderArtifacts(
            annotated_video_path=manifest.video_path,
            issue_thumbnail_paths=[],
            issue_clip_paths=[],
        )

    pose_by_frame = {frame.frame_index: frame for frame in pose_sequence.frames}
    ordered_pose_frames = sorted(pose_sequence.frames, key=lambda frame: frame.frame_index)
    pose_cursor = 0
    last_pose_frame = None
    boundary_labels = _rep_boundaries(reps)
    thumbnail_targets = _thumbnail_targets(issues, run_dir)
    issue_thumbnail_paths: list[dict[str, Any]] = []

    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            if pose_cursor < len(ordered_pose_frames):
                while (
                    pose_cursor + 1 < len(ordered_pose_frames)
                    and ordered_pose_frames[pose_cursor + 1].frame_index <= frame_index
                ):
                    pose_cursor += 1
                candidate = ordered_pose_frames[pose_cursor]
                if candidate.frame_index <= frame_index:
                    last_pose_frame = candidate

            exact_pose = pose_by_frame.get(frame_index)
            active_pose = exact_pose or last_pose_frame
            active_issues = _active_issues(frame_index, issues)
            if active_pose is not None and active_pose.landmarks:
                points = _frame_landmark_points(active_pose.landmarks, width, height)
                _draw_pose(frame, points, _highlighted_joints(active_issues))
                _draw_angle_labels(frame, points, active_issues)

            _draw_overlays(
                frame,
                frame_index,
                len(reps.reps),
                len(issues.issues),
                boundary_labels,
                _rep_phase(frame_index, reps),
                active_issues,
                manifest.quality_warnings,
            )
            for thumbnail in thumbnail_targets.get(frame_index, []):
                if cv2.imwrite(thumbnail["path"], frame):
                    issue_thumbnail_paths.append(thumbnail)
            writer.write(frame)
            frame_index += 1
    finally:
        writer.release()
        capture.release()
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)

    if raw_output_path != output_path:
        encoded = _encode_bt709_output(raw_output_path, output_path, source_path)
        if not encoded:
            raw_output_path.replace(output_path)
        raw_output_path.unlink(missing_ok=True)

    return RenderArtifacts(
        annotated_video_path=str(output_path),
        issue_thumbnail_paths=issue_thumbnail_paths,
        issue_clip_paths=_issue_clip_paths(
            output_path,
            issues,
            run_dir,
            fps,
            width,
            height,
        ),
    )
