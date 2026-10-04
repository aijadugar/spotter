from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from spotter.exercise_catalog import EXERCISES, INTENDED_EXERCISES


Exercise = str
GOALS = {"strength", "hypertrophy", "endurance", "mobility", "beginner_practice"}
EXPERIENCE_LEVELS = {"beginner", "intermediate"}
EQUIPMENT = {"bodyweight", "dumbbell", "barbell", "unknown"}


class ContractValidationError(ValueError):
    """Raised when a pipeline artifact does not match its JSON contract."""


@dataclass(frozen=True)
class UserProfile:
    goal: str
    experience_level: str
    intended_exercise: str = "auto"
    intended_variation: str | None = None
    known_limitations: list[str] = field(default_factory=list)
    equipment: str = "unknown"


@dataclass(frozen=True)
class VideoManifest:
    video_path: str | None
    fps: float
    duration_sec: float
    total_frames: int
    sampled_frames: int
    width: int
    height: int
    codec: str | None
    container: str | None
    brightness_mean: float | None
    blur_laplacian_var: float | None
    quality_warnings: list[str]
    analysis_allowed: bool


@dataclass(frozen=True)
class PoseFrame:
    frame_index: int
    timestamp_sec: float
    landmarks: dict[str, dict[str, float]]
    world_landmarks: dict[str, dict[str, float]]
    pose_quality: dict[str, Any]


@dataclass(frozen=True)
class PoseSequence:
    frames: list[PoseFrame]
    normalized: bool
    smoothing_method: str
    pose_valid_ratio: float


@dataclass(frozen=True)
class ExerciseClassification:
    exercise: Exercise
    confidence: float
    window_predictions: list[dict[str, Any]]
    fallback_required: bool


@dataclass(frozen=True)
class Rep:
    rep_id: int
    start_frame: int
    mid_frame: int
    end_frame: int
    start_sec: float
    mid_sec: float
    end_sec: float


@dataclass(frozen=True)
class Reps:
    exercise: Exercise
    reps: list[Rep]
    partial_reps: list[dict[str, Any]]


@dataclass(frozen=True)
class RepAnalysisItem:
    rep_id: int
    duration_sec: float
    range_of_motion_score: float
    stability_score: float
    symmetry_score: float
    metrics: dict[str, Any]
    variation_hints: list[str]


@dataclass(frozen=True)
class RepAnalysis:
    exercise: Exercise
    items: list[RepAnalysisItem]
    aggregate_metrics: dict[str, Any]


@dataclass(frozen=True)
class Variation:
    exercise: Exercise
    detected_variation: str
    variation_confidence: float
    not_issues: list[str]


@dataclass(frozen=True)
class IssueMarker:
    rep_id: int
    issue: str
    severity: float
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    affected_joints: list[str]
    evidence: dict[str, Any]


@dataclass(frozen=True)
class IssueMarkers:
    issues: list[IssueMarker]


@dataclass(frozen=True)
class CoachSummary:
    summary: str
    what_you_did: list[str]
    what_looked_good: list[str]
    what_changed_across_reps: list[str]
    valid_variation_vs_issue: list[str]
    top_fixes: list[str]
    next_session_plan: list[str]
    confidence_notes: list[str]


@dataclass(frozen=True)
class Verification:
    passed: bool
    checks: dict[str, bool]
    notes: list[str]


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    timestamp: str
    profile_key: str
    exercise: Exercise
    rep_count: int
    aggregate_metrics: dict[str, Any]
    issue_counts: dict[str, int]
    form_score: float
    source_run_id: str


@dataclass(frozen=True)
class ProgressPlan:
    focus: str
    targets: list[str]
    next_session_cues: list[str]
    encouragement: str
    confidence_notes: list[str]


@dataclass(frozen=True)
class SpeechResult:
    audio_path: str | None
    language: str
    lines: list[str]
    backend: str
    briefing_audio_path: str | None = None
    cues: list[dict[str, Any]] | None = None
    chars_used: int = 0
    chars_skipped: int = 0


def to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    return value


def validate_contract(name: str, value: Any) -> None:
    payload = to_dict(value)
    validators = {
        "user_profile.json": _validate_user_profile,
        "video_manifest.json": _validate_video_manifest,
        "pose_sequence.json": _validate_pose_sequence,
        "exercise_classification.json": _validate_exercise_classification,
        "rep_debug.json": _validate_rep_debug,
        "reps.json": _validate_reps,
        "rep_analysis.json": _validate_rep_analysis,
        "variation.json": _validate_variation,
        "issue_markers.json": _validate_issue_markers,
        "coach_summary.json": _validate_coach_summary,
        "verification.json": _validate_verification,
        "session_record.json": _validate_session_record,
        "progress_plan.json": _validate_progress_plan,
        "speech.json": _validate_speech_result,
        "final_report.json": _validate_final_report,
        "manifest.json": _validate_run_manifest,
    }
    try:
        validator = validators[name]
    except KeyError as exc:
        raise ContractValidationError(f"Unknown contract: {name}") from exc
    validator(payload, name)


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractValidationError(f"{path} must be an object")
    return value


def _require_fields(payload: dict[str, Any], required: set[str], path: str) -> None:
    missing = sorted(required - payload.keys())
    if missing:
        raise ContractValidationError(f"{path} missing required field(s): {', '.join(missing)}")


def _require_type(value: Any, expected_type: type | tuple[type, ...], path: str) -> None:
    if not isinstance(value, expected_type):
        raise ContractValidationError(f"{path} has invalid type")


def _require_bool(value: Any, path: str) -> None:
    if not isinstance(value, bool):
        raise ContractValidationError(f"{path} must be a boolean")


def _require_number(value: Any, path: str, *, minimum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractValidationError(f"{path} must be a number")
    if minimum is not None and value < minimum:
        raise ContractValidationError(f"{path} must be >= {minimum}")


def _require_int(value: Any, path: str, *, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ContractValidationError(f"{path} must be >= {minimum}")


def _require_score(value: Any, path: str) -> None:
    _require_number(value, path)
    if value < 0 or value > 1:
        raise ContractValidationError(f"{path} must be between 0 and 1")


def _require_enum(value: Any, allowed: set[str], path: str) -> None:
    if value not in allowed:
        raise ContractValidationError(f"{path} has invalid enum value: {value!r}")


def _require_string_list(value: Any, path: str) -> None:
    _require_type(value, list, path)
    for index, item in enumerate(value):
        _require_type(item, str, f"{path}[{index}]")


def _require_time_range(start_frame: Any, end_frame: Any, start_sec: Any, end_sec: Any, path: str) -> None:
    _require_int(start_frame, f"{path}.start_frame", minimum=0)
    _require_int(end_frame, f"{path}.end_frame", minimum=0)
    _require_number(start_sec, f"{path}.start_sec", minimum=0)
    _require_number(end_sec, f"{path}.end_sec", minimum=0)
    if start_frame > end_frame:
        raise ContractValidationError(f"{path} frame range must be ordered")
    if start_sec > end_sec:
        raise ContractValidationError(f"{path} timestamp range must be ordered")


def _validate_user_profile(value: Any, path: str) -> None:
    payload = _require_mapping(value, path)
    _require_fields(
        payload,
        {
            "goal",
            "experience_level",
            "intended_exercise",
            "intended_variation",
            "known_limitations",
            "equipment",
        },
        path,
    )
    _require_enum(payload["goal"], GOALS, f"{path}.goal")
    _require_enum(payload["experience_level"], EXPERIENCE_LEVELS, f"{path}.experience_level")
    _require_enum(payload["intended_exercise"], INTENDED_EXERCISES, f"{path}.intended_exercise")
    if payload["intended_variation"] is not None:
        _require_type(payload["intended_variation"], str, f"{path}.intended_variation")
    _require_string_list(payload["known_limitations"], f"{path}.known_limitations")
    _require_enum(payload["equipment"], EQUIPMENT, f"{path}.equipment")


def _validate_video_manifest(value: Any, path: str) -> None:
    payload = _require_mapping(value, path)
    _require_fields(
        payload,
        {
            "video_path",
            "fps",
            "duration_sec",
            "total_frames",
            "sampled_frames",
            "width",
            "height",
            "codec",
            "container",
            "brightness_mean",
            "blur_laplacian_var",
            "quality_warnings",
            "analysis_allowed",
        },
        path,
    )
    if payload["video_path"] is not None:
        _require_type(payload["video_path"], str, f"{path}.video_path")
    _require_number(payload["fps"], f"{path}.fps", minimum=0)
    _require_number(payload["duration_sec"], f"{path}.duration_sec", minimum=0)
    _require_int(payload["total_frames"], f"{path}.total_frames", minimum=0)
    _require_int(payload["sampled_frames"], f"{path}.sampled_frames", minimum=0)
    if payload["sampled_frames"] > payload["total_frames"]:
        raise ContractValidationError(f"{path}.sampled_frames must be <= total_frames")
    _require_int(payload["width"], f"{path}.width", minimum=0)
    _require_int(payload["height"], f"{path}.height", minimum=0)
    if payload["codec"] is not None:
        _require_type(payload["codec"], str, f"{path}.codec")
    if payload["container"] is not None:
        _require_type(payload["container"], str, f"{path}.container")
    if payload["brightness_mean"] is not None:
        _require_number(payload["brightness_mean"], f"{path}.brightness_mean", minimum=0)
    if payload["blur_laplacian_var"] is not None:
        _require_number(payload["blur_laplacian_var"], f"{path}.blur_laplacian_var", minimum=0)
    _require_string_list(payload["quality_warnings"], f"{path}.quality_warnings")
    _require_bool(payload["analysis_allowed"], f"{path}.analysis_allowed")


def _validate_pose_sequence(value: Any, path: str) -> None:
    payload = _require_mapping(value, path)
    _require_fields(payload, {"frames", "normalized", "smoothing_method", "pose_valid_ratio"}, path)
    _require_type(payload["frames"], list, f"{path}.frames")
    previous_frame = -1
    previous_timestamp = -1.0
    for index, frame_value in enumerate(payload["frames"]):
        frame_path = f"{path}.frames[{index}]"
        frame = _require_mapping(frame_value, frame_path)
        _require_fields(
            frame,
            {"frame_index", "timestamp_sec", "landmarks", "world_landmarks", "pose_quality"},
            frame_path,
        )
        _require_int(frame["frame_index"], f"{frame_path}.frame_index", minimum=0)
        _require_number(frame["timestamp_sec"], f"{frame_path}.timestamp_sec", minimum=0)
        if frame["frame_index"] < previous_frame:
            raise ContractValidationError(f"{frame_path}.frame_index must be ordered")
        if frame["timestamp_sec"] < previous_timestamp:
            raise ContractValidationError(f"{frame_path}.timestamp_sec must be ordered")
        previous_frame = frame["frame_index"]
        previous_timestamp = frame["timestamp_sec"]
        _require_mapping(frame["landmarks"], f"{frame_path}.landmarks")
        _require_mapping(frame["world_landmarks"], f"{frame_path}.world_landmarks")
        _require_mapping(frame["pose_quality"], f"{frame_path}.pose_quality")
    _require_bool(payload["normalized"], f"{path}.normalized")
    _require_type(payload["smoothing_method"], str, f"{path}.smoothing_method")
    _require_score(payload["pose_valid_ratio"], f"{path}.pose_valid_ratio")


def _validate_exercise_classification(value: Any, path: str) -> None:
    payload = _require_mapping(value, path)
    _require_fields(payload, {"exercise", "confidence", "window_predictions", "fallback_required"}, path)
    _require_enum(payload["exercise"], EXERCISES, f"{path}.exercise")
    _require_score(payload["confidence"], f"{path}.confidence")
    _require_type(payload["window_predictions"], list, f"{path}.window_predictions")
    for index, prediction_value in enumerate(payload["window_predictions"]):
        prediction_path = f"{path}.window_predictions[{index}]"
        prediction = _require_mapping(prediction_value, prediction_path)
        _require_fields(prediction, {"start_sec", "end_sec", "label", "confidence"}, prediction_path)
        _require_number(prediction["start_sec"], f"{prediction_path}.start_sec", minimum=0)
        _require_number(prediction["end_sec"], f"{prediction_path}.end_sec", minimum=0)
        if prediction["start_sec"] > prediction["end_sec"]:
            raise ContractValidationError(f"{prediction_path} timestamps must be ordered")
        _require_enum(prediction["label"], EXERCISES, f"{prediction_path}.label")
        _require_score(prediction["confidence"], f"{prediction_path}.confidence")
    _require_bool(payload["fallback_required"], f"{path}.fallback_required")


def _validate_rep(rep_value: Any, path: str) -> None:
    rep = _require_mapping(rep_value, path)
    _require_fields(
        rep,
        {"rep_id", "start_frame", "mid_frame", "end_frame", "start_sec", "mid_sec", "end_sec"},
        path,
    )
    _require_int(rep["rep_id"], f"{path}.rep_id", minimum=1)
    _require_time_range(rep["start_frame"], rep["end_frame"], rep["start_sec"], rep["end_sec"], path)
    _require_int(rep["mid_frame"], f"{path}.mid_frame", minimum=0)
    _require_number(rep["mid_sec"], f"{path}.mid_sec", minimum=0)
    if not rep["start_frame"] <= rep["mid_frame"] <= rep["end_frame"]:
        raise ContractValidationError(f"{path}.mid_frame must be inside rep frame range")
    if not rep["start_sec"] <= rep["mid_sec"] <= rep["end_sec"]:
        raise ContractValidationError(f"{path}.mid_sec must be inside rep timestamp range")


def _validate_reps(value: Any, path: str) -> None:
    payload = _require_mapping(value, path)
    _require_fields(payload, {"exercise", "reps", "partial_reps"}, path)
    _require_enum(payload["exercise"], EXERCISES, f"{path}.exercise")
    _require_type(payload["reps"], list, f"{path}.reps")
    for index, rep in enumerate(payload["reps"]):
        _validate_rep(rep, f"{path}.reps[{index}]")
    _require_type(payload["partial_reps"], list, f"{path}.partial_reps")


def _validate_rep_debug(value: Any, path: str) -> None:
    payload = _require_mapping(value, path)
    _require_fields(
        payload,
        {
            "accepted_reps",
            "body_line_mean",
            "extrema",
            "raw_signal_range",
            "selected_signal",
            "thresholds",
            "usable_signal_samples",
        },
        path,
    )
    _require_type(payload["accepted_reps"], list, f"{path}.accepted_reps")
    _require_number(payload["body_line_mean"], f"{path}.body_line_mean")
    _require_type(payload["extrema"], list, f"{path}.extrema")
    _require_number(payload["raw_signal_range"], f"{path}.raw_signal_range", minimum=0)
    _require_type(payload["selected_signal"], str, f"{path}.selected_signal")
    _require_mapping(payload["thresholds"], f"{path}.thresholds")
    _require_int(payload["usable_signal_samples"], f"{path}.usable_signal_samples", minimum=0)


def _validate_rep_analysis(value: Any, path: str) -> None:
    payload = _require_mapping(value, path)
    _require_fields(payload, {"exercise", "items", "aggregate_metrics"}, path)
    _require_enum(payload["exercise"], EXERCISES, f"{path}.exercise")
    _require_type(payload["items"], list, f"{path}.items")
    for index, item_value in enumerate(payload["items"]):
        item_path = f"{path}.items[{index}]"
        item = _require_mapping(item_value, item_path)
        _require_fields(
            item,
            {
                "rep_id",
                "duration_sec",
                "range_of_motion_score",
                "stability_score",
                "symmetry_score",
                "metrics",
                "variation_hints",
            },
            item_path,
        )
        _require_int(item["rep_id"], f"{item_path}.rep_id", minimum=1)
        _require_number(item["duration_sec"], f"{item_path}.duration_sec", minimum=0)
        _require_score(item["range_of_motion_score"], f"{item_path}.range_of_motion_score")
        _require_score(item["stability_score"], f"{item_path}.stability_score")
        _require_score(item["symmetry_score"], f"{item_path}.symmetry_score")
        _require_mapping(item["metrics"], f"{item_path}.metrics")
        _require_string_list(item["variation_hints"], f"{item_path}.variation_hints")
    _require_mapping(payload["aggregate_metrics"], f"{path}.aggregate_metrics")


def _validate_variation(value: Any, path: str) -> None:
    payload = _require_mapping(value, path)
    _require_fields(payload, {"exercise", "detected_variation", "variation_confidence", "not_issues"}, path)
    _require_enum(payload["exercise"], EXERCISES, f"{path}.exercise")
    _require_type(payload["detected_variation"], str, f"{path}.detected_variation")
    _require_score(payload["variation_confidence"], f"{path}.variation_confidence")
    _require_string_list(payload["not_issues"], f"{path}.not_issues")


def _validate_issue_markers(value: Any, path: str) -> None:
    payload = _require_mapping(value, path)
    _require_fields(payload, {"issues"}, path)
    _require_type(payload["issues"], list, f"{path}.issues")
    for index, issue_value in enumerate(payload["issues"]):
        issue_path = f"{path}.issues[{index}]"
        issue = _require_mapping(issue_value, issue_path)
        _require_fields(
            issue,
            {
                "rep_id",
                "issue",
                "severity",
                "start_frame",
                "end_frame",
                "start_sec",
                "end_sec",
                "affected_joints",
                "evidence",
            },
            issue_path,
        )
        _require_int(issue["rep_id"], f"{issue_path}.rep_id", minimum=1)
        _require_type(issue["issue"], str, f"{issue_path}.issue")
        _require_score(issue["severity"], f"{issue_path}.severity")
        _require_time_range(
            issue["start_frame"],
            issue["end_frame"],
            issue["start_sec"],
            issue["end_sec"],
            issue_path,
        )
        _require_string_list(issue["affected_joints"], f"{issue_path}.affected_joints")
        _require_mapping(issue["evidence"], f"{issue_path}.evidence")


def _validate_coach_summary(value: Any, path: str) -> None:
    payload = _require_mapping(value, path)
    _require_fields(
        payload,
        {
            "summary",
            "what_you_did",
            "what_looked_good",
            "what_changed_across_reps",
            "valid_variation_vs_issue",
            "top_fixes",
            "next_session_plan",
            "confidence_notes",
        },
        path,
    )
    _require_type(payload["summary"], str, f"{path}.summary")
    _require_string_list(payload["what_you_did"], f"{path}.what_you_did")
    _require_string_list(payload["what_looked_good"], f"{path}.what_looked_good")
    _require_string_list(
        payload["what_changed_across_reps"],
        f"{path}.what_changed_across_reps",
    )
    _require_string_list(
        payload["valid_variation_vs_issue"],
        f"{path}.valid_variation_vs_issue",
    )
    _require_string_list(payload["top_fixes"], f"{path}.top_fixes")
    _require_string_list(payload["next_session_plan"], f"{path}.next_session_plan")
    _require_string_list(payload["confidence_notes"], f"{path}.confidence_notes")


def _validate_verification(value: Any, path: str) -> None:
    payload = _require_mapping(value, path)
    _require_fields(payload, {"passed", "checks", "notes"}, path)
    _require_bool(payload["passed"], f"{path}.passed")
    checks = _require_mapping(payload["checks"], f"{path}.checks")
    for key, value in checks.items():
        _require_type(key, str, f"{path}.checks key")
        _require_bool(value, f"{path}.checks.{key}")
    _require_string_list(payload["notes"], f"{path}.notes")


def _validate_session_record(value: Any, path: str) -> None:
    payload = _require_mapping(value, path)
    _require_fields(
        payload,
        {
            "session_id",
            "timestamp",
            "profile_key",
            "exercise",
            "rep_count",
            "aggregate_metrics",
            "issue_counts",
            "form_score",
            "source_run_id",
        },
        path,
    )
    _require_type(payload["session_id"], str, f"{path}.session_id")
    _require_type(payload["timestamp"], str, f"{path}.timestamp")
    _require_type(payload["profile_key"], str, f"{path}.profile_key")
    _require_enum(payload["exercise"], EXERCISES, f"{path}.exercise")
    _require_int(payload["rep_count"], f"{path}.rep_count", minimum=0)
    _require_mapping(payload["aggregate_metrics"], f"{path}.aggregate_metrics")
    _require_mapping(payload["issue_counts"], f"{path}.issue_counts")
    for key, count in payload["issue_counts"].items():
        _require_type(key, str, f"{path}.issue_counts key")
        _require_int(count, f"{path}.issue_counts[{key}]", minimum=0)
    _require_score(payload["form_score"], f"{path}.form_score")
    _require_type(payload["source_run_id"], str, f"{path}.source_run_id")


def _validate_progress_plan(value: Any, path: str) -> None:
    payload = _require_mapping(value, path)
    _require_fields(
        payload,
        {"focus", "targets", "next_session_cues", "encouragement", "confidence_notes"},
        path,
    )
    _require_type(payload["focus"], str, f"{path}.focus")
    _require_string_list(payload["targets"], f"{path}.targets")
    _require_string_list(payload["next_session_cues"], f"{path}.next_session_cues")
    _require_type(payload["encouragement"], str, f"{path}.encouragement")
    _require_string_list(payload["confidence_notes"], f"{path}.confidence_notes")


def _validate_speech_result(value: Any, path: str) -> None:
    payload = _require_mapping(value, path)
    _require_fields(payload, {"audio_path", "language", "lines", "backend"}, path)
    if payload["audio_path"] is not None:
        _require_type(payload["audio_path"], str, f"{path}.audio_path")
    _require_type(payload["language"], str, f"{path}.language")
    _require_string_list(payload["lines"], f"{path}.lines")
    _require_type(payload["backend"], str, f"{path}.backend")
    # Optional new fields
    if "briefing_audio_path" in payload and payload["briefing_audio_path"] is not None:
        _require_type(payload["briefing_audio_path"], str, f"{path}.briefing_audio_path")
    if "cues" in payload and payload["cues"] is not None:
        _require_type(payload["cues"], list, f"{path}.cues")
        for i, cue in enumerate(payload["cues"]):
            cue_path = f"{path}.cues[{i}]"
            _require_mapping(cue, cue_path)
            _require_fields(cue, {"issue_id", "rep_number", "text", "audio_path"}, cue_path)
            _require_type(cue["issue_id"], str, f"{cue_path}.issue_id")
            _require_int(cue["rep_number"], f"{cue_path}.rep_number", minimum=1)
            _require_type(cue["text"], str, f"{cue_path}.text")
            if cue["audio_path"] is not None:
                _require_type(cue["audio_path"], str, f"{cue_path}.audio_path")
    if "chars_used" in payload:
        _require_int(payload["chars_used"], f"{path}.chars_used", minimum=0)
    if "chars_skipped" in payload:
        _require_int(payload["chars_skipped"], f"{path}.chars_skipped", minimum=0)


def _validate_final_report(value: Any, path: str) -> None:
    payload = _require_mapping(value, path)
    _require_fields(
        payload,
        {
            "run_id",
            "profile",
            "video_manifest",
            "exercise",
            "reps",
            "rep_analysis",
            "variation",
            "issue_markers",
            "coach_summary",
            "verification",
            "artifacts",
        },
        path,
    )
    _require_type(payload["run_id"], str, f"{path}.run_id")
    _validate_user_profile(payload["profile"], f"{path}.profile")
    _validate_video_manifest(payload["video_manifest"], f"{path}.video_manifest")
    _validate_exercise_classification(payload["exercise"], f"{path}.exercise")
    _validate_reps(payload["reps"], f"{path}.reps")
    _validate_rep_analysis(payload["rep_analysis"], f"{path}.rep_analysis")
    _validate_variation(payload["variation"], f"{path}.variation")
    _validate_issue_markers(payload["issue_markers"], f"{path}.issue_markers")
    _validate_coach_summary(payload["coach_summary"], f"{path}.coach_summary")
    _validate_verification(payload["verification"], f"{path}.verification")
    artifacts = _require_mapping(payload["artifacts"], f"{path}.artifacts")
    _require_fields(artifacts, {"run_dir", "annotated_video_path"}, f"{path}.artifacts")
    _require_type(artifacts["run_dir"], str, f"{path}.artifacts.run_dir")
    if artifacts["annotated_video_path"] is not None:
        _require_type(artifacts["annotated_video_path"], str, f"{path}.artifacts.annotated_video_path")
    if "issue_thumbnail_paths" in artifacts:
        _require_type(artifacts["issue_thumbnail_paths"], list, f"{path}.artifacts.issue_thumbnail_paths")
        for index, thumbnail_value in enumerate(artifacts["issue_thumbnail_paths"]):
            thumbnail_path = f"{path}.artifacts.issue_thumbnail_paths[{index}]"
            thumbnail = _require_mapping(thumbnail_value, thumbnail_path)
            _require_fields(thumbnail, {"issue", "rep_id", "frame", "path"}, thumbnail_path)
            _require_type(thumbnail["issue"], str, f"{thumbnail_path}.issue")
            _require_int(thumbnail["rep_id"], f"{thumbnail_path}.rep_id", minimum=1)
            _require_int(thumbnail["frame"], f"{thumbnail_path}.frame", minimum=0)
            _require_type(thumbnail["path"], str, f"{thumbnail_path}.path")
    if "issue_clip_paths" in artifacts:
        _require_type(artifacts["issue_clip_paths"], list, f"{path}.artifacts.issue_clip_paths")
        for index, clip_value in enumerate(artifacts["issue_clip_paths"]):
            clip_path = f"{path}.artifacts.issue_clip_paths[{index}]"
            clip = _require_mapping(clip_value, clip_path)
            _require_fields(
                clip,
                {"issue", "rep_id", "start_sec", "end_sec", "clip_start_sec", "clip_end_sec", "path"},
                clip_path,
            )
            _require_type(clip["issue"], str, f"{clip_path}.issue")
            _require_int(clip["rep_id"], f"{clip_path}.rep_id", minimum=1)
            _require_number(clip["start_sec"], f"{clip_path}.start_sec", minimum=0)
            _require_number(clip["end_sec"], f"{clip_path}.end_sec", minimum=0)
            _require_number(clip["clip_start_sec"], f"{clip_path}.clip_start_sec", minimum=0)
            _require_number(clip["clip_end_sec"], f"{clip_path}.clip_end_sec", minimum=0)
            if clip["start_sec"] > clip["end_sec"]:
                raise ContractValidationError(f"{clip_path} timestamps must be ordered")
            if clip["clip_start_sec"] > clip["clip_end_sec"]:
                raise ContractValidationError(f"{clip_path} clip timestamps must be ordered")
            _require_type(clip["path"], str, f"{clip_path}.path")


def _validate_run_manifest(value: Any, path: str) -> None:
    payload = _require_mapping(value, path)
    _require_fields(payload, {"run_id", "mock_mode", "artifacts"}, path)
    _require_type(payload["run_id"], str, f"{path}.run_id")
    _require_bool(payload["mock_mode"], f"{path}.mock_mode")
    _require_type(payload["artifacts"], list, f"{path}.artifacts")
    for index, artifact_value in enumerate(payload["artifacts"]):
        artifact_path = f"{path}.artifacts[{index}]"
        artifact = _require_mapping(artifact_value, artifact_path)
        _require_fields(artifact, {"name", "path", "contract"}, artifact_path)
        _require_type(artifact["name"], str, f"{artifact_path}.name")
        _require_type(artifact["path"], str, f"{artifact_path}.path")
        _require_type(artifact["contract"], str, f"{artifact_path}.contract")
