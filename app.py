from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any

import gradio as gr
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent / "src"))

from spotter.exercise_catalog import USER_SELECTABLE_EXERCISES
from spotter.pipeline import run_pipeline

BASE_DIR = Path(__file__).parent
WEB_DIR = BASE_DIR / "web"
RUNS_ROOT = BASE_DIR / "runs"
DEMO_DIR = BASE_DIR / "demo" / "clips"
MAX_UPLOAD_BYTES = int(os.getenv("SPOTTER_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))

APP_DESCRIPTION = (
    "Upload a short workout clip, tune the athlete context, and generate an annotated "
    "form-review report with structured artifacts."
)

# Demo clip cache: keyed by (clip file hash, profile dict, pipeline version) -> analysis response.
# Disabled by default; set SPOTTER_DEMO_CACHE=0 to disable.
_DEMO_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}
_DEMO_CACHE_LOCK = threading.Lock()
_DEMO_CACHE_ENABLED = os.getenv("SPOTTER_DEMO_CACHE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# Pipeline version for cache invalidation when model/code versions change
_PIPELINE_VERSION = "20261004-v2"  # Increment when pipeline behavior changes


def _warmup_models() -> None:
    """Warm up ML models at server startup to avoid per-request cold starts."""
    # Import here to avoid circular imports
    from spotter.steps.pose_backends.registry import warmup_mediapipe
    from spotter.ml.exercise_router_inference import load_router_model
    from spotter.slm.providers import get_coach_summary_model
    from spotter.steps.session_memory import get_session_memory
    from spotter.steps.speech import get_speech_synthesizer

    print("=== Warming up ML models at startup ===")
    t0 = time.time()
    warmup_mediapipe()
    print(f"  MediaPipe pose ready: {time.time() - t0:.2f}s")
    load_router_model()
    print(f"  Router ready: {time.time() - t0:.2f}s")
    get_coach_summary_model()
    print(f"  Coach model ready: {time.time() - t0:.2f}s")
    get_session_memory()
    print(f"  Session memory ready: {time.time() - t0:.2f}s")
    get_speech_synthesizer()
    print(f"  Speech ready: {time.time() - t0:.2f}s")
    print(f"=== Total warmup: {time.time() - t0:.2f}s ===")


# Run warmup at module import (server startup)
_warmup_models()


server = gr.Server(
    title="Spotter",
    summary="Video-based workout form review API",
    description=APP_DESCRIPTION,
)
server.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@server.get("/healthz", include_in_schema=False)
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "spotter"}


@server.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@server.get("/api/config")
def config() -> dict[str, Any]:
    return {
        "description": APP_DESCRIPTION,
        "goals": [
            "strength",
            "hypertrophy",
            "endurance",
            "mobility",
            "beginner_practice",
        ],
        "experience_levels": ["beginner", "intermediate"],
        "exercises": ["auto", *USER_SELECTABLE_EXERCISES],
        "limitations": ["wrist_discomfort", "knee_discomfort", "shoulder_discomfort"],
        "equipment": ["bodyweight", "dumbbell", "barbell", "unknown"],
    }


_DEMO_PROFILE = {
    "goal": "beginner_practice",
    "experience_level": "beginner",
    "intended_exercise": "auto",
    "intended_variation": None,
    "known_limitations": [],
    "equipment": "bodyweight",
}

_DEMO_CLIPS: list[dict[str, str]] = [
    {
        "id": "clean_squat",
        "file": "demo_clips/clean_squat.mp4",
        "label": "Clean squat",
        "description": "Good-form rep set with steady depth and posture.",
    },
    {
        "id": "wide_squat_fault",
        "file": "demo_clips/wide_squat_fault.mp4",
        "label": "Wide-stance squat",
        "description": "A clear, visible form fault the coach should flag.",
    },
    {
        "id": "unsupported_clip",
        "file": "demo_clips/unsupported_clip.mp4",
        "label": "Unsupported movement",
        "description": "An unclear or unsupported exercise the router should reject as unknown.",
    },
]


@server.get("/api/demo/clips")
def demo_clips() -> dict[str, Any]:
    available = []
    for clip in _DEMO_CLIPS:
        path = BASE_DIR / clip["file"]
        available.append(
            {
                "id": clip["id"],
                "label": clip["label"],
                "description": clip["description"],
                "available": path.is_file(),
            }
        )
    return {"clips": available, "cache_enabled": _DEMO_CACHE_ENABLED}


@server.get("/api/demo/analyze/{clip_id}")
async def demo_analyze(clip_id: str) -> Any:
    clip = next((c for c in _DEMO_CLIPS if c["id"] == clip_id), None)
    if clip is None:
        raise HTTPException(status_code=404, detail="Demo clip not found.")
    clip_path = BASE_DIR / clip["file"]
    if not clip_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Demo clip {clip['file']!r} is not present in the repo. "
                "Add it to demo/clips/ to enable this demo."
            ),
        )

    cache_key = _demo_cache_key(clip_path, _DEMO_PROFILE)
    if _DEMO_CACHE_ENABLED:
        with _DEMO_CACHE_LOCK:
            cached = _DEMO_CACHE.get(cache_key)
            if cached is not None:
                cached["cached"] = True
                return cached

    response = await _run_demo_pipeline(str(clip_path))
    if _DEMO_CACHE_ENABLED:
        with _DEMO_CACHE_LOCK:
            _DEMO_CACHE[cache_key] = response
    return response


def _demo_cache_key(clip_path: Path, profile: dict[str, Any]) -> tuple[str, str, str]:
    digest = hashlib.sha256()
    with clip_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    profile_key = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    return digest.hexdigest(), profile_key, _PIPELINE_VERSION


async def _run_demo_pipeline(video_path: str) -> dict[str, Any]:
    events: Queue[dict[str, Any] | None] = Queue()

    def worker() -> None:
        try:
            result = _run_analysis_pipeline(
                video_path,
                _DEMO_PROFILE,
                bypass_verifier=True,
                progress=events.put,
            )
            events.put({"type": "complete", "result": _analysis_response(result)})
        except Exception as exc:  # pragma: no cover - surfaced to browser clients
            events.put({"type": "error", "detail": _friendly_error(exc)})
        finally:
            events.put(None)

    thread = Thread(target=worker, daemon=True)
    thread.start()
    return StreamingResponse(
        _stream_events(events),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache"},
    )


def _stream_events(events: Queue[dict[str, Any] | None]) -> Any:
    while True:
        event = events.get()
        if event is None:
            break
        yield f"{json.dumps(event)}\n"


def _friendly_error(exc: Exception) -> str:
    message = str(exc).strip()
    lowered = message.lower()
    if "no person" in lowered or "no pose" in lowered or "pose_not_detected" in lowered:
        return (
            "No person was clearly visible in this clip. Try a video where the "
            "whole body is in frame, shot from the side, with good lighting."
        )
    if "too_short" in lowered or "video too short" in lowered or "duration" in lowered and "short" in lowered:
        return (
            "The video is too short (minimum 10 seconds). Record a longer clip and try again."
        )
    if "too_long" in lowered or "video too long" in lowered or "duration" in lowered and "long" in lowered:
        return (
            "The video is too long (maximum 60 seconds). Trim the clip and try again."
        )
    if "video_decode_failed" in lowered or "decode" in lowered or "corrupt" in lowered or "unsupported format" in lowered:
        return (
            "The video file could not be decoded. Make sure it's a valid MP4/MOV/WebM file and try again."
        )
    if "unsupported" in lowered or "unknown" in lowered or "fallback_required" in lowered:
        return (
            "The router could not identify a supported exercise in this clip. "
            "Try a different angle or select the exercise manually."
        )
    if "download" in lowered or "model" in lowered:
        return (
            "A model download failed. Check your internet connection and run it again."
        )
    if "provider" in lowered or "llm" in lowered or "coach" in lowered:
        return (
            "The coach summary provider was unavailable, so Spotter used its "
            "deterministic fallback summary instead. The report is still complete."
        )
    if "timeout" in lowered or "timed out" in lowered:
        return (
            "The analysis took too long and was stopped. Try a shorter clip "
            "(under 60 seconds) and run it again."
        )
    return (
        "Something went wrong while analyzing this clip. Try another video, "
        "or check the connection and run it again."
    )


@server.get("/api/artifacts/{run_id}/{filename}", include_in_schema=False)
def artifact(run_id: str, filename: str) -> FileResponse:
    artifact_path = (RUNS_ROOT / run_id / filename).resolve()
    runs_root = RUNS_ROOT.resolve()
    if runs_root not in artifact_path.parents or not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return FileResponse(artifact_path)


def _artifact_url(run_id: str, path: str | None) -> str | None:
    if not path:
        return None

    if not isinstance(path, str):
        return None

    artifact_path = Path(path).resolve()
    run_root = (RUNS_ROOT / run_id).resolve()
    if run_root not in artifact_path.parents or not artifact_path.is_file():
        return None
    return f"/api/artifacts/{run_id}/{artifact_path.name}"


def _artifact_link(run_id: str, name: str, path: str | None) -> dict[str, str] | None:
    url = _artifact_url(run_id, path)
    if url is None:
        return None
    return {"name": name, "url": url}


def _artifact_urls(result: dict[str, Any]) -> list[dict[str, str]]:
    run_id = result["run_id"]
    run_dir = Path(str(result["run_dir"]))
    links: list[dict[str, str]] = []
    artifact_files = [
        "final_report.json",
        "video_manifest.json",
        "pose_sequence.json",
        "exercise_classification.json",
        "reps.json",
        "rep_debug.json",
        "rep_analysis.json",
        "variation.json",
        "issue_markers.json",
        "coach_summary.json",
        "verification.json",
        "progress_plan.json",
        "session_record.json",
        "speech.json",
        "manifest.json",
    ]
    for filename in artifact_files:
        link = _artifact_link(run_id, filename, str(run_dir / filename))
        if link is not None:
            links.append(link)

    video_link = _artifact_link(
        run_id,
        "annotated_video.mp4",
        result.get("annotated_video_path"),
    )
    if video_link is not None:
        links.append(video_link)

    for thumbnail in result.get("issue_thumbnail_paths", []):
        if not isinstance(thumbnail, dict):
            continue
        path = thumbnail.get("path")
        issue = thumbnail.get("issue", "issue")
        rep_id = thumbnail.get("rep_id", "?")
        if not isinstance(path, str):
            continue
        link = _artifact_link(run_id, f"thumbnail_rep_{rep_id}_{issue}.jpg", path)
        if link is not None:
            links.append(link)

    for clip in result.get("issue_clip_paths", []):
        if not isinstance(clip, dict):
            continue
        path = clip.get("path")
        issue = clip.get("issue", "issue")
        rep_id = clip.get("rep_id", "?")
        if not isinstance(path, str):
            continue
        link = _artifact_link(run_id, f"clip_rep_{rep_id}_{issue}.mp4", path)
        if link is not None:
            links.append(link)

    return links


def _parse_limitations(limitations: str) -> list[str]:
    try:
        parsed_limitations = json.loads(limitations)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="Limitations must be valid JSON."
        ) from exc

    if not isinstance(parsed_limitations, list) or not all(
        isinstance(item, str) for item in parsed_limitations
    ):
        raise HTTPException(
            status_code=400, detail="Limitations must be a JSON list of strings."
        )
    return parsed_limitations


def _profile_input(
    *,
    goal: str,
    experience_level: str,
    intended_exercise: str,
    intended_variation: str,
    limitations: str,
    equipment: str,
) -> dict[str, Any]:
    return {
        "goal": goal,
        "experience_level": experience_level,
        "intended_exercise": intended_exercise,
        "intended_variation": intended_variation or None,
        "known_limitations": _parse_limitations(limitations),
        "equipment": equipment,
    }


def _parse_bool_form(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _safe_upload_stem(filename: str) -> str:
    stem = Path(filename).stem or "upload"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip(".-_")
    return safe_stem[:48] or "upload"


def _run_analysis_pipeline(
    video_path: str | None,
    profile_input: dict[str, Any],
    bypass_verifier: bool = True,
    progress: Any | None = None,
) -> dict[str, Any]:
    return run_pipeline(
        video_path=video_path,
        profile_input=profile_input,
        bypass_verifier=bypass_verifier,
        progress=progress,
    )


async def _save_upload(video: UploadFile | None) -> str | None:
    video_path: str | None = None
    if video is not None and video.filename:
        suffix = Path(video.filename).suffix or ".mp4"
        prefix = f"spotter-{_safe_upload_stem(video.filename)}-"
        with tempfile.NamedTemporaryFile(
            delete=False,
            prefix=prefix,
            suffix=suffix,
        ) as temp_video:
            bytes_written = 0
            while True:
                chunk = video.file.read(65536)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    Path(temp_video.name).unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Upload exceeds the maximum size of {MAX_UPLOAD_BYTES // 1024 // 1024} MB. "
                            "Trim the clip and try again."
                        ),
                    )
                temp_video.write(chunk)
            video_path = temp_video.name
    return video_path


def _analysis_response(result: dict[str, Any]) -> dict[str, Any]:
    issue_thumbnail_urls = []
    for thumbnail in result.get("issue_thumbnail_paths", []):
        if not isinstance(thumbnail, dict):
            continue
        path = thumbnail.get("path")
        if not isinstance(path, str):
            continue
        url = _artifact_url(result["run_id"], path)
        if url is not None:
            issue_thumbnail_urls.append({**thumbnail, "url": url})

    issue_clip_urls = []
    for clip in result.get("issue_clip_paths", []):
        if not isinstance(clip, dict):
            continue
        path = clip.get("path")
        if not isinstance(path, str):
            continue
        url = _artifact_url(result["run_id"], path)
        if url is not None:
            issue_clip_urls.append({**clip, "url": url})

    return {
        "run_id": result["run_id"],
        "run_dir": result["run_dir"],
        "annotated_video_url": _artifact_url(
            result["run_id"], result["annotated_video_path"]
        ),
        "issue_thumbnail_urls": issue_thumbnail_urls,
        "issue_clip_urls": issue_clip_urls,
        "artifact_urls": _artifact_urls(result),
        "final_report_url": f"/api/artifacts/{result['run_id']}/final_report.json",
        "report": result["final_report"],
    }


@server.post("/api/analyze")
async def analyze_api(
    video: UploadFile | None = File(default=None),
    goal: str = Form(default="beginner_practice"),
    experience_level: str = Form(default="beginner"),
    intended_exercise: str = Form(default="auto"),
    intended_variation: str = Form(default=""),
    limitations: str = Form(default="[]"),
    equipment: str = Form(default="bodyweight"),
    bypass_verifier: str = Form(default="true"),
) -> dict[str, Any]:
    profile = _profile_input(
        goal=goal,
        experience_level=experience_level,
        intended_exercise=intended_exercise,
        intended_variation=intended_variation,
        limitations=limitations,
        equipment=equipment,
    )
    video_path = await _save_upload(video)
    try:
        result = _run_analysis_pipeline(
            video_path,
            profile,
            _parse_bool_form(bypass_verifier),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_friendly_error(exc)) from exc
    finally:
        if video_path is not None:
            Path(video_path).unlink(missing_ok=True)

    return _analysis_response(result)


@server.post("/api/analyze/stream")
async def analyze_stream_api(
    video: UploadFile | None = File(default=None),
    goal: str = Form(default="beginner_practice"),
    experience_level: str = Form(default="beginner"),
    intended_exercise: str = Form(default="auto"),
    intended_variation: str = Form(default=""),
    limitations: str = Form(default="[]"),
    equipment: str = Form(default="bodyweight"),
    bypass_verifier: str = Form(default="true"),
) -> StreamingResponse:
    profile = _profile_input(
        goal=goal,
        experience_level=experience_level,
        intended_exercise=intended_exercise,
        intended_variation=intended_variation,
        limitations=limitations,
        equipment=equipment,
    )
    video_path = await _save_upload(video)
    events: Queue[dict[str, Any] | None] = Queue()

    def worker() -> None:
        try:
            result = _run_analysis_pipeline(
                video_path,
                profile,
                _parse_bool_form(bypass_verifier),
                events.put,
            )
            events.put({"type": "complete", "result": _analysis_response(result)})
        except Exception as exc:  # pragma: no cover - surfaced to browser clients
            events.put({"type": "error", "detail": _friendly_error(exc)})
        finally:
            if video_path is not None:
                Path(video_path).unlink(missing_ok=True)
            events.put(None)

    def event_stream() -> Any:
        thread = Thread(target=worker, daemon=True)
        thread.start()
        while True:
            event = events.get()
            if event is None:
                break
            yield f"{json.dumps(event)}\n"
        thread.join(timeout=1)

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache"},
    )


if __name__ == "__main__":
    server.launch(_frontend=False)
