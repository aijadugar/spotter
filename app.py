from __future__ import annotations

import json
from queue import Queue
import re
import shutil
import sys
import tempfile
from pathlib import Path
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

APP_DESCRIPTION = (
    "Upload a short workout clip, tune the athlete context, and generate an annotated "
    "form-review report with structured artifacts."
)


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
            shutil.copyfileobj(video.file, temp_video)
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
            events.put({"type": "error", "detail": str(exc)})
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
