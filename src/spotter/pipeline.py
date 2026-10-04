from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from spotter.artifacts import write_json
from spotter.contracts import UserProfile, Verification, to_dict
from spotter.env import env_truthy, load_local_env
from spotter.exercises import create_exercise_strategy
from spotter.knowledge_cards import retrieve_cards_with_metadata
from spotter.steps import (
    annotated_renderer,
    coach_summary,
    coach_summary_fallback,
    exercise_classifier,
    pose_cleaning,
    pose_landmarker,
    progress_plan as progress_plan_step,
    session_memory,
    session_record,
    speech,
    verifier,
    video_qc,
)


RUNS_DIR = Path("runs")
ProgressCallback = Callable[[dict[str, Any]], None]
BYPASS_VERIFIER_ENV = "SPOTTER_COACH_SUMMARY_BYPASS_VERIFIER"
DEFAULT_BYPASS_VERIFIER = True


def _bypass_verifier_enabled(requested: bool | None) -> bool:
    if requested is not None:
        return requested
    configured = os.getenv(BYPASS_VERIFIER_ENV)
    if configured is None:
        return DEFAULT_BYPASS_VERIFIER
    return env_truthy(configured)


def _disabled_verification() -> Verification:
    return Verification(
        passed=True,
        checks={"verifier_disabled": True},
        notes=["Coach summary verifier is disabled for this run."],
    )


def _env_mock_mode(video_path: str | None) -> bool:
    configured = os.getenv("SPOTTER_MOCK_MODE")
    if configured is None:
        return video_path is None

    value = configured.strip().lower()
    return value not in {"0", "false", "no", "off"}


def run_pipeline(
    video_path: str | None,
    profile_input: dict[str, Any],
    *,
    mock: bool | None = None,
    bypass_verifier: bool | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    load_local_env()
    mock_mode = _env_mock_mode(video_path) if mock is None else mock
    bypass_verifier_enabled = _bypass_verifier_enabled(bypass_verifier)

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_dir = RUNS_DIR / run_id
    artifact_index: list[dict[str, str]] = []

    def write_artifact(filename: str, payload: Any) -> None:
        path = write_json(run_dir, filename, payload)
        artifact_index.append(
            {
                "name": filename,
                "path": str(path),
                "contract": filename,
            }
        )

    def emit(step: str, status: str, text: str, **payload: Any) -> None:
        if progress is None:
            return
        progress(
            {
                "type": "progress",
                "step": step,
                "status": status,
                "text": text,
                "payload": payload,
            }
        )

    profile = UserProfile(
        goal=profile_input["goal"],
        experience_level=profile_input["experience_level"],
        intended_exercise=profile_input.get("intended_exercise", "auto"),
        intended_variation=profile_input.get("intended_variation"),
        known_limitations=profile_input.get("known_limitations", []),
        equipment=profile_input.get("equipment", "unknown"),
    )
    write_artifact("user_profile.json", profile)

    emit(
        "quality",
        "active",
        "First up, I am checking if the video is clear enough to coach from.",
    )
    manifest = video_qc.run(video_path)
    write_artifact("video_manifest.json", manifest)
    emit(
        "quality",
        "done",
        (
            "Quick note: the video has a few things to watch."
            if manifest.quality_warnings
            else "Nice, your video quality looks solid."
        ),
        warnings=manifest.quality_warnings,
        analysis_allowed=manifest.analysis_allowed,
    )

    emit(
        "pose",
        "active",
        "Now I am mapping your posture and tracking the key body landmarks.",
    )
    pose_sequence = pose_landmarker.run(manifest, mock=mock_mode)
    cleaned_pose_sequence = pose_cleaning.run(pose_sequence)
    write_artifact("pose_sequence.json", cleaned_pose_sequence)
    pose_source = (
        cleaned_pose_sequence.frames[0].pose_quality.get("source")
        if cleaned_pose_sequence.frames
        else "none"
    )
    emit(
        "pose",
        "done",
        "Posture tracking is done. I found the key landmarks I need.",
        frame_count=len(cleaned_pose_sequence.frames),
        pose_source=pose_source,
        pose_valid_ratio=cleaned_pose_sequence.pose_valid_ratio,
    )

    emit("exercise", "active", "Let me figure out which exercise you are doing.")
    classification = exercise_classifier.run(cleaned_pose_sequence, profile, mock=mock_mode)
    write_artifact("exercise_classification.json", classification)
    emit(
        "exercise",
        "done",
        f"Looks like you are doing {classification.exercise.replace('_', ' ')}.",
        exercise=classification.exercise,
        confidence=classification.confidence,
    )

    exercise = create_exercise_strategy(
        classification.exercise,
        video_manifest=manifest,
        pose_sequence=cleaned_pose_sequence,
        profile=profile,
    )

    emit("reps", "active", "Counting your reps now. One clean rep at a time.")
    reps, rep_debug = exercise.count()
    write_artifact("reps.json", reps)
    write_artifact("rep_debug.json", rep_debug)
    emit(
        "reps",
        "done",
        (
            f"I counted {len(reps.reps)} {classification.exercise.replace('_', ' ')} "
            "reps in this set."
        ),
        rep_count=len(reps.reps),
        exercise=classification.exercise,
    )

    emit(
        "issues",
        "active",
        "Almost there. I am checking the moments that may need a small fix.",
    )
    analysis = exercise.analyze_reps(reps)
    write_artifact("rep_analysis.json", analysis)

    variation = exercise.resolve_variation(analysis)
    write_artifact("variation.json", variation)

    issues = exercise.mark_issues(reps, analysis, variation)
    write_artifact("issue_markers.json", issues)
    emit(
        "issues",
        "done",
        (
            f"I found {len(issues.issues)} coaching point"
            f"{'' if len(issues.issues) == 1 else 's'} worth reviewing."
            if issues.issues
            else "Good news, I did not spot any clear form issues in this set."
        ),
        issue_count=len(issues.issues),
    )

    emit("render", "active", "I am preparing your annotated video and issue clips.")
    render_artifacts = annotated_renderer.run(
        manifest,
        cleaned_pose_sequence,
        reps,
        issues,
        run_dir,
    )
    emit(
        "render",
        "done",
        (
            "Your annotated video is ready."
            if render_artifacts.annotated_video_path
            else "I could not render an annotated video, but the report is ready."
        ),
        annotated_video_path=render_artifacts.annotated_video_path,
        issue_clip_count=len(render_artifacts.issue_clip_paths),
    )

    emit(
        "coach",
        "active",
        "I am turning the scan into coaching notes you can use right away.",
    )
    analysis_mode = "mock" if mock_mode else "real"
    mock_steps = ["coach_summary"]
    if not bypass_verifier_enabled:
        mock_steps.append("verifier")
    if mock_mode:
        mock_steps.insert(0, "exercise_classifier")

    knowledge_retrieval = retrieve_cards_with_metadata(
        profile=profile,
        classification=classification,
        variation=variation,
        issues=issues,
    )
    summary_cards = knowledge_retrieval.cards
    coach_result = coach_summary.run_with_metadata(
        profile,
        classification,
        reps,
        analysis,
        variation,
        issues,
        cards=summary_cards,
    )
    summary = coach_result.summary
    coach_summary_source = coach_result.source
    coach_summary_provider = coach_result.provider
    coach_summary_model = coach_result.model
    coach_summary_verifier_bypassed = bypass_verifier_enabled
    if bypass_verifier_enabled:
        verification = _disabled_verification()
    else:
        verification = verifier.run(
            summary,
            issues,
            variation,
            classification=classification,
            analysis=analysis,
            reps=reps,
        )
        if not verification.passed:
            summary = coach_summary_fallback.build_fallback_summary(
                profile=profile,
                classification=classification,
                reps=reps,
                analysis=analysis,
                variation=variation,
                issues=issues,
                cards=summary_cards,
                failure_reason="; ".join(verification.notes) or "verification_failed",
            )
            coach_summary_source = "fallback_after_verification"
            coach_summary_provider = coach_result.provider
            coach_summary_model = coach_result.model
            verification = verifier.run(
                summary,
                issues,
                variation,
                classification=classification,
                analysis=analysis,
                reps=reps,
            )
    write_artifact("coach_summary.json", summary)
    write_artifact("verification.json", verification)
    emit(
        "coach",
        "done",
        "Coach notes are ready.",
        verification_passed=verification.passed,
    )

    profile_key = os.getenv("SPOTTER_PROFILE_KEY", "default")
    memory = session_memory.get_session_memory()
    history = memory.get_history(profile_key)

    emit("memory", "active", "Checking what we worked on in past sessions.")
    emit(
        "memory",
        "done",
        f"I found {len(history)} previous session(s).",
        history_count=len(history),
    )

    findings = progress_plan_step.SessionFindings(
        exercise=classification.exercise,
        rep_count=len(reps.reps),
        aggregate_metrics=analysis.aggregate_metrics,
        issue_labels=[issue.issue for issue in issues.issues],
    )
    planner = progress_plan_step.get_progress_planner()
    try:
        plan = planner.plan(history, findings)
    except Exception as exc:  # pragma: no cover - defensive
        plan = progress_plan_step.build_fallback_plan(
            history, findings, failure_reason=f"planner_error:{exc}"
        )
    write_artifact("progress_plan.json", plan)
    emit("plan", "done", "Here is your plan for next time.", focus=plan.focus)

    record = session_record.build_session_record(
        run_id=run_id,
        profile_key=profile_key,
        timestamp=datetime.now(timezone.utc).isoformat(),
        classification=classification,
        reps=reps,
        analysis=analysis,
        issues=issues,
    )
    memory.append(record)
    write_artifact("session_record.json", record)

    lines = speech.build_speech_lines(summary, plan)
    speech_result = speech.get_speech_synthesizer().synthesize(
        lines, language=speech.default_language(), out_dir=run_dir
    )
    write_artifact("speech.json", speech_result)

    final_report = {
        "run_id": run_id,
        "profile": to_dict(profile),
        "video_manifest": to_dict(manifest),
        "exercise": to_dict(classification),
        "reps": to_dict(reps),
        "rep_analysis": to_dict(analysis),
        "variation": to_dict(variation),
        "issue_markers": to_dict(issues),
        "coach_summary": to_dict(summary),
        "verification": to_dict(verification),
        "progress_plan": to_dict(plan),
        "session_record": to_dict(record),
        "speech": to_dict(speech_result),
        "artifacts": {
            "run_dir": str(run_dir),
            "annotated_video_path": render_artifacts.annotated_video_path,
            "issue_thumbnail_paths": render_artifacts.issue_thumbnail_paths,
            "issue_clip_paths": render_artifacts.issue_clip_paths,
            "rep_debug_path": str(run_dir / "rep_debug.json"),
            "analysis_mode": analysis_mode,
            "pose_source": pose_source,
            "mock_steps": mock_steps,
            "coach_summary_source": coach_summary_source,
            "coach_summary_provider": coach_summary_provider,
            "coach_summary_model": coach_summary_model,
            "coach_summary_verifier_bypassed": coach_summary_verifier_bypassed,
            "coach_summary_verifier_bypass_requested": bypass_verifier_enabled,
            "knowledge_card_pack_paths": list(knowledge_retrieval.loaded_pack_paths),
            "knowledge_external_cards_loaded": knowledge_retrieval.external_cards_loaded,
            "knowledge_external_cards_retrieved": knowledge_retrieval.external_cards_retrieved,
        },
    }
    write_artifact("final_report.json", final_report)

    run_manifest = {
        "run_id": run_id,
        "mock_mode": mock_mode,
        "artifacts": artifact_index,
    }
    write_json(run_dir, "manifest.json", run_manifest)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "annotated_video_path": render_artifacts.annotated_video_path,
        "issue_thumbnail_paths": render_artifacts.issue_thumbnail_paths,
        "issue_clip_paths": render_artifacts.issue_clip_paths,
        "manifest_path": str(run_dir / "manifest.json"),
        "final_report": final_report,
    }
