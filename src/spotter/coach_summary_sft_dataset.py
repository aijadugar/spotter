from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any

from spotter.contracts import (
    CoachSummary,
    ExerciseClassification,
    IssueMarker,
    IssueMarkers,
    Rep,
    RepAnalysis,
    RepAnalysisItem,
    Reps,
    UserProfile,
    Variation,
)
from spotter.knowledge_cards import retrieve_cards
from spotter.slm.prompting import build_summary_evidence


SYSTEM_PROMPT = (
    "You are Spotter's grounded coach-summary model. "
    "Use only the provided structured evidence and knowledge cards. "
    "Return coach_summary.json as JSON only."
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _profile_from_dict(payload: dict[str, Any]) -> UserProfile:
    return UserProfile(
        goal=str(payload["goal"]),
        experience_level=str(payload["experience_level"]),
        intended_exercise=str(payload.get("intended_exercise", "auto")),
        intended_variation=payload.get("intended_variation"),
        known_limitations=[str(item) for item in payload.get("known_limitations", [])],
        equipment=str(payload.get("equipment", "unknown")),
    )


def _classification_from_dict(payload: dict[str, Any]) -> ExerciseClassification:
    return ExerciseClassification(
        exercise=str(payload["exercise"]),
        confidence=float(payload["confidence"]),
        window_predictions=list(payload.get("window_predictions", [])),
        fallback_required=bool(payload.get("fallback_required", False)),
    )


def _reps_from_dict(payload: dict[str, Any]) -> Reps:
    return Reps(
        exercise=str(payload["exercise"]),
        reps=[
            Rep(
                rep_id=int(item["rep_id"]),
                start_frame=int(item["start_frame"]),
                mid_frame=int(item["mid_frame"]),
                end_frame=int(item["end_frame"]),
                start_sec=float(item["start_sec"]),
                mid_sec=float(item["mid_sec"]),
                end_sec=float(item["end_sec"]),
            )
            for item in payload.get("reps", [])
        ],
        partial_reps=list(payload.get("partial_reps", [])),
    )


def _analysis_from_dict(payload: dict[str, Any]) -> RepAnalysis:
    return RepAnalysis(
        exercise=str(payload["exercise"]),
        items=[
            RepAnalysisItem(
                rep_id=int(item["rep_id"]),
                duration_sec=float(item["duration_sec"]),
                range_of_motion_score=float(item["range_of_motion_score"]),
                stability_score=float(item["stability_score"]),
                symmetry_score=float(item["symmetry_score"]),
                metrics=dict(item.get("metrics", {})),
                variation_hints=[str(value) for value in item.get("variation_hints", [])],
            )
            for item in payload.get("items", [])
        ],
        aggregate_metrics=dict(payload.get("aggregate_metrics", {})),
    )


def _variation_from_dict(payload: dict[str, Any]) -> Variation:
    return Variation(
        exercise=str(payload["exercise"]),
        detected_variation=str(payload["detected_variation"]),
        variation_confidence=float(payload["variation_confidence"]),
        not_issues=[str(item) for item in payload.get("not_issues", [])],
    )


def _issues_from_dict(payload: dict[str, Any]) -> IssueMarkers:
    return IssueMarkers(
        issues=[
            IssueMarker(
                rep_id=int(item["rep_id"]),
                issue=str(item["issue"]),
                severity=float(item["severity"]),
                start_frame=int(item["start_frame"]),
                end_frame=int(item["end_frame"]),
                start_sec=float(item["start_sec"]),
                end_sec=float(item["end_sec"]),
                affected_joints=[str(value) for value in item.get("affected_joints", [])],
                evidence=dict(item.get("evidence", {})),
            )
            for item in payload.get("issues", [])
        ]
    )


def _summary_from_dict(payload: dict[str, Any]) -> CoachSummary:
    return CoachSummary(
        summary=str(payload["summary"]),
        what_you_did=[str(item) for item in payload.get("what_you_did", [])],
        what_looked_good=[str(item) for item in payload.get("what_looked_good", [])],
        what_changed_across_reps=[
            str(item) for item in payload.get("what_changed_across_reps", [])
        ],
        valid_variation_vs_issue=[
            str(item) for item in payload.get("valid_variation_vs_issue", [])
        ],
        top_fixes=[str(item) for item in payload.get("top_fixes", [])],
        next_session_plan=[str(item) for item in payload.get("next_session_plan", [])],
        confidence_notes=[str(item) for item in payload.get("confidence_notes", [])],
    )


def build_sft_row_from_run_dir(run_dir: Path) -> dict[str, Any]:
    profile = _profile_from_dict(_load_json(run_dir / "user_profile.json"))
    classification = _classification_from_dict(
        _load_json(run_dir / "exercise_classification.json")
    )
    reps = _reps_from_dict(_load_json(run_dir / "reps.json"))
    analysis = _analysis_from_dict(_load_json(run_dir / "rep_analysis.json"))
    variation = _variation_from_dict(_load_json(run_dir / "variation.json"))
    issues = _issues_from_dict(_load_json(run_dir / "issue_markers.json"))
    summary = _summary_from_dict(_load_json(run_dir / "coach_summary.json"))

    cards = retrieve_cards(
        profile=profile,
        classification=classification,
        variation=variation,
        issues=issues,
    )
    evidence = build_summary_evidence(
        profile=profile,
        classification=classification,
        reps=reps,
        analysis=analysis,
        variation=variation,
        issues=issues,
        cards=cards,
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(evidence, ensure_ascii=False, indent=2),
            },
            {
                "role": "assistant",
                "content": json.dumps(asdict(summary), ensure_ascii=False, indent=2),
            },
        ],
        "metadata": {
            "run_dir": str(run_dir),
            "exercise": classification.exercise,
            "goal": profile.goal,
            "equipment": profile.equipment,
            "issue_count": len(issues.issues),
            "variation": variation.detected_variation,
        },
    }


def collect_run_dirs(runs_dir: Path) -> list[Path]:
    run_dirs = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        required = [
            "user_profile.json",
            "exercise_classification.json",
            "reps.json",
            "rep_analysis.json",
            "variation.json",
            "issue_markers.json",
            "coach_summary.json",
        ]
        if all((child / filename).is_file() for filename in required):
            run_dirs.append(child)
    return run_dirs


def split_sft_rows(
    rows: list[dict[str, Any]],
    *,
    eval_count: int,
    seed: int = 7,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = list(rows)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    eval_count = max(0, min(eval_count, len(ordered)))
    eval_rows = ordered[:eval_count]
    train_rows = ordered[eval_count:]
    return train_rows, eval_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

