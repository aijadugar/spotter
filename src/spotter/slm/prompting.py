from __future__ import annotations

import json
from typing import Any

from spotter.contracts import (
    ExerciseClassification,
    IssueMarkers,
    RepAnalysis,
    Reps,
    UserProfile,
    Variation,
    to_dict,
)
from spotter.knowledge_cards import KnowledgeCard, prioritized_coaching_points


def _compact_rep_metric(item: Any) -> dict[str, Any]:
    payload = to_dict(item)
    return {
        key: payload[key]
        for key in (
            "rep_id",
            "duration_sec",
            "range_of_motion_score",
            "stability_score",
            "symmetry_score",
            "variation_hints",
        )
        if key in payload
    }


def _compact_issue(issue: Any) -> dict[str, Any]:
    payload = to_dict(issue)
    compact = {
        key: payload[key]
        for key in ("rep_id", "issue", "severity", "start_sec", "end_sec")
        if key in payload
    }
    evidence = payload.get("evidence")
    if isinstance(evidence, dict):
        compact["evidence_keys"] = sorted(str(key) for key in evidence.keys())[:8]
    return compact


def _compact_card(card: KnowledgeCard) -> dict[str, Any]:
    return {
        "card_id": card.card_id,
        "card_type": card.card_type,
        "title": card.title,
        "coaching_points": list(card.coaching_points[:2]),
    }


def _selected_aggregate_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "avg_rom_score",
            "avg_stability_score",
            "avg_symmetry_score",
            "fatigue_trend_rom_delta",
            "pose_valid_ratio",
        )
        if key in metrics
    }


def build_summary_evidence(
    *,
    profile: UserProfile,
    classification: ExerciseClassification,
    reps: Reps,
    analysis: RepAnalysis,
    variation: Variation,
    issues: IssueMarkers,
    cards: list[KnowledgeCard],
) -> dict[str, Any]:
    issue_counts: dict[str, int] = {}
    top_issue_intervals = []
    for issue in sorted(issues.issues, key=lambda item: item.severity, reverse=True):
        issue_counts[issue.issue] = issue_counts.get(issue.issue, 0) + 1
        if len(top_issue_intervals) < 3:
            top_issue_intervals.append(_compact_issue(issue))

    sorted_issues = sorted(issues.issues, key=lambda item: item.severity, reverse=True)
    rep_metrics = [_compact_rep_metric(item) for item in analysis.items[:4]]
    return {
        "user_profile": {
            "goal": profile.goal,
            "experience_level": profile.experience_level,
            "intended_exercise": profile.intended_exercise,
            "equipment": profile.equipment,
        },
        "exercise_classification": {
            "exercise": classification.exercise,
            "confidence": classification.confidence,
            "fallback_required": classification.fallback_required,
        },
        "variation": {
            "detected_variation": variation.detected_variation,
            "variation_confidence": variation.variation_confidence,
            "not_issues": variation.not_issues,
        },
        "rep_summary": {
            "rep_count": len(reps.reps),
            "aggregate_metrics": _selected_aggregate_metrics(analysis.aggregate_metrics),
            "rep_metrics": rep_metrics,
        },
        "issue_summary": {
            "issue_counts": issue_counts,
            "issues": [_compact_issue(issue) for issue in sorted_issues[:3]],
            "top_issue_intervals": top_issue_intervals[:2],
        },
        "priority_cues": prioritized_coaching_points(cards, limit=4),
        "knowledge_cards": [_compact_card(card) for card in cards[:3]],
    }


def build_coach_summary_prompt(
    *,
    profile: UserProfile,
    classification: ExerciseClassification,
    reps: Reps,
    analysis: RepAnalysis,
    variation: Variation,
    issues: IssueMarkers,
    cards: list[KnowledgeCard],
) -> str:
    evidence = build_summary_evidence(
        profile=profile,
        classification=classification,
        reps=reps,
        analysis=analysis,
        variation=variation,
        issues=issues,
        cards=cards,
    )
    expected_schema = {
        "summary": "string",
        "what_you_did": ["string"],
        "what_looked_good": ["string"],
        "what_changed_across_reps": ["string"],
        "valid_variation_vs_issue": ["string"],
        "top_fixes": ["string"],
        "next_session_plan": ["string"],
        "confidence_notes": ["string"],
    }

    instructions = {
        "response_format": {
            "type": "json_object",
            "schema": expected_schema,
        },
        "task": "Generate a grounded coach summary from structured exercise-analysis artifacts.",
        "rules": [
            "Return exactly one JSON object that matches response_format.schema.",
            "The first output character must be `{`.",
            "Do not return an array, string, null, schema description, or example payload.",
            "Use only the evidence JSON and retrieved knowledge cards.",
            "Do not infer new issues that are absent from issue_summary.issues.",
            "Do not diagnose injuries, pain, mobility deficits, or pathology.",
            "Do not claim injury prevention.",
            "Do not treat a valid detected variation or not-issue label as an error.",
            "Use priority_cues as the first source for phrasing top fixes and next-session guidance.",
            "When referencing an issue or variation, include its exact label in "
            "backticks at least once.",
            "If evidence is limited or confidence is low, say so in confidence_notes.",
            "Return JSON only. No markdown fences. No extra commentary.",
        ],
    }
    final_contract = {
        "final_output_contract": {
            "required_keys": list(expected_schema.keys()),
            "array_keys": [
                key for key, value in expected_schema.items() if isinstance(value, list)
            ],
            "rules": [
                "Return exactly one JSON object.",
                "Do not echo evidence JSON.",
                "Do not add text before or after the JSON object.",
            ],
        }
    }
    json_dump_kwargs = {"ensure_ascii": False, "separators": (",", ":")}
    return "\n\n".join(
        [
            json.dumps(instructions, **json_dump_kwargs),
            json.dumps(evidence, **json_dump_kwargs),
            json.dumps(final_contract, **json_dump_kwargs),
            'Output the final coach summary JSON object now. Start with {"summary":',
        ]
    )
