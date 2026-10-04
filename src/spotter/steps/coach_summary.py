from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from spotter.contracts import (
    CoachSummary,
    ExerciseClassification,
    IssueMarkers,
    RepAnalysis,
    Reps,
    UserProfile,
    Variation,
    validate_contract,
)
from spotter.knowledge_cards import KnowledgeCard, retrieve_cards
from spotter.slm.prompting import build_coach_summary_prompt
from spotter.slm.providers import CoachSummaryModel, get_coach_summary_model
from spotter.steps.coach_summary_fallback import build_fallback_summary

_SUMMARY_KEYS = {
    "summary",
    "what_you_did",
    "what_looked_good",
    "what_changed_across_reps",
    "valid_variation_vs_issue",
    "top_fixes",
    "next_session_plan",
    "confidence_notes",
}


@dataclass(frozen=True)
class CoachSummaryResult:
    summary: CoachSummary
    provider: str
    model: str
    source: str


def _text_preview(text: str, *, limit: int = 240) -> str:
    preview = " ".join(text.strip().split())
    if len(preview) > limit:
        return f"{preview[:limit]}..."
    return preview or "<empty>"


def _unwrap_json_payload(payload: Any) -> Any:
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        payload = payload[0]
    if isinstance(payload, dict):
        for key in ("coach_summary", "summary_json", "output"):
            nested_payload = payload.get(key)
            if isinstance(nested_payload, dict):
                return nested_payload
    return payload


def _summary_key_count(payload: Any) -> int:
    if not isinstance(payload, dict):
        return 0
    return len(_SUMMARY_KEYS.intersection(payload))


def _best_json_object_candidate(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    best_payload: dict[str, Any] | None = None
    best_score = 0
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        payload = _unwrap_json_payload(payload)
        score = _summary_key_count(payload)
        if isinstance(payload, dict) and score > best_score:
            best_payload = payload
            best_score = score
            if score == len(_SUMMARY_KEYS):
                break
    return best_payload


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        payload = _best_json_object_candidate(text)
        if payload is None:
            raise ValueError(
                "Coach summary model output was not valid JSON. "
                f"Parser error: {exc}. Output preview: {_text_preview(text)}"
            ) from exc
    else:
        payload = _unwrap_json_payload(payload)

    if not isinstance(payload, dict):
        raise ValueError(
            "Coach summary model output must be a JSON object. "
            f"Got {type(payload).__name__}. Output preview: {_text_preview(text)}"
        )
    if _summary_key_count(payload) == 0:
        raise ValueError(
            "Coach summary model output did not contain a coach summary JSON object. "
            f"Output preview: {_text_preview(text)}"
        )
    return payload


def _summary_list_field(payload: dict[str, Any], key: str) -> list[str]:
    value = payload[key]
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError(f"Coach summary model output field {key!r} must be a string array")
    if len(value) >= 8 and all(isinstance(item, str) and len(item) <= 1 for item in value):
        return ["".join(value)]
    return [str(item) for item in value]


def _summary_from_payload(payload: dict[str, Any]) -> CoachSummary:
    missing_fields = sorted(_SUMMARY_KEYS.difference(payload))
    if missing_fields:
        raise ValueError(
            "Coach summary model output is missing required field(s): "
            f"{', '.join(missing_fields)}"
        )
    summary = CoachSummary(
        summary=str(payload["summary"]),
        what_you_did=_summary_list_field(payload, "what_you_did"),
        what_looked_good=_summary_list_field(payload, "what_looked_good"),
        what_changed_across_reps=_summary_list_field(payload, "what_changed_across_reps"),
        valid_variation_vs_issue=_summary_list_field(payload, "valid_variation_vs_issue"),
        top_fixes=_summary_list_field(payload, "top_fixes"),
        next_session_plan=_summary_list_field(payload, "next_session_plan"),
        confidence_notes=_summary_list_field(payload, "confidence_notes"),
    )
    validate_contract("coach_summary.json", summary)
    return summary


def _fallback(
    *,
    profile: UserProfile,
    classification: ExerciseClassification,
    reps: Reps,
    analysis: RepAnalysis,
    variation: Variation,
    issues: IssueMarkers,
    cards: list[KnowledgeCard],
    failure_reason: str | None = None,
) -> CoachSummary:
    return build_fallback_summary(
        profile=profile,
        classification=classification,
        reps=reps,
        analysis=analysis,
        variation=variation,
        issues=issues,
        cards=cards,
        failure_reason=failure_reason,
    )


def run(
    profile: UserProfile,
    classification: ExerciseClassification,
    reps: Reps,
    analysis: RepAnalysis,
    variation: Variation,
    issues: IssueMarkers,
    *,
    cards: list[KnowledgeCard] | None = None,
    model: CoachSummaryModel | None = None,
) -> CoachSummary:
    return run_with_metadata(
        profile,
        classification,
        reps,
        analysis,
        variation,
        issues,
        cards=cards,
        model=model,
    ).summary


def run_with_metadata(
    profile: UserProfile,
    classification: ExerciseClassification,
    reps: Reps,
    analysis: RepAnalysis,
    variation: Variation,
    issues: IssueMarkers,
    *,
    cards: list[KnowledgeCard] | None = None,
    model: CoachSummaryModel | None = None,
) -> CoachSummaryResult:
    cards = cards or retrieve_cards(
        profile=profile,
        classification=classification,
        variation=variation,
        issues=issues,
    )
    provider = get_coach_summary_model() if model is None else model
    if provider is None:
        return CoachSummaryResult(
            summary=_fallback(
                profile=profile,
                classification=classification,
                reps=reps,
                analysis=analysis,
                variation=variation,
                issues=issues,
                cards=cards,
                failure_reason="remote provider unavailable",
            ),
            provider="none",
            model="none",
            source="fallback_initial",
        )

    prompt = build_coach_summary_prompt(
        profile=profile,
        classification=classification,
        reps=reps,
        analysis=analysis,
        variation=variation,
        issues=issues,
        cards=cards,
    )

    try:
        generation = provider.generate_summary(prompt)
        payload = _extract_json_object(generation.text)
        return CoachSummaryResult(
            summary=_summary_from_payload(payload),
            provider=generation.provider,
            model=generation.model,
            source="model_or_local",
        )
    except Exception as exc:
        return CoachSummaryResult(
            summary=_fallback(
                profile=profile,
                classification=classification,
                reps=reps,
                analysis=analysis,
                variation=variation,
                issues=issues,
                cards=cards,
                failure_reason=str(exc),
            ),
            provider="fallback",
            model="fallback",
            source="fallback_initial",
        )
