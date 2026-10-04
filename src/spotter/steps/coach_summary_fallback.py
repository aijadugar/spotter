from __future__ import annotations

from spotter.contracts import (
    CoachSummary,
    ExerciseClassification,
    IssueMarkers,
    RepAnalysis,
    Reps,
    UserProfile,
    Variation,
)
from spotter.knowledge_cards import KnowledgeCard, get_card_by_label


def _metric_score(metric: float, *, high: float = 0.8, medium: float = 0.65) -> str:
    if metric >= high:
        return "looked steady"
    if metric >= medium:
        return "looked fairly consistent"
    return "showed room to tighten up"


def _issue_cards(issues: IssueMarkers) -> list[KnowledgeCard]:
    cards: list[KnowledgeCard] = []
    seen: set[str] = set()
    for issue in issues.issues:
        card = get_card_by_label(issue.issue)
        if card is None or card.card_id in seen:
            continue
        seen.add(card.card_id)
        cards.append(card)
    return cards


def _top_issue_labels(issues: IssueMarkers) -> list[str]:
    ranked = sorted(issues.issues, key=lambda item: item.severity, reverse=True)
    labels: list[str] = []
    for issue in ranked:
        if issue.issue not in labels:
            labels.append(issue.issue)
        if len(labels) == 3:
            break
    return labels


def _confidence_notes(
    classification: ExerciseClassification,
    analysis: RepAnalysis,
    variation: Variation,
    reps: Reps,
) -> list[str]:
    notes: list[str] = []
    if classification.confidence < 0.7:
        notes.append(
            "Exercise classification confidence is limited at "
            f"{classification.confidence:.0%}, so treat the summary as a cautious read."
        )
    if variation.variation_confidence < 0.7:
        notes.append(
            "Variation confidence is "
            f"{variation.variation_confidence:.0%}, so the variation call should be "
            "treated as contextual rather than absolute."
        )
    pose_valid_ratio = float(analysis.aggregate_metrics.get("pose_valid_ratio", 1.0))
    if pose_valid_ratio < 0.85:
        notes.append(
            f"Pose coverage is {pose_valid_ratio:.0%}, so some coaching points may be "
            "based on limited landmark evidence."
        )
    if not reps.reps:
        notes.append("No full reps were segmented, so the summary stays conservative.")
    if not notes:
        notes.append(
            "This summary stays grounded to the current JSON evidence and may "
            "miss details outside that evidence."
        )
    return notes


def build_fallback_summary(
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
    del cards
    rep_count = len(reps.reps)
    avg_rom = float(analysis.aggregate_metrics.get("avg_rom_score", 0.0))
    avg_stability = float(analysis.aggregate_metrics.get("avg_stability_score", 0.0))
    avg_symmetry = float(analysis.aggregate_metrics.get("avg_symmetry_score", 0.0))
    fatigue_delta = float(analysis.aggregate_metrics.get("fatigue_trend_rom_delta", 0.0))
    issue_labels = _top_issue_labels(issues)
    issue_cards = _issue_cards(issues)
    issue_count = len(issues.issues)

    what_you_did = [
        (
            f"You completed {rep_count} detected `"
            f"{classification.exercise}` reps with the variation labeled as "
            f"`{variation.detected_variation}`."
        )
    ]
    if profile.goal:
        what_you_did.append(f"Your selected training goal was `{profile.goal}`.")

    what_looked_good = [
        f"Range of motion {_metric_score(avg_rom)} overall ({avg_rom:.0%}).",
        f"Rep stability {_metric_score(avg_stability)} overall ({avg_stability:.0%}).",
        f"Left-right symmetry {_metric_score(avg_symmetry)} overall ({avg_symmetry:.0%}).",
    ]
    if issue_count == 0:
        what_looked_good.append(
            "No sustained issue markers were detected in the current JSON evidence."
        )

    if rep_count <= 1:
        what_changed = [
            "There was not enough rep-to-rep data to describe a clear trend "
            "across reps."
        ]
    elif fatigue_delta <= -0.08:
        what_changed = [
            f"Range of motion trended down across reps (delta {fatigue_delta:.2f}), "
            "which suggests the later reps were less consistent."
        ]
    elif fatigue_delta >= 0.08:
        what_changed = [
            f"Range of motion improved slightly across reps (delta {fatigue_delta:.2f}) "
            "as the set went on."
        ]
    else:
        what_changed = ["Rep-to-rep range stayed fairly stable across the set."]

    variation_notes = [
        f"The detected variation was `{variation.detected_variation}`, so it should be "
        "treated as context rather than a fault by default."
    ]
    if variation.not_issues:
        variation_notes.append(
            "The variation step marked "
            + ", ".join(f"`{label}`" for label in variation.not_issues)
            + " as not-issue context."
        )
    if issue_labels:
        variation_notes.append(
            "The actual issue markers in this set were "
            + ", ".join(f"`{label}`" for label in issue_labels)
            + "."
        )
    else:
        variation_notes.append(
            "No issue labels were present, so there is nothing to overcorrect."
        )

    top_fixes: list[str] = []
    for card in issue_cards[:3]:
        top_fixes.append(card.coaching_points[0])
    if not top_fixes:
        top_fixes.append(
            "Keep the same camera angle and repeat the set to confirm the current pattern."
        )

    next_session_plan = [
        "Start with 1 easy set of controlled reps using the same camera angle.",
        (
            "Keep your top focus on "
            + (
                ", ".join(f"`{label}`" for label in issue_labels)
                if issue_labels
                else "repeatable control"
            )
            + "."
        ),
        "Compare the next run against this report to see whether the same labels show up again.",
    ]

    confidence_notes = _confidence_notes(classification, analysis, variation, reps)
    if failure_reason:
        confidence_notes.append(
            "Fallback summary was used because the generated summary did not pass "
            f"verification: {failure_reason}"
        )

    issue_text = (
        "No issue markers were present."
        if not issue_labels
        else "The highest-priority issue labels were "
        + ", ".join(f"`{label}`" for label in issue_labels)
        + "."
    )
    summary = (
        "This grounded summary is based on structured artifacts for "
        f"`{classification.exercise}` rather than direct video interpretation. "
        f"{issue_text} The detected variation was `{variation.detected_variation}`."
    )

    return CoachSummary(
        summary=summary,
        what_you_did=what_you_did,
        what_looked_good=what_looked_good,
        what_changed_across_reps=what_changed,
        valid_variation_vs_issue=variation_notes,
        top_fixes=top_fixes,
        next_session_plan=next_session_plan,
        confidence_notes=confidence_notes,
    )
