from __future__ import annotations

import re

from spotter.contracts import (
    CoachSummary,
    ExerciseClassification,
    IssueMarkers,
    RepAnalysis,
    Reps,
    Variation,
    Verification,
)
from spotter.knowledge_cards import known_issue_labels


DIAGNOSIS_PATTERNS = (
    "diagnos",
    "injury",
    "tendonitis",
    "tear",
    "impingement",
    "pathology",
    "medical assessment",
)
INJURY_PREVENTION_PATTERNS = (
    "prevent injury",
    "injury prevention",
    "avoid injury",
)
NEGATIVE_VARIATION_CONTEXT = (
    "is an issue",
    "is a fault",
    "is a problem",
    "fault",
    "problem",
    "error",
    "wrong",
    "incorrect",
    "should be fixed",
)
SAFE_VARIATION_CONTEXT = (
    "not-issue context",
    "not an issue",
    "rather than a fault",
    "context rather than a fault",
)


def _contains_negative_variation_language(lines: list[str], labels: list[str]) -> bool:
    for line in lines:
        lowered = line.lower()
        if not any(label in lowered for label in labels):
            continue
        if any(token in lowered for token in SAFE_VARIATION_CONTEXT):
            continue
        if any(token in lowered for token in NEGATIVE_VARIATION_CONTEXT):
            return True
    return False


def _summary_sections(summary: CoachSummary) -> list[str]:
    return [
        summary.summary,
        *summary.what_you_did,
        *summary.what_looked_good,
        *summary.what_changed_across_reps,
        *summary.valid_variation_vs_issue,
        *summary.top_fixes,
        *summary.next_session_plan,
        *summary.confidence_notes,
    ]


def _normalized_text(summary: CoachSummary) -> str:
    return " ".join(_summary_sections(summary)).lower()


def _mentioned_labels(summary: CoachSummary) -> set[str]:
    text = " ".join(_summary_sections(summary))
    labels = set(re.findall(r"`([a-z0-9_]+)`", text))
    lowered = text.lower()
    for label in known_issue_labels():
        if label in lowered:
            labels.add(label)
    return labels


def _confidence_notes_required(
    classification: ExerciseClassification,
    analysis: RepAnalysis,
    variation: Variation,
    reps: Reps,
    issues: IssueMarkers,
) -> bool:
    if classification.confidence < 0.7:
        return True
    if variation.variation_confidence < 0.7:
        return True
    if float(analysis.aggregate_metrics.get("pose_valid_ratio", 1.0)) < 0.85:
        return True
    if len(reps.reps) == 0:
        return True
    if len(issues.issues) == 0:
        return True
    return False


def run(
    summary: CoachSummary,
    issues: IssueMarkers,
    variation: Variation,
    *,
    classification: ExerciseClassification,
    analysis: RepAnalysis,
    reps: Reps,
) -> Verification:
    allowed_issues = {issue.issue for issue in issues.issues}
    mentioned_labels = _mentioned_labels(summary)
    mentioned_issues = mentioned_labels & known_issue_labels()

    no_issue_outside_json = mentioned_issues <= allowed_issues

    variation_lines = summary.valid_variation_vs_issue + summary.top_fixes
    variation_text = " ".join(variation_lines).lower()
    variation_not_overcorrected = True
    variation_labels = [variation.detected_variation, *variation.not_issues]
    if _contains_negative_variation_language(variation_lines, variation_labels):
        variation_not_overcorrected = False
    if variation.detected_variation and variation.detected_variation not in variation_text:
        variation_not_overcorrected = False

    normalized = _normalized_text(summary)
    no_diagnosis = all(pattern not in normalized for pattern in DIAGNOSIS_PATTERNS)
    no_injury_prevention_claim = all(
        pattern not in normalized for pattern in INJURY_PREVENTION_PATTERNS
    )
    confidence_present = (
        not _confidence_notes_required(classification, analysis, variation, reps, issues)
    ) or bool(summary.confidence_notes)

    checks = {
        "no_issue_outside_json": no_issue_outside_json,
        "variation_not_overcorrected": variation_not_overcorrected,
        "no_diagnosis": no_diagnosis,
        "no_injury_prevention_claim": no_injury_prevention_claim,
        "confidence_notes_present_when_required": confidence_present,
    }

    notes: list[str] = []
    if not no_issue_outside_json:
        extra = sorted(mentioned_issues - allowed_issues)
        notes.append(f"Summary mentioned issue labels not present in JSON: {', '.join(extra)}.")
    if not variation_not_overcorrected:
        notes.append("Summary did not keep valid variation context separate from issue correction.")
    if not no_diagnosis:
        notes.append("Summary used diagnosis-style language.")
    if not no_injury_prevention_claim:
        notes.append("Summary made an injury-prevention claim.")
    if not confidence_present:
        notes.append("Summary is missing required confidence notes.")

    return Verification(
        passed=all(checks.values()),
        checks=checks,
        notes=notes,
    )
