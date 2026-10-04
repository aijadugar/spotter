from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Callable, Literal

from spotter.contracts import IssueMarker, PoseFrame, PoseSequence, Rep, RepAnalysisItem, Variation
from spotter.steps.rep_signals import average_axis, smooth_signal


Comparison = Literal["lt", "gt"]
Phase = Literal["all", "bottom", "top"]
MetricGetter = Callable[[PoseFrame], float | None]


@dataclass(frozen=True)
class IssueRule:
    issue: str
    metric_name: str
    threshold: float
    comparison: Comparison
    affected_joints: tuple[str, ...]
    getter: MetricGetter
    phase: Phase = "all"
    suppress_when_not_issue: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrameScore:
    frame: PoseFrame
    value: float
    severity: float


def frames_for_rep(sequence: PoseSequence | None, rep: Rep) -> list[PoseFrame]:
    if sequence is None:
        return []
    return [
        frame
        for frame in sequence.frames
        if rep.start_frame <= frame.frame_index <= rep.end_frame
    ]


def phase_mask(frames: list[PoseFrame], exercise: str, phase: Phase) -> list[bool]:
    if phase == "all":
        return [True for _ in frames]

    if exercise == "shoulder_press":
        values = [average_axis(frame, ("left_wrist", "right_wrist"), "y") for frame in frames]
        top_is_low = True
    elif exercise == "push_up":
        values = [
            average_axis(frame, ("left_shoulder", "right_shoulder"), "y") for frame in frames
        ]
        top_is_low = False
    else:
        values = [average_axis(frame, ("left_hip", "right_hip"), "y") for frame in frames]
        top_is_low = False

    usable_values = [value for value in values if value is not None]
    if not usable_values:
        return [False for _ in frames]

    minimum = min(usable_values)
    maximum = max(usable_values)
    value_range = maximum - minimum
    if value_range <= 1e-6:
        return [True for _ in frames]

    if phase == "top":
        cutoff = minimum + value_range * 0.35 if top_is_low else maximum - value_range * 0.35
        return [
            value is not None and (value <= cutoff if top_is_low else value >= cutoff)
            for value in values
        ]

    cutoff = maximum - value_range * 0.35 if not top_is_low else minimum + value_range * 0.35
    return [
        value is not None and (value >= cutoff if not top_is_low else value <= cutoff)
        for value in values
    ]


def violates(value: float, rule: IssueRule) -> bool:
    if rule.comparison == "lt":
        return value < rule.threshold
    return value > rule.threshold


def severity(value: float, rule: IssueRule) -> float:
    denominator = max(abs(rule.threshold), 1e-6)
    if rule.comparison == "lt":
        return min(1.0, max(0.0, (rule.threshold - value) / denominator))
    return min(1.0, max(0.0, (value - rule.threshold) / denominator))


def minimum_run_length(frames: list[PoseFrame]) -> int:
    return max(2, min(5, round(len(frames) * 0.08)))


def group_violations(
    scores: list[FrameScore | None],
    minimum_length: int,
) -> list[list[FrameScore]]:
    groups: list[list[FrameScore]] = []
    current: list[FrameScore] = []
    for score_item in scores:
        if score_item is None:
            if len(current) >= minimum_length:
                groups.append(current)
            current = []
            continue
        current.append(score_item)

    if len(current) >= minimum_length:
        groups.append(current)
    return groups


def confidence(group: list[FrameScore], rep_item: RepAnalysisItem, variation: Variation) -> float:
    visibility_values = [
        frame_score.frame.pose_quality.get("mean_visibility")
        for frame_score in group
        if frame_score.frame.pose_quality.get("mean_visibility") is not None
    ]
    visibility = mean(float(value) for value in visibility_values) if visibility_values else 0.6
    support = mean(frame_score.severity for frame_score in group)
    value = (visibility * 0.5) + (support * 0.3) + (variation.variation_confidence * 0.2)
    value *= max(0.5, min(1.0, rep_item.stability_score))
    return round(min(1.0, max(0.0, value)), 2)


def marker_from_group(
    rule: IssueRule,
    group: list[FrameScore],
    rep_item: RepAnalysisItem,
    variation: Variation,
) -> IssueMarker:
    metric_values = [frame_score.value for frame_score in group]
    peak_score = max(group, key=lambda frame_score: frame_score.severity)
    marker_severity = round(
        min(1.0, max(0.0, mean(frame_score.severity for frame_score in group))),
        2,
    )
    start = group[0].frame
    end = group[-1].frame
    evidence = {
        rule.metric_name: round(peak_score.value, 4),
        "threshold": rule.threshold,
        "confidence": confidence(group, rep_item, variation),
        "peak_frame": peak_score.frame.frame_index,
        "variation_context": {
            "detected_variation": variation.detected_variation,
            "variation_confidence": variation.variation_confidence,
            "not_issues": list(variation.not_issues),
        },
        "mean_metric_value": round(mean(metric_values), 4),
        "supporting_frames": len(group),
    }
    if rep_item.variation_hints:
        evidence["rep_variation_hints"] = list(rep_item.variation_hints)

    return IssueMarker(
        rep_id=rep_item.rep_id,
        issue=rule.issue,
        severity=marker_severity,
        start_frame=start.frame_index,
        end_frame=end.frame_index,
        start_sec=start.timestamp_sec,
        end_sec=end.timestamp_sec,
        affected_joints=list(rule.affected_joints),
        evidence=evidence,
    )


def frame_scores_for_rule(
    frames: list[PoseFrame],
    exercise: str,
    rule: IssueRule,
) -> list[FrameScore | None]:
    raw_values = [rule.getter(frame) for frame in frames]
    values = smooth_signal(raw_values, window_radius=2)
    active_phase = phase_mask(frames, exercise, rule.phase)
    scores: list[FrameScore | None] = []

    for frame, value, in_phase in zip(frames, values, active_phase, strict=False):
        if value is None or not in_phase or not violates(value, rule):
            scores.append(None)
            continue
        scores.append(FrameScore(frame=frame, value=value, severity=severity(value, rule)))

    return scores
