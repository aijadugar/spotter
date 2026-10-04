from __future__ import annotations

from statistics import mean, pstdev
from typing import Any

from spotter.contracts import (
    IssueMarker,
    IssueMarkers,
    PoseFrame,
    PoseSequence,
    Rep,
    RepAnalysis,
    RepAnalysisItem,
    Reps,
    UserProfile,
    VideoManifest,
    Variation,
)
from spotter.exercise_catalog import get_exercise_spec
from spotter.exercises.shared.analyzer import (
    ExerciseMetricResult,
    mean_optional,
    round_optional,
    safe_ratio,
    score,
    usable,
    value_series,
)
from spotter.exercises.shared.issue_marker import (
    IssueRule,
    frame_scores_for_rule,
    frames_for_rep,
    group_violations,
    marker_from_group,
    minimum_run_length,
)
from spotter.exercises.shared.rep_counter import ExerciseRepCounter
from spotter.steps.rep_signals import average_axis


class ExerciseStrategy(ExerciseRepCounter):
    def __init__(
        self,
        *,
        video_manifest: VideoManifest,
        pose_sequence: PoseSequence,
        profile: UserProfile,
    ) -> None:
        self.video_manifest = video_manifest
        self.pose_sequence = pose_sequence
        self.profile = profile

    def metrics(self, frames: list[PoseFrame]) -> ExerciseMetricResult:
        raise NotImplementedError

    def detect_variation(self, analysis: RepAnalysis) -> tuple[str, float, list[str]]:
        raise NotImplementedError

    def profile_not_issues(self, variation: str) -> list[str]:
        return []

    def issue_rules(self) -> tuple[IssueRule, ...]:
        return ()

    def analyze_reps(self, reps: Reps) -> RepAnalysis:
        draft_items: list[tuple[Rep, dict[str, Any], float, float, float, list[str]]] = []
        for rep in reps.reps:
            rep_frames = self.frames_for_rep(rep)
            primary_signal = self.primary_signal(rep_frames)
            common_metrics = self.common_rep_metrics(rep, rep_frames, primary_signal)
            exercise_metrics, rom_score, stability_score, symmetry_score, hints = self.metrics(rep_frames)
            metrics = {**common_metrics, **exercise_metrics}
            draft_items.append((rep, metrics, rom_score, stability_score, symmetry_score, hints))

        average_duration = (
            mean(item[1]["rep_duration_sec"] for item in draft_items)
            if draft_items
            else 0.0
        )
        items: list[RepAnalysisItem] = []
        for rep, metrics, rom_score, stability_score, symmetry_score, hints in draft_items:
            duration = metrics["rep_duration_sec"]
            metrics["tempo_consistency_score"] = (
                score(1.0 - abs(duration - average_duration) / max(average_duration, 0.1))
                if average_duration
                else 0.0
            )
            items.append(
                RepAnalysisItem(
                    rep_id=rep.rep_id,
                    duration_sec=duration,
                    range_of_motion_score=rom_score,
                    stability_score=stability_score,
                    symmetry_score=symmetry_score,
                    metrics=metrics,
                    variation_hints=sorted(set(hints)),
                )
            )

        aggregate_metrics = {
            "avg_rom_score": (
                round(mean(item.range_of_motion_score for item in items), 2) if items else 0.0
            ),
            "avg_stability_score": (
                round(mean(item.stability_score for item in items), 2) if items else 0.0
            ),
            "avg_symmetry_score": (
                round(mean(item.symmetry_score for item in items), 2) if items else 0.0
            ),
            "avg_rep_duration_sec": (
                round(mean(item.duration_sec for item in items), 2) if items else 0.0
            ),
            "avg_tempo_consistency_score": self.aggregate_numeric(items, "tempo_consistency_score")
            or 0.0,
            "avg_landmark_confidence": (
                self.aggregate_numeric(items, "landmark_confidence") or self.pose_sequence.pose_valid_ratio
            ),
            "fatigue_trend_rom_delta": self.fatigue_trend(items),
            "pose_valid_ratio": self.pose_sequence.pose_valid_ratio,
        }

        for metric_name in (
            "hand_width_ratio",
            "stance_width_ratio",
            "bottom_pause_sec",
            "lockout_quality",
            "wrist_height_asymmetry",
            "wrist_travel",
            "knee_support_score",
        ):
            aggregate_value = self.aggregate_numeric(items, metric_name)
            if aggregate_value is not None:
                aggregate_metrics[f"avg_{metric_name}"] = aggregate_value

        return RepAnalysis(
            exercise=self.exercise,
            items=items,
            aggregate_metrics=aggregate_metrics,
        )

    def resolve_variation(self, analysis: RepAnalysis) -> Variation:
        if self.profile.intended_variation:
            variation = self.profile.intended_variation
            confidence = 0.95
            not_issues = self.profile_not_issues(variation)
        else:
            variation, confidence, not_issues = self.detect_variation(analysis)

        if analysis.aggregate_metrics.get("avg_rom_score", 1.0) < 0.7:
            not_issues.append("low_rom_requires_user_intent_check")
        if analysis.aggregate_metrics.get("pose_valid_ratio", 1.0) < 0.8:
            not_issues.append("low_pose_confidence_limits_variation_call")

        return Variation(
            exercise=self.exercise,
            detected_variation=variation,
            variation_confidence=confidence,
            not_issues=sorted(set(not_issues)),
        )

    def mark_issues(
        self,
        reps: Reps,
        analysis: RepAnalysis,
        variation: Variation,
    ) -> IssueMarkers:
        rep_by_id = {rep.rep_id: rep for rep in reps.reps}
        issues: list[IssueMarker] = []

        for item in analysis.items:
            rep = rep_by_id.get(item.rep_id)
            if rep is None:
                continue

            rep_frames = frames_for_rep(self.pose_sequence, rep)
            if not rep_frames:
                fallback = self.fallback_rep_marker(reps, item, variation)
                if fallback is not None:
                    issues.append(fallback)
                continue

            min_run_length = minimum_run_length(rep_frames)
            for rule in self.issue_rules():
                if set(rule.suppress_when_not_issue) & set(variation.not_issues):
                    continue

                scores = frame_scores_for_rule(rep_frames, self.exercise, rule)
                for group in group_violations(scores, min_run_length):
                    issues.append(marker_from_group(rule, group, item, variation))

        return IssueMarkers(
            issues=sorted(
                issues,
                key=lambda issue: (issue.start_frame, issue.end_frame, issue.rep_id, issue.issue),
            )
        )

    def frames_for_rep(self, rep: Rep) -> list[PoseFrame]:
        rep_frames = [
            frame
            for frame in self.pose_sequence.frames
            if rep.start_frame <= frame.frame_index <= rep.end_frame
        ]
        if rep_frames:
            return rep_frames

        if not self.pose_sequence.frames:
            return []
        closest = min(
            self.pose_sequence.frames,
            key=lambda frame: min(
                abs(frame.frame_index - rep.start_frame),
                abs(frame.frame_index - rep.mid_frame),
                abs(frame.frame_index - rep.end_frame),
            ),
        )
        return [closest]

    def primary_signal(self, frames: list[PoseFrame]) -> list[float | None]:
        if self.exercise == "shoulder_press":
            return value_series(
                frames,
                lambda frame: average_axis(frame, ("left_wrist", "right_wrist"), "y"),
            )
        if self.exercise == "push_up":
            return value_series(
                frames,
                lambda frame: mean_optional(
                    [
                        average_axis(frame, ("left_shoulder", "right_shoulder"), "y"),
                        average_axis(frame, ("left_hip", "right_hip"), "y"),
                    ]
                ),
            )
        return value_series(frames, lambda frame: average_axis(frame, ("left_hip", "right_hip"), "y"))

    def common_rep_metrics(
        self,
        rep: Rep,
        frames: list[PoseFrame],
        primary_signal: list[float | None],
    ) -> dict[str, Any]:
        eccentric_duration = round(max(0.0, rep.mid_sec - rep.start_sec), 2)
        concentric_duration = round(max(0.0, rep.end_sec - rep.mid_sec), 2)
        duration = round(max(0.0, rep.end_sec - rep.start_sec), 2)
        smoothness_score, jerk_score = self.smoothness_score(primary_signal)
        stability_axis = value_series(
            frames,
            lambda frame: average_axis(frame, ("left_hip", "right_hip"), "x"),
        )
        stability_noise = self.std(stability_axis) or 0.0

        return {
            "rep_duration_sec": duration,
            "eccentric_duration_sec": eccentric_duration,
            "concentric_duration_sec": concentric_duration,
            "tempo_ratio": round_optional(safe_ratio(eccentric_duration, concentric_duration), 2),
            "top_pause_sec": self.pause_duration(frames, primary_signal, target="top"),
            "bottom_pause_sec": self.pause_duration(frames, primary_signal, target="bottom"),
            "smoothness_score": smoothness_score,
            "jerk_score": round_optional(jerk_score, 4),
            "landmark_confidence": self.mean_visibility(frames),
            "hip_lateral_drift": round_optional(stability_noise, 4),
        }

    def fallback_rep_marker(
        self,
        reps: Reps,
        item: RepAnalysisItem,
        variation: Variation,
    ) -> IssueMarker | None:
        exercise_spec = get_exercise_spec(reps.exercise)
        if item.stability_score >= 0.78 or exercise_spec.mock_issue is None:
            return None

        rep = next((rep for rep in reps.reps if rep.rep_id == item.rep_id), None)
        if rep is None:
            return None

        issue_spec = exercise_spec.mock_issue
        metric_value = (
            item.range_of_motion_score
            if issue_spec.evidence_metric == "range_of_motion_score"
            else item.metrics.get(issue_spec.evidence_metric)
        )
        return IssueMarker(
            rep_id=item.rep_id,
            issue=issue_spec.issue,
            severity=round(1.0 - item.stability_score, 2),
            start_frame=rep.mid_frame,
            end_frame=rep.end_frame,
            start_sec=rep.mid_sec,
            end_sec=rep.end_sec,
            affected_joints=list(issue_spec.affected_joints),
            evidence={
                issue_spec.evidence_metric: metric_value,
                "threshold": issue_spec.threshold,
                "confidence": round(max(0.0, min(1.0, 1.0 - item.stability_score)), 2),
                "variation_context": {
                    "detected_variation": variation.detected_variation,
                    "variation_confidence": variation.variation_confidence,
                    "not_issues": list(variation.not_issues),
                },
                "fallback": "rep_level_metrics",
            },
        )

    def metric(self, analysis: RepAnalysis, name: str) -> float | None:
        value = analysis.aggregate_metrics.get(name)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def confidence(
        self,
        base: float,
        analysis: RepAnalysis,
        supporting_metric: float | None,
    ) -> float:
        rep_bonus = min(0.12, len(analysis.items) * 0.03)
        metric_bonus = 0.0 if supporting_metric is None else min(0.1, abs(supporting_metric) * 0.03)
        pose_bonus = min(0.08, float(analysis.aggregate_metrics.get("pose_valid_ratio", 0.0)) * 0.08)
        return round(min(0.95, base + rep_bonus + metric_bonus + pose_bonus), 2)

    def std(self, values: list[float | None]) -> float | None:
        usable_values = usable(values)
        if len(usable_values) < 2:
            return 0.0 if usable_values else None
        return pstdev(usable_values)

    def mean_visibility(self, frames: list[PoseFrame]) -> float:
        values: list[float | None] = []
        for frame in frames:
            if "mean_visibility" in frame.pose_quality:
                values.append(float(frame.pose_quality["mean_visibility"]))
                continue
            landmark_values = [
                landmark.get("visibility")
                for landmark in frame.landmarks.values()
                if landmark.get("visibility") is not None
            ]
            values.extend(float(value) for value in landmark_values)
        return score(mean_optional(values) if values else 0.0)

    def smoothness_score(self, signal_values: list[float | None]) -> tuple[float, float | None]:
        usable_values = usable(signal_values)
        if len(usable_values) < 4:
            return 0.5, None

        deltas = [
            usable_values[index] - usable_values[index - 1]
            for index in range(1, len(usable_values))
        ]
        jerks = [deltas[index] - deltas[index - 1] for index in range(1, len(deltas))]
        if not jerks:
            return 0.5, None
        jerk = mean(abs(value) for value in jerks)
        return score(1.0 - jerk * 8.0), jerk

    def pause_duration(
        self,
        frames: list[PoseFrame],
        signal_values: list[float | None],
        *,
        target: str,
    ) -> float:
        usable_values = usable(signal_values)
        if len(usable_values) < 3 or len(frames) < 3:
            return 0.0

        min_value = min(usable_values)
        max_value = max(usable_values)
        tolerance = max((max_value - min_value) * 0.08, 0.01)
        if target == "bottom":
            active = [value is not None and value >= max_value - tolerance for value in signal_values]
        else:
            active = [value is not None and value <= min_value + tolerance for value in signal_values]

        longest = 0
        current = 0
        for item in active:
            if item:
                current += 1
                longest = max(longest, current)
            else:
                current = 0

        if longest <= 1:
            return 0.0
        frame_duration = (frames[-1].timestamp_sec - frames[0].timestamp_sec) / max(
            1, len(frames) - 1
        )
        return round(longest * frame_duration, 2)

    def aggregate_numeric(self, items: list[RepAnalysisItem], metric_name: str) -> float | None:
        values = [
            item.metrics.get(metric_name)
            for item in items
            if isinstance(item.metrics.get(metric_name), (int, float))
        ]
        if not values:
            return None
        return round(sum(float(value) for value in values) / len(values), 4)

    def fatigue_trend(self, items: list[RepAnalysisItem]) -> float:
        if len(items) < 2:
            return 0.0
        first = items[0].range_of_motion_score
        last = items[-1].range_of_motion_score
        return round(last - first, 4)
