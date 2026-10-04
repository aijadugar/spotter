from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any

from spotter.contracts import PoseSequence, Rep, Reps
from spotter.steps.rep_signals import SignalSample, normalize_optional, samples_from_values, smooth_signal
from spotter.steps.rep_state_machine import RepSegment, find_local_extrema, segment_low_high_low


MIN_CYCLE_FRAMES = 12
MIN_PHASE_FRAMES = 4
MIN_USABLE_SIGNAL_SAMPLES = 9
MIN_SIGNAL_RANGE = 0.22


def mean_optional(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)


def combine(primary: list[float | None], secondary: list[float | None], *, weight: float) -> list[float | None]:
    normalized_secondary = normalize_optional(secondary)
    combined: list[float | None] = []
    for primary_value, secondary_value in zip(primary, normalized_secondary, strict=False):
        if primary_value is None:
            combined.append(None)
            continue
        if secondary_value is None:
            combined.append(primary_value)
            continue
        combined.append(primary_value + secondary_value * weight)
    return combined


def normalized_samples(
    sequence: PoseSequence,
    raw_signal: list[float | None],
) -> tuple[list[SignalSample], float]:
    smoothed_signal = smooth_signal(raw_signal)
    normalized_signal = normalize_optional(smoothed_signal)
    samples = samples_from_values(sequence, normalized_signal)
    signal_range = max((value for value in normalized_signal if value is not None), default=0.0) - min(
        (value for value in normalized_signal if value is not None),
        default=0.0,
    )
    return samples, round(signal_range, 4)


def segments_to_reps(segments: list[RepSegment]) -> list[Rep]:
    return [
        Rep(
            rep_id=index + 1,
            start_frame=segment.start.frame_index,
            mid_frame=segment.middle.frame_index,
            end_frame=segment.end.frame_index,
            start_sec=round(segment.start.timestamp_sec, 3),
            mid_sec=round(segment.middle.timestamp_sec, 3),
            end_sec=round(segment.end.timestamp_sec, 3),
        )
        for index, segment in enumerate(segments)
    ]


def partial_reps(
    sequence: PoseSequence,
    segments: list[RepSegment],
    samples: list[SignalSample],
    *,
    signal_range: float,
) -> list[dict[str, Any]]:
    if not samples:
        return [{"reason": "low_signal_quality"}]

    partials: list[dict[str, Any]] = []
    if not segments:
        if signal_range >= MIN_SIGNAL_RANGE * 0.7:
            partials.append(
                {
                    "reason": "insufficient_rom",
                    "start_frame": samples[0].frame_index,
                    "end_frame": samples[-1].frame_index,
                    "start_sec": round(samples[0].timestamp_sec, 3),
                    "end_sec": round(samples[-1].timestamp_sec, 3),
                }
            )
        return partials

    first_segment = segments[0]
    if first_segment.start.frame_index - samples[0].frame_index >= MIN_PHASE_FRAMES:
        partials.append(
            {
                "reason": "starts_mid_rep",
                "start_frame": samples[0].frame_index,
                "end_frame": first_segment.start.frame_index,
                "start_sec": round(samples[0].timestamp_sec, 3),
                "end_sec": round(first_segment.start.timestamp_sec, 3),
            }
        )

    last_segment = segments[-1]
    if samples[-1].frame_index - last_segment.end.frame_index >= MIN_PHASE_FRAMES:
        partials.append(
            {
                "reason": "ends_mid_rep",
                "start_frame": last_segment.end.frame_index,
                "end_frame": samples[-1].frame_index,
                "start_sec": round(last_segment.end.timestamp_sec, 3),
                "end_sec": round(samples[-1].timestamp_sec, 3),
            }
        )
    return partials


class ExerciseRepCounter(ABC):
    exercise: str
    pose_sequence: PoseSequence
    min_cycle_frames = MIN_CYCLE_FRAMES
    min_phase_frames = MIN_PHASE_FRAMES
    min_signal_range = MIN_SIGNAL_RANGE
    min_usable_signal_samples = MIN_USABLE_SIGNAL_SAMPLES

    @abstractmethod
    def build_signal(self) -> tuple[list[SignalSample], dict[str, Any]]:
        """Build the exercise-specific normalized motion signal."""

    def count(self) -> tuple[Reps, dict[str, Any]]:
        sequence = self.pose_sequence
        samples, debug = self.build_signal()
        signal_range = debug["raw_signal_range"]
        extrema = find_local_extrema(samples)
        min_amplitude = max(self.min_signal_range, signal_range * 0.35)
        segments = (
            segment_low_high_low(
                extrema,
                min_cycle_frames=self.min_cycle_frames,
                min_phase_frames=self.min_phase_frames,
                min_amplitude=min_amplitude,
            )
            if len(samples) >= self.min_usable_signal_samples
            else []
        )
        partials = partial_reps(sequence, segments, samples, signal_range=signal_range)
        if sequence.pose_valid_ratio < 0.8:
            partials.append({"reason": "low_pose_valid_ratio"})

        reps = Reps(
            exercise=self.exercise,
            reps=segments_to_reps(segments),
            partial_reps=partials,
        )
        debug_payload = {
            **debug,
            "thresholds": {
                "min_cycle_frames": self.min_cycle_frames,
                "min_phase_frames": self.min_phase_frames,
                "min_amplitude": round(min_amplitude, 4),
            },
            "extrema": [
                {
                    "kind": extrema_item.kind,
                    "frame_index": extrema_item.sample.frame_index,
                    "timestamp_sec": round(extrema_item.sample.timestamp_sec, 3),
                    "value": round(extrema_item.sample.value, 4),
                }
                for extrema_item in extrema
            ],
            "accepted_reps": [
                {
                    "start": asdict(segment.start),
                    "middle": asdict(segment.middle),
                    "end": asdict(segment.end),
                    "amplitude": round(segment.amplitude, 4),
                }
                for segment in segments
            ],
        }
        return reps, debug_payload
