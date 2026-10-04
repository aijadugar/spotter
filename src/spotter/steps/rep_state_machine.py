from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from spotter.steps.rep_signals import SignalSample


ExtremaKind = Literal["min", "max"]


@dataclass(frozen=True)
class Extrema:
    kind: ExtremaKind
    sample: SignalSample


@dataclass(frozen=True)
class RepSegment:
    start: SignalSample
    middle: SignalSample
    end: SignalSample
    amplitude: float


def find_local_extrema(samples: list[SignalSample], *, min_gap_frames: int = 4) -> list[Extrema]:
    if len(samples) < 2:
        return []

    extrema: list[Extrema] = []
    first, second = samples[0], samples[1]
    if first.value < second.value:
        extrema.append(Extrema(kind="min", sample=first))
    elif first.value > second.value:
        extrema.append(Extrema(kind="max", sample=first))

    for index in range(1, len(samples) - 1):
        previous = samples[index - 1]
        current = samples[index]
        following = samples[index + 1]
        if current.value >= previous.value and current.value > following.value:
            kind: ExtremaKind | None = "max"
        elif current.value <= previous.value and current.value < following.value:
            kind = "min"
        else:
            kind = None
        if kind is None:
            continue
        if (
            extrema
            and current.frame_index - extrema[-1].sample.frame_index < min_gap_frames
            and extrema[-1].kind == kind
        ):
            previous_extrema = extrema[-1]
            if kind == "max" and current.value > previous_extrema.sample.value:
                extrema[-1] = Extrema(kind=kind, sample=current)
            elif kind == "min" and current.value < previous_extrema.sample.value:
                extrema[-1] = Extrema(kind=kind, sample=current)
            continue
        extrema.append(Extrema(kind=kind, sample=current))

    previous, last = samples[-2], samples[-1]
    if previous.value < last.value:
        kind: ExtremaKind | None = "max"
    elif previous.value > last.value:
        kind = "min"
    else:
        kind = None
    if kind is not None:
        if (
            extrema
            and last.frame_index - extrema[-1].sample.frame_index < min_gap_frames
            and extrema[-1].kind == kind
        ):
            replace = kind == "max" and last.value > extrema[-1].sample.value
            replace = replace or (kind == "min" and last.value < extrema[-1].sample.value)
            if replace:
                extrema[-1] = Extrema(kind=kind, sample=last)
        else:
            extrema.append(Extrema(kind=kind, sample=last))
    return extrema


def segment_low_high_low(
    extrema: list[Extrema],
    *,
    min_cycle_frames: int,
    min_phase_frames: int,
    min_amplitude: float,
) -> list[RepSegment]:
    segments: list[RepSegment] = []
    start_index = 0
    while start_index < len(extrema):
        first = extrema[start_index]
        if first.kind != "min":
            start_index += 1
            continue

        accepted_segment: RepSegment | None = None
        accepted_end_index: int | None = None

        for peak_index in range(start_index + 1, len(extrema)):
            second = extrema[peak_index]
            if second.kind != "max":
                continue
            if second.sample.frame_index - first.sample.frame_index < min_phase_frames:
                continue

            for end_index in range(peak_index + 1, len(extrema)):
                third = extrema[end_index]
                if third.kind != "min":
                    continue
                if third.sample.frame_index - second.sample.frame_index < min_phase_frames:
                    continue
                if third.sample.frame_index - first.sample.frame_index < min_cycle_frames:
                    continue

                left_amplitude = second.sample.value - first.sample.value
                right_amplitude = second.sample.value - third.sample.value
                amplitude = min(left_amplitude, right_amplitude)
                if amplitude < min_amplitude:
                    continue

                accepted_segment = RepSegment(
                    start=first.sample,
                    middle=second.sample,
                    end=third.sample,
                    amplitude=amplitude,
                )
                accepted_end_index = end_index
                break

            if accepted_segment is not None:
                break

        if accepted_segment is None or accepted_end_index is None:
            start_index += 1
            continue

        segments.append(accepted_segment)
        start_index = accepted_end_index
    return segments
