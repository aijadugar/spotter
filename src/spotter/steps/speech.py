from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

from spotter.contracts import CoachSummary, ProgressPlan, SpeechResult

TTS_BACKEND_ENV = "SPOTTER_TTS_BACKEND"
DEFAULT_LANGUAGE_ENV = "SPOTTER_TTS_LANGUAGE"


class SpeechSynthesizer(Protocol):
    def synthesize(
        self,
        lines: list[str],
        language: str,
        out_dir: Path,
        *,
        summary: CoachSummary | None = None,
        plan: ProgressPlan | None = None,
        issues: list[Any] | None = None,
    ) -> SpeechResult:
        ...


def build_speech_lines(summary: CoachSummary, plan: ProgressPlan) -> list[str]:
    lines = [summary.summary]
    lines.extend(summary.what_looked_good[:1])
    lines.extend(summary.top_fixes[:1])
    lines.append(plan.focus)
    lines.extend(plan.next_session_cues[:1])
    lines.append(plan.encouragement)
    return [line for line in lines if line]


class NullSpeechSynthesizer:
    def synthesize(
        self,
        lines: list[str],
        language: str,
        out_dir: Path,
        *,
        summary: CoachSummary | None = None,
        plan: ProgressPlan | None = None,
        issues: list[Any] | None = None,
    ) -> SpeechResult:
        del out_dir, summary, plan, issues
        return SpeechResult(
            audio_path=None,
            language=language,
            lines=lines,
            backend="none",
            chars_used=0,
            chars_skipped=0,
        )


def get_speech_synthesizer() -> SpeechSynthesizer:
    backend = os.getenv(TTS_BACKEND_ENV, "none").strip().lower()
    if backend == "open":
        from spotter.steps.speech_open import OpenTtsSynthesizer

        return OpenTtsSynthesizer()
    if backend == "elevenlabs":
        from spotter.steps.speech_elevenlabs import ElevenLabsSynthesizer

        return ElevenLabsSynthesizer()
    return NullSpeechSynthesizer()


def default_language() -> str:
    return os.getenv(DEFAULT_LANGUAGE_ENV, "en")
