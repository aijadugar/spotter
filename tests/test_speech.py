from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.contracts import CoachSummary, ProgressPlan
from spotter.steps.speech import (
    NullSpeechSynthesizer,
    build_speech_lines,
    get_speech_synthesizer,
)


def _summary() -> CoachSummary:
    return CoachSummary(
        summary="Solid set.",
        what_you_did=["2 squats."],
        what_looked_good=["Control."],
        what_changed_across_reps=[],
        valid_variation_vs_issue=[],
        top_fixes=["Go a little deeper."],
        next_session_plan=["Slow down."],
        confidence_notes=[],
    )


def _plan() -> ProgressPlan:
    return ProgressPlan(
        focus="Depth.",
        targets=["Reach depth."],
        next_session_cues=["Slow descent."],
        encouragement="Nice work.",
        confidence_notes=[],
    )


class SpeechTests(unittest.TestCase):
    def test_lines_include_summary_fix_and_plan_cue(self) -> None:
        lines = build_speech_lines(_summary(), _plan())
        self.assertIn("Solid set.", lines)
        self.assertIn("Go a little deeper.", lines)
        self.assertIn("Slow descent.", lines)

    def test_null_synthesizer_returns_no_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = NullSpeechSynthesizer().synthesize(
                ["hello"], language="en", out_dir=Path(temp_dir)
            )
            self.assertIsNone(result.audio_path)
            self.assertEqual(result.backend, "none")
            self.assertEqual(result.lines, ["hello"])

    def test_factory_defaults_to_null_backend(self) -> None:
        self.assertIsInstance(get_speech_synthesizer(), NullSpeechSynthesizer)


if __name__ == "__main__":
    unittest.main()
