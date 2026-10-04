from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.contracts import (
    ContractValidationError,
    ProgressPlan,
    SessionRecord,
    SpeechResult,
    validate_contract,
)


def _record() -> SessionRecord:
    return SessionRecord(
        session_id="20261003T090000Z-abc12345",
        timestamp="2026-10-03T09:00:00Z",
        profile_key="friend",
        exercise="squat",
        rep_count=8,
        aggregate_metrics={
            "avg_rom_score": 0.72,
            "avg_stability_score": 0.80,
            "avg_symmetry_score": 0.77,
            "avg_rep_duration_sec": 1.9,
        },
        issue_counts={"shallow_depth": 2},
        form_score=0.76,
        source_run_id="20261003T090000Z-abc12345",
    )


class CompanionContractTests(unittest.TestCase):
    def test_session_record_contract_accepts_valid_payload(self) -> None:
        validate_contract("session_record.json", _record())

    def test_session_record_rejects_form_score_out_of_range(self) -> None:
        bad = SessionRecord(**{**_record().__dict__, "form_score": 1.4})
        with self.assertRaises(ContractValidationError):
            validate_contract("session_record.json", bad)

    def test_progress_plan_contract_accepts_valid_payload(self) -> None:
        plan = ProgressPlan(
            focus="Control depth on the last reps.",
            targets=["Reach parallel depth on 6 of 8 reps."],
            next_session_cues=["Slow the descent."],
            encouragement="You are trending up.",
            confidence_notes=["Based on 3 sessions."],
        )
        validate_contract("progress_plan.json", plan)

    def test_progress_plan_rejects_non_string_target(self) -> None:
        plan = ProgressPlan(
            focus="x",
            targets=["ok"],
            next_session_cues=[],
            encouragement="y",
            confidence_notes=[],
        )
        payload = plan.__dict__
        payload["targets"] = [1]
        with self.assertRaises(ContractValidationError):
            validate_contract("progress_plan.json", payload)

    def test_speech_contract_allows_null_audio_path(self) -> None:
        result = SpeechResult(
            audio_path=None,
            language="en",
            lines=["Nice set."],
            backend="none",
        )
        validate_contract("speech.json", result)


if __name__ == "__main__":
    unittest.main()
