from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.contracts import ExerciseClassification, IssueMarkers, RepAnalysis, Reps
from spotter.steps.progress_plan import SessionFindings, build_fallback_plan
from spotter.steps.session_record import build_session_record


class PipelineCompanionTests(unittest.TestCase):
    def test_findings_to_plan_to_record_chain(self) -> None:
        findings = SessionFindings(
            exercise="squat",
            rep_count=8,
            aggregate_metrics={"avg_rom_score": 0.7, "avg_stability_score": 0.8},
            issue_labels=["shallow_depth"],
        )
        plan = build_fallback_plan([], findings)
        record = build_session_record(
            run_id="run-1",
            profile_key="friend",
            timestamp="2026-10-03T09:00:00Z",
            classification=ExerciseClassification("squat", 0.9, [], False),
            reps=Reps("squat", [], []),
            analysis=RepAnalysis("squat", [], {"avg_rom_score": 0.7}),
            issues=IssueMarkers([]),
        )
        self.assertTrue(plan.focus)
        self.assertEqual(record.source_run_id, "run-1")


if __name__ == "__main__":
    unittest.main()
