from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.contracts import (
    ExerciseClassification,
    IssueMarker,
    IssueMarkers,
    Rep,
    RepAnalysis,
    RepAnalysisItem,
    Reps,
)
from spotter.steps.session_record import build_session_record, form_score_from_metrics


def _classification() -> ExerciseClassification:
    return ExerciseClassification(
        exercise="squat", confidence=0.9, window_predictions=[], fallback_required=False
    )


def _reps() -> Reps:
    return Reps(
        exercise="squat",
        reps=[Rep(1, 0, 5, 10, 0.0, 0.2, 0.4), Rep(2, 11, 15, 20, 0.5, 0.7, 0.9)],
        partial_reps=[],
    )


def _analysis() -> RepAnalysis:
    return RepAnalysis(
        exercise="squat",
        items=[RepAnalysisItem(1, 0.4, 0.7, 0.8, 0.9, {}, [])],
        aggregate_metrics={
            "avg_rom_score": 0.7,
            "avg_stability_score": 0.8,
            "avg_symmetry_score": 0.9,
        },
    )


def _issues() -> IssueMarkers:
    return IssueMarkers(
        issues=[
            IssueMarker(2, "shallow_depth", 0.8, 11, 14, 0.5, 0.6, ["left_hip"], {}),
            IssueMarker(2, "shallow_depth", 0.6, 15, 18, 0.7, 0.8, ["right_hip"], {}),
        ]
    )


class SessionRecordTests(unittest.TestCase):
    def test_form_score_is_mean_of_available_metrics(self) -> None:
        self.assertAlmostEqual(
            form_score_from_metrics({"avg_rom_score": 0.6, "avg_stability_score": 0.9}),
            0.75,
        )

    def test_form_score_defaults_when_metrics_missing(self) -> None:
        self.assertEqual(form_score_from_metrics({}), 0.0)

    def test_build_record_counts_issues(self) -> None:
        record = build_session_record(
            run_id="run-1",
            profile_key="friend",
            timestamp="2026-10-03T09:00:00Z",
            classification=_classification(),
            reps=_reps(),
            analysis=_analysis(),
            issues=_issues(),
        )
        self.assertEqual(record.rep_count, 2)
        self.assertEqual(record.issue_counts, {"shallow_depth": 2})
        self.assertEqual(record.exercise, "squat")
        self.assertEqual(record.source_run_id, "run-1")
        self.assertAlmostEqual(record.form_score, 0.8)


if __name__ == "__main__":
    unittest.main()
