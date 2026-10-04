from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.contracts import SessionRecord
from spotter.steps.progress_plan import SessionFindings, build_fallback_plan


def _record(session_id: str, timestamp: str, rom: float, depth: int) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        timestamp=timestamp,
        profile_key="friend",
        exercise="squat",
        rep_count=8,
        aggregate_metrics={
            "avg_rom_score": rom,
            "avg_stability_score": 0.8,
            "avg_symmetry_score": 0.78,
        },
        issue_counts={"shallow_depth": depth},
        form_score=rom,
        source_run_id=session_id,
    )


def _findings(depth: int = 2) -> SessionFindings:
    return SessionFindings(
        exercise="squat",
        rep_count=8,
        aggregate_metrics={
            "avg_rom_score": 0.71,
            "avg_stability_score": 0.82,
            "avg_symmetry_score": 0.79,
        },
        issue_labels=["shallow_depth"] * depth,
    )


class FallbackPlanTests(unittest.TestCase):
    def test_plan_focus_uses_most_frequent_issue(self) -> None:
        history = [_record("s1", "2026-10-01T09:00:00Z", 0.6, 1)]
        plan = build_fallback_plan(history, _findings(depth=2))
        self.assertIn("shallow_depth", plan.focus)

    def test_plan_targets_use_rom_metric(self) -> None:
        history = [_record("s1", "2026-10-01T09:00:00Z", 0.6, 1)]
        plan = build_fallback_plan(history, _findings())
        self.assertTrue(any("range" in target.lower() or "depth" in target.lower() for target in plan.targets))

    def test_plan_notes_failure_reason_when_provided(self) -> None:
        plan = build_fallback_plan([], _findings(), failure_reason="model_unavailable")
        self.assertTrue(any("model_unavailable" in note for note in plan.confidence_notes))

    def test_plan_with_no_history_still_returns_plan(self) -> None:
        plan = build_fallback_plan([], _findings(depth=0))
        self.assertTrue(plan.encouragement)
        self.assertTrue(plan.targets)


if __name__ == "__main__":
    unittest.main()
