from __future__ import annotations

from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.contracts import ProgressPlan
from spotter.steps.progress_plan import SessionFindings
from spotter.steps.progress_plan_model import (
    ModelProgressPlanner,
    build_progress_plan_prompt,
    extract_plan_payload,
    verify_plan,
)


class _GoodModel:
    def generate_summary(self, prompt: str):
        del prompt
        from spotter.slm.providers import CoachSummaryGeneration

        return CoachSummaryGeneration(
            text=json.dumps(
                {
                    "focus": "Control `shallow_depth`.",
                    "targets": ["Reach depth on 6 of 8 reps."],
                    "next_session_cues": ["Slow the descent."],
                    "encouragement": "Steady progress.",
                    "confidence_notes": ["Based on 2 sessions."],
                }
            ),
            provider="test",
            model="test-model",
        )


class _BadModel:
    def generate_summary(self, prompt: str):
        del prompt
        raise RuntimeError("boom")


def _findings() -> SessionFindings:
    return SessionFindings(
        exercise="squat",
        rep_count=8,
        aggregate_metrics={"avg_rom_score": 0.7},
        issue_labels=["shallow_depth"],
    )


class ModelPlannerTests(unittest.TestCase):
    def test_prompt_includes_detected_issue_labels(self) -> None:
        prompt = build_progress_plan_prompt([], _findings())
        self.assertIn("shallow_depth", prompt)

    def test_extract_plan_payload_handles_code_fence(self) -> None:
        text = "```json\n{\"focus\":\"x\"}\n```"
        self.assertEqual(extract_plan_payload(text), {"focus": "x"})

    def test_model_planner_returns_plan(self) -> None:
        planner = ModelProgressPlanner(model=_GoodModel())
        plan = planner.plan([], _findings())
        self.assertIsInstance(plan, ProgressPlan)
        self.assertEqual(plan.targets, ["Reach depth on 6 of 8 reps."])

    def test_model_planner_falls_back_on_failure(self) -> None:
        planner = ModelProgressPlanner(model=_BadModel())
        plan = planner.plan([], _findings())
        self.assertTrue(plan.targets)

    def test_verifier_blocks_plan_issue_not_detected(self) -> None:
        plan = ProgressPlan(
            focus="Fix the `knee_valgus` immediately.",
            targets=[],
            next_session_cues=[],
            encouragement="ok",
            confidence_notes=[],
        )
        result = verify_plan(plan, allowed_issues={"shallow_depth"})
        self.assertFalse(result.passed)

    def test_verifier_blocks_medical_language(self) -> None:
        plan = ProgressPlan(
            focus="This could be an injury risk.",
            targets=[],
            next_session_cues=[],
            encouragement="ok",
            confidence_notes=[],
        )
        result = verify_plan(plan, allowed_issues=set())
        self.assertFalse(result.passed)


if __name__ == "__main__":
    unittest.main()
