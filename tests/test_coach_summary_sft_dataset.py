from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.coach_summary_sft_dataset import (  # noqa: E402
    build_sft_row_from_run_dir,
    collect_run_dirs,
    split_sft_rows,
    write_jsonl,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class CoachSummarySftDatasetTests(unittest.TestCase):
    def test_build_sft_row_from_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run-001"
            run_dir.mkdir()
            _write_json(
                run_dir / "user_profile.json",
                {
                    "goal": "beginner_practice",
                    "experience_level": "beginner",
                    "intended_exercise": "push_up",
                    "intended_variation": None,
                    "known_limitations": [],
                    "equipment": "bodyweight",
                },
            )
            _write_json(
                run_dir / "exercise_classification.json",
                {
                    "exercise": "push_up",
                    "confidence": 0.8,
                    "window_predictions": [],
                    "fallback_required": False,
                },
            )
            _write_json(
                run_dir / "reps.json",
                {
                    "exercise": "push_up",
                    "reps": [
                        {
                            "rep_id": 1,
                            "start_frame": 0,
                            "mid_frame": 10,
                            "end_frame": 20,
                            "start_sec": 0.0,
                            "mid_sec": 0.3,
                            "end_sec": 0.7,
                        }
                    ],
                    "partial_reps": [],
                },
            )
            _write_json(
                run_dir / "rep_analysis.json",
                {
                    "exercise": "push_up",
                    "items": [
                        {
                            "rep_id": 1,
                            "duration_sec": 0.7,
                            "range_of_motion_score": 0.8,
                            "stability_score": 0.82,
                            "symmetry_score": 0.84,
                            "metrics": {"body_line_score": 0.79},
                            "variation_hints": ["wide_grip_push_up"],
                        }
                    ],
                    "aggregate_metrics": {"pose_valid_ratio": 0.93},
                },
            )
            _write_json(
                run_dir / "variation.json",
                {
                    "exercise": "push_up",
                    "detected_variation": "wide_grip_push_up",
                    "variation_confidence": 0.77,
                    "not_issues": ["wide_hand_placement"],
                },
            )
            _write_json(
                run_dir / "issue_markers.json",
                {
                    "issues": [
                        {
                            "rep_id": 1,
                            "issue": "hip_sag",
                            "severity": 0.8,
                            "start_frame": 10,
                            "end_frame": 16,
                            "start_sec": 0.3,
                            "end_sec": 0.53,
                            "affected_joints": ["left_hip", "right_hip"],
                            "evidence": {"body_line_score": 0.61},
                        }
                    ]
                },
            )
            _write_json(
                run_dir / "coach_summary.json",
                {
                    "summary": "Example grounded summary.",
                    "what_you_did": ["You completed 1 `push_up` rep."],
                    "what_looked_good": ["The setup looked organized."],
                    "what_changed_across_reps": ["Not enough reps for a strong trend."],
                    "valid_variation_vs_issue": [
                        "The detected variation was `wide_grip_push_up` and `wide_hand_placement` stayed context only."
                    ],
                    "top_fixes": ["Keep shoulders, hips, and ankles moving as one line."],
                    "next_session_plan": ["Repeat the set with the same setup."],
                    "confidence_notes": ["Confidence is moderate."],
                },
            )

            row = build_sft_row_from_run_dir(run_dir)

        self.assertEqual(len(row["messages"]), 3)
        self.assertEqual(row["messages"][0]["role"], "system")
        self.assertIn("knowledge_cards", row["messages"][1]["content"])
        self.assertIn("Example grounded summary.", row["messages"][2]["content"])
        self.assertEqual(row["metadata"]["exercise"], "push_up")

    def test_collect_split_and_write_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_dir = Path(temp_dir) / "runs"
            runs_dir.mkdir()
            for index in range(2):
                run_dir = runs_dir / f"run-{index:03d}"
                run_dir.mkdir()
                for filename in [
                    "user_profile.json",
                    "exercise_classification.json",
                    "reps.json",
                    "rep_analysis.json",
                    "variation.json",
                    "issue_markers.json",
                    "coach_summary.json",
                ]:
                    _write_json(run_dir / filename, {})
            collected = collect_run_dirs(runs_dir)
            train_rows, eval_rows = split_sft_rows(
                [{"id": 1}, {"id": 2}, {"id": 3}],
                eval_count=1,
                seed=5,
            )
            output_path = Path(temp_dir) / "dataset.jsonl"
            write_jsonl(output_path, train_rows)

            written_lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(collected), 2)
        self.assertEqual(len(eval_rows), 1)
        self.assertEqual(len(train_rows), 2)
        self.assertEqual(len(written_lines), 2)


if __name__ == "__main__":
    unittest.main()
