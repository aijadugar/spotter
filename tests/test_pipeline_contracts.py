from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter import pipeline
from spotter.contracts import (
    CoachSummary,
    ContractValidationError,
    UserProfile,
    validate_contract,
)
from spotter.exercise_catalog import USER_SELECTABLE_EXERCISES
from spotter.steps.coach_summary import CoachSummaryResult


PROFILE_INPUT = {
    "goal": "beginner_practice",
    "experience_level": "beginner",
    "intended_exercise": "auto",
    "intended_variation": None,
    "known_limitations": [],
    "equipment": "bodyweight",
}

EXPECTED_ARTIFACT_KEYS = {
    "user_profile.json": [
        "equipment",
        "experience_level",
        "goal",
        "intended_exercise",
        "intended_variation",
        "known_limitations",
    ],
    "video_manifest.json": [
        "analysis_allowed",
        "blur_laplacian_var",
        "brightness_mean",
        "codec",
        "container",
        "duration_sec",
        "fps",
        "height",
        "quality_warnings",
        "sampled_frames",
        "total_frames",
        "video_path",
        "width",
    ],
    "pose_sequence.json": [
        "frames",
        "normalized",
        "pose_valid_ratio",
        "smoothing_method",
    ],
    "exercise_classification.json": [
        "confidence",
        "exercise",
        "fallback_required",
        "window_predictions",
    ],
    "rep_debug.json": [
        "accepted_reps",
        "body_line_mean",
        "extrema",
        "raw_signal_range",
        "selected_signal",
        "thresholds",
        "usable_signal_samples",
    ],
    "reps.json": ["exercise", "partial_reps", "reps"],
    "rep_analysis.json": ["aggregate_metrics", "exercise", "items"],
    "variation.json": [
        "detected_variation",
        "exercise",
        "not_issues",
        "variation_confidence",
    ],
    "issue_markers.json": ["issues"],
    "coach_summary.json": [
        "confidence_notes",
        "next_session_plan",
        "summary",
        "top_fixes",
        "valid_variation_vs_issue",
        "what_changed_across_reps",
        "what_looked_good",
        "what_you_did",
    ],
    "verification.json": ["checks", "notes", "passed"],
    "progress_plan.json": [
        "confidence_notes",
        "encouragement",
        "focus",
        "next_session_cues",
        "targets",
    ],
    "session_record.json": [
        "aggregate_metrics",
        "exercise",
        "form_score",
        "issue_counts",
        "profile_key",
        "rep_count",
        "session_id",
        "source_run_id",
        "timestamp",
    ],
    "speech.json": ["audio_path", "backend", "language", "lines"],
    "final_report.json": [
        "artifacts",
        "coach_summary",
        "exercise",
        "issue_markers",
        "profile",
        "progress_plan",
        "rep_analysis",
        "reps",
        "run_id",
        "session_record",
        "speech",
        "variation",
        "verification",
        "video_manifest",
    ],
    "manifest.json": ["artifacts", "mock_mode", "run_id"],
}


class PipelineContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_runs_dir = pipeline.RUNS_DIR
        pipeline.RUNS_DIR = Path(self.temp_dir.name) / "runs"
        self.original_db_path = pipeline.session_memory.DEFAULT_DB_PATH
        pipeline.session_memory.DEFAULT_DB_PATH = Path(self.temp_dir.name) / "runs" / "history.db"

    def tearDown(self) -> None:
        pipeline.session_memory.DEFAULT_DB_PATH = self.original_db_path
        pipeline.RUNS_DIR = self.original_runs_dir
        self.temp_dir.cleanup()

    def _write_video(
        self,
        filename: str,
        *,
        fps: float = 30.0,
        duration_sec: float = 10.0,
        size: tuple[int, int] = (640, 480),
    ) -> Path:
        path = Path(self.temp_dir.name) / filename
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            size,
        )
        self.assertTrue(writer.isOpened())
        width, height = size
        for frame_index in range(int(fps * duration_sec)):
            frame = np.full((height, width, 3), 130, dtype=np.uint8)
            offset = frame_index % 120
            cv2.rectangle(frame, (40 + offset, 80), (260 + offset, 300), (245, 245, 245), -1)
            cv2.line(
                frame,
                (0, frame_index % height),
                (width - 1, height - 1),
                (20, 20, 20),
                3,
            )
            writer.write(frame)
        writer.release()
        return path

    def _assert_pipeline_artifacts(self, result: dict[str, object]) -> None:
        run_dir = Path(str(result["run_dir"]))
        manifest_path = run_dir / "manifest.json"
        self.assertTrue(manifest_path.exists())

        for artifact_name, keys in EXPECTED_ARTIFACT_KEYS.items():
            artifact_path = run_dir / artifact_name
            self.assertTrue(artifact_path.exists(), artifact_name)
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(sorted(payload.keys()), keys, artifact_name)
            if artifact_name == "final_report.json":
                self.assertIn("issue_thumbnail_paths", payload["artifacts"])
                self.assertIsInstance(payload["artifacts"]["issue_thumbnail_paths"], list)
                self.assertIn("issue_clip_paths", payload["artifacts"])
                self.assertIsInstance(payload["artifacts"]["issue_clip_paths"], list)
                self.assertIn("knowledge_card_pack_paths", payload["artifacts"])
                self.assertIsInstance(payload["artifacts"]["knowledge_card_pack_paths"], list)
                self.assertIn("knowledge_external_cards_loaded", payload["artifacts"])
                self.assertIn("knowledge_external_cards_retrieved", payload["artifacts"])

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertTrue(manifest["mock_mode"])
        self.assertEqual(
            [artifact["name"] for artifact in manifest["artifacts"]],
            [
                "user_profile.json",
                "video_manifest.json",
                "pose_sequence.json",
                "exercise_classification.json",
                "reps.json",
                "rep_debug.json",
                "rep_analysis.json",
                "variation.json",
                "issue_markers.json",
                "coach_summary.json",
                "verification.json",
                "progress_plan.json",
                "session_record.json",
                "speech.json",
                "final_report.json",
            ],
        )

    def test_pipeline_runs_end_to_end_without_video(self) -> None:
        result = pipeline.run_pipeline(video_path=None, profile_input=PROFILE_INPUT, mock=True)

        self._assert_pipeline_artifacts(result)
        report = result["final_report"]
        self.assertEqual(report["exercise"]["exercise"], "squat")
        self.assertEqual(report["video_manifest"]["quality_warnings"], ["video_decode_failed"])
        self.assertFalse(report["video_manifest"]["analysis_allowed"])

    def test_pipeline_runs_end_to_end_with_fixture_video_path(self) -> None:
        fixture = self._write_video("sample.mp4")
        result = pipeline.run_pipeline(
            video_path=str(fixture), profile_input=PROFILE_INPUT, mock=True
        )

        self._assert_pipeline_artifacts(result)
        report = result["final_report"]
        self.assertEqual(report["video_manifest"]["video_path"], str(fixture))
        self.assertEqual(report["video_manifest"]["quality_warnings"], [])
        self.assertTrue(report["video_manifest"]["analysis_allowed"])
        self.assertEqual(report["video_manifest"]["width"], 640)
        self.assertEqual(report["video_manifest"]["height"], 480)

    def test_pipeline_emits_progress_after_steps(self) -> None:
        events: list[dict[str, object]] = []

        result = pipeline.run_pipeline(
            video_path=None,
            profile_input=PROFILE_INPUT,
            mock=True,
            progress=events.append,
        )

        self._assert_pipeline_artifacts(result)
        done_events = [
            event
            for event in events
            if event.get("type") == "progress" and event.get("status") == "done"
        ]
        self.assertEqual(
            [event["step"] for event in done_events],
            ["quality", "pose", "exercise", "reps", "issues", "render", "coach", "memory", "plan"],
        )
        payload_by_step = {str(event["step"]): event.get("payload", {}) for event in done_events}
        self.assertEqual(payload_by_step["exercise"]["exercise"], "squat")
        self.assertEqual(payload_by_step["reps"]["rep_count"], 0)
        self.assertEqual(payload_by_step["issues"]["issue_count"], 0)
        self.assertIn("annotated_video_path", payload_by_step["render"])

    def test_pipeline_can_disable_verifier_and_keep_model_summary(self) -> None:
        model_summary = CoachSummary(
            summary="Model summary kept.",
            what_you_did=["You completed 1 `squat` rep."],
            what_looked_good=["The setup looked steady."],
            what_changed_across_reps=["Not enough reps for a trend."],
            valid_variation_vs_issue=["The detected variation was `wide_squat_stance`."],
            top_fixes=["Sit slightly deeper before standing up."],
            next_session_plan=["Repeat the set with the same camera angle."],
            confidence_notes=["Confidence is limited."],
        )
        with (
            patch.dict(os.environ, {"SPOTTER_COACH_SUMMARY_BYPASS_VERIFIER": "1"}),
            patch(
                "spotter.pipeline.coach_summary.run_with_metadata",
                return_value=CoachSummaryResult(
                    summary=model_summary,
                    provider="hf_inference",
                    model="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
                    source="model_or_local",
                ),
            ),
            patch(
                "spotter.pipeline.verifier.run",
            ) as verifier_run,
        ):
            result = pipeline.run_pipeline(video_path=None, profile_input=PROFILE_INPUT, mock=True)

        self._assert_pipeline_artifacts(result)
        report = result["final_report"]
        self.assertEqual(report["coach_summary"]["summary"], "Model summary kept.")
        self.assertEqual(report["artifacts"]["coach_summary_source"], "model_or_local")
        self.assertEqual(report["artifacts"]["coach_summary_provider"], "hf_inference")
        self.assertEqual(
            report["artifacts"]["coach_summary_model"],
            "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        )
        self.assertTrue(report["artifacts"]["coach_summary_verifier_bypassed"])
        self.assertTrue(report["verification"]["passed"])
        self.assertEqual(report["verification"]["checks"], {"verifier_disabled": True})
        self.assertEqual(
            report["verification"]["notes"],
            ["Coach summary verifier is disabled for this run."],
        )
        self.assertTrue(report["artifacts"]["coach_summary_verifier_bypass_requested"])
        verifier_run.assert_not_called()

    def test_contract_validation_rejects_missing_required_field(self) -> None:
        payload = {
            "goal": "beginner_practice",
            "experience_level": "beginner",
            "intended_exercise": "auto",
            "intended_variation": None,
            "known_limitations": [],
        }

        with self.assertRaisesRegex(ContractValidationError, "missing required"):
            validate_contract("user_profile.json", payload)

    def test_contract_validation_rejects_invalid_enum_value(self) -> None:
        profile = UserProfile(
            goal="beginner_practice",
            experience_level="expert",
            intended_exercise="auto",
            intended_variation=None,
            known_limitations=[],
            equipment="bodyweight",
        )

        with self.assertRaisesRegex(ContractValidationError, "invalid enum"):
            validate_contract("user_profile.json", profile)

    def test_catalog_exercises_are_valid_profile_inputs(self) -> None:
        for exercise in USER_SELECTABLE_EXERCISES:
            with self.subTest(exercise=exercise):
                profile = UserProfile(
                    goal="beginner_practice",
                    experience_level="beginner",
                    intended_exercise=exercise,
                    intended_variation=None,
                    known_limitations=[],
                    equipment="bodyweight",
                )

                validate_contract("user_profile.json", profile)

    def test_pipeline_runs_for_each_catalog_exercise(self) -> None:
        for exercise in USER_SELECTABLE_EXERCISES:
            with self.subTest(exercise=exercise):
                result = pipeline.run_pipeline(
                    video_path=None,
                    profile_input={**PROFILE_INPUT, "intended_exercise": exercise},
                    mock=True,
                )

                report = result["final_report"]
                self.assertEqual(report["exercise"]["exercise"], exercise)

    def test_pipeline_runs_with_manual_unknown_exercise(self) -> None:
        result = pipeline.run_pipeline(
            video_path=None,
            profile_input={**PROFILE_INPUT, "intended_exercise": "unknown"},
            mock=True,
        )

        self._assert_pipeline_artifacts(result)
        report = result["final_report"]
        self.assertEqual(report["exercise"]["exercise"], "unknown")
        self.assertFalse(report["exercise"]["fallback_required"])
        self.assertEqual(report["reps"]["reps"], [])
        self.assertEqual(report["reps"]["partial_reps"], [{"reason": "unknown_exercise"}])

    def test_mock_mode_defaults_to_real_when_video_path_is_present(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(pipeline._env_mock_mode("sample.mp4"))
            self.assertTrue(pipeline._env_mock_mode(None))


if __name__ == "__main__":
    unittest.main()
