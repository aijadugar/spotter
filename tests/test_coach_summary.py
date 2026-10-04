from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.contracts import (
    CoachSummary,
    ExerciseClassification,
    IssueMarker,
    IssueMarkers,
    Rep,
    RepAnalysis,
    RepAnalysisItem,
    Reps,
    UserProfile,
    Variation,
)
from spotter.env import load_local_env
import spotter.slm.providers as slm_providers
from spotter.knowledge_cards import (
    clear_catalog_cache,
    prioritized_coaching_points,
    retrieve_cards,
    retrieve_cards_with_metadata,
)
from spotter.slm.prompting import build_summary_evidence
from spotter.steps import coach_summary, coach_summary_fallback, verifier


class _BadModel:
    def generate_summary(self, prompt: str):
        del prompt
        raise RuntimeError("synthetic model failure")


class _GoodModel:
    def generate_summary(self, prompt: str):
        del prompt
        from spotter.slm.providers import CoachSummaryGeneration

        return CoachSummaryGeneration(
            text=(
                '{"summary":"Structured summary.","what_you_did":["You completed 2 `push_up` reps."],'
                '"what_looked_good":["Tempo looked steady."],'
                '"what_changed_across_reps":["Later reps drifted into `hip_sag`."],'
                '"valid_variation_vs_issue":["The detected variation was `wide_grip_push_up` with `wide_hand_placement` as context."],'
                '"top_fixes":["Keep the hips in line through the later reps."],'
                '"next_session_plan":["Repeat the set with slower reps."],'
                '"confidence_notes":["Confidence is limited."]}'
            ),
            provider="hf_inference",
            model="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        )


def _profile() -> UserProfile:
    return UserProfile(
        goal="beginner_practice",
        experience_level="beginner",
        intended_exercise="push_up",
        intended_variation=None,
        known_limitations=[],
        equipment="bodyweight",
    )


def _classification() -> ExerciseClassification:
    return ExerciseClassification(
        exercise="push_up",
        confidence=0.66,
        window_predictions=[],
        fallback_required=False,
    )


def _reps() -> Reps:
    return Reps(
        exercise="push_up",
        reps=[
            Rep(1, 0, 10, 20, 0.0, 0.33, 0.67),
            Rep(2, 21, 30, 40, 0.7, 1.0, 1.33),
        ],
        partial_reps=[],
    )


def _analysis() -> RepAnalysis:
    return RepAnalysis(
        exercise="push_up",
        items=[
            RepAnalysisItem(
                rep_id=1,
                duration_sec=0.67,
                range_of_motion_score=0.82,
                stability_score=0.84,
                symmetry_score=0.88,
                metrics={"body_line_score": 0.9},
                variation_hints=["wide_grip_push_up"],
            ),
            RepAnalysisItem(
                rep_id=2,
                duration_sec=0.63,
                range_of_motion_score=0.68,
                stability_score=0.71,
                symmetry_score=0.82,
                metrics={"body_line_score": 0.6},
                variation_hints=["wide_grip_push_up"],
            ),
        ],
        aggregate_metrics={
            "avg_rom_score": 0.75,
            "avg_stability_score": 0.78,
            "avg_symmetry_score": 0.85,
            "fatigue_trend_rom_delta": -0.12,
            "pose_valid_ratio": 0.79,
        },
    )


def _variation() -> Variation:
    return Variation(
        exercise="push_up",
        detected_variation="wide_grip_push_up",
        variation_confidence=0.68,
        not_issues=["wide_hand_placement"],
    )


def _issues() -> IssueMarkers:
    return IssueMarkers(
        issues=[
            IssueMarker(
                rep_id=2,
                issue="hip_sag",
                severity=0.82,
                start_frame=24,
                end_frame=31,
                start_sec=0.8,
                end_sec=1.03,
                affected_joints=["left_hip", "right_hip"],
                evidence={"body_line_score": 0.59, "confidence": 0.82},
            )
        ]
    )


class CoachSummaryTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_catalog_cache()

    def test_load_local_env_populates_missing_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "HF_TOKEN=test-token\n"
                "SPOTTER_COACH_SUMMARY_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                current = Path.cwd()
                try:
                    os.chdir(temp_dir)
                    load_local_env()
                finally:
                    os.chdir(current)

                self.assertEqual(os.getenv("HF_TOKEN"), "test-token")
                self.assertEqual(
                    os.getenv("SPOTTER_COACH_SUMMARY_MODEL"),
                    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
                )

    def test_get_coach_summary_model_can_use_local_transformers_provider(self) -> None:
        local_payload = '{"summary":"ok"}'
        with (
            patch.dict(
                os.environ,
                {
                    "SPOTTER_COACH_SUMMARY_PROVIDER": "local_transformers",
                    "SPOTTER_COACH_SUMMARY_MODEL": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
                    "SPOTTER_COACH_SUMMARY_MAX_TOKENS": "123",
                    "SPOTTER_COACH_SUMMARY_TEMPERATURE": "0",
                },
                clear=True,
            ),
            patch.object(slm_providers, "load_local_env"),
            patch.object(
                slm_providers,
                "_generate_local_transformers_summary",
                return_value=local_payload,
            ) as generate,
        ):
            provider = slm_providers.get_coach_summary_model()

            self.assertIsNotNone(provider)
            generation = provider.generate_summary("coach prompt")

        self.assertEqual(generation.provider, "local_transformers")
        self.assertEqual(generation.model, "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
        self.assertEqual(generation.text, local_payload)
        generate.assert_called_once_with(
            model="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
            prompt="coach prompt",
            max_tokens=123,
            max_input_tokens=2048,
            temperature=0.0,
            token=None,
        )

    def test_get_coach_summary_model_can_use_llama_cpp_provider(self) -> None:
        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self) -> bytes:
                return (
                    b'{"choices":[{"message":{"content":"{\\"summary\\":\\"ok\\"}"}}]}'
                )

        with (
            patch.dict(
                os.environ,
                {
                    "SPOTTER_COACH_SUMMARY_PROVIDER": "llama_cpp",
                    "SPOTTER_COACH_SUMMARY_MODEL": "local-nemotron-3-nano-4b-gguf",
                    "SPOTTER_COACH_SUMMARY_MAX_TOKENS": "321",
                    "SPOTTER_COACH_SUMMARY_TEMPERATURE": "0",
                    "SPOTTER_LLAMA_CPP_BASE_URL": "http://127.0.0.1:8090",
                    "SPOTTER_LLAMA_CPP_TIMEOUT": "9",
                },
                clear=True,
            ),
            patch.object(slm_providers, "load_local_env"),
            patch.object(
                slm_providers.urllib.request,
                "urlopen",
                return_value=_Response(),
            ) as urlopen,
        ):
            provider = slm_providers.get_coach_summary_model()

            self.assertIsNotNone(provider)
            generation = provider.generate_summary("coach prompt")

        self.assertEqual(generation.provider, "llama_cpp")
        self.assertEqual(generation.model, "local-nemotron-3-nano-4b-gguf")
        self.assertEqual(generation.text, '{"summary":"ok"}')
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8090/v1/chat/completions")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 9.0)
        self.assertIn(b'"max_tokens": 321', request.data)
        self.assertIn(b'"content": "coach prompt"', request.data)

    def test_card_retrieval_is_deterministic_and_grounded(self) -> None:
        cards = retrieve_cards(
            profile=_profile(),
            classification=_classification(),
            variation=_variation(),
            issues=_issues(),
        )

        card_ids = [card.card_id for card in cards]
        self.assertEqual(
            card_ids[:5],
            [
                "exercise:push_up",
                "variation:wide_grip_push_up",
                "issue:hip_sag",
                "equipment:bodyweight",
                "goal:beginner_practice",
            ],
        )
        self.assertIn("safety:no_diagnosis", card_ids)
        self.assertIn("goal_overlay:push_up:beginner_practice", card_ids)

    def test_retrieval_metadata_reports_external_card_usage(self) -> None:
        retrieval = retrieve_cards_with_metadata(
            profile=_profile(),
            classification=_classification(),
            variation=_variation(),
            issues=_issues(),
        )

        self.assertTrue(retrieval.loaded_pack_paths)
        self.assertGreaterEqual(retrieval.external_cards_loaded, 1)
        self.assertGreaterEqual(retrieval.external_cards_retrieved, 1)

    def test_external_pack_can_override_known_card_by_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pack_path = Path(temp_dir) / "override-pack.json"
            pack_path.write_text(
                json.dumps(
                    {
                        "cards": [
                            {
                                "card_id": "exercise:push_up",
                                "card_type": "exercise",
                                "labels": ["push_up"],
                                "title": "Push-up Override",
                                "summary": "Override summary for deterministic retrieval testing.",
                                "evidence_rules": [
                                    "Use only structured evidence."
                                ],
                                "coaching_points": [
                                    "Return the overridden card."
                                ]
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"SPOTTER_KNOWLEDGE_CARD_PACKS": str(pack_path)},
                clear=False,
            ):
                clear_catalog_cache()
                cards = retrieve_cards(
                    profile=_profile(),
                    classification=_classification(),
                    variation=_variation(),
                    issues=_issues(),
                )

            push_up_card = next(card for card in cards if card.card_id == "exercise:push_up")
            self.assertEqual(push_up_card.title, "Push-up Override")
            self.assertEqual(push_up_card.source_kind, "external")
            self.assertEqual(push_up_card.source_path, str(pack_path.resolve()))

    def test_prompt_evidence_includes_prioritized_cues(self) -> None:
        cards = retrieve_cards(
            profile=_profile(),
            classification=_classification(),
            variation=_variation(),
            issues=_issues(),
        )

        evidence = build_summary_evidence(
            profile=_profile(),
            classification=_classification(),
            reps=_reps(),
            analysis=_analysis(),
            variation=_variation(),
            issues=_issues(),
            cards=cards,
        )

        self.assertTrue(evidence["priority_cues"])
        self.assertIn(
            "Keep shoulders, hips, and ankles moving as one line.",
            evidence["priority_cues"],
        )

    def test_prompt_evidence_omits_raw_issue_interval_evidence(self) -> None:
        issues = IssueMarkers(
            issues=[
                IssueMarker(
                    rep_id=4,
                    issue="shallow_depth",
                    severity=1.0,
                    start_frame=239,
                    end_frame=242,
                    start_sec=7.967,
                    end_sec=8.067,
                    affected_joints=["left_hip", "right_hip"],
                    evidence={
                        "mean_metric_value": 0.0,
                        "peak_frame": 0,
                        "supporting_frames": [239, 240, 241],
                        "threshold": 0.93,
                    },
                )
            ]
        )
        cards = retrieve_cards(
            profile=_profile(),
            classification=_classification(),
            variation=_variation(),
            issues=issues,
        )

        evidence = build_summary_evidence(
            profile=_profile(),
            classification=_classification(),
            reps=_reps(),
            analysis=_analysis(),
            variation=_variation(),
            issues=issues,
            cards=cards,
        )
        interval = evidence["issue_summary"]["top_issue_intervals"][0]

        self.assertNotIn("evidence", interval)
        self.assertEqual(
            interval["evidence_keys"],
            ["mean_metric_value", "peak_frame", "supporting_frames", "threshold"],
        )

    def test_prioritized_coaching_points_prefers_issue_and_context_cards(self) -> None:
        cards = retrieve_cards(
            profile=_profile(),
            classification=_classification(),
            variation=_variation(),
            issues=_issues(),
        )

        points = prioritized_coaching_points(cards, limit=4)

        self.assertLessEqual(len(points), 4)
        self.assertIn("Keep shoulders, hips, and ankles moving as one line.", points)

    def test_coach_summary_falls_back_when_model_fails(self) -> None:
        cards = retrieve_cards(
            profile=_profile(),
            classification=_classification(),
            variation=_variation(),
            issues=_issues(),
        )

        summary = coach_summary.run(
            _profile(),
            _classification(),
            _reps(),
            _analysis(),
            _variation(),
            _issues(),
            cards=cards,
            model=_BadModel(),
        )

        self.assertTrue(summary.confidence_notes)
        self.assertIn("Fallback summary was used", " ".join(summary.confidence_notes))
        self.assertIn("`wide_grip_push_up`", " ".join(summary.valid_variation_vs_issue))

    def test_coach_summary_metadata_includes_provider_and_model(self) -> None:
        result = coach_summary.run_with_metadata(
            _profile(),
            _classification(),
            _reps(),
            _analysis(),
            _variation(),
            _issues(),
            cards=retrieve_cards(
                profile=_profile(),
                classification=_classification(),
                variation=_variation(),
                issues=_issues(),
            ),
            model=_GoodModel(),
        )

        self.assertEqual(result.provider, "hf_inference")
        self.assertEqual(result.model, "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
        self.assertEqual(result.source, "model_or_local")

    def test_extract_json_object_reports_model_output_preview(self) -> None:
        with self.assertRaisesRegex(ValueError, "Output preview: I cannot return JSON"):
            coach_summary._extract_json_object("I cannot return JSON for that request.")

        with self.assertRaisesRegex(ValueError, "Got list"):
            coach_summary._extract_json_object("[]")

    def test_extract_json_object_accepts_common_wrappers(self) -> None:
        payload = {"summary": "ok"}

        self.assertEqual(coach_summary._extract_json_object(json.dumps([payload])), payload)
        self.assertEqual(
            coach_summary._extract_json_object(json.dumps({"coach_summary": payload})),
            payload,
        )

    def test_extract_json_object_ignores_evidence_echo_before_summary(self) -> None:
        payload = {
            "summary": "ok",
            "what_you_did": [],
            "what_looked_good": [],
            "what_changed_across_reps": [],
            "valid_variation_vs_issue": [],
            "top_fixes": [],
            "next_session_plan": [],
            "confidence_notes": [],
        }
        output = (
            '{"rep_id":5,"issue":"shallow_depth","severity":1.0}'
            f"\n{json.dumps(payload)}"
        )

        self.assertEqual(coach_summary._extract_json_object(output), payload)

    def test_summary_from_payload_wraps_string_fields_instead_of_splitting_chars(self) -> None:
        payload = {
            "summary": "Structured summary.",
            "what_you_did": "The athlete executed the squat with control.",
            "what_looked_good": "Tempo stayed steady.",
            "what_changed_across_reps": "Depth was similar across reps.",
            "valid_variation_vs_issue": "No valid variation was overcorrected.",
            "top_fixes": "Sit slightly deeper while staying controlled.",
            "next_session_plan": "Repeat the set with slower reps.",
            "confidence_notes": "Confidence is limited by the camera angle.",
        }

        summary = coach_summary._summary_from_payload(payload)

        self.assertEqual(
            summary.what_you_did,
            ["The athlete executed the squat with control."],
        )
        self.assertEqual(summary.what_looked_good, ["Tempo stayed steady."])

    def test_summary_from_payload_repairs_character_array_fields(self) -> None:
        payload = {
            "summary": "Structured summary.",
            "what_you_did": list("The athlete executed the squat with control."),
            "what_looked_good": ["Tempo stayed steady."],
            "what_changed_across_reps": ["Depth was similar across reps."],
            "valid_variation_vs_issue": ["No valid variation was overcorrected."],
            "top_fixes": ["Sit slightly deeper while staying controlled."],
            "next_session_plan": ["Repeat the set with slower reps."],
            "confidence_notes": ["Confidence is limited by the camera angle."],
        }

        summary = coach_summary._summary_from_payload(payload)

        self.assertEqual(
            summary.what_you_did,
            ["The athlete executed the squat with control."],
        )

    def test_verifier_rejects_issue_not_in_json(self) -> None:
        summary = CoachSummary(
            summary="The strongest issue was `incomplete_depth`.",
            what_you_did=["You completed 2 `push_up` reps."],
            what_looked_good=["Tempo looked steady."],
            what_changed_across_reps=["Later reps lost range."],
            valid_variation_vs_issue=["The detected variation was `wide_grip_push_up`."],
            top_fixes=["Address `incomplete_depth` first."],
            next_session_plan=["Repeat the set with slower reps."],
            confidence_notes=["Confidence is limited."],
        )

        result = verifier.run(
            summary,
            _issues(),
            _variation(),
            classification=_classification(),
            analysis=_analysis(),
            reps=_reps(),
        )

        self.assertFalse(result.passed)
        self.assertFalse(result.checks["no_issue_outside_json"])

    def test_verifier_rejects_diagnosis_and_variation_overcorrection(self) -> None:
        summary = CoachSummary(
            summary="This `wide_grip_push_up` pattern shows a shoulder injury risk.",
            what_you_did=["You completed 2 `push_up` reps."],
            what_looked_good=["The set started under control."],
            what_changed_across_reps=["Later reps drifted into `hip_sag`."],
            valid_variation_vs_issue=[
                "Your `wide_grip_push_up` with `wide_hand_placement` is a problem "
                "that should be fixed."
            ],
            top_fixes=["Correct `wide_hand_placement` before anything else."],
            next_session_plan=["Repeat the set."],
            confidence_notes=["Confidence is limited."],
        )

        result = verifier.run(
            summary,
            _issues(),
            _variation(),
            classification=_classification(),
            analysis=_analysis(),
            reps=_reps(),
        )

        self.assertFalse(result.passed)
        self.assertFalse(result.checks["variation_not_overcorrected"])
        self.assertFalse(result.checks["no_diagnosis"])

    def test_fallback_summary_does_not_false_positive_on_issue_marker_phrase(self) -> None:
        summary = coach_summary_fallback.build_fallback_summary(
            profile=_profile(),
            classification=ExerciseClassification(
                exercise="squat",
                confidence=0.92,
                window_predictions=[],
                fallback_required=False,
            ),
            reps=Reps(
                exercise="squat",
                reps=[Rep(1, 0, 10, 20, 0.0, 0.33, 0.67)],
                partial_reps=[],
            ),
            analysis=RepAnalysis(
                exercise="squat",
                items=[],
                aggregate_metrics={
                    "avg_rom_score": 0.57,
                    "avg_stability_score": 0.68,
                    "avg_symmetry_score": 0.56,
                    "fatigue_trend_rom_delta": -0.10,
                    "pose_valid_ratio": 0.93,
                },
            ),
            variation=Variation(
                exercise="squat",
                detected_variation="wide_squat_stance",
                variation_confidence=0.82,
                not_issues=["wide_stance"],
            ),
            issues=IssueMarkers(
                issues=[
                    IssueMarker(
                        rep_id=1,
                        issue="shallow_depth",
                        severity=0.81,
                        start_frame=10,
                        end_frame=14,
                        start_sec=0.33,
                        end_sec=0.46,
                        affected_joints=["left_hip", "right_hip"],
                        evidence={"confidence": 0.81},
                    )
                ]
            ),
            cards=[],
        )

        result = verifier.run(
            summary,
            IssueMarkers(
                issues=[
                    IssueMarker(
                        rep_id=1,
                        issue="shallow_depth",
                        severity=0.81,
                        start_frame=10,
                        end_frame=14,
                        start_sec=0.33,
                        end_sec=0.46,
                        affected_joints=["left_hip", "right_hip"],
                        evidence={"confidence": 0.81},
                    )
                ]
            ),
            Variation(
                exercise="squat",
                detected_variation="wide_squat_stance",
                variation_confidence=0.82,
                not_issues=["wide_stance"],
            ),
            classification=ExerciseClassification(
                exercise="squat",
                confidence=0.92,
                window_predictions=[],
                fallback_required=False,
            ),
            analysis=RepAnalysis(
                exercise="squat",
                items=[],
                aggregate_metrics={
                    "avg_rom_score": 0.57,
                    "avg_stability_score": 0.68,
                    "avg_symmetry_score": 0.56,
                    "fatigue_trend_rom_delta": -0.10,
                    "pose_valid_ratio": 0.93,
                },
            ),
            reps=Reps(
                exercise="squat",
                reps=[Rep(1, 0, 10, 20, 0.0, 0.33, 0.67)],
                partial_reps=[],
            ),
        )

        self.assertTrue(result.checks["variation_not_overcorrected"])


if __name__ == "__main__":
    unittest.main()
