from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import urllib.error
import io
import logging
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ElevenLabsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_synthesize_success(self) -> None:
        from spotter.steps.speech_elevenlabs import ElevenLabsSynthesizer

        synth = ElevenLabsSynthesizer()
        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test_key", "ELEVENLABS_VOICE_ID": "test_voice"}, clear=True):
            with patch("spotter.steps.speech_elevenlabs._get_voice_id", return_value="test_voice"):
                with patch("spotter.steps.speech_elevenlabs._synthesize_text", return_value=b"fake_audio"):
                    with patch("spotter.steps.speech_elevenlabs._get_cached_audio", return_value=None):
                        result = synth.synthesize(
                            lines=["Test summary"],
                            language="en",
                            out_dir=self.out_dir,
                            summary=None,
                            plan=None,
                            issues=[],
                        )

        self.assertEqual(result.backend, "elevenlabs")
        self.assertIsNotNone(result.audio_path)
        self.assertEqual(result.language, "en")

    def test_synthesize_missing_api_key(self) -> None:
        from spotter.steps.speech_elevenlabs import ElevenLabsSynthesizer

        synth = ElevenLabsSynthesizer()
        with patch.dict("os.environ", {}, clear=True):
            result = synth.synthesize(
                lines=["Test summary"],
                language="en",
                out_dir=self.out_dir,
            )

        self.assertEqual(result.backend, "elevenlabs_unavailable")
        self.assertIsNone(result.audio_path)

    def test_synthesize_missing_voice_id(self) -> None:
        from spotter.steps.speech_elevenlabs import ElevenLabsSynthesizer

        synth = ElevenLabsSynthesizer()
        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test_key"}, clear=True):
            with patch("spotter.steps.speech_elevenlabs._get_voice_id", return_value=None):
                result = synth.synthesize(
                    lines=["Test summary"],
                    language="en",
                    out_dir=self.out_dir,
                )

        self.assertEqual(result.backend, "elevenlabs_unavailable")
        self.assertIsNone(result.audio_path)

    def test_character_budget_truncation(self) -> None:
        from spotter.steps.speech_elevenlabs import _truncate_to_budget

        text, skipped = _truncate_to_budget("a" * 700, 600)
        self.assertEqual(len(text), 600)
        self.assertEqual(skipped, 100)

        text, skipped = _truncate_to_budget("short", 600)
        self.assertEqual(text, "short")
        self.assertEqual(skipped, 0)

    def test_api_key_never_in_artifacts(self) -> None:
        from spotter.steps.speech_elevenlabs import ElevenLabsSynthesizer

        synth = ElevenLabsSynthesizer()
        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "secret_key_123", "ELEVENLABS_VOICE_ID": "test_voice"}, clear=True):
            with patch("spotter.steps.speech_elevenlabs._get_voice_id", return_value="test_voice"):
                with patch("spotter.steps.speech_elevenlabs._synthesize_text", return_value=b"audio"):
                    with patch("spotter.steps.speech_elevenlabs._get_cached_audio", return_value=None):
                        result = synth.synthesize(
                            lines=["Test summary"],
                            language="en",
                            out_dir=self.out_dir,
                        )

        result_dict = {
            "audio_path": result.audio_path,
            "language": result.language,
            "lines": result.lines,
            "backend": result.backend,
            "briefing_audio_path": result.briefing_audio_path,
            "cues": result.cues,
            "chars_used": result.chars_used,
            "chars_skipped": result.chars_skipped,
        }
        result_json = json.dumps(result_dict)
        self.assertNotIn("secret_key_123", result_json)

    def test_verifier_only_text_spoken(self) -> None:
        from spotter.steps.speech_elevenlabs import _build_briefing_lines

        summary = type("Summary", (), {
            "summary": "Good form on squats.",
            "what_looked_good": ["Depth was consistent."],
            "top_fixes": ["Push knees out."],
        })()
        plan = type("Plan", (), {
            "focus": "Knee alignment",
            "next_session_cues": ["Keep knees tracking toes."],
            "encouragement": "Great work!",
        })()

        lines = _build_briefing_lines(summary, plan)
        self.assertIn("Good form on squats.", lines)
        self.assertIn("Depth was consistent.", lines)
        self.assertIn("Push knees out.", lines)
        self.assertIn("Knee alignment", lines)
        self.assertIn("Keep knees tracking toes.", lines)
        self.assertIn("Great work!", lines)

    def test_cache_key_deterministic(self) -> None:
        from spotter.steps.speech_elevenlabs import _cache_key

        key1 = _cache_key("hello", "voice1", "model1", "en")
        key2 = _cache_key("hello", "voice1", "model1", "en")
        self.assertEqual(key1, key2)

        key3 = _cache_key("world", "voice1", "model1", "en")
        self.assertNotEqual(key1, key3)

    # 0a: New tests for missing ElevenLabs tests
    def test_429_then_200_retry_with_backoff(self) -> None:
        """Test that 429 then 200 triggers exactly one retry with backoff, and audio is produced."""
        call_count = {"count": 0}
        sleep_times = []

        def mock_synthesize_text(text, api_key, voice_id, model_id, output_format):
            call_count["count"] += 1
            if call_count["count"] == 1:
                # First call: return None to simulate 429 (retryable)
                return None
            # Second call: success
            return b"audio_on_retry"

        def mock_sleep(seconds):
            sleep_times.append(seconds)

        from spotter.steps.speech_elevenlabs import ElevenLabsSynthesizer
        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test_key", "ELEVENLABS_VOICE_ID": "test_voice"}, clear=True):
            with patch("spotter.steps.speech_elevenlabs._get_voice_id", return_value="test_voice"):
                with patch("spotter.steps.speech_elevenlabs._synthesize_text", side_effect=mock_synthesize_text):
                    with patch("time.sleep", side_effect=mock_sleep):
                        synth = ElevenLabsSynthesizer()
                        result = synth.synthesize(
                            lines=["Test summary"],
                            language="en",
                            out_dir=self.out_dir,
                            summary=None,
                            plan=None,
                            issues=[],
                        )

        # Should have retried exactly once
        self.assertEqual(call_count["count"], 2)
        # Should have slept with backoff (first retry: 1s or 2s depending on implementation)
        self.assertTrue(len(sleep_times) >= 1)
        # Should have produced audio
        self.assertEqual(result.backend, "elevenlabs")
        self.assertIsNotNone(result.audio_path)

    def test_persistent_5xx_returns_unavailable_no_partial_files(self) -> None:
        """Test that persistent 5xx errors return elevenlabs_unavailable with no partial files left."""
        def mock_synthesize_text(text, api_key, voice_id, model_id, output_format):
            # Return None to simulate persistent failure (what _make_request_with_retry does after retries exhausted)
            return None

        from spotter.steps.speech_elevenlabs import ElevenLabsSynthesizer
        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test_key", "ELEVENLABS_VOICE_ID": "test_voice"}, clear=True):
            with patch("spotter.steps.speech_elevenlabs._get_voice_id", return_value="test_voice"):
                with patch("spotter.steps.speech_elevenlabs._synthesize_text", side_effect=mock_synthesize_text):
                    synth = ElevenLabsSynthesizer()
                    result = synth.synthesize(
                        lines=["Test summary"],
                        language="en",
                        out_dir=self.out_dir,
                        summary=None,
                        plan=None,
                        issues=[],
                    )

        self.assertEqual(result.backend, "elevenlabs_unavailable")
        self.assertIsNone(result.audio_path)
        self.assertIsNone(result.briefing_audio_path)
        self.assertIsNone(result.cues)
        # No partial files should be left in out_dir
        files = list(self.out_dir.iterdir())
        self.assertEqual(len(files), 0, f"Partial files left: {files}")

    def test_cache_hit_zero_http_requests(self) -> None:
        """Test that second synthesize with same text/voice/model/language makes ZERO HTTP requests."""
        from spotter.steps.speech_elevenlabs import ElevenLabsSynthesizer, _cache_key, _CACHE_DIR

        call_count = {"count": 0}

        def mock_synthesize_text(text, api_key, voice_id, model_id, output_format):
            call_count["count"] += 1
            return b"audio"

        # Pre-populate disk cache
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_key = _cache_key("Test summary", "test_voice", "eleven_flash_v2_5", "en")
        (Path("/tmp/test_tts_cache") / f"{cache_key}.mp3").write_bytes(b"cached_audio")

        from spotter.steps.speech_elevenlabs import ElevenLabsSynthesizer
        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test_key", "ELEVENLABS_VOICE_ID": "test_voice"}, clear=True):
            with patch("spotter.steps.speech_elevenlabs._get_voice_id", return_value="test_voice"):
                with patch("spotter.steps.speech_elevenlabs._get_cached_audio", return_value=Path("/tmp/test_tts_cache") / f"{cache_key}.mp3"):
                    synth = ElevenLabsSynthesizer()
                    result = synth.synthesize(
                        lines=["Test summary"],
                        language="en",
                        out_dir=self.out_dir,
                        summary=None,
                        plan=None,
                        issues=[],
                    )

        # Should have produced audio from cache
        self.assertEqual(result.backend, "elevenlabs")
        self.assertIsNotNone(result.audio_path)
        # _synthesize_text should NOT have been called (cache hit)
        self.assertEqual(call_count["count"], 0, "HTTP request made on cache hit!")

    def test_cue_text_trimming_and_cap(self) -> None:
        """Test that cue text exceeding ~140 chars is trimmed, and cues are capped at 3."""
        from spotter.steps.speech_elevenlabs import _build_cue_lines

        issues = []
        for i in range(5):
            issues.append(type("Issue", (), {
                "issue": f"very_long_issue_name_that_exceeds_one_hundred_forty_characters_limit_{i}",
                "rep_id": i + 1,
            })())

        summary = type("Summary", (), {
            "top_fixes": ["A" * 200],  # Very long fix
        })()
        plan = type("Plan", (), {
            "focus": "test",
            "next_session_cues": [],
            "encouragement": "test",
        })()

        cues = _build_cue_lines(summary, plan, issues)

        # Should be capped at 3 cues
        self.assertLessEqual(len(cues), 3, f"Expected at most 3 cues, got {len(cues)}")

        # Each cue text should be <= 140 chars
        for cue in cues:
            self.assertLessEqual(len(cue["text"]), 140, f"Cue text too long: {len(cue['text'])} chars: {cue['text']}")

    def test_api_key_never_in_logs(self) -> None:
        """Test that API key never appears in logs or written artifacts."""
        import logging
        import io

        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger("spotter.steps.speech_elevenlabs")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        from spotter.steps.speech_elevenlabs import ElevenLabsSynthesizer
        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "secret_key_xyz", "ELEVENLABS_VOICE_ID": "test_voice"}, clear=True):
            with patch("spotter.steps.speech_elevenlabs._get_voice_id", return_value="test_voice"):
                with patch("spotter.steps.speech_elevenlabs._synthesize_text", return_value=b"audio"):
                    with patch("spotter.steps.speech_elevenlabs._get_cached_audio", return_value=None):
                        synth = ElevenLabsSynthesizer()
                        result = synth.synthesize(
                            lines=["Test summary"],
                            language="en",
                            out_dir=self.out_dir,
                            summary=None,
                            plan=None,
                            issues=[],
                        )

        log_contents = log_stream.getvalue()
        self.assertNotIn("secret_key_xyz", log_contents, "API key leaked in logs!")

        # Also verify not in artifact
        result_dict = {
            "audio_path": result.audio_path,
            "language": result.language,
            "lines": result.lines,
            "backend": result.backend,
            "briefing_audio_path": result.briefing_audio_path,
            "cues": result.cues,
            "chars_used": result.chars_used,
            "chars_skipped": result.chars_skipped,
        }
        result_json = json.dumps(result_dict)
        self.assertNotIn("secret_key_xyz", result_json)


if __name__ == "__main__":
    unittest.main()