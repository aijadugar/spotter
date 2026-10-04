from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.steps.session_memory import get_session_memory
from spotter.steps.speech import get_speech_synthesizer
from spotter.steps.speech_elevenlabs import ElevenLabsSynthesizer
from spotter.steps.speech_open import OpenTtsSynthesizer


class OptionalBackendTests(unittest.TestCase):
    def test_memory_backend_backboard_delegates_to_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict("os.environ", {"SPOTTER_MEMORY_BACKEND": "backboard"}):
                memory = get_session_memory(Path(temp_dir) / "history.db")
            self.assertEqual(memory.get_history("nobody"), [])

    def test_speech_backend_open_is_selectable(self) -> None:
        with patch.dict("os.environ", {"SPOTTER_TTS_BACKEND": "open"}):
            self.assertIsInstance(get_speech_synthesizer(), OpenTtsSynthesizer)

    def test_speech_backend_elevenlabs_without_key_has_no_audio(self) -> None:
        with (
            patch.dict("os.environ", {"SPOTTER_TTS_BACKEND": "elevenlabs"}, clear=False),
            patch.dict("os.environ", {"ELEVENLABS_API_KEY": "", "ELEVENLABS_VOICE_ID": ""}),
        ):
            synthesizer = get_speech_synthesizer()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = synthesizer.synthesize(["hello"], language="en", out_dir=Path(temp_dir))
        self.assertIsNone(result.audio_path)

    def test_open_tts_without_piper_has_no_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("spotter.steps.speech_open.shutil.which", return_value=None):
                result = OpenTtsSynthesizer().synthesize(
                    ["hello"], language="en", out_dir=Path(temp_dir)
                )
        self.assertIsNone(result.audio_path)
        self.assertEqual(result.backend, "open_unavailable")

    def test_elevenlabs_without_config_is_unavailable(self) -> None:
        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "", "ELEVENLABS_VOICE_ID": ""}):
            with tempfile.TemporaryDirectory() as temp_dir:
                result = ElevenLabsSynthesizer().synthesize(
                    ["hello"], language="en", out_dir=Path(temp_dir)
                )
        self.assertEqual(result.backend, "elevenlabs_unavailable")


if __name__ == "__main__":
    unittest.main()
