from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.request

from spotter.contracts import SpeechResult

API_KEY_ENV = "ELEVENLABS_API_KEY"
VOICE_ENV = "ELEVENLABS_VOICE_ID"
MODEL_ID = "eleven_multilingual_v2"


class ElevenLabsSynthesizer:
    """Optional hosted voice. Keys come from the user's environment.

    When no key or voice id is configured, or the request fails, the adapter
    degrades to text only for that run.
    """

    def synthesize(self, lines: list[str], language: str, out_dir: Path) -> SpeechResult:
        api_key = os.getenv(API_KEY_ENV)
        voice_id = os.getenv(VOICE_ENV)
        text = " ".join(line for line in lines if line).strip()
        if not api_key or not voice_id or not text:
            return SpeechResult(
                audio_path=None,
                language=language,
                lines=lines,
                backend="elevenlabs_unavailable",
            )
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        payload = json.dumps({"text": text, "model_id": MODEL_ID}).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        )
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "coach_feedback.mp3"
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                out_path.write_bytes(response.read())
        except Exception:
            return SpeechResult(
                audio_path=None,
                language=language,
                lines=lines,
                backend="elevenlabs_unavailable",
            )
        return SpeechResult(
            audio_path=str(out_path), language=language, lines=lines, backend="elevenlabs"
        )
