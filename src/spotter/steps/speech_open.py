from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from spotter.contracts import SpeechResult


class OpenTtsSynthesizer:
    """Open TTS adapter. Uses a local `piper` binary when available.

    The voice model is configured by the user (for example via SPOTTER_TTS_VOICE).
    When the binary or model is missing, the adapter degrades to text only.
    """

    def synthesize(self, lines: list[str], language: str, out_dir: Path) -> SpeechResult:
        text = " ".join(line for line in lines if line).strip()
        piper = shutil.which("piper")
        if piper is None or not text:
            return SpeechResult(
                audio_path=None,
                language=language,
                lines=lines,
                backend="open_unavailable",
            )
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "coach_feedback.wav"
        try:
            process = subprocess.run(
                [piper, "--output_file", str(out_path)],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=120,
            )
        except Exception:
            return SpeechResult(
                audio_path=None, language=language, lines=lines, backend="open_unavailable"
            )
        if process.returncode != 0 or not out_path.is_file():
            return SpeechResult(
                audio_path=None, language=language, lines=lines, backend="open_unavailable"
            )
        return SpeechResult(
            audio_path=str(out_path), language=language, lines=lines, backend="open"
        )
