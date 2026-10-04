from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from spotter.contracts import CoachSummary, ProgressPlan, SpeechResult

API_KEY_ENV = "ELEVENLABS_API_KEY"
VOICE_ENV = "ELEVENLABS_VOICE_ID"
MODEL_ENV = "ELEVENLABS_MODEL_ID"
OUTPUT_FORMAT_ENV = "ELEVENLABS_OUTPUT_FORMAT"
LANGUAGE_ENV = "SPOTTER_TTS_LANGUAGE"
MAX_CHARS_ENV = "SPOTTER_TTS_MAX_CHARS_PER_RUN"

DEFAULT_MODEL = "eleven_flash_v2_5"  # Lowest cost, supports Hindi + Marathi
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"
DEFAULT_MAX_CHARS = 600
DEFAULT_LANGUAGE = "en"
BASE_URL = "https://api.elevenlabs.io/v1"

# Process-level cache for voice lookup
_VOICE_CACHE: dict[str, str | None] = {}

# Disk cache directory for audio files
_CACHE_DIR = Path(os.getenv("SPOTTER_TTS_CACHE_DIR", "/tmp/spotter-tts-cache"))


def _get_voice_id(api_key: str) -> str | None:
    """Get voice ID from env, or fetch a sensible default from voices API."""
    voice_id = os.getenv(VOICE_ENV)
    if voice_id:
        return voice_id

    # Check cache
    if "default" in _VOICE_CACHE:
        return _VOICE_CACHE["default"]

    # Fetch voices and pick a sensible default
    try:
        url = f"{BASE_URL}/voices/search"
        request = urllib.request.Request(
            url,
            headers={"xi-api-key": api_key},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            voices = data.get("voices", [])
            if voices:
                # Pick first English-capable voice
                for v in voices:
                    verified = v.get("verified_languages", [])
                    if "en" in verified:
                        _VOICE_CACHE["default"] = v["voice_id"]
                        return v["voice_id"]
                # Fallback to first voice
                _VOICE_CACHE["default"] = voices[0]["voice_id"]
                return voices[0]["voice_id"]
    except Exception:
        pass

    _VOICE_CACHE["default"] = None
    return None


def _cache_key(text: str, voice_id: str, model_id: str, language: str) -> str:
    """Generate deterministic cache key."""
    content = f"{text}|{voice_id}|{model_id}|{language}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _get_cached_audio(cache_key: str) -> Path | None:
    """Check if audio exists in disk cache."""
    cache_path = _CACHE_DIR / f"{cache_key}.mp3"
    if cache_path.exists():
        return cache_path
    return None


def _save_to_cache(cache_key: str, audio_bytes: bytes) -> Path:
    """Save audio to disk cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f"{cache_key}.mp3"
    cache_path.write_bytes(audio_bytes)
    return cache_path


def _truncate_to_budget(text: str, max_chars: int) -> tuple[str, int]:
    """Truncate text to character budget. Returns (truncated_text, chars_skipped)."""
    if len(text) <= max_chars:
        return text, 0
    return text[:max_chars], len(text) - max_chars


def _make_request_with_retry(
    url: str,
    payload: bytes,
    headers: dict[str, str],
    max_retries: int = 1,
) -> bytes | None:
    """Make HTTP request with one retry on 429/5xx."""
    for attempt in range(max_retries + 1):
        try:
            request = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return None
        except Exception:
            return None
    return None


def _build_briefing_lines(summary: CoachSummary, plan: ProgressPlan) -> list[str]:
    """Build briefing lines from verified summary and plan (max ~40s spoken)."""
    lines = [summary.summary]
    if summary.what_looked_good:
        lines.append(summary.what_looked_good[0])
    if summary.top_fixes:
        lines.append(summary.top_fixes[0])
    lines.append(plan.focus)
    if plan.next_session_cues:
        lines.append(plan.next_session_cues[0])
    lines.append(plan.encouragement)
    return [line for line in lines if line]


def _build_cue_lines(summary: CoachSummary, plan: ProgressPlan, issues: list[Any]) -> list[dict[str, Any]]:
    """Build cue clips for top 3 issues (max ~140 chars each)."""
    cues = []
    for issue in issues[:3]:
        rep_num = getattr(issue, "rep_id", 1)
        issue_name = getattr(issue, "issue", "form issue")
        fix = summary.top_fixes[0] if summary.top_fixes else "Check your form"
        text = f"Rep {rep_num}: {issue_name.replace('_', ' ')}. {fix}"
        if len(text) > 140:
            text = text[:137] + "..."
        cues.append({
            "issue_id": issue_name,
            "rep_number": rep_num,
            "text": text,
        })
    return cues


class ElevenLabsSynthesizer:
    """Optional hosted voice. Keys come from the user's environment.

    When no key or voice id is configured, or the request fails, the adapter
    degrades to text only for that run. Never crashes the pipeline.
    """

    def synthesize(
        self,
        lines: list[str],
        language: str,
        out_dir: Path,
        *,
        summary: CoachSummary | None = None,
        plan: ProgressPlan | None = None,
        issues: list[Any] | None = None,
    ) -> SpeechResult:
        api_key = os.getenv(API_KEY_ENV)
        if not api_key or not lines:
            return SpeechResult(
                audio_path=None,
                language=language,
                lines=lines,
                backend="elevenlabs_unavailable",
                chars_used=0,
                chars_skipped=0,
            )

        voice_id = _get_voice_id(api_key)
        if not voice_id:
            return SpeechResult(
                audio_path=None,
                language=language,
                lines=lines,
                backend="elevenlabs_unavailable",
                chars_used=0,
                chars_skipped=0,
            )

        model_id = os.getenv(MODEL_ENV, DEFAULT_MODEL)
        output_format = os.getenv(OUTPUT_FORMAT_ENV, DEFAULT_OUTPUT_FORMAT)
        max_chars = int(os.getenv(MAX_CHARS_ENV, str(DEFAULT_MAX_CHARS)))

        # Build text to speak (verified content only)
        briefing_text = " ".join(_build_briefing_lines(summary, plan)) if summary and plan else " ".join(lines)
        cues = _build_cue_lines(summary, plan, issues or []) if summary and plan else []

        # Combine all text for character budget
        all_text = briefing_text + " " + " ".join(c["text"] for c in cues)
        all_text, total_skipped = _truncate_to_budget(all_text, max_chars)

        # Generate briefing audio (first priority)
        briefing_cache_key = _cache_key(briefing_text, voice_id, model_id, language)
        briefing_audio_path = _get_cached_audio(briefing_cache_key)

        if briefing_audio_path is None:
            audio_bytes = _synthesize_text(briefing_text, api_key, voice_id, model_id, output_format)
            if audio_bytes is None:
                return SpeechResult(
                    audio_path=None,
                    language=language,
                    lines=lines,
                    backend="elevenlabs_unavailable",
                    chars_used=0,
                    chars_skipped=total_skipped,
                )
            briefing_audio_path = _save_to_cache(briefing_cache_key, audio_bytes)

        # Generate cue audios
        cue_results = []
        chars_used = len(briefing_text)
        for cue in cues:
            if chars_used + len(cue["text"]) > max_chars:
                break
            chars_used += len(cue["text"])

            cue_cache_key = _cache_key(cue["text"], voice_id, model_id, language)
            cue_audio_path = _get_cached_audio(cue_cache_key)

            if cue_audio_path is None:
                audio_bytes = _synthesize_text(cue["text"], api_key, voice_id, model_id, output_format)
                if audio_bytes is not None:
                    cue_audio_path = _save_to_cache(cue_cache_key, audio_bytes)

            cue["audio_path"] = str(cue_audio_path) if cue_audio_path else None
            cue_results.append(cue)

        # Copy briefing to run directory for artifact serving
        run_briefing_path = out_dir / "briefing.mp3"
        if briefing_audio_path.exists():
            run_briefing_path.write_bytes(briefing_audio_path.read_bytes())

        return SpeechResult(
            audio_path=str(run_briefing_path) if run_briefing_path.exists() else None,
            language=language,
            lines=lines,
            backend="elevenlabs",
            briefing_audio_path=str(run_briefing_path) if run_briefing_path.exists() else None,
            cues=cue_results,
            chars_used=chars_used,
            chars_skipped=total_skipped,
        )


def _synthesize_text(
    text: str,
    api_key: str,
    voice_id: str,
    model_id: str,
    output_format: str,
) -> bytes | None:
    """Synthesize text to speech via ElevenLabs API."""
    url = f"{BASE_URL}/text-to-speech/{voice_id}?output_format={output_format}"
    payload = json.dumps(
        {"text": text, "model_id": model_id},
        ensure_ascii=False,
    ).encode("utf-8")
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    return _make_request_with_retry(url, payload, headers)