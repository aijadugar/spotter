from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import urllib.request
import urllib.error

from spotter.contracts import SessionRecord


BACKBOARD_BASE_URL = "https://app.backboard.io/api"
DEFAULT_TIMEOUT = 5.0  # seconds
BACKBOARD_DIR = Path.home() / ".spotter" / "backboard"


class BackboardMemory:
    """Backboard hosted memory adapter with local SQLite fallback.

    - One assistant per Spotter install, created lazily, id cached locally
    - SQLite stays source of truth; append() writes locally first, then pushes
      one compact entry (timeout <= 5s)
    - get_history() merges local + remote deduped by session id
    - No key or any failure -> behave exactly like local SQLite, never fail a run
    - Sends ONLY derived metrics (date, exercise, rep count, per-issue counts, plan focus)
    - Never sends video, frames, landmarks, names, or free text
    - Opt-in: SPOTTER_MEMORY_BACKEND=backboard
    """

    def __init__(self, fallback) -> None:
        self.fallback = fallback
        self._assistant_id: str | None = None
        self._assistant_id_lock = threading.Lock()
        self._local_cache_path = BACKBOARD_DIR / "assistant_id.txt"
        self._enabled = os.getenv("SPOTTER_MEMORY_BACKEND", "").strip().lower() == "backboard"
        self._api_key = os.getenv("BACKBOARD_API_KEY", "").strip()

        # Load cached assistant ID
        if self._local_cache_path.exists():
            try:
                self._assistant_id = self._local_cache_path.read_text().strip()
            except Exception:
                pass

    def _headers(self) -> dict[str, str]:
        return {
            "X-API-Key": self._api_key,
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> tuple[int, dict | None]:
        """Make HTTP request with timeout. Returns (status_code, json_response)."""
        if not self._enabled or not self._api_key:
            return 0, None

        url = f"{BACKBOARD_BASE_URL}{path}"
        data = json.dumps(body).encode("utf-8") if body else None

        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_data = resp.read().decode("utf-8")
                if resp_data:
                    return resp.status, json.loads(resp_data)
                return resp.status, None
        except urllib.error.HTTPError as e:
            return e.code, None
        except (urllib.error.URLError, TimeoutError, Exception):
            return 0, None

    def _ensure_assistant(self) -> str | None:
        """Create or get cached assistant ID."""
        if self._assistant_id:
            return self._assistant_id

        with self._assistant_id_lock:
            # Double-check after acquiring lock
            if self._assistant_id:
                return self._assistant_id

            if not self._enabled or not self._api_key:
                return None

            # Create assistant
            body = {
                "name": "Spotter Progress Memory",
                "instructions": (
                    "You are a memory store for workout progress tracking. "
                    "Store compact session summaries with derived metrics only. "
                    "Never store video, frames, landmarks, names, or free text."
                ),
            }
            status, resp = self._request("POST", "/assistants", body)
            if status in (200, 201) and resp and "id" in resp:
                self._assistant_id = resp["id"]
                # Cache locally
                try:
                    BACKBOARD_DIR.mkdir(parents=True, exist_ok=True)
                    self._local_cache_path.write_text(self._assistant_id)
                except Exception:
                    pass
                return self._assistant_id
            return None

    def _compact_session_record(self, record: SessionRecord) -> dict[str, Any]:
        """Extract only derived metrics for Backboard."""
        return {
            "session_id": record.session_id,
            "timestamp": record.timestamp,
            "exercise": record.exercise,
            "rep_count": record.rep_count,
            "form_score": record.form_score,
            "issue_counts": record.issue_counts,
            "plan_focus": getattr(record, "plan_focus", None),
        }

    def append(self, record: SessionRecord) -> None:
        """Write locally first, then push to Backboard (non-blocking)."""
        # Always write to local SQLite first
        self.fallback.append(record)

        if not self._enabled or not self._api_key:
            return

        # Push to Backboard in background (fire-and-forget)
        def _push():
            try:
                assistant_id = self._ensure_assistant()
                if not assistant_id:
                    return

                compact = self._compact_session_record(record)
                body = {
                    "assistant_id": assistant_id,
                    "content": json.dumps(compact),
                    "metadata": {
                        "type": "spotter_session",
                        "session_id": record.session_id,
                    },
                }
                # Use threads/messages to store memory
                self._request("POST", "/threads/messages", body, timeout=DEFAULT_TIMEOUT)
            except Exception:
                # Silently ignore any errors in background push
                pass

        thread = threading.Thread(target=_push, daemon=True)
        thread.start()

    def get_history(self, profile_key: str, limit: int = 20) -> list[SessionRecord]:
        """Merge local + remote, deduped by session_id."""
        local_records = self.fallback.get_history(profile_key, limit * 2)  # fetch more for dedupe

        if not self._enabled or not self._api_key:
            return local_records[:limit]

        # Try to fetch from Backboard
        assistant_id = self._ensure_assistant()
        if not assistant_id:
            return local_records[:limit]

        # Fetch recent messages from the assistant's thread
        # Note: Backboard API may not have a direct "list memories" endpoint
        # We'll use the thread history approach
        status, resp = self._request("GET", f"/assistants/{assistant_id}")
        if status != 200 or not resp:
            return local_records[:limit]

        # If there's a thread_id in response, fetch messages
        thread_id = resp.get("thread_id")
        if not thread_id:
            return local_records[:limit]

        status, messages_resp = self._request("GET", f"/threads/{thread_id}/messages")
        if status != 200 or not messages_resp:
            return local_records[:limit]

        # Parse remote records
        remote_records: list[SessionRecord] = []
        for msg in messages_resp.get("messages", []):
            try:
                content = msg.get("content", "{}")
                if isinstance(content, str):
                    data = json.loads(content)
                else:
                    data = content

                if data.get("type") != "spotter_session":
                    continue

                # Convert to SessionRecord
                remote_records.append(
                    SessionRecord(
                        session_id=data.get("session_id", str(uuid.uuid4())),
                        timestamp=data.get("timestamp", ""),
                        profile_key=profile_key,
                        exercise=data.get("exercise", "unknown"),
                        rep_count=data.get("rep_count", 0),
                        aggregate_metrics={},
                        issue_counts=data.get("issue_counts", {}),
                        form_score=data.get("form_score", 0.0),
                        source_run_id=data.get("session_id", ""),
                    )
                )
            except Exception:
                continue

        # Merge: local is source of truth, remote supplements
        seen_ids = set()
        merged: list[SessionRecord] = []
        for record in local_records + remote_records:
            if record.session_id not in seen_ids:
                seen_ids.add(record.session_id)
                merged.append(record)

        # Sort by timestamp descending, limit
        merged.sort(key=lambda r: r.timestamp, reverse=True)
        return merged[:limit]

    def delete_cloud_memory(self) -> bool:
        """Delete all cloud memory for this assistant."""
        if not self._enabled or not self._api_key or not self._assistant_id:
            return False

        # Delete the assistant (removes associated memory)
        status, _ = self._request("DELETE", f"/assistants/{self._assistant_id}")
        if status in (200, 204):
            # Clear local cache
            try:
                self._local_cache_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._assistant_id = None
            return True
        return False

    def get_sync_status(self) -> dict[str, Any]:
        """Get sync status for UI."""
        return {
            "enabled": self._enabled,
            "has_key": bool(self._api_key),
            "assistant_id": self._assistant_id,
            "backend": "backboard" if self._enabled else "local",
        }


def create_memory_backend(profile_key: str, db_path: Path | None = None):
    """Factory to create the appropriate memory backend."""
    from spotter.steps.session_memory import SqliteSessionMemory

    local_memory = SqliteSessionMemory(db_path=db_path)

    if os.getenv("SPOTTER_MEMORY_BACKEND", "").strip().lower() == "backboard":
        return BackboardMemory(local_memory)

    return local_memory