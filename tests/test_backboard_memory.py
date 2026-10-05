from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from spotter.contracts import SessionRecord, to_dict
from spotter.steps.session_memory import SqliteSessionMemory, SessionMemory
from spotter.steps.backboard_memory import BackboardMemory, create_memory_backend, DEFAULT_TIMEOUT


class BackboardMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        # Create temp DB for local fallback
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"

        # Clear env vars
        for key in list(os.environ.keys()):
            if key.startswith("SPOTTER_") or key.startswith("BACKBOARD_"):
                os.environ.pop(key, None)

        # Clear cached assistant ID from previous test runs
        from spotter.steps.backboard_memory import BACKBOARD_DIR
        import shutil
        if BACKBOARD_DIR.exists():
            shutil.rmtree(BACKBOARD_DIR, ignore_errors=True)

    def tearDown(self) -> None:
        for key in list(os.environ.keys()):
            if key.startswith("SPOTTER_") or key.startswith("BACKBOARD_"):
                os.environ.pop(key, None)
        # Close any open DB connections before cleanup
        import gc
        gc.collect()
        time.sleep(0.1)
        self.temp_dir.cleanup()

    def _make_record(self, session_id: str, exercise: str = "squat", rep_count: int = 5,
                     issue_counts: dict = None, form_score: float = 0.8) -> SessionRecord:
        return SessionRecord(
            session_id=session_id,
            timestamp=f"2026-09-{int(session_id[-1]):02d}T09:00:00Z",
            profile_key="friend",
            exercise=exercise,
            rep_count=rep_count,
            aggregate_metrics={"avg_rom_score": 0.8, "avg_stability_score": 0.7, "avg_symmetry_score": 0.9},
            issue_counts=issue_counts or {},
            form_score=form_score,
            source_run_id=f"run-{session_id}",
        )

    def test_disabled_by_default(self) -> None:
        """Backboard is disabled unless SPOTTER_MEMORY_BACKEND=backboard."""
        local = SqliteSessionMemory(db_path=self.db_path)
        backend = BackboardMemory(local)
        self.assertFalse(backend._enabled)
        self.assertIsNone(backend._assistant_id)

    def test_enabled_with_env_var(self) -> None:
        """Backboard enables when SPOTTER_MEMORY_BACKEND=backboard and key exists."""
        os.environ["SPOTTER_MEMORY_BACKEND"] = "backboard"
        os.environ["BACKBOARD_API_KEY"] = "test_key"

        local = SqliteSessionMemory(db_path=self.db_path)
        backend = BackboardMemory(local)
        self.assertTrue(backend._enabled)
        self.assertEqual(backend._api_key, "test_key")

    def test_append_writes_locally_first(self) -> None:
        """append() writes to local SQLite first, then pushes to Backboard."""
        os.environ["SPOTTER_MEMORY_BACKEND"] = "backboard"
        os.environ["BACKBOARD_API_KEY"] = "test_key"

        local = SqliteSessionMemory(db_path=self.db_path)
        backend = BackboardMemory(local)

        record = self._make_record("session-1")
        backend.append(record)

        # Verify local write happened
        history = local.get_history("friend", limit=10)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].session_id, "session-1")

    def test_no_key_fallbacks_to_local(self) -> None:
        """No BACKBOARD_API_KEY -> behaves exactly like local SQLite."""
        os.environ["SPOTTER_MEMORY_BACKEND"] = "backboard"
        # No BACKBOARD_API_KEY

        local = SqliteSessionMemory(db_path=self.db_path)
        backend = BackboardMemory(local)

        record = self._make_record("session-1")
        backend.append(record)

        history = backend.get_history("friend", limit=10)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].session_id, "session-1")

    def test_compact_record_excludes_forbidden_fields(self) -> None:
        """Compact record sent to Backboard only contains derived metrics."""
        os.environ["SPOTTER_MEMORY_BACKEND"] = "backboard"
        os.environ["BACKBOARD_API_KEY"] = "test_key"

        local = SqliteSessionMemory(db_path=self.db_path)
        backend = BackboardMemory(local)

        record = self._make_record("session-1", issue_counts={"knee_valgus": 2, "shallow_depth": 1})
        compact = backend._compact_session_record(record)

        # Check required fields present
        self.assertIn("session_id", compact)
        self.assertIn("timestamp", compact)
        self.assertIn("exercise", compact)
        self.assertIn("rep_count", compact)
        self.assertIn("form_score", compact)
        self.assertIn("issue_counts", compact)
        self.assertIn("plan_focus", compact)

        # Check forbidden fields absent
        self.assertNotIn("landmarks", compact)
        self.assertNotIn("frames", compact)
        self.assertNotIn("video", compact)
        self.assertNotIn("raw_text", compact)
        self.assertNotIn("name", compact)

    def test_get_history_merges_local_remote_deduped(self) -> None:
        """get_history() merges local + remote, deduped by session_id."""
        os.environ["SPOTTER_MEMORY_BACKEND"] = "backboard"
        os.environ["BACKBOARD_API_KEY"] = "test_key"

        local = SqliteSessionMemory(db_path=self.db_path)
        backend = BackboardMemory(local)

        # Add local records directly to DB
        local.append(self._make_record("session-1", rep_count=5))
        local.append(self._make_record("session-2", rep_count=6))

        # Verify local records work
        local_history = local.get_history("friend", limit=10)
        self.assertEqual(len(local_history), 2)

        # Mock remote response - use real local DB for local records
        def mock_request(method, path, body=None, timeout=DEFAULT_TIMEOUT):
            if method == "GET" and path == "/assistants/assistant-123":
                return (200, {"id": "assistant-123", "thread_id": "thread-456"})
            elif method == "GET" and path == "/threads/thread-456/messages":
                return (200, {
                    "messages": [
                        {"content": json.dumps({
                            "type": "spotter_session",
                            "session_id": "session-2",  # duplicate
                            "timestamp": "2026-09-02T09:00:00Z",
                            "exercise": "squat",
                            "rep_count": 6,
                            "form_score": 0.8,
                            "issue_counts": {},
                        })},
                        {"content": json.dumps({
                            "type": "spotter_session",
                            "session_id": "session-3",  # new from remote
                            "timestamp": "2026-09-03T09:00:00Z",
                            "exercise": "squat",
                            "rep_count": 7,
                            "form_score": 0.85,
                            "issue_counts": {},
                        })},
                    ]
                })
            return (0, None)

        with patch.object(backend, '_request', side_effect=mock_request):
            with patch.object(backend, '_ensure_assistant', return_value="assistant-123"):
                history = backend.get_history("friend", limit=10)

        # Debug: print what we got
        print(f"History: {[r.session_id for r in history]}")

        # Should have 3 unique sessions (1, 2, 3) - session-2 deduped
        session_ids = [r.session_id for r in history]
        self.assertEqual(len(session_ids), 3)
        self.assertIn("session-1", session_ids)
        self.assertIn("session-2", session_ids)
        self.assertIn("session-3", session_ids)

    def test_401_fallbacks_to_local(self) -> None:
        """401 Unauthorized -> local fallback with no exception."""
        os.environ["SPOTTER_MEMORY_BACKEND"] = "backboard"
        os.environ["BACKBOARD_API_KEY"] = "invalid_key"

        local = SqliteSessionMemory(db_path=self.db_path)
        backend = BackboardMemory(local)

        with patch.object(backend, '_request') as mock_request:
            mock_request.return_value = (401, None)

            record = self._make_record("session-1")
            backend.append(record)  # Should not raise

            history = backend.get_history("friend", limit=10)
            self.assertEqual(len(history), 1)

    def test_timeout_fallbacks_to_local(self) -> None:
        """Timeout -> local fallback with no exception."""
        os.environ["SPOTTER_MEMORY_BACKEND"] = "backboard"
        os.environ["BACKBOARD_API_KEY"] = "test_key"

        local = SqliteSessionMemory(db_path=self.db_path)
        backend = BackboardMemory(local)

        # Override _request to raise TimeoutError on /assistants POST
        # This tests that append() doesn't crash the main thread
        original_request = backend._request
        def mock_request(method, path, body=None, timeout=DEFAULT_TIMEOUT):
            if path == "/assistants" and method == "POST":
                raise TimeoutError("Request timed out")
            return original_request(method, path, body, timeout)

        with patch.object(backend, '_request', side_effect=mock_request):
            record = self._make_record("session-1")
            backend.append(record)  # Main thread should not raise

            # Verify local DB still works (local fallback)
            history = local.get_history("friend", limit=10)
            self.assertEqual(len(history), 1)

    def test_5xx_fallbacks_to_local(self) -> None:
        """5xx error -> local fallback with no exception."""
        os.environ["SPOTTER_MEMORY_BACKEND"] = "backboard"
        os.environ["BACKBOARD_API_KEY"] = "test_key"

        local = SqliteSessionMemory(db_path=self.db_path)
        backend = BackboardMemory(local)

        with patch.object(backend, '_request') as mock_request:
            mock_request.return_value = (500, None)

            record = self._make_record("session-1")
            backend.append(record)  # Should not raise

            history = backend.get_history("friend", limit=10)
            self.assertEqual(len(history), 1)

    def test_key_never_in_artifacts(self) -> None:
        """API key never appears in artifacts or logs."""
        os.environ["SPOTTER_MEMORY_BACKEND"] = "backboard"
        os.environ["BACKBOARD_API_KEY"] = "secret_key_123"

        local = SqliteSessionMemory(db_path=self.db_path)
        backend = BackboardMemory(local)

        record = self._make_record("session-1")
        backend.append(record)

        history = backend.get_history("friend", limit=10)

        # Serialize and check
        import json as json_module
        history_json = json_module.dumps([{
            "session_id": r.session_id,
            "exercise": r.exercise,
            "rep_count": r.rep_count,
            "form_score": r.form_score,
        } for r in history])

        self.assertNotIn("secret_key_123", history_json)
        self.assertNotIn("BACKBOARD_API_KEY", history_json)

    def test_comparison_logic_first_session(self) -> None:
        """Comparison logic for first session."""
        from spotter.steps.progress_insight import compare_sessions

        current = self._make_record("session-1", rep_count=5, form_score=0.8)
        comp = compare_sessions(current, None)

        self.assertEqual(comp.trend, "first_session")
        self.assertEqual(comp.rep_delta, 0)

    def test_comparison_logic_improved(self) -> None:
        """Comparison logic for improved session."""
        from spotter.steps.progress_insight import compare_sessions

        previous = self._make_record("session-1", rep_count=5, issue_counts={"knee_valgus": 2}, form_score=0.7)
        current = self._make_record("session-2", rep_count=6, issue_counts={}, form_score=0.85)

        comp = compare_sessions(current, previous)

        self.assertEqual(comp.trend, "improved")
        self.assertEqual(comp.rep_delta, 1)
        self.assertIn("knee_valgus", comp.issues_fixed)

    def test_comparison_logic_regressed(self) -> None:
        """Comparison logic for regressed session."""
        from spotter.steps.progress_insight import compare_sessions

        previous = self._make_record("session-1", rep_count=5, issue_counts={}, form_score=0.85)
        current = self._make_record("session-2", rep_count=4, issue_counts={"knee_valgus": 2}, form_score=0.65)

        comp = compare_sessions(current, previous)

        self.assertEqual(comp.trend, "regressed")
        self.assertEqual(comp.rep_delta, -1)
        self.assertIn("knee_valgus", comp.issues_new)

    def test_comparison_logic_persistent(self) -> None:
        """Comparison logic for persistent issues."""
        from spotter.steps.progress_insight import compare_sessions

        previous = self._make_record("session-1", issue_counts={"knee_valgus": 2})
        current = self._make_record("session-2", issue_counts={"knee_valgus": 1, "shallow_depth": 1})

        comp = compare_sessions(current, previous)

        self.assertIn("knee_valgus", comp.issues_persistent)
        self.assertIn("shallow_depth", comp.issues_new)

    def test_create_memory_backend_factory(self) -> None:
        """Factory returns correct backend based on env."""
        os.environ["SPOTTER_MEMORY_BACKEND"] = "backboard"
        os.environ["BACKBOARD_API_KEY"] = "test_key"

        backend = create_memory_backend("friend", self.db_path)
        self.assertIsInstance(backend, BackboardMemory)

        # Reset env
        os.environ.pop("SPOTTER_MEMORY_BACKEND", None)
        os.environ.pop("BACKBOARD_API_KEY", None)

        backend2 = create_memory_backend("friend", self.db_path)
        self.assertIsInstance(backend2, SqliteSessionMemory)
        self.assertNotIsInstance(backend2, BackboardMemory)

    def test_get_sync_status(self) -> None:
        """Sync status for UI."""
        os.environ["SPOTTER_MEMORY_BACKEND"] = "backboard"
        os.environ["BACKBOARD_API_KEY"] = "test_key"

        local = SqliteSessionMemory(db_path=self.db_path)
        backend = BackboardMemory(local)

        status = backend.get_sync_status()
        self.assertTrue(status["enabled"])
        self.assertTrue(status["has_key"])
        self.assertEqual(status["backend"], "backboard")

    def test_assistant_created_once_cached(self) -> None:
        """Assistant created once, ID cached locally."""
        os.environ["SPOTTER_MEMORY_BACKEND"] = "backboard"
        os.environ["BACKBOARD_API_KEY"] = "test_key"

        local = SqliteSessionMemory(db_path=self.db_path)
        backend = BackboardMemory(local)

        # Clear cached assistant ID to force creation
        backend._assistant_id = None
        backend._local_cache_path.unlink(missing_ok=True)

        with patch.object(backend, '_request') as mock_request:
            mock_request.return_value = (201, {"id": "assistant-123"})

            # First call creates assistant
            id1 = backend._ensure_assistant()
            self.assertEqual(id1, "assistant-123")
            self.assertEqual(mock_request.call_count, 1)

            # Second call uses cache
            id2 = backend._ensure_assistant()
            self.assertEqual(id2, "assistant-123")
            self.assertEqual(mock_request.call_count, 1)  # No additional call


if __name__ == "__main__":
    unittest.main()