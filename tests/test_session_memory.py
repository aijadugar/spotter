from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.contracts import SessionRecord
from spotter.steps.session_memory import SqliteSessionMemory, get_session_memory


def _record(session_id: str, timestamp: str, depth: int, form: float) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        timestamp=timestamp,
        profile_key="friend",
        exercise="squat",
        rep_count=8,
        aggregate_metrics={"avg_rom_score": 0.7, "avg_stability_score": 0.8, "avg_symmetry_score": 0.75},
        issue_counts={"shallow_depth": depth},
        form_score=form,
        source_run_id=session_id,
    )


class SessionMemoryTests(unittest.TestCase):
    def test_history_is_empty_for_unknown_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = SqliteSessionMemory(Path(temp_dir) / "history.db")
            self.assertEqual(memory.get_history("nobody"), [])

    def test_append_and_get_history_round_trip_is_chronological(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = SqliteSessionMemory(Path(temp_dir) / "history.db")
            memory.append(_record("s2", "2026-10-02T09:00:00Z", 2, 0.7))
            memory.append(_record("s1", "2026-10-01T09:00:00Z", 1, 0.6))

            history = memory.get_history("friend")

            self.assertEqual([item.session_id for item in history], ["s1", "s2"])
            self.assertEqual(history[1].issue_counts, {"shallow_depth": 2})
            self.assertEqual(history[0].form_score, 0.6)

    def test_history_respects_limit_and_keeps_most_recent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = SqliteSessionMemory(Path(temp_dir) / "history.db")
            for index in range(5):
                memory.append(_record(f"s{index}", f"2026-10-0{index + 1}T09:00:00Z", index, 0.5))

            history = memory.get_history("friend", limit=2)

            self.assertEqual([item.session_id for item in history], ["s3", "s4"])

    def test_profiles_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = SqliteSessionMemory(Path(temp_dir) / "history.db")
            memory.append(_record("s1", "2026-10-01T09:00:00Z", 1, 0.6))
            self.assertEqual(memory.get_history("someone_else"), [])

    def test_factory_defaults_to_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = get_session_memory(Path(temp_dir) / "history.db")
            self.assertIsInstance(memory, SqliteSessionMemory)


if __name__ == "__main__":
    unittest.main()
