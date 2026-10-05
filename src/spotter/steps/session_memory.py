from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from spotter.contracts import SessionRecord, to_dict

DEFAULT_DB_PATH = Path("runs") / "history.db"
MEMORY_BACKEND_ENV = "SPOTTER_MEMORY_BACKEND"


class SessionMemory(Protocol):
    def get_history(self, profile_key: str, limit: int = 20) -> list[SessionRecord]:
        ...

    def append(self, record: SessionRecord) -> None:
        ...


class SqliteSessionMemory:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_records (
                    session_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    profile_key TEXT NOT NULL,
                    exercise TEXT NOT NULL,
                    rep_count INTEGER NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )

    def append(self, record: SessionRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO session_records
                    (session_id, timestamp, profile_key, exercise, rep_count, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.session_id,
                    record.timestamp,
                    record.profile_key,
                    record.exercise,
                    record.rep_count,
                    json.dumps(to_dict(record), ensure_ascii=False),
                ),
            )

    def get_history(self, profile_key: str, limit: int = 20) -> list[SessionRecord]:
        if limit <= 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM session_records
                WHERE profile_key = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (profile_key, limit),
            ).fetchall()
        records = [SessionRecord(**json.loads(row[0])) for row in rows]
        return list(reversed(records))

    def close(self) -> None:
        """Close any open connections (no-op for SQLite, kept for interface compatibility)."""
        pass


def get_session_memory(db_path: Path | str | None = None) -> SessionMemory:
    backend = os.getenv(MEMORY_BACKEND_ENV, "sqlite").strip().lower()
    if backend == "backboard":
        from spotter.steps.backboard_memory import BackboardSessionMemory

        return BackboardSessionMemory(fallback=SqliteSessionMemory(db_path or DEFAULT_DB_PATH))
    return SqliteSessionMemory(db_path or DEFAULT_DB_PATH)
