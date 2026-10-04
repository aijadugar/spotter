from __future__ import annotations

from pathlib import Path
import json
import os
import sqlite3

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).parent
DB_PATH = Path(os.getenv("SPOTTER_HISTORY_DB", "runs/history.db"))

app = FastAPI(title="Spotter Companion Progress")


def _history(profile_key: str) -> list[dict]:
    if not DB_PATH.is_file():
        return []
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT payload FROM session_records
            WHERE profile_key = ?
            ORDER BY timestamp ASC
            """,
            (profile_key,),
        ).fetchall()
    return [json.loads(row[0]) for row in rows]


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/history")
def history(profile_key: str = Query("default")) -> list[dict]:
    return _history(profile_key)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html")
