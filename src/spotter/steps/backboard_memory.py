from __future__ import annotations

from spotter.contracts import SessionRecord

try:  # pragma: no cover - typing only
    from spotter.steps.session_memory import SessionMemory
except ImportError:  # pragma: no cover
    SessionMemory = object  # type: ignore[assignment,misc]


class BackboardSessionMemory:
    """Optional hosted memory adapter.

    Backboard stores cross-session records so the plan can reference history across
    devices. The exact HTTP contract must be confirmed against the Backboard API.
    Until then this adapter delegates to a local fallback store, so enabling
    SPOTTER_MEMORY_BACKEND=backboard never breaks a run.
    """

    def __init__(self, fallback) -> None:
        self.fallback = fallback

    def get_history(self, profile_key: str, limit: int = 20) -> list[SessionRecord]:
        return self.fallback.get_history(profile_key, limit)

    def append(self, record: SessionRecord) -> None:
        self.fallback.append(record)
