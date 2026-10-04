# Spotter Companion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Spotter's one-shot form review into a longitudinal, voice-enabled companion with a small open model that generates grounded next-session plans.

**Architecture:** Keep the existing pose/rep/issue/coach pipeline untouched. Append three new stages after the coach summary is verified: `session_memory` (load/append structured history), `progress_plan` (history + today's findings to a next-session plan, owned by a fine-tuned model with a deterministic fallback), and `speech` (verified text to audio, off by default). A separate `render/` FastAPI app charts the history store.

**Tech Stack:** Python 3.10+, stdlib `sqlite3`, existing `dataclasses` contracts, Gradio/FastAPI app, `unittest` tests. Optional external services: Tinker (LoRA fine-tune), Backboard (memory), ElevenLabs (voice), Render (dashboard).

## Global Constraints

- Python requires-python `>=3.10`; target-version `py310`; ruff line-length 100.
- Structured evidence first, language second: the plan model may only reference detected issues.
- Offline-first: every new stage degrades gracefully and the pipeline never fails because a new stage is unavailable.
- Not a medical device: never emit diagnosis, injury-prevention, or fall-risk language.
- New artifacts: `session_record.json`, `progress_plan.json`, `speech.json`.
- Environment switches mirror the existing `SPOTTER_COACH_SUMMARY_PROVIDER` pattern: `SPOTTER_MEMORY_BACKEND`, `SPOTTER_PROGRESS_PLAN_PROVIDER`, `SPOTTER_TTS_BACKEND`.
- Tests run without heavy deps (no torch/mediapipe) for the new modules; run a single test file with `python3 tests/test_<name>.py -v`.

---

### Task 1: Add SessionRecord, ProgressPlan, SpeechResult contracts

**Files:**
- Modify: `src/spotter/contracts.py`
- Test: `tests/test_companion_contracts.py`

**Interfaces:**
- Consumes: existing `_require_*` helpers and `validate_contract` registry in `src/spotter/contracts.py`.
- Produces: `SessionRecord`, `ProgressPlan`, `SpeechResult` dataclasses; validators registered under `session_record.json`, `progress_plan.json`, `speech.json`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_companion_contracts.py`:

```python
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.contracts import (
    ContractValidationError,
    ProgressPlan,
    SessionRecord,
    SpeechResult,
    validate_contract,
)


def _record() -> SessionRecord:
    return SessionRecord(
        session_id="20261003T090000Z-abc12345",
        timestamp="2026-10-03T09:00:00Z",
        profile_key="friend",
        exercise="squat",
        rep_count=8,
        aggregate_metrics={
            "avg_rom_score": 0.72,
            "avg_stability_score": 0.80,
            "avg_symmetry_score": 0.77,
            "avg_rep_duration_sec": 1.9,
        },
        issue_counts={"shallow_depth": 2},
        form_score=0.76,
        source_run_id="20261003T090000Z-abc12345",
    )


class CompanionContractTests(unittest.TestCase):
    def test_session_record_contract_accepts_valid_payload(self) -> None:
        validate_contract("session_record.json", _record())

    def test_session_record_rejects_form_score_out_of_range(self) -> None:
        bad = SessionRecord(**{**_record().__dict__, "form_score": 1.4})
        with self.assertRaises(ContractValidationError):
            validate_contract("session_record.json", bad)

    def test_progress_plan_contract_accepts_valid_payload(self) -> None:
        plan = ProgressPlan(
            focus="Control depth on the last reps.",
            targets=["Reach parallel depth on 6 of 8 reps."],
            next_session_cues=["Slow the descent."],
            encouragement="You are trending up.",
            confidence_notes=["Based on 3 sessions."],
        )
        validate_contract("progress_plan.json", plan)

    def test_progress_plan_rejects_non_string_target(self) -> None:
        plan = ProgressPlan(
            focus="x",
            targets=["ok"],
            next_session_cues=[],
            encouragement="y",
            confidence_notes=[],
        )
        payload = plan.__dict__
        payload["targets"] = [1]
        with self.assertRaises(ContractValidationError):
            validate_contract("progress_plan.json", payload)

    def test_speech_contract_allows_null_audio_path(self) -> None:
        result = SpeechResult(
            audio_path=None,
            language="en",
            lines=["Nice set."],
            backend="none",
        )
        validate_contract("speech.json", result)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/test_companion_contracts.py -v`
Expected: FAIL with `ImportError: cannot import name 'ProgressPlan'`.

- [ ] **Step 3: Add the dataclasses**

In `src/spotter/contracts.py`, after the `Verification` dataclass (around line 150), add:

```python
@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    timestamp: str
    profile_key: str
    exercise: Exercise
    rep_count: int
    aggregate_metrics: dict[str, Any]
    issue_counts: dict[str, int]
    form_score: float
    source_run_id: str


@dataclass(frozen=True)
class ProgressPlan:
    focus: str
    targets: list[str]
    next_session_cues: list[str]
    encouragement: str
    confidence_notes: list[str]


@dataclass(frozen=True)
class SpeechResult:
    audio_path: str | None
    language: str
    lines: list[str]
    backend: str
```

- [ ] **Step 4: Add the validators**

In `src/spotter/contracts.py`, inside `validate_contract`, add to the `validators` dict:

```python
        "session_record.json": _validate_session_record,
        "progress_plan.json": _validate_progress_plan,
        "speech.json": _validate_speech_result,
```

Then add these functions near the other validators (after `_validate_verification`):

```python
def _validate_session_record(value: Any, path: str) -> None:
    payload = _require_mapping(value, path)
    _require_fields(
        payload,
        {
            "session_id",
            "timestamp",
            "profile_key",
            "exercise",
            "rep_count",
            "aggregate_metrics",
            "issue_counts",
            "form_score",
            "source_run_id",
        },
        path,
    )
    _require_type(payload["session_id"], str, f"{path}.session_id")
    _require_type(payload["timestamp"], str, f"{path}.timestamp")
    _require_type(payload["profile_key"], str, f"{path}.profile_key")
    _require_enum(payload["exercise"], EXERCISES, f"{path}.exercise")
    _require_int(payload["rep_count"], f"{path}.rep_count", minimum=0)
    _require_mapping(payload["aggregate_metrics"], f"{path}.aggregate_metrics")
    _require_mapping(payload["issue_counts"], f"{path}.issue_counts")
    for key, count in payload["issue_counts"].items():
        _require_type(key, str, f"{path}.issue_counts key")
        _require_int(count, f"{path}.issue_counts[{key}]", minimum=0)
    _require_score(payload["form_score"], f"{path}.form_score")
    _require_type(payload["source_run_id"], str, f"{path}.source_run_id")


def _validate_progress_plan(value: Any, path: str) -> None:
    payload = _require_mapping(value, path)
    _require_fields(
        payload,
        {"focus", "targets", "next_session_cues", "encouragement", "confidence_notes"},
        path,
    )
    _require_type(payload["focus"], str, f"{path}.focus")
    _require_string_list(payload["targets"], f"{path}.targets")
    _require_string_list(payload["next_session_cues"], f"{path}.next_session_cues")
    _require_type(payload["encouragement"], str, f"{path}.encouragement")
    _require_string_list(payload["confidence_notes"], f"{path}.confidence_notes")


def _validate_speech_result(value: Any, path: str) -> None:
    payload = _require_mapping(value, path)
    _require_fields(payload, {"audio_path", "language", "lines", "backend"}, path)
    if payload["audio_path"] is not None:
        _require_type(payload["audio_path"], str, f"{path}.audio_path")
    _require_type(payload["language"], str, f"{path}.language")
    _require_string_list(payload["lines"], f"{path}.lines")
    _require_type(payload["backend"], str, f"{path}.backend")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 tests/test_companion_contracts.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add src/spotter/contracts.py tests/test_companion_contracts.py
git commit -m "feat: add session, plan, and speech contracts"
```

---

### Task 2: SQLite session memory

**Files:**
- Create: `src/spotter/steps/session_memory.py`
- Test: `tests/test_session_memory.py`

**Interfaces:**
- Consumes: `SessionRecord`, `to_dict` from `spotter.contracts`.
- Produces: `SessionMemory` protocol; `SqliteSessionMemory(db_path)` with `append(record)` and `get_history(profile_key, limit=20)` returning chronological `list[SessionRecord]`; `get_session_memory(db_path=None) -> SessionMemory`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_session_memory.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/test_session_memory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spotter.steps.session_memory'`.

- [ ] **Step 3: Implement the module**

Create `src/spotter/steps/session_memory.py`:

```python
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

        return sqlite3.connect(self.db_path)

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


def get_session_memory(db_path: Path | str | None = None) -> SessionMemory:
    backend = os.getenv(MEMORY_BACKEND_ENV, "sqlite").strip().lower()
    if backend == "backboard":
        from spotter.steps.backboard_memory import BackboardSessionMemory

        return BackboardSessionMemory(fallback=SqliteSessionMemory(db_path or DEFAULT_DB_PATH))
    return SqliteSessionMemory(db_path or DEFAULT_DB_PATH)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 tests/test_session_memory.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/spotter/steps/session_memory.py tests/test_session_memory.py
git commit -m "feat: add sqlite session memory"
```

---

### Task 3: Deterministic progress planner

**Files:**
- Create: `src/spotter/steps/progress_plan.py`
- Test: `tests/test_progress_plan.py`

**Interfaces:**
- Consumes: `SessionRecord`, `ProgressPlan` from `spotter.contracts`.
- Produces: `SessionFindings` dataclass; `build_fallback_plan(history, findings, failure_reason=None) -> ProgressPlan`; `get_progress_planner() -> ProgressPlanner`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_progress_plan.py`:

```python
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.contracts import SessionRecord
from spotter.steps.progress_plan import SessionFindings, build_fallback_plan


def _record(session_id: str, timestamp: str, rom: float, depth: int) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        timestamp=timestamp,
        profile_key="friend",
        exercise="squat",
        rep_count=8,
        aggregate_metrics={
            "avg_rom_score": rom,
            "avg_stability_score": 0.8,
            "avg_symmetry_score": 0.78,
        },
        issue_counts={"shallow_depth": depth},
        form_score=rom,
        source_run_id=session_id,
    )


def _findings(depth: int = 2) -> SessionFindings:
    return SessionFindings(
        exercise="squat",
        rep_count=8,
        aggregate_metrics={
            "avg_rom_score": 0.71,
            "avg_stability_score": 0.82,
            "avg_symmetry_score": 0.79,
        },
        issue_labels=["shallow_depth"] * depth,
    )


class FallbackPlanTests(unittest.TestCase):
    def test_plan_focus_uses_most_frequent_issue(self) -> None:
        history = [_record("s1", "2026-10-01T09:00:00Z", 0.6, 1)]
        plan = build_fallback_plan(history, _findings(depth=2))
        self.assertIn("shallow_depth", plan.focus)

    def test_plan_targets_use_rom_metric(self) -> None:
        history = [_record("s1", "2026-10-01T09:00:00Z", 0.6, 1)]
        plan = build_fallback_plan(history, _findings())
        self.assertTrue(any("range" in target.lower() or "depth" in target.lower() for target in plan.targets))

    def test_plan_notes_failure_reason_when_provided(self) -> None:
        plan = build_fallback_plan([], _findings(), failure_reason="model_unavailable")
        self.assertTrue(any("model_unavailable" in note for note in plan.confidence_notes))

    def test_plan_with_no_history_still_returns_plan(self) -> None:
        plan = build_fallback_plan([], _findings(depth=0))
        self.assertTrue(plan.encouragement)
        self.assertTrue(plan.targets)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/test_progress_plan.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spotter.steps.progress_plan'`.

- [ ] **Step 3: Implement the deterministic planner**

Create `src/spotter/steps/progress_plan.py`:

```python
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from typing import Any, Protocol

from spotter.contracts import ProgressPlan, SessionRecord

PROGRESS_PLAN_PROVIDER_ENV = "SPOTTER_PROGRESS_PLAN_PROVIDER"


@dataclass(frozen=True)
class SessionFindings:
    exercise: str
    rep_count: int
    aggregate_metrics: dict[str, Any]
    issue_labels: list[str]


class ProgressPlanner(Protocol):
    def plan(self, history: list[SessionRecord], findings: SessionFindings) -> ProgressPlan:
        ...


def _metric(findings: SessionFindings, key: str, default: float = 0.0) -> float:
    value = findings.aggregate_metrics.get(key, default)
    return float(value) if isinstance(value, int | float) else default


def _trend(history: list[SessionRecord], key: str) -> float | None:
    if not history:
        return None
    first = history[0].aggregate_metrics.get(key)
    last = history[-1].aggregate_metrics.get(key)
    if not isinstance(first, int | float) or not isinstance(last, int | float):
        return None
    return float(last) - float(first)


def _top_issue(history: list[SessionRecord], findings: SessionFindings) -> str | None:
    counter: Counter[str] = Counter(findings.issue_labels)
    for record in history:
        counter.update(record.issue_counts.keys())
    if not counter:
        return None
    return counter.most_common(1)[0][0]


class FallbackProgressPlanner:
    def plan(self, history: list[SessionRecord], findings: SessionFindings) -> ProgressPlan:
        return build_fallback_plan(history, findings)


def build_fallback_plan(
    history: list[SessionRecord],
    findings: SessionFindings,
    failure_reason: str | None = None,
) -> ProgressPlan:
    rom = _metric(findings, "avg_rom_score", 0.7)
    stability = _metric(findings, "avg_stability_score", 0.7)
    symmetry = _metric(findings, "avg_symmetry_score", 0.7)

    top_issue = _top_issue(history, findings)
    if top_issue:
        focus = f"Keep working on `{top_issue}` this session."
    else:
        focus = "Keep the same controlled tempo and build consistency."

    targets: list[str] = []
    if rom < 0.8:
        targets.append("Aim for a little more range of motion on every rep.")
    if stability < 0.8:
        targets.append("Keep the core braced so the movement stays steady.")
    if symmetry < 0.8:
        targets.append("Check left and right sides stay even.")
    if not targets:
        targets.append("Repeat the set and hold this quality.")

    cues = ["Move at a slow, controlled tempo."]
    if top_issue:
        cues.append(f"Watch for `{top_issue}`, especially in the last reps.")

    rom_trend = _trend(history, "avg_rom_score")
    if rom_trend is not None and rom_trend > 0:
        encouragement = "Your range of motion is trending up. Keep it going."
    elif history:
        encouragement = "Progress is not always linear. Consistency is what counts."
    else:
        encouragement = "Good first session. The next one builds on it."

    notes = [f"Based on {len(history)} previous session(s)."]
    if failure_reason:
        notes.append(f"Deterministic plan used because: {failure_reason}.")
    return ProgressPlan(
        focus=focus,
        targets=targets,
        next_session_cues=cues,
        encouragement=encouragement,
        confidence_notes=notes,
    )


def get_progress_planner() -> ProgressPlanner:
    provider = os.getenv(PROGRESS_PLAN_PROVIDER_ENV, "fallback").strip().lower()
    if provider in {"fallback", "deterministic", ""}:
        return FallbackProgressPlanner()
    from spotter.steps.progress_plan_model import ModelProgressPlanner

    return ModelProgressPlanner()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 tests/test_progress_plan.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/spotter/steps/progress_plan.py tests/test_progress_plan.py
git commit -m "feat: add deterministic progress planner"
```

---

### Task 4: Model-backed progress planner and plan verifier

**Files:**
- Create: `src/spotter/steps/progress_plan_model.py`
- Modify: `src/spotter/steps/verifier.py`
- Test: `tests/test_progress_plan_model.py`

**Interfaces:**
- Consumes: `FallbackProgressPlanner`, `SessionFindings`, `build_fallback_plan` from `progress_plan`; `CoachSummaryModel` protocol from `spotter.slm.providers`.
- Produces: `build_progress_plan_prompt(history, findings) -> str`; `extract_plan_payload(text) -> dict`; `ModelProgressPlanner`; `verify_plan(plan, allowed_issues) -> Verification`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_progress_plan_model.py`:

```python
from __future__ import annotations

from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.contracts import ProgressPlan
from spotter.steps.progress_plan import SessionFindings
from spotter.steps.progress_plan_model import (
    ModelProgressPlanner,
    build_progress_plan_prompt,
    extract_plan_payload,
    verify_plan,
)


class _GoodModel:
    def generate_summary(self, prompt: str):
        del prompt
        from spotter.slm.providers import CoachSummaryGeneration

        return CoachSummaryGeneration(
            text=json.dumps(
                {
                    "focus": "Control `shallow_depth`.",
                    "targets": ["Reach depth on 6 of 8 reps."],
                    "next_session_cues": ["Slow the descent."],
                    "encouragement": "Steady progress.",
                    "confidence_notes": ["Based on 2 sessions."],
                }
            ),
            provider="test",
            model="test-model",
        )


class _BadModel:
    def generate_summary(self, prompt: str):
        del prompt
        raise RuntimeError("boom")


def _findings() -> SessionFindings:
    return SessionFindings(
        exercise="squat",
        rep_count=8,
        aggregate_metrics={"avg_rom_score": 0.7},
        issue_labels=["shallow_depth"],
    )


class ModelPlannerTests(unittest.TestCase):
    def test_prompt_includes_detected_issue_labels(self) -> None:
        prompt = build_progress_plan_prompt([], _findings())
        self.assertIn("shallow_depth", prompt)

    def test_extract_plan_payload_handles_code_fence(self) -> None:
        text = "```json\n{\"focus\":\"x\"}\n```"
        self.assertEqual(extract_plan_payload(text), {"focus": "x"})

    def test_model_planner_returns_plan(self) -> None:
        planner = ModelProgressPlanner(model=_GoodModel())
        plan = planner.plan([], _findings())
        self.assertIsInstance(plan, ProgressPlan)
        self.assertEqual(plan.targets, ["Reach depth on 6 of 8 reps."])

    def test_model_planner_falls_back_on_failure(self) -> None:
        planner = ModelProgressPlanner(model=_BadModel())
        plan = planner.plan([], _findings())
        self.assertTrue(plan.targets)

    def test_verifier_blocks_plan_issue_not_detected(self) -> None:
        plan = ProgressPlan(
            focus="Fix the `knee_valgus` immediately.",
            targets=[],
            next_session_cues=[],
            encouragement="ok",
            confidence_notes=[],
        )
        result = verify_plan(plan, allowed_issues={"shallow_depth"})
        self.assertFalse(result.passed)

    def test_verifier_blocks_medical_language(self) -> None:
        plan = ProgressPlan(
            focus="This could be an injury risk.",
            targets=[],
            next_session_cues=[],
            encouragement="ok",
            confidence_notes=[],
        )
        result = verify_plan(plan, allowed_issues=set())
        self.assertFalse(result.passed)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/test_progress_plan_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spotter.steps.progress_plan_model'`.

- [ ] **Step 3: Implement the model planner**

Create `src/spotter/steps/progress_plan_model.py`:

```python
from __future__ import annotations

import json
import re

from spotter.contracts import ProgressPlan, SessionRecord, Verification
from spotter.knowledge_cards import known_issue_labels
from spotter.steps.progress_plan import (
    SessionFindings,
    build_fallback_plan,
)

FORBIDDEN_PLAN_PATTERNS = (
    "diagnos",
    "injury",
    "tear",
    "impingement",
    "pathology",
    "fall risk",
    "risk of falling",
    "medical assessment",
)


def build_progress_plan_prompt(
    history: list[SessionRecord],
    findings: SessionFindings,
) -> str:
    history_lines = [
        (
            f"- {record.timestamp}: {record.rep_count} reps, "
            f"form {record.form_score:.2f}, issues {record.issue_counts}"
        )
        for record in history
    ] or ["- no previous sessions"]
    return (
        "You are a strength coach. Return ONLY a JSON object with keys "
        "focus, targets, next_session_cues, encouragement, confidence_notes.\n"
        f"Exercise: {findings.exercise}\n"
        f"Reps this session: {findings.rep_count}\n"
        f"Aggregate metrics: {json.dumps(findings.aggregate_metrics)}\n"
        f"Detected issues: {json.dumps(findings.issue_labels)}\n"
        "Previous sessions:\n"
        + "\n".join(history_lines)
        + "\nOnly mention issues from the detected issues list."
    )


def extract_plan_payload(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("Progress plan output must be a JSON object.")
    return payload


def _plan_from_payload(payload: dict) -> ProgressPlan:
    def as_list(value) -> list[str]:
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, list):
            return [str(item) for item in value]
        return []

    return ProgressPlan(
        focus=str(payload.get("focus", "")).strip() or "Keep building consistency.",
        targets=as_list(payload.get("targets")),
        next_session_cues=as_list(payload.get("next_session_cues")),
        encouragement=str(payload.get("encouragement", "")).strip() or "Keep going.",
        confidence_notes=as_list(payload.get("confidence_notes")),
    )


class ModelProgressPlanner:
    def __init__(self, model=None) -> None:
        if model is None:
            from spotter.slm.providers import get_coach_summary_model

            model = get_coach_summary_model()
        self.model = model

    def plan(self, history: list[SessionRecord], findings: SessionFindings) -> ProgressPlan:
        allowed_issues = allowed_issues_for(history, findings)
        try:
            generation = self.model.generate_summary(
                build_progress_plan_prompt(history, findings)
            )
            plan = _plan_from_payload(extract_plan_payload(generation.text))
        except Exception as exc:
            return build_fallback_plan(history, findings, failure_reason=f"model_failed:{exc}")
        verification = verify_plan(plan, allowed_issues=allowed_issues)
        if not verification.passed:
            return build_fallback_plan(
                history,
                findings,
                failure_reason="; ".join(verification.notes),
            )
        return plan


def allowed_issues_for(
    history: list[SessionRecord], findings: SessionFindings
) -> set[str]:
    allowed = set(findings.issue_labels)
    for record in history:
        allowed.update(record.issue_counts.keys())
    return allowed


def verify_plan(plan: ProgressPlan, allowed_issues: set[str]) -> Verification:
    text = " ".join(
        [
            plan.focus,
            *plan.targets,
            *plan.next_session_cues,
            plan.encouragement,
            *plan.confidence_notes,
        ]
    )
    lowered = text.lower()
    known = known_issue_labels()
    mentioned = {label for label in known if label in lowered}
    no_issue_outside_json = mentioned <= allowed_issues
    no_medical = all(pattern not in lowered for pattern in FORBIDDEN_PLAN_PATTERNS)
    checks = {
        "no_issue_outside_json": no_issue_outside_json,
        "no_medical_language": no_medical,
    }
    notes = []
    if not no_issue_outside_json:
        notes.append(f"Plan mentioned undetected issue labels: {sorted(mentioned - allowed_issues)}.")
    if not no_medical:
        notes.append("Plan used medical or injury language.")
    return Verification(passed=all(checks.values()), checks=checks, notes=notes)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 tests/test_progress_plan_model.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/spotter/steps/progress_plan_model.py tests/test_progress_plan_model.py
git commit -m "feat: add model-backed progress planner with verifier"
```

---

### Task 5: Session record builder

**Files:**
- Create: `src/spotter/steps/session_record.py`
- Test: `tests/test_session_record.py`

**Interfaces:**
- Consumes: `SessionRecord`, `ExerciseClassification`, `Reps`, `RepAnalysis`, `IssueMarkers`.
- Produces: `build_session_record(*, run_id, profile_key, timestamp, classification, reps, analysis, issues) -> SessionRecord`; `form_score_from_metrics(metrics) -> float`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_session_record.py`:

```python
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.contracts import (
    ExerciseClassification,
    IssueMarker,
    IssueMarkers,
    Rep,
    RepAnalysis,
    RepAnalysisItem,
    Reps,
)
from spotter.steps.session_record import build_session_record, form_score_from_metrics


def _classification() -> ExerciseClassification:
    return ExerciseClassification(
        exercise="squat", confidence=0.9, window_predictions=[], fallback_required=False
    )


def _reps() -> Reps:
    return Reps(
        exercise="squat",
        reps=[Rep(1, 0, 5, 10, 0.0, 0.2, 0.4), Rep(2, 11, 15, 20, 0.5, 0.7, 0.9)],
        partial_reps=[],
    )


def _analysis() -> RepAnalysis:
    return RepAnalysis(
        exercise="squat",
        items=[RepAnalysisItem(1, 0.4, 0.7, 0.8, 0.9, {}, [])],
        aggregate_metrics={
            "avg_rom_score": 0.7,
            "avg_stability_score": 0.8,
            "avg_symmetry_score": 0.9,
        },
    )


def _issues() -> IssueMarkers:
    return IssueMarkers(
        issues=[
            IssueMarker(2, "shallow_depth", 0.8, 11, 14, 0.5, 0.6, ["left_hip"], {}),
            IssueMarker(2, "shallow_depth", 0.6, 15, 18, 0.7, 0.8, ["right_hip"], {}),
        ]
    )


class SessionRecordTests(unittest.TestCase):
    def test_form_score_is_mean_of_available_metrics(self) -> None:
        self.assertAlmostEqual(form_score_from_metrics({"avg_rom_score": 0.6, "avg_stability_score": 0.9}), 0.75)

    def test_form_score_defaults_when_metrics_missing(self) -> None:
        self.assertEqual(form_score_from_metrics({}), 0.0)

    def test_build_record_counts_issues(self) -> None:
        record = build_session_record(
            run_id="run-1",
            profile_key="friend",
            timestamp="2026-10-03T09:00:00Z",
            classification=_classification(),
            reps=_reps(),
            analysis=_analysis(),
            issues=_issues(),
        )
        self.assertEqual(record.rep_count, 2)
        self.assertEqual(record.issue_counts, {"shallow_depth": 2})
        self.assertEqual(record.exercise, "squat")
        self.assertEqual(record.source_run_id, "run-1")
        self.assertAlmostEqual(record.form_score, 0.8)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/test_session_record.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the builder**

Create `src/spotter/steps/session_record.py`:

```python
from __future__ import annotations

from collections import Counter
from typing import Any

from spotter.contracts import (
    ExerciseClassification,
    IssueMarkers,
    RepAnalysis,
    Reps,
    SessionRecord,
)

METRIC_KEYS = ("avg_rom_score", "avg_stability_score", "avg_symmetry_score")


def form_score_from_metrics(metrics: dict[str, Any]) -> float:
    values = [
        float(metrics[key])
        for key in METRIC_KEYS
        if isinstance(metrics.get(key), int | float)
    ]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def build_session_record(
    *,
    run_id: str,
    profile_key: str,
    timestamp: str,
    classification: ExerciseClassification,
    reps: Reps,
    analysis: RepAnalysis,
    issues: IssueMarkers,
) -> SessionRecord:
    issue_counts = Counter(issue.issue for issue in issues.issues)
    return SessionRecord(
        session_id=run_id,
        timestamp=timestamp,
        profile_key=profile_key,
        exercise=classification.exercise,
        rep_count=len(reps.reps),
        aggregate_metrics=dict(analysis.aggregate_metrics),
        issue_counts=dict(issue_counts),
        form_score=form_score_from_metrics(analysis.aggregate_metrics),
        source_run_id=run_id,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 tests/test_session_record.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/spotter/steps/session_record.py tests/test_session_record.py
git commit -m "feat: build session records from pipeline findings"
```

---

### Task 6: Speech stage

**Files:**
- Create: `src/spotter/steps/speech.py`
- Test: `tests/test_speech.py`

**Interfaces:**
- Consumes: `CoachSummary`, `ProgressPlan`, `SpeechResult`.
- Produces: `build_speech_lines(summary, plan) -> list[str]`; `SpeechSynthesizer` protocol; `NullSpeechSynthesizer`; `get_speech_synthesizer() -> SpeechSynthesizer`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_speech.py`:

```python
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.contracts import CoachSummary, ProgressPlan
from spotter.steps.speech import (
    NullSpeechSynthesizer,
    build_speech_lines,
    get_speech_synthesizer,
)


def _summary() -> CoachSummary:
    return CoachSummary(
        summary="Solid set.",
        what_you_did=["2 squats."],
        what_looked_good=["Control."],
        what_changed_across_reps=[],
        valid_variation_vs_issue=[],
        top_fixes=["Go a little deeper."],
        next_session_plan=["Slow down."],
        confidence_notes=[],
    )


def _plan() -> ProgressPlan:
    return ProgressPlan(
        focus="Depth.",
        targets=["Reach depth."],
        next_session_cues=["Slow descent."],
        encouragement="Nice work.",
        confidence_notes=[],
    )


class SpeechTests(unittest.TestCase):
    def test_lines_include_summary_fix_and_plan_cue(self) -> None:
        lines = build_speech_lines(_summary(), _plan())
        self.assertIn("Solid set.", lines)
        self.assertIn("Go a little deeper.", lines)
        self.assertIn("Slow descent.", lines)

    def test_null_synthesizer_returns_no_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = NullSpeechSynthesizer().synthesize(
                ["hello"], language="en", out_dir=Path(temp_dir)
            )
            self.assertIsNone(result.audio_path)
            self.assertEqual(result.backend, "none")
            self.assertEqual(result.lines, ["hello"])

    def test_factory_defaults_to_null_backend(self) -> None:
        self.assertIsInstance(get_speech_synthesizer(), NullSpeechSynthesizer)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/test_speech.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the speech stage**

Create `src/spotter/steps/speech.py`:

```python
from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from spotter.contracts import CoachSummary, ProgressPlan, SpeechResult

TTS_BACKEND_ENV = "SPOTTER_TTS_BACKEND"
DEFAULT_LANGUAGE_ENV = "SPOTTER_TTS_LANGUAGE"


class SpeechSynthesizer(Protocol):
    def synthesize(
        self, lines: list[str], language: str, out_dir: Path
    ) -> SpeechResult:
        ...


def build_speech_lines(summary: CoachSummary, plan: ProgressPlan) -> list[str]:
    lines = [summary.summary]
    lines.extend(summary.what_looked_good[:1])
    lines.extend(summary.top_fixes[:1])
    lines.append(plan.focus)
    lines.extend(plan.next_session_cues[:1])
    lines.append(plan.encouragement)
    return [line for line in lines if line]


class NullSpeechSynthesizer:
    def synthesize(
        self, lines: list[str], language: str, out_dir: Path
    ) -> SpeechResult:
        del out_dir
        return SpeechResult(audio_path=None, language=language, lines=lines, backend="none")


def get_speech_synthesizer() -> SpeechSynthesizer:
    backend = os.getenv(TTS_BACKEND_ENV, "none").strip().lower()
    if backend == "open":
        from spotter.steps.speech_open import OpenTtsSynthesizer

        return OpenTtsSynthesizer()
    if backend == "elevenlabs":
        from spotter.steps.speech_elevenlabs import ElevenLabsSynthesizer

        return ElevenLabsSynthesizer()
    return NullSpeechSynthesizer()


def default_language() -> str:
    return os.getenv(DEFAULT_LANGUAGE_ENV, "en")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 tests/test_speech.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/spotter/steps/speech.py tests/test_speech.py
git commit -m "feat: add speech stage with null default"
```

---

### Task 7: Wire new stages into the pipeline

**Files:**
- Modify: `src/spotter/pipeline.py`
- Test: `tests/test_pipeline_companion.py`

**Interfaces:**
- Consumes: all prior tasks plus `datetime`/`os` already imported in `pipeline.py`.
- Produces: run output gains `progress_plan`, `session_record`, `speech`; artifacts `progress_plan.json`, `session_record.json`, `speech.json`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline_companion.py` (component-level; no heavy deps):

```python
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.contracts import ExerciseClassification, IssueMarkers, RepAnalysis, Reps
from spotter.steps.progress_plan import SessionFindings, build_fallback_plan
from spotter.steps.session_record import build_session_record


class PipelineCompanionTests(unittest.TestCase):
    def test_findings_to_plan_to_record_chain(self) -> None:
        findings = SessionFindings(
            exercise="squat",
            rep_count=8,
            aggregate_metrics={"avg_rom_score": 0.7, "avg_stability_score": 0.8},
            issue_labels=["shallow_depth"],
        )
        plan = build_fallback_plan([], findings)
        record = build_session_record(
            run_id="run-1",
            profile_key="friend",
            timestamp="2026-10-03T09:00:00Z",
            classification=ExerciseClassification("squat", 0.9, [], False),
            reps=Reps("squat", [], []),
            analysis=RepAnalysis("squat", [], {"avg_rom_score": 0.7}),
            issues=IssueMarkers([]),
        )
        self.assertTrue(plan.focus)
        self.assertEqual(record.source_run_id, "run-1")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `python3 tests/test_pipeline_companion.py -v`
Expected: PASS immediately (this guards the wiring contracts before editing the pipeline).

- [ ] **Step 3: Add imports**

In `src/spotter/pipeline.py`, extend the `from spotter.steps import (...)` block:

```python
    progress_plan as progress_plan_step,
    session_memory,
    session_record,
    speech,
```

- [ ] **Step 4: Insert the new stages**

In `src/spotter/pipeline.py`, after the existing `emit("coach", "done", ...)` block (just before `final_report = {`), insert:

```python
    profile_key = os.getenv("SPOTTER_PROFILE_KEY", "default")
    memory = session_memory.get_session_memory()
    history = memory.get_history(profile_key)

    emit("memory", "active", "Checking what we worked on in past sessions.")
    emit("memory", "done", f"I found {len(history)} previous session(s).", history_count=len(history))

    findings = progress_plan_step.SessionFindings(
        exercise=classification.exercise,
        rep_count=len(reps.reps),
        aggregate_metrics=analysis.aggregate_metrics,
        issue_labels=[issue.issue for issue in issues.issues],
    )
    planner = progress_plan_step.get_progress_planner()
    try:
        plan = planner.plan(history, findings)
    except Exception as exc:  # pragma: no cover - defensive
        plan = progress_plan_step.build_fallback_plan(
            history, findings, failure_reason=f"planner_error:{exc}"
        )
    write_artifact("progress_plan.json", plan)
    emit("plan", "done", "Here is your plan for next time.", focus=plan.focus)

    record = session_record.build_session_record(
        run_id=run_id,
        profile_key=profile_key,
        timestamp=datetime.now(timezone.utc).isoformat(),
        classification=classification,
        reps=reps,
        analysis=analysis,
        issues=issues,
    )
    memory.append(record)
    write_artifact("session_record.json", record)

    lines = speech.build_speech_lines(summary, plan)
    speech_result = speech.get_speech_synthesizer().synthesize(
        lines, language=speech.default_language(), out_dir=run_dir
    )
    write_artifact("speech.json", speech_result)
```

- [ ] **Step 5: Add to the final report**

In `src/spotter/pipeline.py`, inside the `final_report` dict, add after `"verification": to_dict(verification),`:

```python
        "progress_plan": to_dict(plan),
        "session_record": to_dict(record),
        "speech": to_dict(speech_result),
```

- [ ] **Step 6: Run tests**

Run: `python3 tests/test_pipeline_companion.py -v`
Expected: PASS.

Run: `python3 -m compileall -q src/spotter/pipeline.py`
Expected: no output (syntax OK).

- [ ] **Step 7: Commit**

```bash
git add src/spotter/pipeline.py tests/test_pipeline_companion.py
git commit -m "feat: wire memory, plan, and speech into the pipeline"
```

---

### Task 8: Progress-plan SFT dataset builder

**Files:**
- Create: `scripts/build_progress_plan_sft_dataset.py`
- Test: `tests/test_progress_plan_dataset.py`

**Interfaces:**
- Consumes: `SessionFindings`, `build_progress_plan_prompt`, `build_fallback_plan`.
- Produces: JSONL rows `{"prompt": str, "completion": str}` at `data/sft/progress_plan_train.jsonl` and `data/sft/progress_plan_eval.jsonl`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_progress_plan_dataset.py`:

```python
from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


class DatasetBuilderTests(unittest.TestCase):
    def test_builder_writes_valid_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir)
            subprocess.run(
                [
                    sys.executable,
                    str(REPO / "scripts" / "build_progress_plan_sft_dataset.py"),
                    "--out-dir",
                    str(out),
                    "--train-count",
                    "5",
                    "--eval-count",
                    "2",
                    "--seed",
                    "7",
                ],
                check=True,
                cwd=REPO,
            )
            train_rows = [
                json.loads(line)
                for line in (out / "progress_plan_train.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(train_rows), 5)
            self.assertIn("prompt", train_rows[0])
            self.assertIn("completion", train_rows[0])
            json.loads(train_rows[0]["completion"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/test_progress_plan_dataset.py -v`
Expected: FAIL (script missing, `check=True` raises).

- [ ] **Step 3: Implement the builder**

Create `scripts/build_progress_plan_sft_dataset.py`:

```python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.contracts import SessionRecord
from spotter.steps.progress_plan import SessionFindings, build_fallback_plan
from spotter.steps.progress_plan_model import build_progress_plan_prompt

METRIC_KEYS = ("avg_rom_score", "avg_stability_score", "avg_symmetry_score")


def _random_findings(rng: random.Random) -> SessionFindings:
    metrics = {key: round(rng.uniform(0.4, 0.98), 2) for key in METRIC_KEYS}
    labels = rng.sample(
        ["shallow_depth", "knee_valgus", "hip_sag", "incomplete_lockout", "torso_lean"],
        k=rng.randint(0, 3),
    )
    return SessionFindings(
        exercise=rng.choice(["squat", "push_up", "shoulder_press"]),
        rep_count=rng.randint(3, 12),
        aggregate_metrics=metrics,
        issue_labels=list(labels),
    )


def _random_history(rng: random.Random, findings: SessionFindings) -> list[SessionRecord]:
    records = []
    for index in range(rng.randint(0, 6)):
        records.append(
            SessionRecord(
                session_id=f"synthetic-{index}",
                timestamp=f"2026-09-{index + 1:02d}T09:00:00Z",
                profile_key="friend",
                exercise=findings.exercise,
                rep_count=findings.rep_count,
                aggregate_metrics={
                    key: round(rng.uniform(0.4, 0.98), 2) for key in METRIC_KEYS
                },
                issue_counts={label: rng.randint(0, 3) for label in findings.issue_labels},
                form_score=round(rng.uniform(0.4, 0.98), 2),
                source_run_id=f"synthetic-{index}",
            )
        )
    return records


def build_rows(count: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for _ in range(count):
        findings = _random_findings(rng)
        history = _random_history(rng, findings)
        plan = build_fallback_plan(history, findings)
        completion = json.dumps(
            {
                "focus": plan.focus,
                "targets": plan.targets,
                "next_session_cues": plan.next_session_cues,
                "encouragement": plan.encouragement,
                "confidence_notes": plan.confidence_notes,
            }
        )
        rows.append(
            {
                "prompt": build_progress_plan_prompt(history, findings),
                "completion": completion,
                "meta": {
                    "exercise": findings.exercise,
                    "issue_labels": list(findings.issue_labels),
                    "history_issue_labels": sorted(
                        {label for record in history for label in record.issue_counts}
                    ),
                },
            }
        )
    return rows


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/sft")
    parser.add_argument("--train-count", type=int, default=400)
    parser.add_argument("--eval-count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    _write(out_dir / "progress_plan_train.jsonl", build_rows(args.train_count, args.seed))
    _write(out_dir / "progress_plan_eval.jsonl", build_rows(args.eval_count, args.seed + 1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 tests/test_progress_plan_dataset.py -v`
Expected: PASS.

- [ ] **Step 5: Generate the real dataset**

Run: `python3 scripts/build_progress_plan_sft_dataset.py`
Expected: writes `data/sft/progress_plan_train.jsonl` and `data/sft/progress_plan_eval.jsonl`.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_progress_plan_sft_dataset.py tests/test_progress_plan_dataset.py data/sft/progress_plan_train.jsonl data/sft/progress_plan_eval.jsonl
git commit -m "feat: add progress-plan sft dataset builder"
```

---

### Task 9: Tinker LoRA fine-tune and evaluation

**Files:**
- Create: `configs/progress_plan_lora.default.json`
- Create: `scripts/train_progress_plan_tinker.py`
- Create: `docs/60-progress-plan-training-report.md`

**Interfaces:**
- Consumes: `data/sft/progress_plan_train.jsonl`, `data/sft/progress_plan_eval.jsonl`.
- Produces: a fine-tuned model id and a results table comparing fine-tuned vs base.

- [ ] **Step 1: Add the LoRA config**

Create `configs/progress_plan_lora.default.json`:

```json
{
  "base_model": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
  "train_file": "data/sft/progress_plan_train.jsonl",
  "eval_file": "data/sft/progress_plan_eval.jsonl",
  "output_dir": "models/progress_plan_lora",
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "learning_rate": 0.0002,
  "num_train_epochs": 2,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 8,
  "max_seq_length": 2048
}
```

- [ ] **Step 2: Add the training script**

Create `scripts/train_progress_plan_tinker.py`. It loads the JSONL, calls the Tinker client, and writes a run report. Confirm the current Tinker SDK and model list before running; this script reads `TINKER_API_KEY` from the environment and never hardcodes it.

```python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/progress_plan_lora.default.json")
    parser.add_argument("--model", default=None, help="Override base model id")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.model:
        config["base_model"] = args.model

    if not os.getenv("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is required and must be provided by the user.")

    import tinker  # provided by the Tinker SDK

    train_rows = _load_jsonl(Path(config["train_file"]))
    client = tinker.Client()
    training = client.create_lora_training(
        base_model=config["base_model"],
        train_data=train_rows,
        lora_rank=config["lora_r"],
        learning_rate=config["learning_rate"],
        epochs=config["num_train_epochs"],
    )
    model_id = training.wait()
    print(json.dumps({"fine_tuned_model": model_id}))


if __name__ == "__main__":
    main()
```

Note: the exact Tinker client method names depend on the SDK version. Verify against the Tinker docs and adjust `create_lora_training`/`wait` to the real API before running. The deliverable is the trained model id plus the report, not the wrapper.

- [ ] **Step 3: Evaluate fine-tuned vs base**

Create `scripts/evaluate_progress_plan.py` (same shape as Task 8's builder test pattern):

- For each eval row, generate with base and fine-tuned models.
- Compute: `schema_valid_rate` (json parses to the five keys), `grounding_rate` (mentioned labels subset of findings labels), `issue_f1` against the fallback plan's issue references, `latency_p50_ms`, `latency_p95_ms`.
- Write `reports/progress_plan_eval.json` and paste the table into `docs/60-progress-plan-training-report.md`.

- [ ] **Step 4: Write the report**

Create `docs/60-progress-plan-training-report.md` with: base model, dataset sizes, hyperparameters, the metrics table, and a short note on where the fine-tune won or lost. This is the open-vs-base artifact used in the write-up.

- [ ] **Step 5: Commit**

```bash
git add configs/progress_plan_lora.default.json scripts/train_progress_plan_tinker.py scripts/evaluate_progress_plan.py docs/60-progress-plan-training-report.md
git commit -m "feat: add progress-plan tinker training and eval"
```

---

### Task 10: Render progress dashboard

**Files:**
- Create: `render/app.py`
- Create: `render/index.html`
- Create: `render/requirements.txt`
- Create: `render/render.yaml`

**Interfaces:**
- Consumes: the SQLite history store (`runs/history.db`), read-only.
- Produces: `GET /` returns the dashboard page; `GET /api/history?profile_key=...` returns JSON records.

- [ ] **Step 1: Implement the service**

Create `render/app.py`:

```python
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
```

Create `render/index.html` with a minimal chart of `form_score` over sessions and the recurring issue labels (plain fetch + inline JS, no build step):

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Spotter Companion Progress</title>
  </head>
  <body>
    <h1>Progress</h1>
    <p>Sessions per week, form score trend, and recurring issues.</p>
    <table id="sessions">
      <thead><tr><th>Date</th><th>Exercise</th><th>Reps</th><th>Form</th></tr></thead>
      <tbody></tbody>
    </table>
    <script>
      fetch("/api/history?profile_key=default")
        .then((r) => r.json())
        .then((rows) => {
          const body = document.querySelector("#sessions tbody");
          rows.forEach((row) => {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td>${row.timestamp}</td><td>${row.exercise}</td>` +
              `<td>${row.rep_count}</td><td>${row.form_score}</td>`;
            body.appendChild(tr);
          });
        });
    </script>
  </body>
</html>
```

Create `render/requirements.txt`:

```text
fastapi>=0.136.3
uvicorn>=0.30.0
```

Create `render/render.yaml`:

```yaml
services:
  - type: web
    name: spotter-companion-progress
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: SPOTTER_HISTORY_DB
        value: runs/history.db
```

- [ ] **Step 2: Run locally to verify**

Run: `python3 -c "import ast; ast.parse(open('render/app.py').read())"`
Expected: no output (syntax OK).

Run (optional, if fastapi/uvicorn available): `cd render && uvicorn app:app --port 8000` and open the preview. Otherwise note the dependency gap.

- [ ] **Step 3: Commit**

```bash
git add render/
git commit -m "feat: add render progress dashboard"
```

---

### Task 11: UI affordances, safety line, and docs

**Files:**
- Modify: `web/index.html`, `web/report.js`
- Modify: `README.md`
- Modify: `docs/01-docs-index.md`

**Interfaces:**
- Consumes: the new `progress_plan.json` and `speech.json` artifacts served by `app.py`'s `_artifact_urls`.
- Produces: a "Play feedback" control and large-text toggle; a visible safety line; docs for the new workflow.

- [ ] **Step 1: Add the safety line**

In `README.md`, immediately under the existing "Spotter is not a medical device." paragraph, add:

```markdown
This build is a practice companion. It is not a medical or fall-risk assessment tool.
```

- [ ] **Step 2: Add artifact links for the new outputs**

In `app.py`'s `_artifact_urls`, add `session_record.json`, `progress_plan.json`, and `speech.json` to `artifact_files` so the report can link them.

- [ ] **Step 3: Add the UI affordances**

In `web/index.html`, add a large-text toggle and a play-feedback button in the report container. In `web/report.js`, when the report payload includes speech with an `audio_path`, wire the button to the artifact URL; when `audio_path` is null, disable it and show "Voice off". Apply a root class for large text that increases `font-size` on the report.

- [ ] **Step 4: Update the docs index**

In `docs/01-docs-index.md`, add a "Companion" section listing `60-progress-plan-training-report.md` and the spec path.

- [ ] **Step 5: Verify**

Run: `python3 -m compileall -q app.py src/spotter`
Expected: no output.

Run: `python3 tests/test_companion_contracts.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add web/ README.md docs/01-docs-index.md app.py
git commit -m "feat: add companion ui affordances and safety docs"
```

---

## Self-Review

- Spec coverage: contracts (Task 1), memory (Task 2), progress_plan + fallback (Tasks 3-4), session record (Task 5), speech (Task 6), pipeline wiring (Task 7), dataset (Task 8), Tinker experiment (Task 9), Render dashboard (Task 10), safety line + UI + docs (Task 11). All spec sections map to a task.
- Placeholder scan: the only intentionally deferred detail is the exact Tinker SDK method names in Task 9, which is called out explicitly because the SDK is external and versioned; the training/eval deliverables and report are concrete.
- Type consistency: `SessionRecord`, `ProgressPlan`, `SpeechResult`, `SessionFindings`, `build_fallback_plan`, `verify_plan`, `build_speech_lines`, `build_session_record`, and `get_session_memory` are used with the same signatures across tasks.
