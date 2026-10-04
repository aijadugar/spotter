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
