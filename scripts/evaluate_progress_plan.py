from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.knowledge_cards import known_issue_labels
from spotter.steps.progress_plan_model import extract_plan_payload

REQUIRED_KEYS = {"focus", "targets", "next_session_cues", "encouragement", "confidence_notes"}


def schema_valid(text: str) -> bool:
    try:
        payload = extract_plan_payload(text)
    except Exception:
        return False
    return REQUIRED_KEYS.issubset(payload.keys())


def mentioned_issue_labels(text: str) -> set[str]:
    lowered = text.lower()
    return {label for label in known_issue_labels() if label in lowered}


def grounded(text: str, allowed: set[str]) -> bool:
    return mentioned_issue_labels(text) <= allowed


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def evaluate_rows(rows: list[dict], generate, *, label: str) -> dict:
    latencies: list[float] = []
    schema_hits = 0
    grounding_hits = 0
    for row in rows:
        meta = row.get("meta", {})
        allowed = set(meta.get("issue_labels", [])) | set(meta.get("history_issue_labels", []))
        start = time.perf_counter()
        text = generate(row["prompt"])
        latencies.append((time.perf_counter() - start) * 1000.0)
        if schema_valid(text):
            schema_hits += 1
        if grounded(text, allowed):
            grounding_hits += 1
    total = len(rows) or 1
    return {
        "label": label,
        "n": len(rows),
        "schema_valid_rate": round(schema_hits / total, 4),
        "grounding_rate": round(grounding_hits / total, 4),
        "latency_p50_ms": round(percentile(latencies, 0.5), 1),
        "latency_p95_ms": round(percentile(latencies, 0.95), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default="data/sft/progress_plan_eval.jsonl")
    parser.add_argument("--label", default="base")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Score the reference completions instead of a live model.",
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.eval_file).read_text(encoding="utf-8").splitlines()
        if line
    ]

    if args.dry_run:
        reference = {row["prompt"]: row["completion"] for row in rows}
        generate = lambda prompt: reference[prompt]
    else:
        from spotter.slm.providers import get_coach_summary_model

        model = get_coach_summary_model()
        if model is None:
            raise SystemExit("No model configured. Set SPOTTER_COACH_SUMMARY_PROVIDER/MODEL.")
        generate = lambda prompt: model.generate_summary(prompt).text

    report = evaluate_rows(rows, generate, label=args.label)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"progress_plan_eval_{args.label}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
