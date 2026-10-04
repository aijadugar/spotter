from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


RELEVANT_KEYWORDS = (
    "exercise",
    "workout",
    "gym",
    "fitness",
    "squat",
    "push up",
    "push-up",
    "press",
    "training",
    "rep",
    "form",
    "technique",
    "strength",
    "core",
    "muscle",
)
BLOCKED_KEYWORDS = (
    "toxin",
    "pollution",
    "work-life balance",
    "social well-being",
    "financial stress",
    "medical diagnosis",
    "pathology",
    "insomnia",
    "anxiety",
    "depression",
    "avoid injury",
    "prevent injury",
)


def _normalized_text(*parts: str) -> str:
    return " ".join(part.strip().lower() for part in parts if part and part.strip())


def is_style_relevant(instruction: str, output: str) -> bool:
    text = _normalized_text(instruction, output)
    if not any(keyword in text for keyword in RELEVANT_KEYWORDS):
        return False
    if any(keyword in text for keyword in BLOCKED_KEYWORDS):
        return False
    return True


def _style_record(instruction: str, output: str, source_dataset: str) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a concise, practical fitness coach. Keep a supportive tone and avoid medical claims.",
            },
            {"role": "user", "content": instruction.strip()},
            {"role": "assistant", "content": output.strip()},
        ],
        "metadata": {
            "source_dataset": source_dataset,
            "style_only": True,
        },
    }


def load_haz_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {"instruction": row["Human"], "output": row["Assistant"]}
            for row in reader
            if row.get("Human") and row.get("Assistant")
        ]


def load_chibbss_rows(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array")
    rows: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        instruction = item.get("instruction")
        output = item.get("output")
        if isinstance(instruction, str) and isinstance(output, str):
            rows.append({"instruction": instruction, "output": output})
    return rows


def convert_rows_to_style_corpus(
    rows: list[dict[str, str]],
    *,
    source_dataset: str,
) -> list[dict[str, Any]]:
    corpus: list[dict[str, Any]] = []
    for row in rows:
        instruction = row.get("instruction", "")
        output = row.get("output", "")
        if not is_style_relevant(instruction, output):
            continue
        corpus.append(_style_record(instruction, output, source_dataset))
    return corpus


def write_style_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

