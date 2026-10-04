from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


FIELD_CANDIDATES = {
    "name": ("name", "exercise", "title", "Exercise_Name"),
    "aliases": ("aliases", "alias", "alternative_names", "Alternative Names"),
    "category": ("category", "Category", "exercise_type", "type"),
    "equipment": ("equipment", "Equipment", "equipments"),
    "primary_muscles": (
        "primary_muscles",
        "primaryMuscles",
        "target_muscle",
        "Target Muscle",
        "muscles",
    ),
    "secondary_muscles": (
        "secondary_muscles",
        "secondaryMuscles",
        "synergist_muscles",
        "Secondary Muscles",
    ),
    "instructions": ("instructions", "Instructions", "steps", "how_to"),
    "description": ("description", "Description", "summary", "overview"),
}

KNOWN_EXERCISE_LABELS = {
    "push up": "push_up",
    "push-up": "push_up",
    "pushup": "push_up",
    "shoulder press": "shoulder_press",
    "overhead press": "shoulder_press",
    "bodyweight squat": "squat",
    "air squat": "squat",
}


def _first_present(row: dict[str, Any], field_names: tuple[str, ...]) -> Any:
    for field_name in field_names:
        if field_name in row and row[field_name] not in (None, ""):
            return row[field_name]
    return None


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_parts = re.split(r"[;\n]|(?:,\s*)", value)
        return [part.strip() for part in raw_parts if part.strip()]
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                values.append(item.strip())
        return values
    return []


def _as_instruction_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        lines = [line.strip("-• ").strip() for line in value.splitlines() if line.strip()]
        if len(lines) > 1:
            return lines
        sentence_parts = [part.strip() for part in value.split(".") if part.strip()]
        return [part if part.endswith(".") else f"{part}." for part in sentence_parts]
    return []


def _slugify(value: str) -> str:
    lowered = value.strip().lower()
    lowered = KNOWN_EXERCISE_LABELS.get(lowered, lowered)
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    return lowered.strip("_")


def _display_name(value: str) -> str:
    words = re.split(r"[_\s-]+", value.strip())
    return " ".join(word.capitalize() for word in words if word)


def _dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _normalize_exercise_label(name: str, aliases: list[str]) -> str:
    alias_candidates = [name, *aliases]
    for candidate in alias_candidates:
        normalized = KNOWN_EXERCISE_LABELS.get(candidate.strip().lower())
        if normalized:
            return normalized
    return _slugify(name)


def _build_summary(
    *,
    exercise_name: str,
    category: str | None,
    equipment: list[str],
    primary_muscles: list[str],
    description: str | None,
) -> str:
    parts = [f"A {exercise_name} summary should stay grounded in structured rep evidence."]
    if category:
        parts.append(f"This movement is commonly categorized as {category.lower()}.")
    if equipment:
        parts.append(f"Typical equipment context: {', '.join(equipment[:3])}.")
    if primary_muscles:
        parts.append(
            f"Common target-muscle context includes {', '.join(primary_muscles[:3])}."
        )
    if description:
        parts.append(description.strip())
    return " ".join(parts)


def normalize_dataset_rows(
    rows: list[dict[str, Any]],
    *,
    source_dataset: str,
) -> list[dict[str, Any]]:
    exercises: list[dict[str, Any]] = []
    for row in rows:
        name_value = _first_present(row, FIELD_CANDIDATES["name"])
        if not isinstance(name_value, str) or not name_value.strip():
            continue

        aliases = _dedupe_keep_order(
            _as_string_list(_first_present(row, FIELD_CANDIDATES["aliases"]))
        )
        exercise_label = _normalize_exercise_label(name_value, aliases)
        title = _display_name(name_value)
        category = _first_present(row, FIELD_CANDIDATES["category"])
        equipment = _dedupe_keep_order(
            _as_string_list(_first_present(row, FIELD_CANDIDATES["equipment"]))
        )
        primary_muscles = _dedupe_keep_order(
            _as_string_list(_first_present(row, FIELD_CANDIDATES["primary_muscles"]))
        )
        secondary_muscles = _dedupe_keep_order(
            _as_string_list(_first_present(row, FIELD_CANDIDATES["secondary_muscles"]))
        )
        instructions = _dedupe_keep_order(
            _as_instruction_list(_first_present(row, FIELD_CANDIDATES["instructions"]))
        )
        description_value = _first_present(row, FIELD_CANDIDATES["description"])
        description = description_value.strip() if isinstance(description_value, str) else None

        exercises.append(
            {
                "source_dataset": source_dataset,
                "exercise_label": exercise_label,
                "title": title,
                "aliases": aliases,
                "category": category.strip() if isinstance(category, str) else None,
                "equipment": equipment,
                "primary_muscles": primary_muscles,
                "secondary_muscles": secondary_muscles,
                "instructions": instructions,
                "description": description,
            }
        )

    exercises.sort(key=lambda item: item["exercise_label"])
    return exercises


def transform_dataset_rows(
    rows: list[dict[str, Any]],
    *,
    source_dataset: str,
) -> dict[str, Any]:
    exercises = normalize_dataset_rows(rows, source_dataset=source_dataset)
    cards: list[dict[str, Any]] = []
    for exercise in exercises:
        exercise_label = str(exercise["exercise_label"])
        title = str(exercise["title"])
        aliases = [str(item) for item in exercise["aliases"]]
        category = exercise["category"]
        equipment = [str(item) for item in exercise["equipment"]]
        primary_muscles = [str(item) for item in exercise["primary_muscles"]]
        secondary_muscles = [str(item) for item in exercise["secondary_muscles"]]
        instructions = [str(item) for item in exercise["instructions"]]
        description = exercise["description"] if isinstance(exercise["description"], str) else None

        labels = _dedupe_keep_order([exercise_label, *(_slugify(alias) for alias in aliases)])

        evidence_rules = [
            "Use structured rep analysis and issue markers instead of inferring directly from raw video.",
            "Treat valid detected variations as context unless issue markers show a separate problem.",
        ]
        if equipment:
            evidence_rules.append(
                f"Use {', '.join(equipment[:3])} only as exercise context, not as a claim about what the user must change."
            )

        coaching_points = instructions[:3]
        if not coaching_points:
            coaching_points = [
                "Keep the summary focused on repeatable setup and execution cues supported by the evidence."
            ]

        allowed_interpretations = []
        if primary_muscles:
            muscle_line = f"Common target muscles: {', '.join(primary_muscles[:3])}."
            if secondary_muscles:
                muscle_line += f" Secondary support may include {', '.join(secondary_muscles[:3])}."
            allowed_interpretations.append(muscle_line)
        if equipment:
            allowed_interpretations.append(
                f"Common equipment context: {', '.join(equipment[:3])}."
            )

        cards.append(
            {
                "card_id": f"exercise:{exercise_label}",
                "card_type": "exercise",
                "labels": labels,
                "title": title,
                "summary": _build_summary(
                    exercise_name=title,
                    category=category if isinstance(category, str) else None,
                    equipment=equipment,
                    primary_muscles=primary_muscles,
                    description=description,
                ),
                "evidence_rules": evidence_rules,
                "coaching_points": coaching_points,
                "allowed_interpretations": allowed_interpretations,
                "related_cards": [],
            }
        )

    cards.sort(key=lambda card: card["card_id"])
    return {
        "source_dataset": source_dataset,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "normalized_exercise_count": len(exercises),
        "card_count": len(cards),
        "cards": cards,
    }


def load_dataset_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(payload)
        return rows

    payload = json.loads(text)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows_value = payload.get("rows") or payload.get("data") or payload.get("train")
        if not isinstance(rows_value, list):
            raise ValueError(
                f"{path} must contain a top-level list or a `rows`/`data`/`train` list"
            )
        rows = rows_value
    else:
        raise ValueError(f"{path} must be a JSON array or object")

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {index} must be a JSON object")
    return rows


def write_card_pack(
    *,
    input_path: Path,
    output_path: Path,
    source_dataset: str,
) -> dict[str, Any]:
    rows = load_dataset_rows(input_path)
    pack = transform_dataset_rows(rows, source_dataset=source_dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
    return pack


def write_normalized_exercises(
    *,
    input_path: Path,
    output_path: Path,
    source_dataset: str,
) -> dict[str, Any]:
    rows = load_dataset_rows(input_path)
    exercises = normalize_dataset_rows(rows, source_dataset=source_dataset)
    payload = {
        "source_dataset": source_dataset,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exercise_count": len(exercises),
        "exercises": exercises,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an exercise dataset export into a Spotter knowledge-card pack."
    )
    parser.add_argument("--input", required=True, help="Path to a JSON or JSONL dataset export.")
    parser.add_argument(
        "--output",
        required=True,
        help="Destination path for the generated Spotter card-pack JSON.",
    )
    parser.add_argument(
        "--normalized-output",
        help="Optional destination path for a normalized exercise-schema JSON export.",
    )
    parser.add_argument(
        "--source-dataset",
        default="unknown_dataset",
        help="Dataset identifier to record in the generated pack metadata.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.normalized_output:
        write_normalized_exercises(
            input_path=Path(args.input),
            output_path=Path(args.normalized_output),
            source_dataset=args.source_dataset,
        )
    pack = write_card_pack(
        input_path=Path(args.input),
        output_path=Path(args.output),
        source_dataset=args.source_dataset,
    )
    print(
        f"Wrote {pack['card_count']} cards from {args.source_dataset} to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
