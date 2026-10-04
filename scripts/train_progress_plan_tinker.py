from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"Dataset not found: {path}. Run the dataset builder first.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/progress_plan_lora.default.json")
    parser.add_argument("--model", default=None, help="Override base model id")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and config only")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.model:
        config["base_model"] = args.model

    train_rows = _load_jsonl(Path(config["train_file"]))
    eval_rows = _load_jsonl(Path(config["eval_file"]))
    print(
        json.dumps(
            {
                "base_model": config["base_model"],
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "lora_r": config["lora_r"],
            }
        )
    )
    if args.dry_run:
        return

    if not os.getenv("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is required and must be provided by the user.")

    try:
        import tinker  # provided by the Tinker SDK
    except ImportError as exc:
        raise SystemExit(
            "The Tinker SDK is not installed. Install it and confirm the current API names "
            "before training; the wrapper below is intentionally minimal."
        ) from exc

    client = tinker.Client()
    training = client.create_lora_training(
        base_model=config["base_model"],
        train_data=[
            {"prompt": row["prompt"], "completion": row["completion"]}
            for row in train_rows
        ],
        lora_rank=config["lora_r"],
        learning_rate=config["learning_rate"],
        epochs=config["num_train_epochs"],
    )
    model_id = training.wait()
    print(json.dumps({"fine_tuned_model": model_id}))


if __name__ == "__main__":
    main()
