from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from spotter.steps.progress_plan import SessionFindings
from spotter.steps.progress_plan_model import build_progress_plan_prompt
from spotter.contracts import SessionRecord
from build_progress_plan_sft_dataset import build_rows, _write

# Pricing from Tinker "Models & Pricing" page for Qwen3.5-4B:
# Training: $0.737 per 1M tokens
TRAIN_PRICE_PER_M_TOKENS = 0.737
COST_LIMIT_USD = 3.00


def _count_tokens(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text))


def _estimate_cost(train_rows: list[dict], eval_rows: list[dict], enc: tiktoken.Encoding, epochs: int) -> float:
    total_tokens = 0
    for row in train_rows:
        total_tokens += _count_tokens(row["prompt"], enc)
        total_tokens += _count_tokens(row["completion"], enc)
    # Training sees each token once per epoch
    total_tokens *= epochs
    return (total_tokens / 1_000_000) * TRAIN_PRICE_PER_M_TOKENS


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"Dataset not found: {path}. Run the dataset builder first.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _build_datasets(config: dict) -> tuple[Path, Path]:
    """Build train/eval datasets using the deterministic planner."""
    out_dir = Path("data/sft")
    train_path = out_dir / "progress_plan_train.jsonl"
    eval_path = out_dir / "progress_plan_eval.jsonl"

    train_count = 500
    eval_count = 88  # ~15% held out

    _write(train_path, build_rows(train_count, seed=42))
    _write(eval_path, build_rows(eval_count, seed=43))

    return train_path, eval_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/progress_plan_lora.default.json")
    parser.add_argument("--model", default=None, help="Override base model id")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and config only (default without TINKER_API_KEY)")
    parser.add_argument("--build-dataset", action="store_true", help="Rebuild datasets from deterministic planner")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.model:
        config["base_model"] = args.model

    # Default to dry-run if no API key
    has_key = bool(os.getenv("TINKER_API_KEY"))
    if not has_key and not args.dry_run:
        args.dry_run = True
        print("No TINKER_API_KEY set — running in dry-run mode")

    # Build datasets if requested or if missing
    train_path = Path(config["train_file"])
    eval_path = Path(config["eval_file"])
    if args.build_dataset or not train_path.exists() or not eval_path.exists():
        print("Building datasets from deterministic planner...")
        train_path, eval_path = _build_datasets(config)
        config["train_file"] = str(train_path)
        config["eval_file"] = str(eval_path)

    train_rows = _load_jsonl(train_path)
    eval_rows = _load_jsonl(eval_path)

    enc = tiktoken.get_encoding("o200k_base")
    estimated_cost = _estimate_cost(train_rows, eval_rows, enc, config["num_train_epochs"])

    print(
        json.dumps(
            {
                "base_model": config["base_model"],
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "lora_r": config["lora_r"],
                "estimated_tokens_millions": round(sum(
                    _count_tokens(r["prompt"], enc) + _count_tokens(r["completion"], enc) for r in train_rows
                ) * config["num_train_epochs"] / 1_000_000, 2),
                "estimated_cost_usd": round(estimated_cost, 4),
                "cost_limit_usd": COST_LIMIT_USD,
                "dry_run": args.dry_run,
            }
        )
    )

    if args.dry_run:
        print(f"Pricing source: Tinker Models & Pricing page — Qwen3.5-4B training: ${TRAIN_PRICE_PER_M_TOKENS} per 1M tokens")
        if estimated_cost > COST_LIMIT_USD:
            print(f"⚠ Estimated cost ${estimated_cost:.4f} exceeds limit ${COST_LIMIT_USD:.2f}")
        return 0

    if estimated_cost > COST_LIMIT_USD:
        raise SystemExit(f"Estimated cost ${estimated_cost:.4f} exceeds limit ${COST_LIMIT_USD:.2f}. Aborting.")

    # Real training run
    try:
        import tinker
        from tinker import types
    except ImportError as exc:
        raise SystemExit(
            "The Tinker SDK is not installed. Install it with: pip install tinker"
        ) from exc

    print("Connecting to Tinker...")
    service_client = tinker.ServiceClient()

    print(f"Creating LoRA training client for {config['base_model']} (rank={config['lora_r']})...")
    training_client = service_client.create_lora_training_client(
        base_model=config["base_model"],
        rank=config["lora_r"],
    )

    tokenizer = training_client.get_tokenizer()

    # Convert dataset rows to Datum objects
    print("Preparing training data...")
    data = []
    for row in train_rows:
        prompt_tokens = tokenizer.encode(row["prompt"])
        completion_tokens = tokenizer.encode(row["completion"])
        full_sequence = prompt_tokens + completion_tokens
        n_prefix = len(prompt_tokens) - 1

        datum = types.Datum(
            model_input=types.ModelInput.from_ints(tokens=full_sequence[:-1]),
            loss_fn_inputs=dict(
                weights=[0.0] * n_prefix + [1.0] * len(completion_tokens),
                target_tokens=full_sequence[1:],
            ),
        )
        data.append(datum)

    print(f"Training on {len(data)} examples for {config['num_train_epochs']} epochs...")
    for epoch in range(config["num_train_epochs"]):
        print(f"Epoch {epoch + 1}/{config['num_train_epochs']}")
        # In a real implementation, you'd batch the data
        # For this script, we do one forward_backward + optim_step per epoch
        fwdbwd_future = training_client.forward_backward_async(data=data, loss_fn="cross_entropy")
        fwdbwd_result = fwdbwd_future.result_async()
        print(f"  Loss: {fwdbwd_result.loss:.4f}")

        optim_future = training_client.optim_step_async(
            types.AdamParams(learning_rate=config["learning_rate"])
        )
        optim_future.result_async()

    # Save weights and get sampling client for evaluation
    print("Saving weights and creating sampling client...")
    sampling_client = training_client.save_weights_and_get_sampling_client(name="progress-plan-final")

    model_id = sampling_client.base_model  # The fine-tuned model identifier
    print(json.dumps({"fine_tuned_model": model_id}))
    return 0


if __name__ == "__main__":
    sys.exit(main())