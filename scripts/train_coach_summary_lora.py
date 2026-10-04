from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


DEFAULT_CONFIG = {
    "base_model": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    "train_file": str(ROOT / "data/sft/coach_summary_train.jsonl"),
    "eval_file": str(ROOT / "data/sft/coach_summary_eval.jsonl"),
    "output_dir": str(ROOT / "models/coach_summary_lora"),
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "learning_rate": 0.0002,
    "num_train_epochs": 2,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 2048,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or run LoRA fine-tuning for Spotter coach-summary generation."
    )
    parser.add_argument(
        "--config-output",
        default=str(ROOT / "configs/coach_summary_lora.default.json"),
        help="Where to write the resolved training config.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Attempt to run training immediately if the required dependencies are installed.",
    )
    return parser


def _write_config(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_training(config: dict[str, object]) -> None:
    try:
        from datasets import load_dataset
        from trl import SFTConfig, SFTTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional training stack
        raise RuntimeError(
            "Training dependencies are missing. Install `datasets`, `transformers`, `peft`, and `trl` to run LoRA training."
        ) from exc

    train_dataset = load_dataset("json", data_files=str(config["train_file"]), split="train")
    eval_dataset = load_dataset("json", data_files=str(config["eval_file"]), split="train")
    tokenizer = AutoTokenizer.from_pretrained(str(config["base_model"]))
    model = AutoModelForCausalLM.from_pretrained(str(config["base_model"]))
    training_args = SFTConfig(
        output_dir=str(config["output_dir"]),
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=int(config["num_train_epochs"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        max_seq_length=int(config["max_seq_length"]),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )
    trainer.train()
    trainer.save_model(str(config["output_dir"]))


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = dict(DEFAULT_CONFIG)
    _write_config(Path(args.config_output), config)

    if args.run:
        _run_training(config)

    print(
        {
            "config_output": args.config_output,
            "run_requested": args.run,
            "base_model": config["base_model"],
            "output_dir": config["output_dir"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
