from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spotter.coach_summary_sft_dataset import (  # noqa: E402
    build_sft_row_from_run_dir,
    collect_run_dirs,
    split_sft_rows,
    write_jsonl,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Spotter-native SFT train/eval JSONL files from run artifacts."
    )
    parser.add_argument(
        "--runs-dir",
        default=str(ROOT / "runs"),
        help="Directory containing Spotter run artifact folders.",
    )
    parser.add_argument(
        "--train-output",
        default=str(ROOT / "data/sft/coach_summary_train.jsonl"),
        help="Destination for the train JSONL file.",
    )
    parser.add_argument(
        "--eval-output",
        default=str(ROOT / "data/sft/coach_summary_eval.jsonl"),
        help="Destination for the eval JSONL file.",
    )
    parser.add_argument(
        "--eval-count",
        type=int,
        default=10,
        help="Number of examples to reserve for eval.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Shuffle seed for train/eval split.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    run_dirs = collect_run_dirs(Path(args.runs_dir))
    rows = [build_sft_row_from_run_dir(run_dir) for run_dir in run_dirs]
    train_rows, eval_rows = split_sft_rows(
        rows,
        eval_count=args.eval_count,
        seed=args.seed,
    )
    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.eval_output), eval_rows)
    print(
        {
            "runs_dir": args.runs_dir,
            "row_count": len(rows),
            "train_count": len(train_rows),
            "eval_count": len(eval_rows),
            "train_output": args.train_output,
            "eval_output": args.eval_output,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
