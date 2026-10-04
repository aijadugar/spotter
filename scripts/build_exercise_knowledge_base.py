from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spotter.knowledge_card_dataset_transformer import (  # noqa: E402
    write_card_pack,
    write_normalized_exercises,
)


DEFAULT_HF_DATASET = "DORTROX/Exercises-Data"
DEFAULT_HF_FILENAME = "dataset.json"


def _download_hf_dataset_file(repo_id: str, filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required to download Hugging Face datasets") from exc

    return Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a normalized exercise knowledge base and Spotter card pack from a real dataset export."
    )
    parser.add_argument(
        "--input",
        help="Local path to a JSON or JSONL exercise dataset export.",
    )
    parser.add_argument(
        "--hf-dataset",
        default=DEFAULT_HF_DATASET,
        help="Hugging Face dataset repo id to download when --input is omitted.",
    )
    parser.add_argument(
        "--hf-filename",
        default=DEFAULT_HF_FILENAME,
        help="Filename inside the Hugging Face dataset repo to download when --input is omitted.",
    )
    parser.add_argument(
        "--normalized-output",
        default=str(ROOT / "data/knowledge/exercises.json"),
        help="Path for the normalized exercise-schema export.",
    )
    parser.add_argument(
        "--card-pack-output",
        default=str(ROOT / "data/knowledge_cards/external_exercise_dataset_pack.json"),
        help="Path for the generated Spotter card-pack JSON.",
    )
    parser.add_argument(
        "--source-dataset",
        help="Optional source identifier to record in metadata. Defaults to the local file path or HF dataset id.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.input:
        input_path = Path(args.input).expanduser().resolve()
        source_dataset = args.source_dataset or str(input_path)
    else:
        input_path = _download_hf_dataset_file(args.hf_dataset, args.hf_filename)
        source_dataset = args.source_dataset or args.hf_dataset

    normalized = write_normalized_exercises(
        input_path=input_path,
        output_path=Path(args.normalized_output),
        source_dataset=source_dataset,
    )
    pack = write_card_pack(
        input_path=input_path,
        output_path=Path(args.card_pack_output),
        source_dataset=source_dataset,
    )

    print(
        "Built exercise knowledge base",
        {
            "source_dataset": source_dataset,
            "input_path": str(input_path),
            "normalized_output": args.normalized_output,
            "card_pack_output": args.card_pack_output,
            "exercise_count": normalized["exercise_count"],
            "card_count": pack["card_count"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
