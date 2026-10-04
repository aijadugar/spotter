from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spotter.public_fitness_style_data import (  # noqa: E402
    convert_rows_to_style_corpus,
    load_chibbss_rows,
    load_haz_rows,
    write_style_jsonl,
)


DATASET_SPECS = {
    "HazSylvia/Fitness_Unformatted": {
        "filename": "FITNESS.csv",
        "loader": load_haz_rows,
    },
    "chibbss/fitness-chat-prompt-completion-dataset": {
        "filename": "fitness-chat-prompt-completion-dataset.json",
        "loader": load_chibbss_rows,
    },
}


def _download_hf_dataset_file(repo_id: str, filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required to download Hugging Face datasets") from exc
    return Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a filtered public fitness style corpus from real Hugging Face datasets."
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data/sft/public_fitness_style.jsonl"),
        help="Destination JSONL path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    corpus = []
    stats = {}
    for dataset_id, spec in DATASET_SPECS.items():
        path = _download_hf_dataset_file(dataset_id, spec["filename"])
        rows = spec["loader"](path)
        converted = convert_rows_to_style_corpus(rows, source_dataset=dataset_id)
        corpus.extend(converted)
        stats[dataset_id] = {
            "input_rows": len(rows),
            "kept_rows": len(converted),
        }

    write_style_jsonl(Path(args.output), corpus)
    print(
        {
            "output": args.output,
            "row_count": len(corpus),
            "datasets": stats,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
