from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_MODEL_DIR = Path("models/exercise_router/active")
DEFAULT_MODEL_CARD = Path("docs/22-router-huggingface-model-card.md")
DEFAULT_TRAINING_REPORT = Path("docs/20-router-training-report.md")
ARTIFACT_FILENAMES = (
    "baseline.joblib",
    "router.joblib",
    "router_selection.json",
    "temporal.pt",
    "evaluation.json",
    "baseline_metrics.json",
    "temporal_metrics.json",
    "hf_upload.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload Spotter exercise-router artifacts to HF Hub.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repo id, e.g. user/pozify-exercise-router")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-card", type=Path, default=DEFAULT_MODEL_CARD)
    parser.add_argument("--training-report", type=Path, default=DEFAULT_TRAINING_REPORT)
    parser.add_argument("--private", action="store_true", help="Create the model repo as private")
    return parser.parse_args()


def upload_file(api: HfApi, *, repo_id: str, local_path: Path, path_in_repo: str) -> None:
    if not local_path.exists():
        print(f"Skipping missing file: {local_path}")
        return
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
    )
    print(f"Uploaded {local_path} -> {path_in_repo}")


def main() -> None:
    args = parse_args()
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    upload_file(api, repo_id=args.repo_id, local_path=args.model_card, path_in_repo="README.md")
    upload_file(
        api,
        repo_id=args.repo_id,
        local_path=args.training_report,
        path_in_repo="training_report.md",
    )
    for filename in ARTIFACT_FILENAMES:
        upload_file(
            api,
            repo_id=args.repo_id,
            local_path=args.model_dir / filename,
            path_in_repo=filename,
        )


if __name__ == "__main__":
    main()
