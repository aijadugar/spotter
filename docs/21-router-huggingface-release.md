# Hugging Face Router Release Guide

This guide covers publishing the Spotter exercise-router artifacts to a Hugging Face model repository
and configuring runtime inference to download the selected artifact from the Hub.

Official references:

- Upload files with `huggingface_hub`: https://huggingface.co/docs/huggingface_hub/en/guides/upload
- Download files with `hf_hub_download`: https://huggingface.co/docs/huggingface_hub/en/guides/download
- Authentication and token login: https://huggingface.co/docs/huggingface_hub/en/package_reference/authentication
- `HF_TOKEN` environment variable: https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables

## Repository Setup

The configured model repository is:

```text
build-small-hackathon/pozify-exercise-router
```

Create a user access token at:

```text
https://huggingface.co/settings/token
```

Use a token that can write to the target model repository. Do not commit the token to this repo.

Recommended local login:

```bash
uv run hf auth login
```

Alternative for one shell session:

```bash
export HF_TOKEN=<your-token>
```

## Upload Artifacts

After Modal evaluation, the local active artifact directory should contain:

```text
models/exercise_router/active/
  router.joblib
  router_selection.json
  temporal.pt
```

Upload the model card and artifacts:

```bash
uv run python scripts/upload_exercise_router_to_hf.py \
  --repo-id build-small-hackathon/pozify-exercise-router \
  --private
```

Omit `--private` if the model repository should be public.

The Modal training script can also publish directly from the `spotter-router-models` volume after
evaluation. It reads `HF_TOKEN`, `SPOTTER_ROUTER_HF_REPO_ID`, and optional
`SPOTTER_ROUTER_HF_PRIVATE=1` from local `.env` and injects them into Modal secrets at runtime:

```bash
uv run modal run scripts/exercise_router_modal.py --stage all
uv run modal run scripts/exercise_router_modal.py --stage publish --repo-id build-small-hackathon/pozify-exercise-router
```

`stage evaluate` and `stage all` publish automatically after writing `router_selection.json`.

The upload script creates the repository if needed and uploads:

- `README.md` model card
- `router.joblib`
- `router_selection.json`
- `temporal.pt`
- `training_report.md`
- `evaluation.json`
- `baseline_metrics.json`
- `temporal_metrics.json`

## Runtime Configuration

The runtime loader uses `build-small-hackathon/pozify-exercise-router` by default. Override the
model repository ID only when you want to test or deploy a different router repo:

```bash
export SPOTTER_ROUTER_HF_REPO_ID=owner/other-spotter-router
```

Optional:

```bash
export SPOTTER_ROUTER_HF_REVISION=main
```

For private repositories, authenticate with `uv run hf auth login` or set `HF_TOKEN` before running
the app. The loader first tries Hugging Face using the default repo or the override above. If Hub
loading fails, it falls back to local files under `models/exercise_router/active/`.

Disable Hub loading explicitly:

```bash
export SPOTTER_ROUTER_DISABLE_HF=1
```

## Expected Artifact Selection

The current active router selects the BiLSTM temporal artifact:

```json
{
  "selected_model": "temporal.pt",
  "selected_artifact": "temporal.pt",
  "reason": "prefer BiLSTM temporal when available; baseline falls back when temporal is missing"
}
```

The baseline artifact is still uploaded for comparison and fallback, but runtime routing uses the
BiLSTM when `router_selection.json` points at `temporal.pt`.
