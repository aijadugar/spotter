# Coach Summary Modal Training

This guide explains the current Modal workflow for training, evaluating, merging, and publishing the
Spotter grounded coach-summary model.

## Scope

The pipeline lives in:

- `scripts/coach_summary_modal.py`

Available stages:

- `prepare-data`
- `train`
- `evaluate`
- `merge`
- `publish`
- `publish-merged`
- `all`

## What This Pipeline Produces

The pipeline fine-tunes a coach-summary model for Spotter's structured JSON-to-summary task.

It can produce:

- a LoRA adapter
- evaluation artifacts
- a merged full Transformers checkpoint
- a Hugging Face repo containing the merged checkpoint

Important limitation:

- Publishing a merged model repo to Hugging Face does **not** currently guarantee that the repo can
  be used through Hugging Face serverless `chat_completion`.
- The current codebase defaults to `build-small-hackathon/pozify-coach-summary1` and falls back
  from `chat_completion` to `text_generation` for non-chat model repos.
- The most predictable fine-tuned path remains local inference through
  `SPOTTER_COACH_SUMMARY_LOCAL_MODEL_DIR`.

## Requirements

1. Install and authenticate Modal:

```bash
uv run modal setup
```

2. Make sure local `.env` contains at least:

- `HF_TOKEN`
- optional `SPOTTER_COACH_SUMMARY_HF_REPO_ID`
- optional `SPOTTER_COACH_SUMMARY_MERGED_HF_REPO_ID`
- optional `SPOTTER_COACH_SUMMARY_MODEL`
- optional `SPOTTER_COACH_SUMMARY_HF_PRIVATE=1`

The script reads local `.env` and injects those values into a Modal secret at runtime.

## Input Data

The pipeline expects these files:

- `data/sft/coach_summary_train.jsonl`
- `data/sft/coach_summary_eval.jsonl`
- `data/sft/public_fitness_style.jsonl`
- `configs/coach_summary_lora.default.json`

The default checked-in config currently points to:

- base model: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`

The GPU-backed Modal stages currently request `A100-80GB` for training, evaluation, and merging.
This is intentional for the Nemotron base model; the previous `A10G` setting can run out of CUDA
memory during fine-tuning or merge.

## Recommended Training Flow

### 1. Build the SFT dataset

```bash
uv run python scripts/build_coach_summary_sft_dataset.py
```

### 2. Copy data into the Modal data volume

```bash
uv run modal run scripts/coach_summary_modal.py --stage prepare-data
```

### 3. Train the adapter

```bash
uv run modal run scripts/coach_summary_modal.py --stage train --epochs 2 --style-weight 0.2
```

### 4. Evaluate

```bash
uv run modal run scripts/coach_summary_modal.py --stage evaluate --limit 5
```

### 5. Merge the adapter into a full model

```bash
uv run modal run scripts/coach_summary_modal.py --stage merge
```

### 6. Publish the merged model

```bash
uv run modal run scripts/coach_summary_modal.py --stage publish-merged --repo-id build-small-hackathon/pozify-coach-summary1
```

### Full end-to-end run

```bash
uv run modal run scripts/coach_summary_modal.py --stage all --epochs 2 --style-weight 0.2 --repo-id build-small-hackathon/pozify-coach-summary1
```

## Cheap Smoke-Test Workflow

If you want to reduce GPU cost before a full run:

```bash
uv run modal run scripts/coach_summary_modal.py --stage prepare-data
uv run modal run scripts/coach_summary_modal.py --stage train --epochs 1 --style-weight 0.2
uv run modal run scripts/coach_summary_modal.py --stage evaluate --limit 5
```

If the format and verifier behavior look acceptable, rerun with `--epochs 2` and then `merge` +
`publish-merged`.

## Model Volume Artifacts

The Modal model volume stores:

- `adapter/`
- `merged_model/`
- `training_config.json`
- `training_summary.json`
- `evaluation.json`
- `hf_upload.json`
- `merge_summary.json`
- `hf_merged_upload.json`
- `README.md`

## Current Runtime Usage

### Default fine-tuned runtime

```bash
export SPOTTER_COACH_SUMMARY_MODEL=build-small-hackathon/pozify-coach-summary1
uv run python app.py
```

### Use the merged fine-tuned model locally

Download the merged repo or copy the merged directory locally, then:

```bash
export SPOTTER_COACH_SUMMARY_LOCAL_MODEL_DIR=/path/to/merged_model
export SPOTTER_COACH_SUMMARY_BASE_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
export SPOTTER_COACH_SUMMARY_ADAPTER_ID=build-small-hackathon/pozify-coach-summary1
uv run python app.py
```

### Base-model runtime

If you need to force the Nemotron base model:

```bash
export SPOTTER_COACH_SUMMARY_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
```

The fine-tuned model may still fall back if hosted inference is unavailable or returns invalid JSON,
so the deterministic fallback summary remains part of the runtime path.

## Evaluation Meaning

The coach-summary evaluation is task-specific. It measures:

- JSON validity rate
- verifier pass rate
- section completeness rate
- failure count and example failures

This is more useful for Spotter than generic text-generation metrics because the product depends on:

- schema correctness
- grounded issue references
- safe variation handling
- no diagnosis / no injury-prevention claims

## Related Docs

- [../README.md](../README.md)
- [31-coach-training-report.md](31-coach-training-report.md)
- [92-archive-coach-model-data-plan.md](92-archive-coach-model-data-plan.md)
