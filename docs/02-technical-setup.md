# Spotter Technical Setup And Runtime

This document holds the command-heavy setup, runtime, training, and verification notes for Spotter.
The main [README](../README.md) stays focused on the project, model strategy, and product story.

## Run The App Locally

This repo uses a `src/` layout, but `uv` is configured with `package = false`.

```bash
uv sync
uv run python app.py
```

Then open `http://127.0.0.1:7860`.

## Mock vs Real Mode

By default:

- if no video is provided, Spotter uses mock mode
- if a real video is uploaded, Spotter runs the full analysis pipeline

Force mock mode:

```bash
SPOTTER_MOCK_MODE=1 uv run python app.py
```

Force real mode:

```bash
SPOTTER_MOCK_MODE=0 uv run python app.py
```

If you already have the MediaPipe task file locally:

```bash
SPOTTER_MEDIAPIPE_POSE_MODEL=/path/to/pose_landmarker_lite.task \
SPOTTER_MOCK_MODE=0 \
uv run python app.py
```

## Coach Summary Runtime Options

### 1. Fine-tuned coach model

The app defaults to the fine-tuned coach-summary model:

```bash
export SPOTTER_COACH_SUMMARY_MODEL=build-small-hackathon/pozify-coach-summary1
uv run python app.py
```

Spotter tries `chat_completion` first and falls back to `text_generation` when Hugging Face reports
that the repo is not a chat model. The deterministic fallback summary remains enabled if hosted
inference is unavailable or the model output fails validation.

For regular Hugging Face Spaces, keep the provider on hosted inference unless you have a dedicated
local model runtime:

```bash
SPOTTER_COACH_SUMMARY_PROVIDER=hf_inference
SPOTTER_COACH_SUMMARY_MODEL=build-small-hackathon/pozify-coach-summary1
```

For Hugging Face ZeroGPU Spaces, local Transformers is selected automatically so the app does not
call the hosted Hugging Face Inference API. You can also set it explicitly:

```bash
SPOTTER_COACH_SUMMARY_PROVIDER=local_transformers
SPOTTER_COACH_SUMMARY_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
SPOTTER_SPACES_GPU_DURATION=300
```

`HF_TOKEN` is only needed for `hf_inference` or for downloading a private/gated local model repo.
Spotter uses the Nemotron implementation bundled with Transformers instead of downloading remote
model code. If fast Mamba kernels are unavailable at runtime, Spotter caps the local prompt context
before generation to avoid the slow naive Mamba path crashing CUDA.

### 2. Use the fine-tuned merged model locally

Download the merged repo locally, then point Spotter at it:

```bash
export SPOTTER_COACH_SUMMARY_LOCAL_MODEL_DIR=/absolute/path/to/merged_model
export SPOTTER_COACH_SUMMARY_BASE_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
export SPOTTER_COACH_SUMMARY_ADAPTER_ID=build-small-hackathon/pozify-coach-summary1
uv run python app.py
```

This is the simplest way to use `build-small-hackathon/pozify-coach-summary1` today without adding a
dedicated inference endpoint.

### 3. Base cloud model override

If you need the Nemotron base-model runtime:

```bash
export SPOTTER_COACH_SUMMARY_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
uv run python app.py
```

### 4. llama.cpp

Spotter can send the coach-summary prompt to a local `llama-server` that exposes the
OpenAI-compatible `/v1/chat/completions` endpoint.

Example:

```bash
llama-server \
  --model /path/to/nemotron-3-nano-4b.gguf \
  --ctx-size 4096 \
  --n-gpu-layers 99 \
  --host 127.0.0.1 \
  --port 8080
```

Then:

```bash
SPOTTER_COACH_SUMMARY_PROVIDER=llama_cpp \
SPOTTER_COACH_SUMMARY_MODEL=local-nemotron-3-nano-4b-gguf \
SPOTTER_LLAMA_CPP_BASE_URL=http://127.0.0.1:8080 \
SPOTTER_COACH_SUMMARY_MAX_TOKENS=700 \
uv run python app.py
```

## Useful Environment Variables

| Variable | Purpose |
| --- | --- |
| `SPOTTER_ROUTER_DEVICE` | Override router device, for example `cpu` or `cuda`. |
| `SPOTTER_SPACES_GPU_DURATION` | `spaces.GPU` duration in seconds, default `120`. |
| `SPOTTER_COACH_SUMMARY_PROVIDER` | `hf_inference`, `local_transformers`, or `llama_cpp`. |
| `SPOTTER_COACH_SUMMARY_MODEL` | Coach model id or llama.cpp model alias. |
| `SPOTTER_COACH_SUMMARY_LOCAL_MODEL_DIR` | Prefer a local merged/model directory for coach summary. |
| `SPOTTER_COACH_SUMMARY_MAX_INPUT_TOKENS` | Max local Transformers prompt tokens, default `2048`. |
| `SPOTTER_COACH_SUMMARY_BYPASS_VERIFIER` | Keep model output even when verifier fails. |

## Exercise Router Training

Run the full router training and publish flow:

```bash
uv run modal run scripts/exercise_router_modal.py \
  --stage all \
  --repo-id build-small-hackathon/pozify-exercise-router
```

Step-by-step:

```bash
uv run modal run scripts/exercise_router_modal.py --stage ingest
uv run modal run scripts/exercise_router_modal.py --stage features
uv run modal run scripts/exercise_router_modal.py --stage train-baseline
uv run modal run scripts/exercise_router_modal.py --stage train-temporal
uv run modal run scripts/exercise_router_modal.py --stage evaluate
uv run modal run scripts/exercise_router_modal.py --stage publish --repo-id build-small-hackathon/pozify-exercise-router
```

The active router artifact is `temporal.pt`; the baseline is retained for comparison and fallback.

## Coach Summary Training

Build the grounded SFT dataset:

```bash
uv run python scripts/build_coach_summary_sft_dataset.py
```

Run the full coach-summary Modal flow:

```bash
uv run modal run scripts/coach_summary_modal.py \
  --stage all \
  --epochs 2 \
  --style-weight 0.2 \
  --repo-id build-small-hackathon/pozify-coach-summary1
```

The checked-in fine-tune config uses `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` as the base model.
The Modal training, evaluation, and merge stages request an `A100-80GB` GPU because the Nemotron
base model can run out of CUDA memory on the previous `A10G` setting.

Step-by-step:

```bash
uv run modal run scripts/coach_summary_modal.py --stage prepare-data
uv run modal run scripts/coach_summary_modal.py --stage train --epochs 2 --style-weight 0.2
uv run modal run scripts/coach_summary_modal.py --stage evaluate --limit 5
uv run modal run scripts/coach_summary_modal.py --stage merge
uv run modal run scripts/coach_summary_modal.py --stage publish-merged --repo-id build-small-hackathon/pozify-coach-summary1
```

Important runtime note:

- the default coach model is `build-small-hackathon/pozify-coach-summary1`
- Hugging Face hosted inference may still reject a repo or produce invalid JSON, so the
  conservative fallback summary stays enabled
- for the most predictable fine-tuned inference path, use `SPOTTER_COACH_SUMMARY_LOCAL_MODEL_DIR`

## Generated Artifacts

Each run creates `runs/<run_id>/` with:

- `manifest.json`
- `user_profile.json`
- `video_manifest.json`
- `pose_sequence.json`
- `exercise_classification.json`
- `reps.json`
- `rep_debug.json`
- `rep_analysis.json`
- `variation.json`
- `issue_markers.json`
- `annotated_video.mp4`
- `coach_summary.json`
- `verification.json`
- `final_report.json`

JSON artifacts are validated before they are written. The final report records:

- analysis mode
- pose source
- knowledge-card provenance
- coach summary provider/model/source
- verifier status and bypass flags

## Development Checks

```bash
uv run ruff check
uv run python -m compileall src scripts tests app.py
uv run python -m unittest discover -s tests
```

Run the real MediaPipe fixture smoke test only when the fixture is available:

```bash
SPOTTER_RUN_REAL_POSE_TESTS=1 \
uv run python -m unittest tests.test_pose_steps.PoseStepTests.test_real_sample_mov_extracts_pose_landmarks
```
