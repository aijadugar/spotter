# Spotter Build Small Hackathon Report

Status note:

- This report is kept as the hackathon narrative document.
- Current runtime defaults use `build-small-hackathon/spotter-coach-summary1`.
- Current coach-summary training now includes a LoRA/merge/publish pipeline on Modal.
- For the current operational commands, prefer [02-technical-setup.md](02-technical-setup.md) and
  [30-coach-modal-training.md](30-coach-modal-training.md).

Date: June 14, 2026

Spotter is a small-model workout form reviewer. A user uploads a short exercise video, adds basic
training context, and receives a rep-by-rep report with timestamps, annotated video, issue markers,
and a grounded coach summary.

The build was aimed at the Hugging Face Build Small Hackathon: stay under the 32B parameter cap, ship
a Gradio Space, train or fine-tune something real, and show the work clearly enough that judges can
reproduce it.

## The Problem

Workout videos contain useful feedback, but most beginners do not know what to inspect. Generic
fitness advice says things like "keep your core tight" or "go deeper", but it rarely says which rep,
which timestamp, what evidence, and whether a visible difference is actually a valid variation.

Spotter treats the video as evidence. It first extracts pose and movement structure, then lets a small
language model explain the structured findings. The language model is not asked to invent the
analysis.

## Product Flow

```text
video + user profile
-> video QC
-> MediaPipe pose extraction
-> pose cleaning
-> trained exercise router
-> rep counter
-> per-rep metrics
-> variation detector
-> issue markers
-> annotated video
-> Nemotron coach summary
-> deterministic verifier
-> final report
```

The supported exercise router classes are `squat`, `push_up`, `shoulder_press`, and `unknown`.
Routing to `unknown` is a feature: the app should reject unsupported or unclear clips instead of
pretending every video is one of the supported movements.

## Models Used

| Component | Model | Why this choice |
| --------- | ----- | --------------- |
| Pose extractor | MediaPipe Pose Landmarker Lite | Fast, practical feature extractor for a Gradio Space. |
| Exercise router | Custom PyTorch BiLSTM | Tiny trainable temporal model over pose windows. |
| Baseline router | scikit-learn HistGradientBoostingClassifier | Strong baseline over engineered vectors and fallback artifact. |
| Coach summary | build-small-hackathon/spotter-coach-summary1 | Current default fine-tuned runtime for structured JSON explanation. |
| llama.cpp path | Nemotron-3-Nano-4B GGUF via `llama-server` | Local-first/off-grid coach summary path with GPU offload. |

The original hackathon build trained the exercise router first and used Nemotron as a grounded
summarizer over JSON evidence. The current codebase now also contains a coach-summary LoRA / merged
model training path on Modal.

## What Was Trained

The main trained artifact is the Spotter exercise router:

- active artifact: `temporal.pt`
- architecture: bidirectional LSTM with one layer
- trainable parameters: 182,796
- input: 30-frame pose windows
- per-frame features: 237
- labels: `squat`, `push_up`, `shoulder_press`, `unknown`
- published repo: `build-small-hackathon/spotter-exercise-router`

A scikit-learn baseline is also trained:

- artifact: `baseline.joblib`
- model: `HistGradientBoostingClassifier`
- input: engineered aggregate vectors from each pose window
- role: reference and fallback

The active selection policy prefers the BiLSTM when available. The baseline remains available for
comparison and fallback.

## Data

Primary dataset:

- `RickyRiccio/Real_Time_Exercise_Recognition_Dataset`

Label normalization:

| Source class pattern | Router label |
| -------------------- | ------------ |
| Squat folders | `squat` |
| Push-up folders | `push_up` |
| Shoulder press / overhead press folders | `shoulder_press` |
| Unsupported exercises, setup motion, idle, stretching, bad angle | `unknown` |

Unsupported Riccio classes such as bicep curl variants are intentionally mapped to `unknown`.

Latest feature cache:

| Metric | Value |
| ------ | ----: |
| Feature examples | 134 |
| Pose windows | 2,224 |
| Failed feature extractions | 0 |
| Push-up windows | 287 |
| Shoulder press windows | 646 |
| Squat windows | 659 |
| Unknown windows | 632 |

Each example is converted into 30-frame windows. The feature schema includes normalized landmarks,
visibility, joint angles, relative distances, deltas, and velocities. The BiLSTM sees the temporal
tensor directly; the baseline sees aggregate statistics such as mean, standard deviation, min, max,
range, and trend.

## Modal Training Pipeline

Modal handles the expensive batch jobs:

1. `ingest`: download the dataset from Hugging Face and build a JSONL manifest.
2. `features`: decode videos, run video QC, extract MediaPipe pose, clean poses, and cache router
   windows as compressed NumPy arrays.
3. `train-baseline`: train the scikit-learn baseline on engineered vectors.
4. `train-temporal`: train the BiLSTM on a Modal A10 GPU.
5. `evaluate`: score every available artifact and write `router_selection.json`.
6. `publish`: upload model card, artifacts, and metrics to Hugging Face.

Reproduction command:

```bash
uv run modal run scripts/exercise_router_modal.py \
  --stage all \
  --repo-id build-small-hackathon/spotter-exercise-router
```

Modal volumes:

| Volume | Contents |
| ------ | -------- |
| `spotter-router-data` | raw videos, manifests, feature caches |
| `spotter-router-models` | trained artifacts, metrics, selection file, upload log |

Training environment:

| Dependency | Version |
| ---------- | ------- |
| Python | 3.10.20 |
| torch | 2.11.0 |
| scikit-learn | 1.7.2 |
| joblib | 1.5.3 |
| numpy | 1.26.4 |
| scipy | 1.15.3 |

BiLSTM hyperparameters:

| Hyperparameter | Value |
| -------------- | ----: |
| Epochs | 73 |
| Hidden units | 73 |
| Dropout | 0.2174 |
| Learning rate | 0.0004 |
| Batch size | 54 |
| Final training loss | 0.0003 |

## Evaluation

Validation during temporal training:

| Model | Validation accuracy | Unknown rejection rate |
| ----- | ------------------: | ---------------------: |
| Baseline | 0.9910 | Not reported at that stage |
| BiLSTM temporal | 0.9843 | 0.9843 |

Final selection evaluation on the cached router windows:

| Model | Artifact | Accuracy | Unknown rejection rate |
| ----- | -------- | -------: | ---------------------: |
| Baseline | `baseline.joblib` | 0.9982 | 0.9968 |
| BiLSTM temporal | `temporal.pt` | 0.9969 | 0.9968 |

The baseline edges out the temporal model on this cache, but the BiLSTM is the selected artifact
because it consumes the pose-window sequence directly and better matches the intended runtime design.
The baseline remains useful as a sanity check.

These numbers should not be read as production-generalization claims. They are router-window metrics
from the current feature cache. More independent held-out videos are needed.

## GPU Runtime

### Router

The router runs through Torch. It defaults to CPU in local runs and on Hugging Face Spaces because
the router is tiny and CPU avoids CUDA/ZeroGPU runtime failures during classification.

```bash
SPOTTER_ROUTER_DEVICE=cuda uv run python app.py
```

Set `SPOTTER_ROUTER_DEVICE=cuda` only when you explicitly want to test CUDA routing. GPU matters
more for local language-model generation than for this small classifier.

### Hugging Face ZeroGPU

Compute-heavy functions are wrapped with `spaces.GPU`, while request state and streaming response
logic stay outside the GPU worker. Useful settings:

```bash
SPOTTER_COACH_SUMMARY_PROVIDER=local_transformers
SPOTTER_COACH_SUMMARY_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
SPOTTER_SPACES_GPU_DURATION=300
```

This keeps the app inside the small-model budget while avoiding the hosted Hugging Face Inference
API.

### llama.cpp

Spotter now supports `llama-server` through its OpenAI-compatible chat completion route. That means
the coach summary can run from a local GGUF model with llama.cpp GPU offload.

Start llama.cpp with a local GGUF:

```bash
llama-server \
  --model /path/to/nemotron-3-nano-4b-q4_k_m.gguf \
  --ctx-size 4096 \
  --n-gpu-layers 99 \
  --host 127.0.0.1 \
  --port 8080
```

Or use a Hugging Face GGUF repo:

```bash
llama-server \
  --hf-repo owner/nemotron-3-nano-4b-gguf:Q4_K_M \
  --ctx-size 4096 \
  --n-gpu-layers 99 \
  --host 127.0.0.1 \
  --port 8080
```

Then point Spotter at it:

```bash
SPOTTER_COACH_SUMMARY_PROVIDER=llama_cpp \
SPOTTER_COACH_SUMMARY_MODEL=local-nemotron-3-nano-4b-gguf \
SPOTTER_LLAMA_CPP_BASE_URL=http://127.0.0.1:8080 \
uv run python app.py
```

This llama.cpp mode is only for the coach summary. Pose extraction, routing, rep counting, issue
markers, rendering, and verification still run in the Spotter Python pipeline.

## Why Not Fine-Tune The LLM?

The hardest product risk was not prose style; it was routing the video into the right analyzer and
not hallucinating feedback. A language-model fine-tune would make the output sound more tailored,
but it would not solve exercise recognition or timestamped evidence.

The chosen split was:

- train a small router where labels and metrics are measurable;
- keep Nemotron as a general instruction model;
- constrain Nemotron with structured evidence JSON and knowledge cards;
- run deterministic verification after generation.

This makes the app easier to debug. If the router is wrong, inspect `exercise_classification.json`.
If an issue marker is wrong, inspect `rep_analysis.json` and `issue_markers.json`. If the summary
adds unsupported claims, the verifier can reject it and the app falls back to a deterministic
summary.

## Generated Artifacts

Every app run writes a folder under `runs/<run_id>/`:

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

This is useful for judges because the final UI is not a black box. The report can be traced back to
the intermediate evidence.

## What Worked

- Training a tiny router was enough for the supported class set.
- Mapping unsupported exercises to `unknown` made the app safer and more honest.
- Modal made the train/evaluate/publish loop simple to rerun.
- Keeping the language model downstream of structured evidence made the app easier to verify.
- The custom Gradio server approach allowed a more product-like UI while staying inside the Space
  requirement.

## Limitations

- Current metrics are based on the cached router-window dataset, not a large independent benchmark.
- The app relies on usable pose extraction and reasonable camera framing.
- Per-rep issue rules are transparent but not biomechanically exhaustive.
- Nemotron is not fine-tuned; it is prompted and verified.
- The llama.cpp path depends on a separately running `llama-server`.
- This is not medical or clinical software.

## Next Steps

- Add more consented custom videos for `unknown` and borderline cases.
- Add independent held-out demo clips with subject/camera separation.
- Export a GGUF-friendly smaller coach model config for faster local/offline runs.
- Add structured JSON schema enforcement directly in llama.cpp requests.
- Add more exercise-specific analyzers after the router/data loop is stable.
