---
license: other
library_name: scikit-learn
tags:
  - pose-estimation
  - exercise-recognition
  - video-classification
  - spotter
datasets:
  - RickyRiccio/Real_Time_Exercise_Recognition_Dataset
---

# Spotter Exercise Router

This repository contains the Spotter exercise-router artifacts for classifying pose windows as
`squat`, `push_up`, `shoulder_press`, or `unknown`.

## Model Details

The active artifact is selected by `router_selection.json`.

Current selected artifact:

```json
{
  "selected_model": "temporal.pt",
  "selected_artifact": "temporal.pt",
  "reason": "prefer BiLSTM temporal when available; baseline falls back when temporal is missing"
}
```

Artifacts:

- `temporal.pt`: selected PyTorch BiLSTM temporal model trained over 30-frame feature tensors.
- `router_selection.json`: active artifact selector used by Spotter runtime loading.
- `router.joblib`: scikit-learn baseline artifact kept for comparison and fallback.
- `training_report.md`: training and evaluation metrics.

## Intended Use

The router is intended for Spotter's local app pipeline. It routes normalized pose sequences to the
appropriate exercise-specific analyzer or rejects unsupported/uncertain clips as `unknown`.

Supported labels:

- `squat`
- `push_up`
- `shoulder_press`
- `unknown`

## Training Data

Primary source:

- `RickyRiccio/Real_Time_Exercise_Recognition_Dataset`

Unsupported classes from the source dataset, including curl variations, are mapped to `unknown`.
Custom unknown clips can include idle standing, setup motion, stretching, partial reps, severe
occlusion, and bad camera angles.

## Features

The router uses 30-frame sliding windows with engineered pose features:

- normalized landmarks
- landmark visibility
- knee, hip, elbow, and shoulder angles
- relative distances such as hand width over shoulder width
- frame deltas and velocities

## Evaluation

The latest training report is included as `training_report.md`.

Summary:

| Model           | Artifact          | Accuracy | Unknown rejection rate |
| --------------- | ----------------- | -------: | ---------------------: |
| Baseline        | `baseline.joblib` |   0.9987 |                 0.9984 |
| BiLSTM temporal | `temporal.pt`     |   0.9964 |                 0.9984 |

## Limitations

- Metrics are based on the current router-window cache, not a broad deployment benchmark.
- The router expects usable pose extraction and full-body framing where relevant.
- Unsupported exercises are intentionally routed to `unknown`.
- Additional independent held-out videos are needed before treating this as production-grade.

## Runtime Loading

Spotter loads this repository by default. Set `SPOTTER_ROUTER_HF_REPO_ID` only to override the default
with another compatible router repo:

```bash
export SPOTTER_ROUTER_HF_REPO_ID=owner/other-spotter-router
```

For private repositories, authenticate with `hf auth login` or set `HF_TOKEN`.
