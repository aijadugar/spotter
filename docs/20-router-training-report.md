# Exercise Router Training Report

Generated after the full Modal training run on June 12, 2026.

## Summary

The exercise router was trained and evaluated for `squat`, `push_up`, `shoulder_press`, and
`unknown`. The final active artifact is the BiLSTM temporal model. The scikit-learn baseline is
retained as a reference and fallback artifact.

| Field                  | Value                                                                     |
| ---------------------- | ------------------------------------------------------------------------- |
| Modal run              | `https://modal.com/apps/nlag/main/ap-9CDLM3tMgOlCYE2JCBe71H`              |
| Hugging Face repo      | `build-small-hackathon/pozify-exercise-router`                           |
| Selected model         | `temporal.pt`                                                             |
| Selected artifact      | `temporal.pt`                                                             |
| Selection rule         | Prefer BiLSTM temporal when available; baseline falls back when missing   |
| Local active path      | `models/exercise_router/active/temporal.pt`                               |
| Baseline artifact path | `models/exercise_router/active/baseline.joblib`                           |
| Router baseline alias  | `models/exercise_router/active/router.joblib`                             |

## Data

| Metric                     | Value |
| -------------------------- | ----: |
| Feature examples           |   134 |
| Window count               | 2,224 |
| Failed feature extractions |     0 |
| Push-up windows            |   287 |
| Shoulder press windows     |   646 |
| Squat windows              |   659 |
| Unknown windows            |   632 |

Unsupported Riccio dataset classes such as bicep curl are mapped to `unknown`.

## Training Setup

| Model    | Setup                                                                                 |
| -------- | ------------------------------------------------------------------------------------- |
| Baseline | scikit-learn `HistGradientBoostingClassifier` over engineered 30-frame window vectors |
| Temporal | PyTorch BiLSTM over 30-frame feature tensors on Modal A10                             |

The Modal image and local verification environment use the Python 3.10 dependency set:

| Dependency     | Version      |
| -------------- | ------------ |
| Python         | `3.10.20`    |
| scikit-learn   | `1.7.2`      |
| joblib         | `1.5.3`      |
| torch          | `2.11.0`     |
| numpy          | `1.26.4`     |
| scipy          | `1.15.3`     |

The BiLSTM hyperparameters follow the Riccio exercise-classification paper:

| Hyperparameter      |  Value |
| ------------------- | -----: |
| Epochs              |     73 |
| Hidden units        |     73 |
| Dropout             | 0.2174 |
| Learning rate       | 0.0004 |
| Batch size          |     54 |
| Final training loss | 0.0003 |

## Model Complexity

Counts were checked from the local trained artifacts with `uv run` under Python 3.10.

| Model           | Count type                                |   Value |
| --------------- | ----------------------------------------- | ------: |
| Baseline        | Neural-network-style trainable parameters |       0 |
| Baseline        | Input features                            |   1,422 |
| Baseline        | Trees                                     |     800 |
| Baseline        | Total tree nodes                          |  21,254 |
| Baseline        | Split nodes                               |  10,227 |
| Baseline        | Leaves                                    |  11,027 |
| Baseline        | Approximate learned scalar state          |  41,708 |
| BiLSTM temporal | Trainable parameters                      | 182,796 |
| BiLSTM temporal | Input features per frame                  |     237 |
| BiLSTM temporal | Hidden units                              |      73 |
| BiLSTM temporal | Layers                                    |       1 |
| BiLSTM temporal | Output classes                            |       4 |

The baseline is a tree-based `HistGradientBoostingClassifier`, so it does not have trainable tensor
parameters in the same sense as a neural network. The BiLSTM parameter count includes both LSTM
directions and the linear classification head.

## Training Metrics

| Model           | Validation accuracy |                  Unknown rejection rate |
| --------------- | ------------------: | --------------------------------------: |
| Baseline        |              0.9910 | Not reported in baseline training stage |
| BiLSTM temporal |              0.9843 |                                  0.9843 |

## Selection Evaluation

The final evaluation scored every available trained artifact on the cached router windows.
The baseline scored slightly higher on this cache, but the active router is BiLSTM so routing uses
the temporal pose-window sequence directly.

| Model           | Artifact          | Accuracy | Unknown rejection rate |
| --------------- | ----------------- | -------: | ---------------------: |
| Baseline        | `baseline.joblib` |   0.9982 |                 0.9968 |
| BiLSTM temporal | `temporal.pt`     |   0.9969 |                 0.9968 |

### Baseline Precision/Recall

| Label            | Precision | Recall |
| ---------------- | --------: | -----: |
| `squat`          |    0.9985 | 0.9970 |
| `push_up`        |    0.9965 | 1.0000 |
| `shoulder_press` |    0.9985 | 1.0000 |
| `unknown`        |    0.9984 | 0.9968 |

### BiLSTM Precision/Recall

| Label            | Precision | Recall |
| ---------------- | --------: | -----: |
| `squat`          |    0.9939 | 0.9970 |
| `push_up`        |    1.0000 | 1.0000 |
| `shoulder_press` |    0.9984 | 0.9954 |
| `unknown`        |    0.9968 | 0.9968 |

## Confusion Matrices

Rows are true labels. Columns are predicted labels.

### Baseline

| True \\ Predicted | `squat` | `push_up` | `shoulder_press` | `unknown` |
| ----------------- | ------: | --------: | ---------------: | --------: |
| `squat`           |     657 |         1 |                0 |         1 |
| `push_up`         |       0 |       287 |                0 |         0 |
| `shoulder_press`  |       0 |         0 |              646 |         0 |
| `unknown`         |       1 |         0 |                1 |       630 |

### BiLSTM Temporal

| True \\ Predicted | `squat` | `push_up` | `shoulder_press` | `unknown` |
| ----------------- | ------: | --------: | ---------------: | --------: |
| `squat`           |     657 |         0 |                1 |         1 |
| `push_up`         |       0 |       287 |                0 |         0 |
| `shoulder_press`  |       2 |         0 |              643 |         1 |
| `unknown`         |       2 |         0 |                0 |       630 |

## Published Artifacts

The full run uploaded the following artifacts to Hugging Face and the fresh Hub download matched
the local Modal volume artifacts.

| Artifact                | SHA-256                                                            |
| ----------------------- | ------------------------------------------------------------------ |
| `baseline.joblib`       | `dbe53cab28ff664d1eb08546b24e0b5a9cd374d70b6a59bb5f17c2e9af58517f` |
| `router.joblib`         | `dbe53cab28ff664d1eb08546b24e0b5a9cd374d70b6a59bb5f17c2e9af58517f` |
| `temporal.pt`           | `db07644553a37ed7e939f22b8eb720b9cf392149bb219c7e5039cfcf2ac583a2` |
| `evaluation.json`       | `b1e5530d62532d18c512fb38f2c1422f08caa5f067afd978589a1621bed11560` |
| `router_selection.json` | `d8263f0cf739c6aa75a27bbf23246277a15222ce24e449fe300ded72921c4177` |

`baseline.joblib` and `router.joblib` are byte-identical. Runtime selection still loads
`temporal.pt` through `router_selection.json`.

## Reproduction Commands

Run the full training, evaluation, and publish flow:

```bash
uv run modal run scripts/exercise_router_modal.py --stage all --repo-id build-small-hackathon/pozify-exercise-router
```

Download the active artifact, baseline artifacts, selection file, and metrics after evaluation:

```bash
uv run modal volume get --force spotter-router-models /temporal.pt models/exercise_router/active/temporal.pt
uv run modal volume get --force spotter-router-models /baseline.joblib models/exercise_router/active/baseline.joblib
uv run modal volume get --force spotter-router-models /router.joblib models/exercise_router/active/router.joblib
uv run modal volume get --force spotter-router-models /router_selection.json models/exercise_router/active/router_selection.json
uv run modal volume get --force spotter-router-models /evaluation.json models/exercise_router/active/evaluation.json
uv run modal volume get --force spotter-router-models /baseline_metrics.json models/exercise_router/active/baseline_metrics.json
uv run modal volume get --force spotter-router-models /temporal_metrics.json models/exercise_router/active/temporal_metrics.json
```

## Verification

The refreshed artifacts were verified under Python 3.10:

```bash
uv run --extra dev ruff check
uv run python -m compileall src scripts tests app.py
uv run python -m unittest discover -s tests
```

The test suite passed with 75 tests, 1 skipped.

## Notes

These metrics are router-window metrics from the current Modal feature cache, not a claim of
generalization to every capture setup. Add more custom `unknown` clips and independent held-out video
sets before treating the router as production-grade.
