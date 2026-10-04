# Progress-Plan Training Report

This report covers the `progress_plan` model: the task, data, hyperparameters, and the
open-versus-base evaluation harness. The task converts structured session history and today's
findings into a next-session plan as strict JSON.

## Task

- Input: rolling session history plus current-session aggregate metrics and detected issue labels.
- Output: `progress_plan.json` with keys `focus`, `targets`, `next_session_cues`, `encouragement`,
  `confidence_notes`.
- Constraint: the plan may reference only issues detected this session or present in history
  (`allowed_issues_for`). Medical and injury language is rejected by `verify_plan`.

## Data

- Builder: `scripts/build_progress_plan_sft_dataset.py`.
- Train: `data/sft/progress_plan_train.jsonl` (400 rows).
- Eval: `data/sft/progress_plan_eval.jsonl` (60 rows).
- Labels are synthetic perturbations over the supported movements; a hand-checked sample is required
  before a real run.

## Hyperparameters

From `configs/progress_plan_lora.default.json`: base `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`,
LoRA r=16, alpha=32, dropout=0.05, lr=2e-4, 2 epochs, batch 1 with gradient accumulation 8,
max sequence length 2048.

## Harness validation

`scripts/evaluate_progress_plan.py --dry-run` scores the reference completions to validate the
metrics pipeline before any model run.

| label | n | schema_valid_rate | grounding_rate | latency_p50_ms | latency_p95_ms |
| --- | --- | --- | --- | --- | --- |
| reference | 60 | 1.0000 | 1.0000 | 0.0 | 0.0 |

The reference scores are a floor: they confirm the harness, not model quality.

## Open versus base (pending Tinker run)

Run the fine-tune, then evaluate the base model and the tuned model on the same eval file:

```bash
python3 scripts/evaluate_progress_plan.py --label base
python3 scripts/evaluate_progress_plan.py --label tuned
```

Fill the table from `reports/progress_plan_eval_base.json` and
`reports/progress_plan_eval_tuned.json`. This table is the open-versus-base claim used in the write-up.

| label | schema_valid_rate | grounding_rate | latency_p50_ms | latency_p95_ms |
| --- | --- | --- | --- | --- |
| base | pending | pending | pending | pending |
| tuned | pending | pending | pending | pending |

## Notes

- Confirm the current Tinker SDK method names before the real run; `scripts/train_progress_plan_tinker.py`
  is intentionally minimal and reads `TINKER_API_KEY` from the environment.
- Add a native-speaker quality rating on 20 to 30 samples; automated metrics do not capture language
  quality.
