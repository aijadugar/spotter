# Coach Summary Training Report

Generated from the current Spotter codebase state on June 15, 2026.

## Summary

The coach-summary model in Spotter is a grounded JSON-to-JSON generation component, not a generic
fitness chatbot. Its job is to convert structured workout-analysis artifacts into a safe,
evidence-bounded `coach_summary.json`, then pass that output through a verifier before it is shown
in the app.

The current training stack is:

- Base model: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`
- Fine-tuning method: BF16 LoRA SFT on Modal
- Training target: Spotter-native structured summary generation
- Runtime fallback: deterministic conservative summary when model generation fails or verification fails

The current codebase already supports:

- dataset building from real Spotter run artifacts
- Modal stages for prepare/train/evaluate/merge/publish
- verifier-based evaluation
- merged full-model publishing
- UI metadata for provider/model/source reporting

The main deployment gap at the time of this report is not training infrastructure. The main gap is
that the published Hugging Face merged repo is still not accepted by the current Hugging Face
Inference API path used by the app, so the app falls back to rule-based summaries when that repo is
selected.

## Objective

The coach-summary model is designed to generate these grounded sections:

- `summary`
- `what_you_did`
- `what_looked_good`
- `what_changed_across_reps`
- `valid_variation_vs_issue`
- `top_fixes`
- `next_session_plan`
- `confidence_notes`

Unlike a free-form assistant, this model is instructed to:

- use only structured evidence and retrieved knowledge cards
- never invent issue labels outside `issue_markers.json`
- avoid diagnosis language
- avoid injury-prevention claims
- preserve valid variation context instead of overcorrecting it

This contract is implemented in the prompt builder and enforced again by the verifier.

## Training Data

The current SFT data is Spotter-native and built from real run artifacts under `runs/`.

### Source artifacts per example

Each SFT row is constructed from:

- `user_profile.json`
- `exercise_classification.json`
- `reps.json`
- `rep_analysis.json`
- `variation.json`
- `issue_markers.json`
- `coach_summary.json`

The dataset builder reconstructs typed contracts from those files, retrieves the same knowledge
cards used at runtime, and serializes a chat-style SFT example with:

- system prompt
- user message containing structured evidence JSON
- assistant message containing the target `coach_summary.json`

### Current local dataset size

From the checked-in data in `data/sft/`:

| File | Rows |
| --- | ---: |
| `coach_summary_train.jsonl` | 22 |
| `coach_summary_eval.jsonl` | 5 |
| `public_fitness_style.jsonl` | 854 |

The style dataset is much larger than the grounded task dataset, so the training script samples
only a small portion of style rows according to `style_weight`.

## Prompt and Grounding Design

The prompt builder creates a structured instruction block plus a structured evidence block.

### Evidence included in the prompt

- `user_profile`
- `exercise_classification`
- `variation`
- `rep_summary`
- `issue_summary`
- `priority_cues`
- `knowledge_cards`

### Core prompt rules

- Use only the evidence JSON and retrieved knowledge cards.
- Do not infer new issues absent from `issue_summary.issues`.
- Do not diagnose injuries or pathology.
- Do not claim injury prevention.
- Do not treat valid variation labels or not-issue labels as errors.
- Include exact issue or variation labels in backticks when referenced.
- Return JSON only.

This is a good fit for small-model fine-tuning because the task is narrow, repetitive, and highly
structured.

## Model and Training Setup

### Current default base model

The codebase now defaults to:

- `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`

This is used in:

- `configs/coach_summary_lora.default.json`
- `scripts/train_coach_summary_lora.py`
- Modal training, evaluation, and merge stages in `scripts/coach_summary_modal.py`

### Default training hyperparameters

| Hyperparameter | Value |
| --- | ---: |
| Base model | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Learning rate | 0.0002 |
| Epochs | 2 |
| Batch size per device | 1 |
| Gradient accumulation | 8 |
| Max sequence length | 2048 |
| Default style weight | 0.2 |

### Modal implementation

The full training pipeline is implemented in `scripts/coach_summary_modal.py` with these stages:

- `prepare-data`
- `train`
- `evaluate`
- `merge`
- `publish`
- `publish-merged`
- `all`

The training stage:

- loads the base model in BF16 so Nemotron-H Mamba fast kernels receive regular projection weights
- tokenizes and truncates rows explicitly before training so long JSON evidence rows cannot bypass
  the sequence-length cap
- fine-tunes the LoRA adapter with the Transformers `Trainer`
- saves adapter weights and tokenizer to the Modal model volume

The merge stage:

- loads base model plus adapter
- calls `merge_and_unload()`
- writes a merged full checkpoint to `merged_model/`

## Evaluation Design

The evaluation stage is not based on generic BLEU or chat preference metrics. It uses task-specific
checks that match Spotter's production constraints.

### Reported evaluation metrics

- JSON validity rate
- verifier pass rate
- section completeness rate
- failure count and example failures

### Verifier checks

The verifier currently enforces:

- `no_issue_outside_json`
- `variation_not_overcorrected`
- `no_diagnosis`
- `no_injury_prevention_claim`
- `confidence_notes_present_when_required`

This is a strong design choice for Spotter because it measures whether the model behaves safely and
stays grounded, not just whether it sounds fluent.

## Latest Observed Training Outcome

From the latest visible Modal training logs for the full coach-summary flow:

### Training summary

| Metric | Value |
| --- | ---: |
| Train rows | 22 |
| Eval rows | 5 |
| Style rows mixed in | 4 |
| Merged train rows | 26 |
| Epochs | 2 |
| Global steps | 8 |
| Training loss | 1.0797 |

### Evaluation summary

| Metric | Value |
| --- | ---: |
| Evaluated count | 5 |
| JSON valid count | 2 |
| JSON validity rate | 0.40 |
| Verifier pass count | 0 |
| Verifier pass rate | 0.00 |
| Section completeness rate | 0.00 |

### Observed failure modes

- missing required top-level fields such as `summary`
- malformed JSON with delimiter errors
- outputs that fail schema extraction before verifier success is even possible

These numbers suggest that the current grounded task setup is correct in principle, but the dataset
is still too small and the model is not yet reliably producing the full required schema.

## Runtime Behavior in the App

At runtime the pipeline does this:

1. Generate a model summary from structured evidence.
2. Parse the model output into `coach_summary.json`.
3. Run the verifier.
4. If generation fails, use `fallback_initial`.
5. If generation succeeds but verification fails, use `fallback_after_verification` unless bypass is enabled.

The final report stores:

- `coach_summary_provider`
- `coach_summary_model`
- `coach_summary_source`
- verifier bypass metadata

This makes the UI traceable and helps distinguish:

- cloud model output
- local model output
- conservative fallback output

## Current Deployment Status

### What works

- grounded prompt contract
- deterministic knowledge-card retrieval
- SFT dataset builder from Spotter artifacts
- Modal train/evaluate/merge/publish pipeline
- verifier integration
- fallback summary integration
- runtime metadata reporting in UI

### What is currently failing

The published merged Hugging Face repo for the fine-tuned coach model is still not accepted by the
Hugging Face inference route currently used in `HFInferenceCoachSummaryModel`.

Observed runtime failure:

- `The requested model 'build-small-hackathon/spotter-coach-summary1' is not a chat model.`

Because of that, the app falls back even when the runtime resolves the correct repo ID.

### Practical implication

The app runtime defaults to the fine-tuned coach-summary model:

- `build-small-hackathon/spotter-coach-summary1`

The deterministic fallback summary remains enabled because hosted inference can still be
unavailable, reject a model route, or return output that fails schema validation. If needed,
`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` can still be used as an explicit base-model override.

## Assessment

### Strengths

- The task definition is well-scoped and matches the product.
- The prompt is strongly grounded on structured artifacts.
- The verifier is aligned with real product risk, not just language quality.
- The dataset builder is tightly coupled to actual runtime contracts.
- The pipeline supports both model generation and deterministic fallback.

### Weaknesses

- The grounded SFT dataset is still very small.
- Schema reliability is not yet strong enough.
- Verifier pass rate is currently too low for production dependence.
- Published merged artifacts are not yet working cleanly with the current Hugging Face inference strategy.
- Modal config reuse can accidentally preserve stale training settings if old volume artifacts are not cleaned.

## Recommendations

### Short term

1. Use `build-small-hackathon/spotter-coach-summary1` as the default runtime coach model.
2. Keep the fallback summary enabled in production.
3. Expand the grounded SFT dataset from more real runs before increasing training complexity.
4. Add stronger output-format controls or post-processing to improve JSON validity.
5. Clean or version Modal model volumes when switching base models.

### Medium term

1. Grow the Spotter-native dataset to at least 100 to 300 high-quality examples.
2. Evaluate `epochs=1` vs `epochs=2` and `style_weight=0.1` vs `0.2`.
3. Add targeted eval slices:
   - no-issue cases
   - valid-variation cases
   - multi-issue cases
   - low-confidence / low-pose-coverage cases
4. Improve inference compatibility for the merged repo or support local merged-model inference directly.

## Reproduction Commands

Build the grounded SFT dataset from run artifacts:

```bash
uv run python scripts/build_coach_summary_sft_dataset.py
```

Run the full Modal training flow:

```bash
uv run modal run scripts/coach_summary_modal.py --stage all --epochs 2 --style-weight 0.2 --repo-id build-small-hackathon/spotter-coach-summary1
```

Run the app with the default fine-tuned runtime model:

```bash
unset SPOTTER_COACH_SUMMARY_MODEL
uv run python app.py
```

Or set the runtime model explicitly:

```bash
export SPOTTER_COACH_SUMMARY_MODEL=build-small-hackathon/spotter-coach-summary1
uv run python app.py
```

## Conclusion

Spotter's coach-summary model training architecture is directionally strong. The core design choice
to train a grounded summary model on structured evidence, then gate it with a verifier, is the
right one for this product.

The current bottleneck is not lack of infrastructure. It is model reliability and inference
deployment maturity:

- too few grounded examples
- low schema-compliance rate
- no verifier passes in the latest visible eval
- published merged repo still not usable through the current HF inference path

The codebase is ready for the next iteration. The next high-leverage step is to improve dataset
size and output reliability while keeping the verifier and fallback architecture intact.
