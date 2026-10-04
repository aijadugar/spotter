from __future__ import annotations

import json
import inspect
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import modal


APP_NAME = "spotter-coach-summary"
DEFAULT_HF_REPO_NAME = "spotter-coach-summary"
HF_REPO_ID_ENV = "SPOTTER_COACH_SUMMARY_HF_REPO_ID"
HF_MERGED_REPO_ID_ENV = "SPOTTER_COACH_SUMMARY_MERGED_HF_REPO_ID"
HF_PRIVATE_ENV = "SPOTTER_COACH_SUMMARY_HF_PRIVATE"
RUNTIME_MODEL_ENV = "SPOTTER_COACH_SUMMARY_MODEL"
DATA_ROOT = Path("/data")
MODEL_ROOT = Path("/models")
ROOT_DATA = Path("/root/data")
ROOT_CONFIGS = Path("/root/configs")
SFT_ROOT = DATA_ROOT / "sft"
MODEL_CARD_PATH = MODEL_ROOT / "README.md"
DEFAULT_CONFIG_PATH = ROOT_CONFIGS / "coach_summary_lora.default.json"
TRAINING_CONFIG_PATH = MODEL_ROOT / "training_config.json"
TRAINING_SUMMARY_PATH = MODEL_ROOT / "training_summary.json"
EVALUATION_PATH = MODEL_ROOT / "evaluation.json"
HF_UPLOAD_PATH = MODEL_ROOT / "hf_upload.json"
MERGE_SUMMARY_PATH = MODEL_ROOT / "merge_summary.json"
HF_MERGED_UPLOAD_PATH = MODEL_ROOT / "hf_merged_upload.json"
DEFAULT_ADAPTER_DIR = MODEL_ROOT / "adapter"
DEFAULT_MERGED_DIR = MODEL_ROOT / "merged_model"
HF_METADATA_FILENAMES = (
    "training_config.json",
    "training_summary.json",
    "evaluation.json",
    "hf_upload.json",
    "merge_summary.json",
    "hf_merged_upload.json",
)
HF_DATA_FILENAMES = (
    "coach_summary_train.jsonl",
    "coach_summary_eval.jsonl",
    "public_fitness_style.jsonl",
)
TRAINING_GPU = "A100-80GB"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("build-essential", "git", "ninja-build")
    .env(
        {
            "CC": "/usr/bin/gcc",
            "CXX": "/usr/bin/g++",
            "CUDA_HOME": "/usr/local/cuda",
            "MAX_JOBS": "4",
            "TORCH_CUDA_ARCH_LIST": "8.0",
        }
    )
    .pip_install(
        "accelerate==1.14.0",
        "bitsandbytes>=0.48.0",
        "datasets>=2.20.0",
        "huggingface-hub>=0.24.0",
        "packaging>=24.0",
        "peft==0.12.0",
        "setuptools>=69.0.0",
        "torch==2.11.0",
        "transformers==5.12.0",
        "wheel>=0.43.0",
    )
    .pip_install("causal-conv1d>=1.5.0", extra_options="--no-build-isolation")
    .pip_install("mamba-ssm>=2.2.4", extra_options="--no-build-isolation")
    .add_local_dir("src", "/root/src", copy=True)
    .add_local_dir("data", "/root/data", copy=True)
    .add_local_dir("configs", "/root/configs", copy=True)
)

app = modal.App(APP_NAME, image=image)
data_volume = modal.Volume.from_name(
    "spotter-coach-summary-data", create_if_missing=True, version=2
)
model_volume = modal.Volume.from_name(
    "spotter-coach-summary-models", create_if_missing=True, version=2
)


def _load_local_env_vars(filename: str = ".env") -> dict[str, str]:
    candidates = (
        Path.cwd() / filename,
        Path(__file__).resolve().parents[1] / filename,
    )
    values: dict[str, str] = {}
    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            if not key:
                continue
            values[key] = value.strip().strip("'").strip('"')
    return values


def _hf_secret() -> modal.Secret:
    env_values = _load_local_env_vars()
    secret_payload: dict[str, str] = {}
    for key in (
        "HF_TOKEN",
        HF_REPO_ID_ENV,
        HF_MERGED_REPO_ID_ENV,
        HF_PRIVATE_ENV,
        RUNTIME_MODEL_ENV,
    ):
        value = os.getenv(key, env_values.get(key))
        if value is not None and str(value).strip():
            secret_payload[key] = str(value).strip()
    return modal.Secret.from_dict(secret_payload)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} contains a non-object JSONL row")
        rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _env_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _supports_kwarg(callable_obj: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters


def _filtered_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in parameters}


def _load_config(*, include_saved_training_config: bool = True) -> dict[str, Any]:
    config = _read_json(DEFAULT_CONFIG_PATH)
    if include_saved_training_config and TRAINING_CONFIG_PATH.exists():
        config.update(_read_json(TRAINING_CONFIG_PATH))
    return config


def _make_generation_config_greedy(model: Any) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return
    generation_config.do_sample = False
    for name in ("temperature", "top_p", "top_k", "typical_p", "epsilon_cutoff", "eta_cutoff"):
        if hasattr(generation_config, name):
            setattr(generation_config, name, None)


def _render_messages(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message["role"].capitalize()
        parts.append(f"{role}: {message['content'].strip()}")
    return "\n\n".join(parts)


def _sample_style_rows(
    *,
    style_rows: list[dict[str, Any]],
    train_count: int,
    style_weight: float,
) -> list[dict[str, Any]]:
    if not style_rows or style_weight <= 0:
        return []
    keep_count = min(len(style_rows), int(round(train_count * style_weight)))
    return style_rows[:keep_count]


def _build_training_dataset_rows(
    *,
    train_rows: list[dict[str, Any]],
    style_rows: list[dict[str, Any]],
    style_weight: float,
) -> list[dict[str, str]]:
    selected_style_rows = _sample_style_rows(
        style_rows=style_rows,
        train_count=len(train_rows),
        style_weight=style_weight,
    )
    merged = [*train_rows, *selected_style_rows]
    return [{"text": _render_messages(row["messages"])} for row in merged]


def _build_eval_dataset_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"text": _render_messages(row["messages"])} for row in rows]


def _model_card_text(
    *,
    repo_id: str,
    config: dict[str, Any],
    training_summary: dict[str, Any] | None,
    evaluation: dict[str, Any] | None,
) -> str:
    lines = [
        f"# {repo_id}",
        "",
        "LoRA adapter for Spotter grounded coach-summary generation.",
        "",
        "## Base Model",
        "",
        f"- `{config.get('base_model', 'unknown')}`",
        "",
        "## Training Data",
        "",
        f"- Train file: `{config.get('train_file')}`",
        f"- Eval file: `{config.get('eval_file')}`",
        f"- Style file: `{config.get('style_file', SFT_ROOT / 'public_fitness_style.jsonl')}`",
        "",
        "## Objective",
        "",
        "Generate grounded `coach_summary.json` outputs from Spotter structured evidence and knowledge cards.",
        "",
    ]
    if training_summary:
        lines.extend(
            [
                "## Training Summary",
                "",
                f"- Train rows: `{training_summary.get('train_row_count')}`",
                f"- Eval rows: `{training_summary.get('eval_row_count')}`",
                f"- Style rows mixed in: `{training_summary.get('style_row_count')}`",
                f"- Output dir: `{training_summary.get('output_dir')}`",
                "",
            ]
        )
    if evaluation:
        lines.extend(
            [
                "## Evaluation",
                "",
                f"- JSON validity rate: `{evaluation.get('json_validity_rate')}`",
                f"- Verifier pass rate: `{evaluation.get('verifier_pass_rate')}`",
                f"- Section completeness rate: `{evaluation.get('section_completeness_rate')}`",
                "",
            ]
        )
    return "\n".join(lines)


def _resolve_repo_id(
    api: Any,
    repo_id: str | None,
    *,
    env_names: tuple[str, ...] = (HF_REPO_ID_ENV,),
) -> str:
    if repo_id:
        return repo_id
    for env_name in env_names:
        configured = os.getenv(env_name)
        if configured:
            return configured
    try:
        whoami = api.whoami()
        if isinstance(whoami, dict):
            username = whoami.get("name") or whoami.get("fullname")
            if isinstance(username, str) and username.strip():
                return f"{username.strip()}/{DEFAULT_HF_REPO_NAME}"
    except Exception:
        pass
    return DEFAULT_HF_REPO_NAME


def _verifier_inputs_from_evidence(payload: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    sys.path.insert(0, "/root/src")
    from spotter.contracts import (
        ExerciseClassification,
        IssueMarker,
        IssueMarkers,
        Rep,
        RepAnalysis,
        RepAnalysisItem,
        Reps,
        Variation,
    )

    classification_payload = payload["exercise_classification"]
    variation_payload = payload["variation"]
    rep_summary_payload = payload["rep_summary"]
    issue_summary_payload = payload["issue_summary"]

    classification = ExerciseClassification(
        exercise=str(classification_payload["exercise"]),
        confidence=float(classification_payload["confidence"]),
        window_predictions=[],
        fallback_required=bool(classification_payload.get("fallback_required", False)),
    )
    variation = Variation(
        exercise=str(variation_payload["exercise"]),
        detected_variation=str(variation_payload["detected_variation"]),
        variation_confidence=float(variation_payload["variation_confidence"]),
        not_issues=[str(item) for item in variation_payload.get("not_issues", [])],
    )
    rep_metrics = rep_summary_payload.get("rep_metrics", [])
    reps = Reps(
        exercise=str(classification.exercise),
        reps=[
            Rep(
                rep_id=int(item.get("rep_id", index + 1)),
                start_frame=0,
                mid_frame=0,
                end_frame=0,
                start_sec=0.0,
                mid_sec=0.0,
                end_sec=float(item.get("duration_sec", 0.0)),
            )
            for index, item in enumerate(rep_metrics)
        ],
        partial_reps=[],
    )
    analysis = RepAnalysis(
        exercise=str(classification.exercise),
        items=[
            RepAnalysisItem(
                rep_id=int(item.get("rep_id", index + 1)),
                duration_sec=float(item.get("duration_sec", 0.0)),
                range_of_motion_score=float(item.get("range_of_motion_score", 0.0)),
                stability_score=float(item.get("stability_score", 0.0)),
                symmetry_score=float(item.get("symmetry_score", 0.0)),
                metrics=dict(item.get("metrics", {})),
                variation_hints=[str(value) for value in item.get("variation_hints", [])],
            )
            for index, item in enumerate(rep_metrics)
        ],
        aggregate_metrics=dict(rep_summary_payload.get("aggregate_metrics", {})),
    )
    issues = IssueMarkers(
        issues=[
            IssueMarker(
                rep_id=int(item.get("rep_id", 0)),
                issue=str(item["issue"]),
                severity=float(item.get("severity", 0.0)),
                start_frame=int(item.get("start_frame", 0)),
                end_frame=int(item.get("end_frame", 0)),
                start_sec=float(item.get("start_sec", 0.0)),
                end_sec=float(item.get("end_sec", 0.0)),
                affected_joints=[str(value) for value in item.get("affected_joints", [])],
                evidence=dict(item.get("evidence", {})),
            )
            for item in issue_summary_payload.get("issues", [])
        ]
    )
    return classification, variation, reps, analysis, issues


def _generate_json_only(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@app.function(
    volumes={str(DATA_ROOT): data_volume},
    timeout=20 * 60,
)
def prepare_data() -> dict[str, Any]:
    SFT_ROOT.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    counts: dict[str, int] = {}
    for filename in HF_DATA_FILENAMES:
        source = ROOT_DATA / "sft" / filename
        if not source.exists():
            continue
        target = SFT_ROOT / filename
        shutil.copyfile(source, target)
        copied.append(filename)
        counts[filename] = len(_read_jsonl(target))

    summary = {
        "ok": bool(copied),
        "copied_files": copied,
        "row_counts": counts,
        "sft_root": str(SFT_ROOT),
    }
    _write_json(DATA_ROOT / "prepare_data_summary.json", summary)
    data_volume.commit()
    return summary


@app.function(
    gpu=TRAINING_GPU,
    volumes={str(DATA_ROOT): data_volume, str(MODEL_ROOT): model_volume},
    secrets=[_hf_secret()],
    timeout=3 * 60 * 60,
)
def train(
    epochs: int | None = None,
    style_weight: float = 0.2,
    output_subdir: str = "adapter",
) -> dict[str, Any]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    config = _load_config(include_saved_training_config=False)
    if epochs is not None:
        config["num_train_epochs"] = epochs
    config["style_weight"] = style_weight
    config["style_file"] = str(SFT_ROOT / "public_fitness_style.jsonl")

    train_rows = _read_jsonl(SFT_ROOT / "coach_summary_train.jsonl")
    eval_rows = _read_jsonl(SFT_ROOT / "coach_summary_eval.jsonl")
    style_rows = _read_jsonl(SFT_ROOT / "public_fitness_style.jsonl")
    if not train_rows or not eval_rows:
        result = {
            "ok": False,
            "error": "Missing SFT train/eval rows. Run prepare_data first.",
        }
        _write_json(TRAINING_SUMMARY_PATH, result)
        model_volume.commit()
        return result

    training_rows = _build_training_dataset_rows(
        train_rows=train_rows,
        style_rows=style_rows,
        style_weight=style_weight,
    )
    eval_dataset_rows = _build_eval_dataset_rows(eval_rows)

    tokenizer = AutoTokenizer.from_pretrained(str(config["base_model"]))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    max_seq_length = int(config.get("max_seq_length", 2048))

    def tokenize_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, list[int]]], dict[str, Any]]:
        tokenized_rows: list[dict[str, list[int]]] = []
        lengths: list[int] = []
        truncated_count = 0
        head_tokens = min(256, max_seq_length // 4)
        tail_tokens = max_seq_length - head_tokens
        for row in rows:
            input_ids = tokenizer(
                str(row["text"]),
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            if tokenizer.eos_token_id is not None:
                input_ids = [*input_ids, int(tokenizer.eos_token_id)]
            lengths.append(len(input_ids))
            if len(input_ids) > max_seq_length:
                truncated_count += 1
                input_ids = [*input_ids[:head_tokens], *input_ids[-tail_tokens:]]
            tokenized_rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                }
            )
        stats = {
            "max_seq_length": max_seq_length,
            "truncated_row_count": truncated_count,
            "max_input_tokens_before_truncation": max(lengths) if lengths else 0,
            "avg_input_tokens_before_truncation": round(sum(lengths) / len(lengths), 2)
            if lengths
            else 0,
        }
        return tokenized_rows, stats

    tokenized_training_rows, train_token_stats = tokenize_rows(training_rows)
    tokenized_eval_rows, eval_token_stats = tokenize_rows(eval_dataset_rows)
    train_dataset = Dataset.from_list(tokenized_training_rows)
    eval_dataset = Dataset.from_list(tokenized_eval_rows)

    model_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": "sdpa",
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(config["base_model"]),
            **model_kwargs,
        )
    except (TypeError, ValueError):
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(
            str(config["base_model"]),
            **model_kwargs,
        )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, peft_config)

    adapter_dir = MODEL_ROOT / output_subdir
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    training_args_kwargs = {
        "output_dir": str(adapter_dir),
        "learning_rate": float(config["learning_rate"]),
        "num_train_epochs": float(config["num_train_epochs"]),
        "per_device_train_batch_size": int(config["per_device_train_batch_size"]),
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "save_strategy": "epoch",
        "logging_steps": 10,
        "bf16": True,
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
        "prediction_loss_only": True,
        "optim": "paged_adamw_8bit",
        "report_to": [],
    }
    if _supports_kwarg(TrainingArguments.__init__, "eval_strategy"):
        training_args_kwargs["eval_strategy"] = "epoch"
    elif _supports_kwarg(TrainingArguments.__init__, "evaluation_strategy"):
        training_args_kwargs["evaluation_strategy"] = "epoch"
    if _supports_kwarg(TrainingArguments.__init__, "gradient_checkpointing_kwargs"):
        training_args_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=TrainingArguments(
            **_filtered_kwargs(TrainingArguments.__init__, training_args_kwargs)
        ),
        data_collator=data_collator,
    )
    train_result = trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    summary = {
        "ok": True,
        "base_model": config["base_model"],
        "adapter_dir": str(adapter_dir),
        "train_row_count": len(train_rows),
        "eval_row_count": len(eval_rows),
        "style_row_count": len(training_rows) - len(train_rows),
        "merged_train_row_count": len(training_rows),
        "style_weight": style_weight,
        "epochs": config["num_train_epochs"],
        "global_step": int(getattr(train_result, "global_step", 0)),
        "training_loss": float(getattr(train_result, "training_loss", 0.0)),
        "output_dir": str(adapter_dir),
        "train_token_stats": train_token_stats,
        "eval_token_stats": eval_token_stats,
    }
    _write_json(TRAINING_CONFIG_PATH, config)
    _write_json(TRAINING_SUMMARY_PATH, summary)
    model_volume.commit()
    return summary


@app.function(
    gpu=TRAINING_GPU,
    volumes={str(DATA_ROOT): data_volume, str(MODEL_ROOT): model_volume},
    secrets=[_hf_secret()],
    timeout=90 * 60,
)
def evaluate(
    adapter_subdir: str = "adapter",
    limit: int | None = None,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/root/src")
    from spotter.steps import verifier
    from spotter.steps.coach_summary import _extract_json_object, _summary_from_payload

    config = _load_config()
    adapter_dir = MODEL_ROOT / adapter_subdir
    eval_rows = _read_jsonl(SFT_ROOT / "coach_summary_eval.jsonl")
    if limit is not None:
        eval_rows = eval_rows[:limit]
    if not adapter_dir.exists():
        result = {"ok": False, "error": f"Adapter dir not found: {adapter_dir}"}
        _write_json(EVALUATION_PATH, result)
        model_volume.commit()
        return result

    tokenizer = AutoTokenizer.from_pretrained(str(config["base_model"]))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        str(config["base_model"]),
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    json_valid_count = 0
    verifier_pass_count = 0
    section_complete_count = 0
    evaluated = 0
    failures: list[dict[str, Any]] = []
    required_sections = {
        "summary",
        "what_you_did",
        "what_looked_good",
        "what_changed_across_reps",
        "valid_variation_vs_issue",
        "top_fixes",
        "next_session_plan",
        "confidence_notes",
    }

    for index, row in enumerate(eval_rows):
        evaluated += 1
        prompt = _render_messages(row["messages"][:2])
        try:
            generated_text = _generate_json_only(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=int(config.get("max_new_tokens", 700)),
            )
            payload = _extract_json_object(generated_text)
            json_valid_count += 1
            if required_sections <= payload.keys():
                section_complete_count += 1
            summary = _summary_from_payload(payload)
            evidence_payload = json.loads(row["messages"][1]["content"])
            classification, variation, reps, analysis, issues = _verifier_inputs_from_evidence(
                evidence_payload
            )
            verification = verifier.run(
                summary,
                issues,
                variation,
                classification=classification,
                analysis=analysis,
                reps=reps,
            )
            if verification.passed:
                verifier_pass_count += 1
            else:
                failures.append(
                    {
                        "index": index,
                        "reason": "verifier_failed",
                        "notes": verification.notes,
                    }
                )
        except Exception as exc:  # pragma: no cover - runtime failure path
            failures.append({"index": index, "reason": repr(exc)})

    result = {
        "ok": True,
        "adapter_dir": str(adapter_dir),
        "evaluated_count": evaluated,
        "json_valid_count": json_valid_count,
        "json_validity_rate": round(json_valid_count / evaluated, 4) if evaluated else 0.0,
        "verifier_pass_count": verifier_pass_count,
        "verifier_pass_rate": round(verifier_pass_count / evaluated, 4) if evaluated else 0.0,
        "section_completeness_rate": round(section_complete_count / evaluated, 4)
        if evaluated
        else 0.0,
        "failure_count": len(failures),
        "failures": failures[:20],
    }
    _write_json(EVALUATION_PATH, result)
    model_volume.commit()
    return result


def _upload_hf_file(
    api: Any,
    *,
    repo_id: str,
    local_path: Path,
    path_in_repo: str,
) -> dict[str, Any]:
    if not local_path.exists():
        return {
            "path": str(local_path),
            "path_in_repo": path_in_repo,
            "uploaded": False,
            "reason": "missing",
        }
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
    )
    return {
        "path": str(local_path),
        "path_in_repo": path_in_repo,
        "uploaded": True,
    }


@app.function(
    gpu=TRAINING_GPU,
    volumes={str(MODEL_ROOT): model_volume},
    secrets=[_hf_secret()],
    timeout=90 * 60,
)
def merge(
    adapter_subdir: str = "adapter",
    merged_subdir: str = "merged_model",
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = _load_config()
    adapter_dir = MODEL_ROOT / adapter_subdir
    merged_dir = MODEL_ROOT / merged_subdir
    if not adapter_dir.exists():
        result = {"ok": False, "error": f"Adapter dir not found: {adapter_dir}"}
        _write_json(MERGE_SUMMARY_PATH, result)
        model_volume.commit()
        return result

    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    base_model = AutoModelForCausalLM.from_pretrained(
        str(config["base_model"]),
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = model.merge_and_unload()
    _make_generation_config_greedy(merged_model)
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    result = {
        "ok": True,
        "base_model": config["base_model"],
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir),
        "dtype": "bfloat16",
    }
    _write_json(MERGE_SUMMARY_PATH, result)
    model_volume.commit()
    return result


@app.function(
    volumes={str(MODEL_ROOT): model_volume, str(DATA_ROOT): data_volume},
    secrets=[_hf_secret()],
    timeout=30 * 60,
)
def publish_to_hf(
    repo_id: str | None = None,
    private: bool | None = None,
    adapter_subdir: str = "adapter",
) -> dict[str, Any]:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    private = _env_truthy(os.getenv(HF_PRIVATE_ENV)) if private is None else private
    if not os.getenv("HF_TOKEN"):
        return {
            "ok": False,
            "error": "HF_TOKEN is required in the Modal environment or local .env",
            "repo_id": repo_id or os.getenv(HF_REPO_ID_ENV) or DEFAULT_HF_REPO_NAME,
        }

    config = _read_json(TRAINING_CONFIG_PATH) if TRAINING_CONFIG_PATH.exists() else {}
    training_summary = _read_json(TRAINING_SUMMARY_PATH) if TRAINING_SUMMARY_PATH.exists() else None
    evaluation = _read_json(EVALUATION_PATH) if EVALUATION_PATH.exists() else None
    api = HfApi()
    repo_id = _resolve_repo_id(api, repo_id, env_names=(HF_REPO_ID_ENV,))
    MODEL_CARD_PATH.write_text(
        _model_card_text(
            repo_id=repo_id,
            config=config,
            training_summary=training_summary,
            evaluation=evaluation,
        ),
        encoding="utf-8",
    )

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    except HfHubHTTPError as exc:
        message = str(exc)
        guidance = (
            "Publish failed while creating or accessing the Hugging Face model repo. "
            "If your token does not have org-level write access, publish to a personal repo id "
            "such as `<your-username>/spotter-coach-summary`, or set "
            f"`{HF_REPO_ID_ENV}` in `.env` to a repo you control."
        )
        return {
            "ok": False,
            "repo_id": repo_id,
            "private": private,
            "error": message,
            "guidance": guidance,
        }
    uploads = [
        _upload_hf_file(
            api,
            repo_id=repo_id,
            local_path=MODEL_CARD_PATH,
            path_in_repo="README.md",
        )
    ]
    adapter_dir = MODEL_ROOT / adapter_subdir
    if adapter_dir.exists():
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(adapter_dir),
            path_in_repo="adapter",
        )
        uploads.append(
            {
                "path": str(adapter_dir),
                "path_in_repo": "adapter/",
                "uploaded": True,
            }
        )
    else:
        uploads.append(
            {
                "path": str(adapter_dir),
                "path_in_repo": "adapter/",
                "uploaded": False,
                "reason": "missing",
            }
        )
    uploads.extend(
        _upload_hf_file(
            api,
            repo_id=repo_id,
            local_path=MODEL_ROOT / filename,
            path_in_repo=filename,
        )
        for filename in HF_METADATA_FILENAMES
    )

    result = {
        "ok": any(item["uploaded"] for item in uploads),
        "repo_id": repo_id,
        "private": private,
        "uploads": uploads,
    }
    _write_json(HF_UPLOAD_PATH, result)
    model_volume.commit()
    return result


@app.function(
    volumes={str(MODEL_ROOT): model_volume, str(DATA_ROOT): data_volume},
    secrets=[_hf_secret()],
    timeout=60 * 60,
)
def publish_merged_to_hf(
    repo_id: str | None = None,
    private: bool | None = None,
    merged_subdir: str = "merged_model",
) -> dict[str, Any]:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    private = _env_truthy(os.getenv(HF_PRIVATE_ENV)) if private is None else private
    resolved_repo_hint = (
        repo_id
        or os.getenv(RUNTIME_MODEL_ENV)
        or os.getenv(HF_MERGED_REPO_ID_ENV)
        or DEFAULT_HF_REPO_NAME
    )
    if not os.getenv("HF_TOKEN"):
        return {
            "ok": False,
            "error": "HF_TOKEN is required in the Modal environment or local .env",
            "repo_id": resolved_repo_hint,
        }

    config = _read_json(TRAINING_CONFIG_PATH) if TRAINING_CONFIG_PATH.exists() else {}
    training_summary = _read_json(TRAINING_SUMMARY_PATH) if TRAINING_SUMMARY_PATH.exists() else None
    evaluation = _read_json(EVALUATION_PATH) if EVALUATION_PATH.exists() else None
    merge_summary = _read_json(MERGE_SUMMARY_PATH) if MERGE_SUMMARY_PATH.exists() else None
    merged_dir = MODEL_ROOT / merged_subdir
    if not merged_dir.exists():
        result = {
            "ok": False,
            "error": f"Merged model dir not found: {merged_dir}",
            "repo_id": resolved_repo_hint,
        }
        _write_json(HF_MERGED_UPLOAD_PATH, result)
        model_volume.commit()
        return result

    api = HfApi()
    repo_id = _resolve_repo_id(
        api,
        repo_id,
        env_names=(RUNTIME_MODEL_ENV, HF_MERGED_REPO_ID_ENV, HF_REPO_ID_ENV),
    )

    MODEL_CARD_PATH.write_text(
        _model_card_text(
            repo_id=repo_id,
            config=config,
            training_summary=training_summary,
            evaluation=evaluation,
        )
        + "\n## Packaging\n\n- Published as a merged, inference-ready Transformers checkpoint.\n",
        encoding="utf-8",
    )

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    except HfHubHTTPError as exc:
        message = str(exc)
        guidance = (
            "Publish failed while creating or accessing the merged Hugging Face model repo. "
            "Set `SPOTTER_COACH_SUMMARY_MODEL` or pass `--repo-id <your-username>/spotter-coach-summary` "
            "to publish to a repo your token can write to."
        )
        result = {
            "ok": False,
            "repo_id": repo_id,
            "private": private,
            "error": message,
            "guidance": guidance,
        }
        _write_json(HF_MERGED_UPLOAD_PATH, result)
        model_volume.commit()
        return result

    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(merged_dir),
    )

    uploads = [
        {
            "path": str(merged_dir),
            "path_in_repo": "./",
            "uploaded": True,
        },
        _upload_hf_file(
            api,
            repo_id=repo_id,
            local_path=MODEL_CARD_PATH,
            path_in_repo="README.md",
        ),
    ]
    uploads.extend(
        _upload_hf_file(
            api,
            repo_id=repo_id,
            local_path=MODEL_ROOT / filename,
            path_in_repo=filename,
        )
        for filename in (
            "training_config.json",
            "training_summary.json",
            "evaluation.json",
            "merge_summary.json",
        )
    )

    result = {
        "ok": True,
        "repo_id": repo_id,
        "private": private,
        "merge_summary": merge_summary,
        "uploads": uploads,
    }
    _write_json(HF_MERGED_UPLOAD_PATH, result)
    model_volume.commit()
    return result


@app.local_entrypoint()
def main(
    stage: str = "evaluate",
    epochs: int | None = None,
    style_weight: float = 0.2,
    limit: int | None = None,
    repo_id: str | None = None,
    private: bool | None = None,
) -> None:
    if stage == "prepare-data":
        print(prepare_data.remote())
    elif stage == "train":
        print(train.remote(epochs=epochs, style_weight=style_weight))
    elif stage == "evaluate":
        print(evaluate.remote(limit=limit))
    elif stage == "merge":
        print(merge.remote())
    elif stage == "publish":
        print(publish_to_hf.remote(repo_id=repo_id, private=private))
    elif stage == "publish-merged":
        print(publish_merged_to_hf.remote(repo_id=repo_id, private=private))
    elif stage == "all":
        print(prepare_data.remote())
        print(train.remote(epochs=epochs, style_weight=style_weight))
        print(evaluate.remote(limit=limit))
        print(merge.remote())
        print(publish_merged_to_hf.remote(repo_id=repo_id, private=private))
    else:
        raise ValueError(
            "stage must be one of: prepare-data, train, evaluate, merge, publish, publish-merged, all"
        )
