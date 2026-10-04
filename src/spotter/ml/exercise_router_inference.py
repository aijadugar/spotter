from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from spotter.env import load_local_env
from spotter.hf_spaces import router_torch_device
from spotter.ml.exercise_router_features import (
    FEATURE_SCHEMA,
    ROUTER_LABELS,
    ROUTER_LANDMARK_SCHEMA,
    window_tensor_feature_names,
    window_vector_feature_names,
    RouterWindow,
)
from spotter.ml.exercise_router_temporal import TorchTemporalRouter


DEFAULT_MODEL_DIR = Path("models/exercise_router/active")
ACTIVE_SELECTION_FILENAME = "router_selection.json"
DEFAULT_HF_REPO_ID = "aijadugar/spotter-exercise-router"
HF_REPO_ID_ENV = "SPOTTER_ROUTER_HF_REPO_ID"
HF_REVISION_ENV = "SPOTTER_ROUTER_HF_REVISION"
HF_DISABLE_ENV = "SPOTTER_ROUTER_DISABLE_HF"
MODEL_FILENAMES = (
    "router.pt",
    "temporal.pt",
    "router.joblib",
    "model.joblib",
    "baseline.joblib",
    "exercise_router.joblib",
)
MIN_FINAL_CONFIDENCE = 0.65
MIN_WINNING_AGREEMENT = 0.60
MIN_TOP_SCORE_MARGIN = 0.15
MIN_POSE_VALID_RATIO = 0.60


@dataclass(frozen=True)
class RouterModelBundle:
    model: Any
    labels: tuple[str, ...] = ROUTER_LABELS
    scaler: Any | None = None
    model_kind: str = "baseline"
    feature_schema: str = FEATURE_SCHEMA
    landmark_schema: str = ROUTER_LANDMARK_SCHEMA


@dataclass(frozen=True)
class WindowRouterPrediction:
    start_sec: float
    end_sec: float
    label: str
    confidence: float
    scores: dict[str, float]


@dataclass(frozen=True)
class AggregatedRouterPrediction:
    label: str
    confidence: float
    fallback_required: bool
    winning_agreement: float
    score_margin: float


def load_router_model(model_dir: Path = DEFAULT_MODEL_DIR) -> RouterModelBundle | None:
    try:
        hf_bundle = load_router_model_from_hf()
    except Exception:
        hf_bundle = None
    if hf_bundle is not None:
        return hf_bundle

    selected_path = _selected_artifact_path(model_dir)
    if selected_path is not None and selected_path.exists():
        return load_router_model_file(selected_path)

    for filename in MODEL_FILENAMES:
        path = model_dir / filename
        if not path.exists():
            continue
        return load_router_model_file(path)
    return None


def load_router_model_from_hf(
    repo_id: str | None = None,
    revision: str | None = None,
) -> RouterModelBundle | None:
    load_local_env()
    if _env_truthy(os.getenv(HF_DISABLE_ENV)):
        return None

    repo_id = repo_id or os.getenv(HF_REPO_ID_ENV) or DEFAULT_HF_REPO_ID
    if not repo_id:
        return None

    revision = revision or os.getenv(HF_REVISION_ENV) or None
    selection_path = _download_hf_artifact(
        repo_id=repo_id,
        filename=ACTIVE_SELECTION_FILENAME,
        revision=revision,
        required=False,
    )
    if selection_path is not None:
        selected_artifact = _selected_artifact_name(selection_path)
        if selected_artifact:
            artifact_path = _download_hf_artifact(
                repo_id=repo_id,
                filename=selected_artifact,
                revision=revision,
                required=True,
            )
            if artifact_path is not None:
                return load_router_model_file(artifact_path)

    for filename in MODEL_FILENAMES:
        artifact_path = _download_hf_artifact(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            required=False,
        )
        if artifact_path is not None:
            return load_router_model_file(artifact_path)
    return None


def load_router_model_file(path: Path) -> RouterModelBundle:
    if path.suffix == ".pt":
        return _load_temporal_router_model_file(path)

    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("joblib is required to load exercise router artifacts") from exc

    artifact = joblib.load(path)
    if isinstance(artifact, RouterModelBundle):
        return artifact
    if isinstance(artifact, dict):
        model = artifact.get("model")
        if model is None:
            raise ValueError(f"Router artifact {path} is missing a 'model' entry")
        model_kind = str(artifact.get("model_kind", artifact.get("kind", "baseline")))
        _validate_router_artifact_metadata(artifact, model_kind=model_kind, path=path)
        label_source = getattr(model, "classes_", None)
        if label_source is None:
            label_source = artifact.get("labels") or artifact.get("classes")
        labels = _labels_from_artifact(label_source, model)
        return RouterModelBundle(
            model=model,
            labels=labels,
            scaler=artifact.get("scaler"),
            model_kind=model_kind,
            feature_schema=str(artifact.get("feature_schema", "")),
            landmark_schema=str(artifact.get("landmark_schema", "")),
        )
    raise ValueError(
        f"Router artifact {path} is missing required schema metadata for {FEATURE_SCHEMA}"
    )


def _load_temporal_router_model_file(path: Path) -> RouterModelBundle:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("torch is required to load temporal exercise router artifacts") from exc

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _validate_router_artifact_metadata(checkpoint, model_kind="temporal", path=path)
    labels = _labels_from_artifact(checkpoint.get("labels"), checkpoint)
    model = TorchTemporalRouter(
        checkpoint=checkpoint,
        labels=labels,
        device=router_torch_device(),
    )
    return RouterModelBundle(
        model=model,
        labels=labels,
        scaler=None,
        model_kind=str(checkpoint.get("model_kind", "temporal")),
        feature_schema=str(checkpoint.get("feature_schema", "")),
        landmark_schema=str(checkpoint.get("landmark_schema", "")),
    )


def _validate_router_artifact_metadata(
    artifact: dict[str, Any],
    *,
    model_kind: str,
    path: Path,
) -> None:
    feature_schema = artifact.get("feature_schema")
    landmark_schema = artifact.get("landmark_schema")
    if feature_schema != FEATURE_SCHEMA or landmark_schema != ROUTER_LANDMARK_SCHEMA:
        raise ValueError(
            f"Router artifact {path} uses incompatible schema: "
            f"feature_schema={feature_schema!r}, landmark_schema={landmark_schema!r}"
        )

    input_size = artifact.get("input_size")
    if input_size is None:
        return
    expected_size = (
        len(window_tensor_feature_names())
        if model_kind == "temporal"
        else len(window_vector_feature_names())
    )
    if int(input_size) != expected_size:
        raise ValueError(
            f"Router artifact {path} input_size={input_size} does not match {expected_size}"
        )


def predict_window_probabilities(
    bundle: RouterModelBundle,
    windows: list[RouterWindow],
) -> list[dict[str, float]]:
    if not windows:
        return []

    labels = _labels_from_artifact(bundle.labels, bundle.model)
    if bundle.feature_schema != FEATURE_SCHEMA or bundle.landmark_schema != ROUTER_LANDMARK_SCHEMA:
        raise ValueError("Router model schema does not match current pose feature schema")
    if bundle.model_kind == "temporal":
        inputs = np.stack([window.tensor for window in windows]).astype(np.float32)
    else:
        inputs = np.stack([window.vector for window in windows]).astype(np.float32)
        if bundle.scaler is not None:
            inputs = bundle.scaler.transform(inputs)

    if hasattr(bundle.model, "predict_proba"):
        raw_scores = np.asarray(bundle.model.predict_proba(inputs), dtype=np.float64)
    elif hasattr(bundle.model, "decision_function"):
        raw_scores = _softmax(np.asarray(bundle.model.decision_function(inputs), dtype=np.float64))
    else:
        predictions = bundle.model.predict(inputs)
        raw_scores = _one_hot_scores(predictions, labels)

    return [_score_map(row, labels) for row in raw_scores]


def window_predictions_from_scores(
    windows: list[RouterWindow],
    score_rows: list[dict[str, float]],
) -> list[WindowRouterPrediction]:
    predictions: list[WindowRouterPrediction] = []
    for window, scores in zip(windows, score_rows, strict=False):
        label = max(ROUTER_LABELS, key=lambda item: scores.get(item, 0.0))
        confidence = max(0.0, min(1.0, scores.get(label, 0.0)))
        predictions.append(
            WindowRouterPrediction(
                start_sec=window.start_sec,
                end_sec=window.end_sec,
                label=label,
                confidence=round(confidence, 4),
                scores={
                    key: round(max(0.0, min(1.0, scores.get(key, 0.0))), 6) for key in ROUTER_LABELS
                },
            )
        )
    return predictions


def aggregate_window_predictions(
    predictions: list[WindowRouterPrediction],
) -> AggregatedRouterPrediction:
    if not predictions:
        return AggregatedRouterPrediction(
            label="unknown",
            confidence=0.0,
            fallback_required=True,
            winning_agreement=0.0,
            score_margin=0.0,
        )

    score_totals = {
        label: sum(prediction.scores.get(label, 0.0) for prediction in predictions)
        for label in ROUTER_LABELS
    }
    ranked_labels = sorted(ROUTER_LABELS, key=lambda label: score_totals[label], reverse=True)
    winning_label = ranked_labels[0]
    second_label = ranked_labels[1]
    window_count = len(predictions)
    winning_score = score_totals[winning_label] / window_count
    second_score = score_totals[second_label] / window_count
    winning_agreement = (
        sum(1 for prediction in predictions if prediction.label == winning_label) / window_count
    )
    winning_confidences = [prediction.scores.get(winning_label, 0.0) for prediction in predictions]
    confidence = min(winning_score, sum(winning_confidences) / len(winning_confidences))
    score_margin = winning_score - second_score
    fallback_required = (
        confidence < MIN_FINAL_CONFIDENCE
        or winning_agreement < MIN_WINNING_AGREEMENT
        or score_margin < MIN_TOP_SCORE_MARGIN
    )
    label = "unknown" if fallback_required and winning_label != "unknown" else winning_label
    return AggregatedRouterPrediction(
        label=label,
        confidence=round(max(0.0, min(1.0, confidence)), 4),
        fallback_required=fallback_required,
        winning_agreement=round(winning_agreement, 4),
        score_margin=round(max(0.0, score_margin), 4),
    )


def contract_window_predictions(
    predictions: list[WindowRouterPrediction],
) -> list[dict[str, float | str]]:
    return [
        {
            "start_sec": prediction.start_sec,
            "end_sec": prediction.end_sec,
            "label": prediction.label,
            "confidence": prediction.confidence,
        }
        for prediction in predictions
    ]


def _labels_from_artifact(labels: Any, model: Any) -> tuple[str, ...]:
    if labels is None and hasattr(model, "classes_"):
        labels = model.classes_
    if labels is None:
        return ROUTER_LABELS
    normalized = tuple(str(label) for label in labels)
    return normalized or ROUTER_LABELS


def _selected_artifact_path(model_dir: Path) -> Path | None:
    selection_path = model_dir / ACTIVE_SELECTION_FILENAME
    if not selection_path.exists():
        return None
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_artifact = selection.get("selected_artifact") or selection.get("selected_model")
    if not isinstance(selected_artifact, str) or not selected_artifact:
        return None
    return model_dir / selected_artifact


def _selected_artifact_name(selection_path: Path) -> str | None:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_artifact = selection.get("selected_artifact") or selection.get("selected_model")
    if not isinstance(selected_artifact, str) or not selected_artifact:
        return None
    return selected_artifact


def _download_hf_artifact(
    *,
    repo_id: str,
    filename: str,
    revision: str | None,
    required: bool,
) -> Path | None:
    try:
        return _hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    except Exception:
        if required:
            raise
        return None


def _hf_hub_download(*, repo_id: str, filename: str, revision: str | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError(
            "huggingface_hub is required to load router artifacts from the Hub"
        ) from exc

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            revision=revision,
        )
    )


def _env_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "on", "true", "yes"}


def _score_map(row: np.ndarray, labels: tuple[str, ...]) -> dict[str, float]:
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    if row.size == 1 and len(labels) == 2:
        row = np.asarray([1.0 - row[0], row[0]], dtype=np.float64)
    scores = {label: 0.0 for label in ROUTER_LABELS}
    for label, score in zip(labels, row, strict=False):
        if label in scores:
            scores[label] = max(0.0, float(score))
    total = sum(scores.values())
    if total <= 1e-12:
        scores["unknown"] = 1.0
        return scores
    return {label: score / total for label, score in scores.items()}


def _softmax(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def _one_hot_scores(predictions: Any, labels: tuple[str, ...]) -> np.ndarray:
    rows: list[list[float]] = []
    for prediction in predictions:
        rows.append([1.0 if str(prediction) == label else 0.0 for label in labels])
    return np.asarray(rows, dtype=np.float64)
