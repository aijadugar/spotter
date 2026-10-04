from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
import zipfile

import modal


APP_NAME = "spotter-exercise-router"
DATASET_ID = "RickyRiccio/Real_Time_Exercise_Recognition_Dataset"
DEFAULT_HF_REPO_ID = "aijadugar/spotter-exercise-router"
HF_REPO_ID_ENV = "SPOTTER_ROUTER_HF_REPO_ID"
HF_PRIVATE_ENV = "SPOTTER_ROUTER_HF_PRIVATE"
DATA_ROOT = Path("/data")
MODEL_ROOT = Path("/models")
DOCS_ROOT = Path("/root/docs")
RAW_ROOT = DATA_ROOT / "raw"
RICCIO_ROOT = RAW_ROOT / "riccio"
CUSTOM_UNKNOWN_ROOT = RAW_ROOT / "custom_unknown"
MANIFEST_PATH = DATA_ROOT / "manifests" / "router_examples.jsonl"
FEATURE_MANIFEST_PATH = DATA_ROOT / "features" / "feature_manifest.jsonl"
VIDEO_SUFFIXES = {".avi", ".m4v", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
ARCHIVE_SUFFIXES = {".zip"}
RICCIO_VIDEO_COLLECTIONS = (
    "final_kaggle_with_additional_video",
    "my_test_video_1",
)
HF_ARTIFACT_FILENAMES = (
    "baseline.joblib",
    "router.joblib",
    "router_selection.json",
    "temporal.pt",
    "evaluation.json",
    "baseline_metrics.json",
    "temporal_metrics.json",
)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libegl1", "libgl1", "libgles2", "libglib2.0-0")
    .pip_install(
        "huggingface-hub>=0.24.0",
        "joblib==1.5.3",
        "mediapipe>=0.10.35",
        "numpy==1.26.4",
        "opencv-python-headless>=4.10.0",
        "scikit-learn==1.7.2",
        "scipy==1.15.3",
        "torch==2.11.0",
    )
    .add_local_dir("src", "/root/src", copy=True)
    .add_local_file(
        "docs/22-router-huggingface-model-card.md",
        "/root/docs/22-router-huggingface-model-card.md",
        copy=True,
    )
    .add_local_file(
        "docs/20-router-training-report.md",
        "/root/docs/20-router-training-report.md",
        copy=True,
    )
)

app = modal.App(APP_NAME, image=image)
data_volume = modal.Volume.from_name(
    "spotter-router-data", create_if_missing=True, version=2
)
model_volume = modal.Volume.from_name(
    "spotter-router-models", create_if_missing=True, version=2
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _env_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _video_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path for path in root.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES
    )


def _riccio_video_files(root: Path) -> list[Path]:
    files: list[Path] = []
    dataset_roots = [
        root / "Real-Time Exercise Recognition Dataset",
        root
        / "Real-Time Exercise Recognition Dataset"
        / "Real-Time Exercise Recognition Dataset",
        root,
    ]
    for collection_name in RICCIO_VIDEO_COLLECTIONS:
        for dataset_root in dataset_roots:
            collection_root = dataset_root / collection_name
            if not collection_root.exists():
                continue
            for class_dir in sorted(
                path for path in collection_root.iterdir() if path.is_dir()
            ):
                files.extend(_video_files(class_dir))
    if files:
        return sorted(files)
    return _video_files(root)


def _extract_archives(root: Path) -> list[str]:
    extracted: list[str] = []
    for archive_path in sorted(root.rglob("*")):
        if archive_path.suffix.lower() not in ARCHIVE_SUFFIXES:
            continue
        target_dir = archive_path.with_suffix("")
        marker = target_dir / ".spotter_extracted"
        if marker.exists():
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        if any(target_dir.iterdir()):
            marker.write_text("ok\n", encoding="utf-8")
            continue
        if archive_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(target_dir)
        marker.write_text("ok\n", encoding="utf-8")
        extracted.append(str(archive_path))
    return extracted


def _label_from_path(path: Path) -> str:
    sys.path.insert(0, "/root/src")
    from spotter.ml.exercise_router_features import normalize_router_label

    for part in reversed(path.parts):
        label = normalize_router_label(part)
        if label != "unknown":
            return label
        if any(
            token in part.lower()
            for token in ("unknown", "curl", "idle", "walk", "stretch")
        ):
            return "unknown"
    return "unknown"


def _example_id(path: Path) -> str:
    safe = "_".join(path.relative_to(RAW_ROOT).parts)
    return "".join(
        char if char.isalnum() or char in {"_", "-", "."} else "_" for char in safe
    )


@app.function(
    volumes={str(DATA_ROOT): data_volume},
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="any",
)
def ingest() -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    RICCIO_ROOT.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=RICCIO_ROOT,
        local_dir_use_symlinks=False,
    )
    extracted_archives = _extract_archives(RICCIO_ROOT)

    examples: list[dict[str, Any]] = []
    for video_path in [
        *_riccio_video_files(RICCIO_ROOT),
        *_video_files(CUSTOM_UNKNOWN_ROOT),
    ]:
        source = (
            "custom_unknown" if CUSTOM_UNKNOWN_ROOT in video_path.parents else "riccio"
        )
        examples.append(
            {
                "id": _example_id(video_path),
                "video_path": str(video_path),
                "label": (
                    "unknown"
                    if source == "custom_unknown"
                    else _label_from_path(video_path)
                ),
                "source": source,
            }
        )

    _write_jsonl(MANIFEST_PATH, examples)
    _write_json(
        DATA_ROOT / "manifests" / "ingest_summary.json",
        {
            "dataset_id": DATASET_ID,
            "example_count": len(examples),
            "extracted_archives": extracted_archives,
            "labels": sorted({example["label"] for example in examples}),
            "custom_unknown_root": str(CUSTOM_UNKNOWN_ROOT),
        },
    )
    data_volume.commit()
    return {"example_count": len(examples), "manifest_path": str(MANIFEST_PATH)}


@app.function(volumes={str(DATA_ROOT): data_volume})
def load_feature_examples(limit: int | None = None) -> list[dict[str, Any]]:
    examples = _read_jsonl(MANIFEST_PATH)
    return examples[:limit] if limit else examples


@app.function(volumes={str(DATA_ROOT): data_volume}, timeout=30 * 60)
def extract_example_features(example: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, "/root/src")

    import numpy as np

    from spotter.ml.exercise_router_features import (
        FEATURE_SCHEMA,
        ROUTER_LANDMARK_SCHEMA,
        extract_router_windows,
        window_tensor_feature_names,
        window_vector_feature_names,
    )
    from spotter.steps import pose_cleaning, pose_landmarker, video_qc

    video_path = Path(example["video_path"])
    manifest = video_qc.run(str(video_path))
    raw_sequence = pose_landmarker.run(manifest, mock=False)
    sequence = pose_cleaning.run(raw_sequence)
    windows = extract_router_windows(sequence)

    feature_path = DATA_ROOT / "features" / f"{example['id']}.npz"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    if windows:
        np.savez_compressed(
            feature_path,
            vectors=np.stack([window.vector for window in windows]),
            tensors=np.stack([window.tensor for window in windows]),
            label=example["label"],
            feature_schema=FEATURE_SCHEMA,
            landmark_schema=ROUTER_LANDMARK_SCHEMA,
            vector_feature_names=np.asarray(window_vector_feature_names()),
            tensor_feature_names=np.asarray(window_tensor_feature_names()),
        )
    else:
        np.savez_compressed(
            feature_path,
            vectors=np.empty((0, 0), dtype=np.float32),
            tensors=np.empty((0, 0, 0), dtype=np.float32),
            label=example["label"],
            feature_schema=FEATURE_SCHEMA,
            landmark_schema=ROUTER_LANDMARK_SCHEMA,
            vector_feature_names=np.asarray(window_vector_feature_names()),
            tensor_feature_names=np.asarray(window_tensor_feature_names()),
        )

    data_volume.commit()
    return {
        "id": example["id"],
        "label": example["label"],
        "video_path": str(video_path),
        "feature_path": str(feature_path),
        "window_count": len(windows),
        "pose_valid_ratio": sequence.pose_valid_ratio,
        "feature_schema": FEATURE_SCHEMA,
        "landmark_schema": ROUTER_LANDMARK_SCHEMA,
    }


@app.function(volumes={str(DATA_ROOT): data_volume})
def write_feature_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    _write_jsonl(FEATURE_MANIFEST_PATH, rows)
    _write_json(
        DATA_ROOT / "features" / "feature_summary.json",
        {
            "example_count": len(rows),
            "window_count": sum(int(row.get("window_count", 0)) for row in rows),
            "failed_count": sum(1 for row in rows if not row.get("ok", True)),
            "feature_schema": rows[0].get("feature_schema") if rows else None,
            "landmark_schema": rows[0].get("landmark_schema") if rows else None,
        },
    )
    data_volume.commit()
    return {
        "feature_manifest_path": str(FEATURE_MANIFEST_PATH),
        "example_count": len(rows),
    }


def _load_feature_arrays() -> tuple[Any, Any, Any]:
    import numpy as np

    sys.path.insert(0, "/root/src")
    from spotter.ml.exercise_router_features import (
        FEATURE_SCHEMA,
        ROUTER_LANDMARK_SCHEMA,
    )

    vectors: list[np.ndarray] = []
    tensors: list[np.ndarray] = []
    labels: list[str] = []
    for row in _read_jsonl(FEATURE_MANIFEST_PATH):
        if not row.get("ok", True) or int(row.get("window_count", 0)) <= 0:
            continue
        data = np.load(row["feature_path"], allow_pickle=False)
        feature_schema = str(data["feature_schema"]) if "feature_schema" in data else ""
        landmark_schema = (
            str(data["landmark_schema"]) if "landmark_schema" in data else ""
        )
        if (
            feature_schema != FEATURE_SCHEMA
            or landmark_schema != ROUTER_LANDMARK_SCHEMA
        ):
            continue
        row_vectors = data["vectors"]
        row_tensors = data["tensors"]
        label = str(data["label"])
        for index in range(row_vectors.shape[0]):
            vectors.append(row_vectors[index])
            tensors.append(row_tensors[index])
            labels.append(label)
    if not vectors:
        return None, None, []
    return np.stack(vectors), np.stack(tensors), labels


@app.function(
    volumes={str(DATA_ROOT): data_volume, str(MODEL_ROOT): model_volume},
    timeout=30 * 60,
)
def train_baseline() -> dict[str, Any]:
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    sys.path.insert(0, "/root/src")
    from spotter.ml.exercise_router_features import (
        FEATURE_SCHEMA,
        ROUTER_LANDMARK_SCHEMA,
        window_vector_feature_names,
    )

    vectors, _tensors, labels = _load_feature_arrays()
    if vectors is None or len(set(labels)) < 2:
        result = {
            "ok": False,
            "error": "At least two labels with extracted windows are required",
        }
        _write_json(MODEL_ROOT / "baseline_metrics.json", result)
        model_volume.commit()
        return result

    label_counts = {label: labels.count(label) for label in sorted(set(labels))}
    stratify = labels if min(label_counts.values()) >= 2 else None
    train_x, valid_x, train_y, valid_y = train_test_split(
        vectors,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.08,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(train_x, train_y)
    predictions = model.predict(valid_x)
    metrics = {
        "ok": True,
        "model_kind": "baseline",
        "accuracy": round(float(accuracy_score(valid_y, predictions)), 4),
        "label_counts": label_counts,
        "window_count": int(vectors.shape[0]),
    }
    joblib.dump(
        {
            "model": model,
            "labels": [str(label) for label in model.classes_],
            "model_kind": "baseline",
            "feature_names": window_vector_feature_names(),
            "feature_schema": FEATURE_SCHEMA,
            "landmark_schema": ROUTER_LANDMARK_SCHEMA,
            "input_size": int(vectors.shape[-1]),
            "metrics": metrics,
        },
        MODEL_ROOT / "baseline.joblib",
    )
    _write_json(MODEL_ROOT / "baseline_metrics.json", metrics)
    model_volume.commit()
    return metrics


@app.function(
    gpu="A10",
    volumes={str(DATA_ROOT): data_volume, str(MODEL_ROOT): model_volume},
    timeout=60 * 60,
)
def train_temporal(epochs: int = 73) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split

    sys.path.insert(0, "/root/src")
    from spotter.ml.exercise_router_features import (
        FEATURE_SCHEMA,
        ROUTER_LABELS,
        ROUTER_LANDMARK_SCHEMA,
        window_tensor_feature_names,
    )
    from spotter.ml.exercise_router_evaluation import (
        evaluation_to_dict,
        evaluate_router_predictions,
    )
    from spotter.ml.exercise_router_temporal import (
        TEMPORAL_ARCHITECTURE,
        TEMPORAL_BATCH_SIZE,
        TEMPORAL_DROPOUT_RATE,
        TEMPORAL_HIDDEN_SIZE,
        TEMPORAL_LEARNING_RATE,
        TEMPORAL_NUM_LAYERS,
        TemporalRouterConfig,
        build_temporal_router_model,
    )

    _vectors, tensors, labels = _load_feature_arrays()
    if tensors is None or len(set(labels)) < 2:
        result = {
            "ok": False,
            "error": "At least two labels with extracted windows are required",
        }
        _write_json(MODEL_ROOT / "temporal_metrics.json", result)
        model_volume.commit()
        return result

    label_to_index = {label: index for index, label in enumerate(ROUTER_LABELS)}
    label_indices = [
        label_to_index.get(label, label_to_index["unknown"]) for label in labels
    ]
    label_counts = {label: labels.count(label) for label in sorted(set(labels))}
    stratify = labels if min(label_counts.values()) >= 2 else None
    train_x, valid_x, train_y, valid_y, _train_labels, valid_labels = train_test_split(
        tensors,
        label_indices,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )
    train_dataset = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        train_dataset, batch_size=TEMPORAL_BATCH_SIZE, shuffle=True, generator=generator
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    config = TemporalRouterConfig(
        input_size=int(tensors.shape[-1]),
        label_count=len(ROUTER_LABELS),
        hidden_size=TEMPORAL_HIDDEN_SIZE,
        num_layers=TEMPORAL_NUM_LAYERS,
        bidirectional=True,
        dropout_rate=TEMPORAL_DROPOUT_RATE,
    )
    model = build_temporal_router_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TEMPORAL_LEARNING_RATE, weight_decay=1e-4
    )
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    last_loss = 0.0
    for _epoch in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())

    valid_predictions = _predict_temporal_labels(
        model=model,
        tensors=valid_x,
        labels=ROUTER_LABELS,
        device=device,
    )
    validation = evaluation_to_dict(
        evaluate_router_predictions(list(valid_labels), valid_predictions)
    )

    torch.save(
        {
            "model_kind": "temporal",
            "architecture": TEMPORAL_ARCHITECTURE,
            "state_dict": model.cpu().state_dict(),
            "labels": list(ROUTER_LABELS),
            "input_size": int(tensors.shape[-1]),
            "feature_names": window_tensor_feature_names(),
            "feature_schema": FEATURE_SCHEMA,
            "landmark_schema": ROUTER_LANDMARK_SCHEMA,
            "hidden_size": TEMPORAL_HIDDEN_SIZE,
            "num_layers": TEMPORAL_NUM_LAYERS,
            "bidirectional": True,
            "dropout_rate": TEMPORAL_DROPOUT_RATE,
            "learning_rate": TEMPORAL_LEARNING_RATE,
            "batch_size": TEMPORAL_BATCH_SIZE,
        },
        MODEL_ROOT / "temporal.pt",
    )
    metrics = {
        "ok": True,
        "model_kind": "temporal",
        "architecture": TEMPORAL_ARCHITECTURE,
        "epochs": epochs,
        "final_training_loss": round(last_loss, 4),
        "window_count": int(tensors.shape[0]),
        "label_counts": label_counts,
        "learning_rate": TEMPORAL_LEARNING_RATE,
        "batch_size": TEMPORAL_BATCH_SIZE,
        "hidden_size": TEMPORAL_HIDDEN_SIZE,
        "dropout_rate": TEMPORAL_DROPOUT_RATE,
        "validation": validation,
    }
    _write_json(MODEL_ROOT / "temporal_metrics.json", metrics)
    model_volume.commit()
    return metrics


def _predict_temporal_labels(
    *,
    model: Any,
    tensors: Any,
    labels: tuple[str, ...],
    device: Any,
) -> list[str]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(torch.tensor(tensors, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    predictions: list[str] = []
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            logits = model(batch_x.to(device))
            predicted_indices = torch.argmax(logits, dim=1).detach().cpu().tolist()
            predictions.extend(labels[index] for index in predicted_indices)
    return predictions


def _evaluate_temporal_checkpoint(checkpoint_path: Path, tensors: Any) -> list[str]:
    import torch

    sys.path.insert(0, "/root/src")
    from spotter.ml.exercise_router_temporal import (
        TemporalRouterConfig,
        build_temporal_router_model,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    labels = tuple(str(label) for label in checkpoint["labels"])
    config = TemporalRouterConfig(
        input_size=int(checkpoint["input_size"]),
        label_count=len(labels),
        hidden_size=int(checkpoint.get("hidden_size", 64)),
        num_layers=int(checkpoint.get("num_layers", 1)),
        bidirectional=bool(checkpoint.get("bidirectional", True)),
        dropout_rate=float(checkpoint.get("dropout_rate", 0.2174)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_temporal_router_model(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return _predict_temporal_labels(
        model=model,
        tensors=tensors,
        labels=labels,
        device=device,
    )


@app.function(volumes={str(DATA_ROOT): data_volume, str(MODEL_ROOT): model_volume})
def evaluate() -> dict[str, Any]:
    import joblib

    sys.path.insert(0, "/root/src")
    from spotter.ml.exercise_router_evaluation import (
        evaluation_to_dict,
        evaluate_router_predictions,
        select_router_candidate,
    )

    vectors, tensors, labels = _load_feature_arrays()
    baseline_path = MODEL_ROOT / "baseline.joblib"
    temporal_path = MODEL_ROOT / "temporal.pt"
    if vectors is None or tensors is None:
        result = {"ok": False, "error": "Feature arrays are required"}
        _write_json(MODEL_ROOT / "evaluation.json", result)
        model_volume.commit()
        return result

    candidates: list[dict[str, Any]] = []
    model_results: dict[str, Any] = {}
    if baseline_path.exists():
        artifact = joblib.load(baseline_path)
        predictions = list(artifact["model"].predict(vectors))
        baseline_evaluation = evaluation_to_dict(
            evaluate_router_predictions(labels, predictions)
        )
        model_results["baseline"] = {
            "ok": True,
            "artifact": "baseline.joblib",
            **baseline_evaluation,
        }
        candidates.append(
            {
                "name": "baseline",
                "source_artifact": baseline_path,
                "selected_artifact": "router.joblib",
                **baseline_evaluation,
            }
        )

    if temporal_path.exists():
        predictions = _evaluate_temporal_checkpoint(temporal_path, tensors)
        temporal_evaluation = evaluation_to_dict(
            evaluate_router_predictions(labels, predictions)
        )
        model_results["temporal"] = {
            "ok": True,
            "artifact": "temporal.pt",
            **temporal_evaluation,
        }
        candidates.append(
            {
                "name": "temporal",
                "source_artifact": temporal_path,
                "selected_artifact": "temporal.pt",
                **temporal_evaluation,
            }
        )

    if not candidates:
        result = {
            "ok": False,
            "error": "At least one trained router artifact is required",
        }
        _write_json(MODEL_ROOT / "evaluation.json", result)
        model_volume.commit()
        return result

    selected = select_router_candidate(candidates)
    if baseline_path.exists():
        shutil.copyfile(baseline_path, MODEL_ROOT / "router.joblib")
    selection = {
        "selected_model": f"{selected['name']}.{ 'joblib' if selected['name'] == 'baseline' else 'pt' }",
        "selected_artifact": selected["selected_artifact"],
        "reason": "prefer BiLSTM temporal when available; baseline falls back when temporal is missing",
    }
    _write_json(MODEL_ROOT / "router_selection.json", selection)
    result = {
        "ok": True,
        "selected_model": selection["selected_model"],
        "selected_artifact": selection["selected_artifact"],
        "models": model_results,
        **{
            key: selected[key]
            for key in (
                "accuracy",
                "precision",
                "recall",
                "unknown_rejection_rate",
                "confusion_matrix",
            )
        },
    }
    _write_json(MODEL_ROOT / "evaluation.json", result)
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
    volumes={str(MODEL_ROOT): model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=20 * 60,
)
def publish_to_hf(
    repo_id: str | None = None,
    private: bool | None = None,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    repo_id = repo_id or os.getenv(HF_REPO_ID_ENV) or DEFAULT_HF_REPO_ID
    private = _env_truthy(os.getenv(HF_PRIVATE_ENV)) if private is None else private
    if not os.getenv("HF_TOKEN"):
        return {
            "ok": False,
            "error": "HF_TOKEN is required in the Modal environment or local .env",
            "repo_id": repo_id,
        }

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    uploads = [
        _upload_hf_file(
            api,
            repo_id=repo_id,
            local_path=DOCS_ROOT / "huggingface-router-model-card.md",
            path_in_repo="README.md",
        ),
        _upload_hf_file(
            api,
            repo_id=repo_id,
            local_path=DOCS_ROOT / "exercise-router-training-report.md",
            path_in_repo="training_report.md",
        ),
    ]
    uploads.extend(
        _upload_hf_file(
            api,
            repo_id=repo_id,
            local_path=MODEL_ROOT / filename,
            path_in_repo=filename,
        )
        for filename in HF_ARTIFACT_FILENAMES
    )

    result = {
        "ok": any(item["uploaded"] for item in uploads),
        "repo_id": repo_id,
        "private": private,
        "uploads": uploads,
    }
    _write_json(MODEL_ROOT / "hf_upload.json", result)
    model_volume.commit()
    return result


@app.local_entrypoint()
def main(
    stage: str = "evaluate",
    limit: int | None = None,
    epochs: int = 73,
    repo_id: str | None = None,
    private: bool | None = None,
) -> None:
    if stage == "ingest":
        print(ingest.remote())
    elif stage == "features":
        examples = load_feature_examples.remote(limit)
        rows: list[dict[str, Any]] = []
        for result in extract_example_features.map(examples, return_exceptions=True):
            if isinstance(result, Exception):
                rows.append({"ok": False, "error": repr(result)})
            else:
                rows.append({"ok": True, **result})
        print(write_feature_manifest.remote(rows))
    elif stage == "train-baseline":
        print(train_baseline.remote())
    elif stage == "train-temporal":
        print(train_temporal.remote(epochs=epochs))
    elif stage == "evaluate":
        print(evaluate.remote())
        print(publish_to_hf.remote(repo_id=repo_id, private=private))
    elif stage == "publish":
        print(publish_to_hf.remote(repo_id=repo_id, private=private))
    elif stage == "all":
        print(ingest.remote())
        examples = load_feature_examples.remote(limit)
        rows = []
        for result in extract_example_features.map(examples, return_exceptions=True):
            rows.append(
                {"ok": False, "error": repr(result)}
                if isinstance(result, Exception)
                else {"ok": True, **result}
            )
        print(write_feature_manifest.remote(rows))
        print(train_baseline.remote())
        print(train_temporal.remote(epochs=epochs))
        print(evaluate.remote())
        print(publish_to_hf.remote(repo_id=repo_id, private=private))
    else:
        raise ValueError(
            "stage must be one of: ingest, features, train-baseline, train-temporal, evaluate, publish, all"
        )
