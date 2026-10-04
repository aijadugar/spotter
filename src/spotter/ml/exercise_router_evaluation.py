from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from spotter.ml.exercise_router_features import ROUTER_LABELS


@dataclass(frozen=True)
class RouterEvaluation:
    accuracy: float
    precision: dict[str, float]
    recall: dict[str, float]
    unknown_rejection_rate: float
    confusion_matrix: dict[str, dict[str, int]]


def evaluate_router_predictions(
    true_labels: list[str],
    predicted_labels: list[str],
) -> RouterEvaluation:
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have the same length")

    matrix = {
        actual: {predicted: 0 for predicted in ROUTER_LABELS}
        for actual in ROUTER_LABELS
    }
    for actual, predicted in zip(true_labels, predicted_labels, strict=False):
        actual_label = actual if actual in ROUTER_LABELS else "unknown"
        predicted_label = predicted if predicted in ROUTER_LABELS else "unknown"
        matrix[actual_label][predicted_label] += 1

    total = len(true_labels)
    correct = sum(matrix[label][label] for label in ROUTER_LABELS)
    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    for label in ROUTER_LABELS:
        predicted_total = sum(matrix[actual][label] for actual in ROUTER_LABELS)
        actual_total = sum(matrix[label].values())
        precision[label] = round(matrix[label][label] / predicted_total, 4) if predicted_total else 0.0
        recall[label] = round(matrix[label][label] / actual_total, 4) if actual_total else 0.0

    unknown_total = sum(matrix["unknown"].values())
    unknown_rejected = matrix["unknown"]["unknown"]
    return RouterEvaluation(
        accuracy=round(correct / total, 4) if total else 0.0,
        precision=precision,
        recall=recall,
        unknown_rejection_rate=(
            round(unknown_rejected / unknown_total, 4) if unknown_total else 0.0
        ),
        confusion_matrix=matrix,
    )


def evaluation_to_dict(evaluation: RouterEvaluation) -> dict[str, Any]:
    return {
        "accuracy": evaluation.accuracy,
        "precision": evaluation.precision,
        "recall": evaluation.recall,
        "unknown_rejection_rate": evaluation.unknown_rejection_rate,
        "confusion_matrix": evaluation.confusion_matrix,
    }


def select_router_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not candidates:
        raise ValueError("At least one router candidate is required")
    return max(candidates, key=router_candidate_sort_key)


def router_candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[float, float, int]:
    return (
        1 if candidate.get("name") == "temporal" else 0,
        float(candidate.get("accuracy", 0.0)),
        float(candidate.get("unknown_rejection_rate", 0.0)),
    )
