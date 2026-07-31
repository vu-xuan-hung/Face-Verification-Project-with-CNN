"""PAD threshold selection and locked-test metrics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def error_rates(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = scores > threshold
    fake = labels == 0
    real = labels == 1
    apcer = float(np.mean(predictions[fake])) if fake.any() else float("nan")
    bpcer = float(np.mean(~predictions[real])) if real.any() else float("nan")
    return {"apcer": apcer, "bpcer": bpcer, "acer": (apcer + bpcer) / 2}


def select_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    max_apcer: float | None = None,
) -> tuple[float, dict[str, float]]:
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    evaluated = [(float(value), error_rates(labels, scores, float(value))) for value in candidates]
    if max_apcer is not None:
        feasible = [item for item in evaluated if item[1]["apcer"] <= max_apcer]
        if not feasible:
            raise ValueError("Validation cannot satisfy max APCER")
        return min(feasible, key=lambda item: (item[1]["bpcer"], item[0]))
    return min(evaluated, key=lambda item: (item[1]["acer"], item[0]))


def evaluate_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    predictions = (scores > threshold).astype(int)
    rates = error_rates(labels, scores, threshold)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "threshold": threshold,
        "sample_count": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        **rates,
    }


def bootstrap_metric_ci(
    rows: list[dict[str, str]],
    threshold: float,
    iterations: int = 500,
    seed: int = 42,
) -> dict[str, list[float]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = "|".join(row.get(field, "") for field in ("subject_id", "session_id", "clip_id"))
        has_group = any(row.get(field, "") for field in ("subject_id", "session_id", "clip_id"))
        groups[key if has_group else row["sample_id"]].append(index)
    group_values = list(groups.values())
    rng = np.random.default_rng(seed)
    metrics = {name: [] for name in ("apcer", "bpcer", "acer", "roc_auc")}
    for _ in range(iterations):
        sampled = rng.integers(0, len(group_values), size=len(group_values))
        indexes = [index for group in sampled for index in group_values[group]]
        labels = np.asarray([int(rows[index]["label"]) for index in indexes])
        scores = np.asarray([float(rows[index]["score"]) for index in indexes])
        if len(set(labels)) < 2:
            continue
        rates = error_rates(labels, scores, threshold)
        for name in ("apcer", "bpcer", "acer"):
            metrics[name].append(rates[name])
        metrics["roc_auc"].append(float(roc_auc_score(labels, scores)))
    if not metrics["roc_auc"]:
        raise ValueError("Cannot bootstrap PAD metrics with one-class groups")
    return {
        name: [float(value) for value in np.quantile(values, [0.025, 0.975])]
        for name, values in metrics.items()
    }
