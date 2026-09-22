"""PAD classification, ISO-style error rates, and calibration helpers."""

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
    roc_curve,
)


def error_rates(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Use production's inclusive boundary: real score >= threshold is bona fide."""
    predictions = scores >= threshold
    attacks = labels == 0
    bona_fide = labels == 1
    apcer = float(np.mean(predictions[attacks])) if attacks.any() else float("nan")
    bpcer = float(np.mean(~predictions[bona_fide])) if bona_fide.any() else float("nan")
    acer = (apcer + bpcer) / 2 if np.isfinite(apcer) and np.isfinite(bpcer) else float("nan")
    return {"apcer": apcer, "bpcer": bpcer, "acer": acer}


def select_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    max_apcer: float | None = None,
    *,
    max_bpcer: float | None = None,
) -> tuple[float, dict[str, float]]:
    candidates = _runnable_thresholds(scores)
    evaluated = [(float(value), error_rates(labels, scores, float(value))) for value in candidates]
    if max_apcer is not None:
        feasible = [item for item in evaluated if item[1]["apcer"] <= max_apcer]
        if not feasible:
            raise ValueError("Calibration cannot satisfy target APCER")
        return min(feasible, key=lambda item: (item[1]["bpcer"], item[0]))
    if max_bpcer is not None:
        feasible = [item for item in evaluated if item[1]["bpcer"] <= max_bpcer]
        if not feasible:
            raise ValueError("Calibration cannot satisfy target BPCER")
        return min(feasible, key=lambda item: (item[1]["apcer"], -item[0]))
    return min(evaluated, key=lambda item: (item[1]["acer"], item[0]))


def evaluate_scores(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    metrics: dict[str, object] = {
        "threshold": threshold,
        "sample_count": len(labels),
        "n_spoof": int(np.sum(labels == 0)),
        "n_real": int(np.sum(labels == 1)),
        "accuracy": float(accuracy_score(labels, predictions)) if len(labels) else None,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        **error_rates(labels, scores, threshold),
    }
    if len(set(labels.tolist())) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["pr_auc"] = float(average_precision_score(labels, scores))
        false_positive, true_positive, thresholds = roc_curve(labels, scores)
        false_negative = 1.0 - true_positive
        index = int(np.nanargmin(np.abs(false_positive - false_negative)))
        metrics["eer"] = float((false_positive[index] + false_negative[index]) / 2)
        metrics["eer_threshold"] = float(thresholds[index])
    else:
        metrics.update(roc_auc=None, pr_auc=None, eer=None, eer_threshold=None)
    return metrics


def threshold_sweep(labels: np.ndarray, scores: np.ndarray) -> list[dict[str, object]]:
    candidates = _runnable_thresholds(scores)
    return [evaluate_scores(labels, scores, float(value)) for value in candidates]


def _runnable_thresholds(scores: np.ndarray) -> np.ndarray:
    """Thresholds accepted by both the model contract and check_pad."""
    lower = np.nextafter(0.5, 1.0)
    upper = np.nextafter(1.0, 0.5)
    observed = scores[(scores > 0.5) & (scores < 1.0)]
    return np.unique(np.concatenate(([lower], observed, [upper])))


def per_attack_apcer(rows: list[dict], threshold: float) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("ground_truth") == "SPOOF" and row.get("pad_score") is not None:
            groups[row.get("attack_type") or "other"].append(float(row["pad_score"]))
    return {
        attack: float(np.mean(np.asarray(scores) >= threshold))
        for attack, scores in sorted(groups.items())
    }


def bootstrap_metric_ci(
    rows: list[dict[str, str]], threshold: float, iterations: int = 500, seed: int = 42
) -> dict[str, list[float]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = "|".join(row.get(field, "") for field in ("subject_id", "session_id", "clip_id"))
        groups[key if key.strip("|") else row["sample_id"]].append(index)
    group_values = list(groups.values())
    rng = np.random.default_rng(seed)
    sampled = {name: [] for name in ("apcer", "bpcer", "acer", "roc_auc")}
    for _ in range(iterations):
        indexes = [i for pick in rng.integers(0, len(group_values), len(group_values)) for i in group_values[pick]]
        labels = np.asarray([int(rows[index]["label"]) for index in indexes])
        scores = np.asarray([float(rows[index]["score"]) for index in indexes])
        if len(set(labels.tolist())) < 2:
            continue
        rates = error_rates(labels, scores, threshold)
        for name in ("apcer", "bpcer", "acer"):
            sampled[name].append(rates[name])
        sampled["roc_auc"].append(float(roc_auc_score(labels, scores)))
    if not sampled["roc_auc"]:
        raise ValueError("Cannot bootstrap PAD metrics with one-class groups")
    return {
        name: [float(value) for value in np.quantile(values, [0.025, 0.975])]
        for name, values in sampled.items()
    }
