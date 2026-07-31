"""Non-visual shortcut probes for anti-spoof datasets."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_SETS = {
    "metadata": ("height", "width", "bytes", "brightness", "contrast", "blur"),
    "brightness": ("brightness",),
    "dimensions": ("height", "width"),
}


def _matrix(rows: list[dict[str, str]], fields: tuple[str, ...]) -> np.ndarray:
    try:
        return np.asarray([[float(row[field]) for field in fields] for row in rows])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid shortcut feature: {exc}") from exc


def _labels(rows: list[dict[str, str]]) -> np.ndarray:
    labels = np.asarray([int(row["label"]) for row in rows])
    if set(labels) != {0, 1}:
        raise ValueError("Shortcut probe requires both labels")
    return labels


def _group_key(row: dict[str, str]) -> str:
    values = [row.get(name, "") for name in ("subject_id", "session_id", "clip_id")]
    return "|".join(values) if any(values) else row["sample_id"]


def bootstrap_auc(
    rows: list[dict[str, str]],
    scores: np.ndarray,
    iterations: int = 500,
    seed: int = 42,
) -> tuple[float, float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[_group_key(row)].append(index)
    groups = list(grouped.values())
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indexes = [index for group in sampled for index in groups[group]]
        labels = np.asarray([int(rows[index]["label"]) for index in indexes])
        if len(set(labels)) == 2:
            estimates.append(roc_auc_score(labels, scores[indexes]))
    if not estimates:
        raise ValueError("Cannot bootstrap AUC: resamples contain one label")
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def run_probe(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    feature_set: str,
    bootstrap_iterations: int = 500,
) -> dict[str, float | str]:
    fields = FEATURE_SETS[feature_set]
    return _run_matrix_probe(
        train_rows,
        test_rows,
        _matrix(train_rows, fields),
        _matrix(test_rows, fields),
        feature_set,
        bootstrap_iterations,
    )


def _run_matrix_probe(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    feature_set: str,
    bootstrap_iterations: int,
) -> dict[str, float | str]:
    train_labels, test_labels = _labels(train_rows), _labels(test_rows)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    )
    model.fit(train_matrix, train_labels)
    scores = model.predict_proba(test_matrix)[:, 1]
    predictions = (scores >= 0.5).astype(int)
    auc = float(roc_auc_score(test_labels, scores))
    ci_low, ci_high = bootstrap_auc(test_rows, scores, bootstrap_iterations)
    majority = float(max(np.mean(test_labels), 1 - np.mean(test_labels)))
    return {
        "feature_set": feature_set,
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "auc": auc,
        "auc_ci_low": ci_low,
        "auc_ci_high": ci_high,
        "majority_accuracy": majority,
        "sample_count": float(len(test_rows)),
    }


def _pixel_vector(row: dict[str, str], root: Path, mode: str) -> np.ndarray:
    path = (root / row["relative_path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Invalid pixel probe path: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode pixel probe image: {path}")
    height, width = image.shape[:2]
    if mode in {"background_only", "center_only"}:
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(
            mask,
            (width // 2, height // 2),
            (max(width * 3 // 10, 1), max(height * 2 // 5, 1)),
            0,
            0,
            360,
            255,
            -1,
        )
        image[mask > 0 if mode == "background_only" else mask == 0] = 0
    if mode == "color_histogram":
        histograms = [
            cv2.calcHist([image], [channel], None, [16], [0, 256]).reshape(-1)
            for channel in range(3)
        ]
        vector = np.concatenate(histograms)
        return vector / max(vector.sum(), 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    size = (8, 8) if mode == "low_resolution" else (16, 16)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA).reshape(-1) / 255.0


def run_pixel_probe(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    root: Path,
    mode: str,
    bootstrap_iterations: int = 500,
) -> dict[str, float | str]:
    root = root.resolve()
    train_matrix = np.asarray([_pixel_vector(row, root, mode) for row in train_rows])
    test_matrix = np.asarray([_pixel_vector(row, root, mode) for row in test_rows])
    return _run_matrix_probe(
        train_rows,
        test_rows,
        train_matrix,
        test_matrix,
        mode,
        bootstrap_iterations,
    )


def category_label_purity(rows: list[dict[str, str]], field: str) -> dict[str, float | str]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        value = row.get(field, "")
        if value:
            counts[value][int(row["label"])] += 1
    samples = sum(sum(value) for value in counts.values())
    if not samples:
        return {"field": field, "coverage": 0.0, "weighted_purity": 1.0}
    correct = sum(max(value) for value in counts.values())
    return {
        "field": field,
        "coverage": samples / len(rows),
        "weighted_purity": correct / samples,
    }


def check_gates(
    result: dict[str, float | str],
    max_auc: float,
    max_auc_ci_high: float,
    max_accuracy_delta: float,
) -> list[str]:
    failures = []
    if float(result["auc"]) > max_auc:
        failures.append("AUC_ABOVE_LIMIT")
    if float(result["auc_ci_high"]) > max_auc_ci_high:
        failures.append("AUC_CI_ABOVE_LIMIT")
    if float(result["accuracy"]) > float(result["majority_accuracy"]) + max_accuracy_delta:
        failures.append("ACCURACY_ABOVE_MAJORITY_LIMIT")
    return failures
