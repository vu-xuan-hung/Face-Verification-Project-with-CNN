"""Optional technical plots generated only when supporting observations exist."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def confusion(path: str | Path, matrix, labels: list[str], title: str) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, str(matrix[row][column]), ha="center", va="center")
    axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Ground truth")
    axis.set_title(title)
    figure.colorbar(image, ax=axis)
    _save(figure, path)


def score_histogram(path: str | Path, groups: dict[str, list[float]], xlabel: str, title: str) -> None:
    groups = {name: values for name, values in groups.items() if values}
    if not groups:
        return
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for name, values in groups.items():
        axis.hist(values, bins=min(30, max(5, len(values))), alpha=0.55, label=name)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Samples")
    axis.set_title(title)
    axis.legend()
    _save(figure, path)


def roc_plot(path: str | Path, labels, scores, *, higher_is_positive=True, title="ROC curve") -> None:
    if len(set(labels)) < 2:
        return
    from sklearn.metrics import auc, roc_curve

    values = np.asarray(scores) if higher_is_positive else -np.asarray(scores)
    false_positive, true_positive, _ = roc_curve(labels, values)
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot(false_positive, true_positive, label=f"AUC={auc(false_positive, true_positive):.4f}")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    axis.legend()
    _save(figure, path)


def error_curve(path: str | Path, rows: list[dict], x: str, left: str, right: str, title: str) -> None:
    if not rows:
        return
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot([row[x] for row in rows], [row.get(left) for row in rows], label=left.upper())
    axis.plot([row[x] for row in rows], [row.get(right) for row in rows], label=right.upper())
    axis.set(xlabel=x.replace("_", " ").title(), ylabel="Error rate", title=title)
    axis.legend()
    _save(figure, path)


def category_counts(path: str | Path, counts: dict[str, int], title: str) -> None:
    if not counts:
        return
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(list(counts), list(counts.values()))
    axis.tick_params(axis="x", rotation=30)
    axis.set(ylabel="Samples", title=title)
    _save(figure, path)


def _save(figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    _pyplot().close(figure)
