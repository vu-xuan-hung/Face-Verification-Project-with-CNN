"""Training-only loader for consented MiniFASNetV2 face-crop collections.

This loader is intentionally not a dataset-release path. It reads collector
class labels and creates a deterministic image-level train/validation split so
that an exploratory fine-tuning run can be reproduced. Without provenance,
the validation metrics are not an independent biometric benchmark.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_LABELS = {"fake": 0, "real": 1}


class TrainingOnlyDataError(ValueError):
    """Raised when the local collector layout cannot be used safely."""


def _read_label(image_path: Path) -> int:
    label_path = image_path.with_suffix(".txt")
    if not label_path.is_file():
        raise TrainingOnlyDataError(f"Missing class label: {label_path}")
    fields = label_path.read_text(encoding="utf-8").strip().split()
    if not fields:
        raise TrainingOnlyDataError(f"Empty class label: {label_path}")
    try:
        label = int(fields[0])
    except ValueError as exc:
        raise TrainingOnlyDataError(f"Invalid class label: {label_path}") from exc
    if label not in {0, 1}:
        raise TrainingOnlyDataError(f"Class label must be 0 or 1: {label_path}")
    return label


def _split_key(relative_path: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{relative_path}".encode("utf-8")).hexdigest()


def build_training_only_splits(
    root: Path,
    *,
    seed: int,
    validation_ratio: float = 0.15,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    """Read ``fake``/``real`` collector folders and make a reproducible split."""
    if not 0 < validation_ratio < 0.5:
        raise TrainingOnlyDataError("validation_ratio must be between 0 and 0.5")
    root = root.resolve()
    by_label: dict[int, list[dict[str, object]]] = {0: [], 1: []}
    for folder, expected_label in EXPECTED_LABELS.items():
        class_root = root / folder
        if not class_root.is_dir():
            raise TrainingOnlyDataError(f"Missing class folder: {class_root}")
        images = sorted(
            path for path in class_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise TrainingOnlyDataError(f"No images below: {class_root}")
        for image_path in images:
            label = _read_label(image_path)
            if label != expected_label:
                raise TrainingOnlyDataError(
                    f"Folder/label mismatch for {image_path}: expected {expected_label}, got {label}"
                )
            relative = image_path.relative_to(root).as_posix()
            by_label[label].append({"_path": str(image_path), "label": label, "relative_path": relative})

    splits = {"train": [], "val": []}
    for label, rows in by_label.items():
        ordered = sorted(rows, key=lambda row: _split_key(str(row["relative_path"]), seed))
        validation_count = max(1, min(len(ordered) - 1, round(len(ordered) * validation_ratio)))
        splits["val"].extend(ordered[:validation_count])
        splits["train"].extend(ordered[validation_count:])
    for rows in splits.values():
        rows.sort(key=lambda row: str(row["relative_path"]))

    summary = {
        "mode": "training_only_data_collect",
        "dataset_root": str(root),
        "seed": seed,
        "validation_ratio": validation_ratio,
        "class_counts": {str(label): count for label, count in sorted(Counter(
            int(row["label"]) for rows in splits.values() for row in rows
        ).items())},
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "limitation": (
            "Image-level split only. No subject/session/clip/device metadata or independent test set; "
            "metrics are exploratory and must not be reported as deployment performance."
        ),
    }
    return splits, summary


def prepare_training_only_face_crop(image: np.ndarray) -> np.ndarray:
    """Resize a collector-saved face crop to MiniFASNetV2's 80x80 BGR input."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TrainingOnlyDataError("Expected an uint8 BGR image")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2:
        raise TrainingOnlyDataError("Expected a non-empty three-channel BGR image")
    resized = cv2.resize(image, (80, 80), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32)
