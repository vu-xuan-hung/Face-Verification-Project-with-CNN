"""
Data loader — reads image/label pairs according to a YAML config.

Config format (data.yaml):
    path: ./data/processed          # base directory
    train: train/images             # relative to path
    val:   val/images
    test:  test/images              # optional
    nc:    2
    names: ["fake", "real"]

Returns a dict with normalised NumPy arrays ready for model training.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Target resolution for all face crops
_TARGET_SIZE = (128, 128)


def load_data_from_config(yaml_path: str | Path) -> dict[str, Any]:
    """Load training, validation, and (optionally) test splits from a YAML config.

    Parameters
    ----------
    yaml_path:
        Path to the dataset configuration YAML file.

    Returns
    -------
    dict with keys:
        ``X_train``, ``y_train``, ``X_val``, ``y_val``,
        ``X_test`` (may be empty), ``y_test`` (may be empty),
        ``classes`` — list of class names.
    """
    yaml_path = Path(yaml_path).resolve()
    with yaml_path.open("r", encoding="utf-8") as f:
        config: dict = yaml.safe_load(f)

    # Resolve base path relative to the YAML file location
    base_path = (yaml_path.parent / config["path"]).resolve()

    def _load_split(split_key: str) -> tuple[np.ndarray, np.ndarray]:
        if split_key not in config:
            return np.array([]), np.array([])

        img_folder = base_path / config[split_key]
        lbl_folder = Path(str(img_folder).replace("images", "labels"))

        logger.info("Loading split '%s' from %s", split_key, img_folder)

        if not img_folder.exists():
            logger.warning("Split directory not found: %s", img_folder)
            return np.array([]), np.array([])

        images: list[np.ndarray] = []
        labels: list[int] = []

        for img_file in sorted(img_folder.iterdir()):
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            img = cv2.imread(str(img_file))
            if img is None:
                logger.warning("Cannot read image: %s", img_file)
                continue
            img = cv2.resize(img, _TARGET_SIZE)
            images.append(img)

            # Corresponding label file
            lbl_file = lbl_folder / (img_file.stem + ".txt")
            if lbl_file.exists():
                first_line = lbl_file.read_text(encoding="utf-8").strip().splitlines()[0]
                class_id = int(first_line.split()[0])
            else:
                class_id = 0  # default to "fake" if label missing
            labels.append(class_id)

        if not images:
            return np.array([]), np.array([])

        X = np.array(images, dtype=np.float32) / 255.0
        y = np.array(labels, dtype=np.int32)
        logger.info("  → %d samples loaded.", len(X))
        return X, y

    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")
    X_test, y_test = _load_split("test")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "classes": config.get("names", ["fake", "real"]),
    }
