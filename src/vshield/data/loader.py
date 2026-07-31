"""Strict anti-spoof data loader with manifest and legacy config support."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from vshield.data.integrity import DatasetIntegrityError, read_csv, read_label, sha256_file

logger = logging.getLogger(__name__)
TARGET_SIZE = (128, 128)


def _empty() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.empty((0, TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=np.float32),
        np.empty((0,), dtype=np.int32),
    )


def _preprocess(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise DatasetIntegrityError(f"Cannot decode image: {path}")
    resized = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def _load_manifest_split(
    config_dir: Path,
    config: dict[str, Any],
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    protocol_value = config.get("protocols", {}).get(split)
    if not protocol_value:
        return _empty()
    protocol_path = (config_dir / protocol_value).resolve()
    dataset_root = (config_dir / config["dataset_root"]).resolve()
    if not protocol_path.is_file():
        raise DatasetIntegrityError(f"Missing protocol: {protocol_path}")

    images: list[np.ndarray] = []
    labels: list[int] = []
    for row in read_csv(protocol_path):
        if row.get("status") != "released" or row.get("split") != split:
            raise DatasetIntegrityError(f"Invalid {split} protocol row: {row.get('sample_id')}")
        path = (dataset_root / row["relative_path"]).resolve()
        if not path.is_relative_to(dataset_root):
            raise DatasetIntegrityError(f"Protocol path escapes dataset root: {path}")
        if not path.is_file():
            raise DatasetIntegrityError(f"Missing protocol image: {path}")
        if config.get("verify_hashes", True) and sha256_file(path) != row["sha256"]:
            raise DatasetIntegrityError(f"Image checksum mismatch: {path}")
        label = int(row["label"])
        if label not in {0, 1}:
            raise DatasetIntegrityError(f"Invalid label for {path}: {label}")
        images.append(_preprocess(path))
        labels.append(label)
    if not images:
        return _empty()
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def _load_legacy_split(
    base_path: Path,
    config: dict[str, Any],
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    if split not in config:
        return _empty()
    image_dir = (base_path / config[split]).resolve()
    if not image_dir.is_dir():
        raise DatasetIntegrityError(f"Split directory not found: {image_dir}")
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        return _empty()
    images = [_preprocess(path) for path in image_paths]
    labels = [read_label(path) for path in image_paths]
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def _validate_release_manifest(config_dir: Path, config: dict[str, Any]) -> None:
    value = config.get("release_manifest")
    if not value:
        raise DatasetIntegrityError("Released dataset requires release_manifest")
    path = (config_dir / value).resolve()
    if not path.is_file():
        raise DatasetIntegrityError(f"Missing dataset release manifest: {path}")
    release = json.loads(path.read_text(encoding="utf-8"))
    if release.get("status") != "released":
        raise DatasetIntegrityError("Dataset release manifest is not released")
    if str(release.get("dataset_version")) != str(config.get("version")):
        raise DatasetIntegrityError("Dataset version does not match release manifest")
    registered = release.get("protocols", {})
    for protocol_value in config.get("protocols", {}).values():
        protocol = (config_dir / protocol_value).resolve()
        expected = registered.get(protocol.name, {}).get("sha256")
        if not expected or not protocol.is_file() or sha256_file(protocol) != expected:
            raise DatasetIntegrityError(f"Protocol is not registered or changed: {protocol}")


def load_data_from_config(
    yaml_path: str | Path,
    *,
    allow_invalid: bool = False,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, Any]:
    yaml_path = Path(yaml_path).resolve()
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise DatasetIntegrityError(f"Invalid YAML config: {yaml_path}")
    status = config.get("status", "candidate")
    if status != "released" and not allow_invalid:
        raise DatasetIntegrityError(
            f"Dataset status is {status!r}; only a released dataset may train a model"
        )
    if status == "released" and "protocols" in config:
        _validate_release_manifest(yaml_path.parent, config)

    if "protocols" in config:

        def loader(split: str) -> tuple[np.ndarray, np.ndarray]:
            return _load_manifest_split(yaml_path.parent, config, split)
    else:
        if "path" not in config:
            raise DatasetIntegrityError("Dataset config requires path or protocols")
        base_path = (yaml_path.parent / config["path"]).resolve()

        def loader(split: str) -> tuple[np.ndarray, np.ndarray]:
            return _load_legacy_split(base_path, config, split)

    unknown_splits = set(splits) - {"train", "val", "test"}
    if unknown_splits:
        raise DatasetIntegrityError(f"Unknown splits requested: {sorted(unknown_splits)}")
    x_train, y_train = loader("train") if "train" in splits else _empty()
    x_val, y_val = loader("val") if "val" in splits else _empty()
    x_test, y_test = loader("test") if "test" in splits else _empty()
    if "train" in splits and not len(x_train):
        raise DatasetIntegrityError("Training split must be non-empty")
    if "val" in splits and not len(x_val):
        raise DatasetIntegrityError("Training and validation splits must be non-empty")
    logger.info("Loaded train=%d val=%d test=%d", len(x_train), len(x_val), len(x_test))
    return {
        "X_train": x_train,
        "y_train": y_train,
        "X_val": x_val,
        "y_val": y_val,
        "X_test": x_test,
        "y_test": y_test,
        "classes": config.get("names", ["fake", "real"]),
        "dataset_version": config.get("version", "unversioned"),
    }
