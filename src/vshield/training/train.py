"""Train anti-spoof model using train/validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import tensorflow as tf

from vshield.data.integrity import DatasetIntegrityError
from vshield.data.loader import load_data_from_config
from vshield.models.augmentation_config import (
    training_augmentation_spec,
    training_augmentation_spec_sha256,
)
from vshield.models.cnn import CNNModel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_training(
    config_path: Path,
    output_path: Path,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass

    data = load_data_from_config(config_path, splits=("train", "val"))
    wrapper = CNNModel(seed=seed)
    history = wrapper.train(data=data, epochs=epochs, batch_size=batch_size)
    wrapper.save(str(output_path))
    augmentation = wrapper.model.get_layer("training_augmentation")
    augmentation_spec = training_augmentation_spec(seed)
    manifest = {
        "model_version": "v2",
        "status": "candidate",
        "model_sha256": _sha256(output_path),
        "dataset_version": data["dataset_version"],
        "config_sha256": _sha256(config_path),
        "seed": seed,
        "augmentation": {
            "canonical_config": augmentation_spec,
            "canonical_sha256": training_augmentation_spec_sha256(seed),
            "keras_config": tf.keras.utils.serialize_keras_object(augmentation),
        },
        "threshold": None,
        "threshold_source": "validation_required",
        "test_accessed_during_training": False,
        "history": {
            key: [float(value) for value in values] for key, values in history.history.items()
        },
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def train() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data-v2.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/models/face_verify_v2.keras")
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        run_training(args.config, args.output, args.epochs, args.batch_size, args.seed)
    except DatasetIntegrityError as exc:
        parser.error(str(exc))


def main() -> None:
    train()


if __name__ == "__main__":
    main()
