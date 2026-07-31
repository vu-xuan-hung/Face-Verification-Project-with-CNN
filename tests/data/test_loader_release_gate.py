import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vshield.data.integrity import (
    MANIFEST_FIELDS,
    DatasetIntegrityError,
    sha256_file,
    write_csv,
)
from vshield.data.loader import load_data_from_config


def test_loader_rejects_invalid_dataset_config(tmp_path: Path):
    config = tmp_path / "data.yaml"
    config.write_text(
        "version: v1\nstatus: invalid_for_model_evaluation\npath: data\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetIntegrityError, match="only a released dataset"):
        load_data_from_config(config)


def test_manifest_loader_uses_bgr_float_and_does_not_touch_test(tmp_path: Path):
    dataset = tmp_path / "dataset"
    protocols = tmp_path / "protocols"
    dataset.mkdir()
    rows = {}
    for split, label in (("train", 0), ("val", 1), ("test", 0)):
        path = dataset / f"{split}.png"
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[:, :, 0] = 255
        assert cv2.imwrite(str(path), image)
        row = dict.fromkeys(MANIFEST_FIELDS, "")
        row.update(
            {
                "sample_id": split,
                "relative_path": path.name,
                "split": split,
                "label": str(label),
                "sha256": sha256_file(path),
                "status": "released",
            }
        )
        rows[split] = row
        write_csv(protocols / f"{split}.csv", [row])

    registry = tmp_path / "release.json"
    registry.write_text(
        json.dumps(
            {
                "dataset_version": "v2",
                "status": "released",
                "protocols": {
                    f"{split}.csv": {"sha256": sha256_file(protocols / f"{split}.csv")}
                    for split in ("train", "val", "test")
                },
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "data.yaml"
    config.write_text(
        "\n".join(
            [
                "version: v2",
                "status: released",
                "dataset_root: dataset",
                "release_manifest: release.json",
                "verify_hashes: true",
                "protocols:",
                "  train: protocols/train.csv",
                "  val: protocols/val.csv",
                "  test: protocols/test.csv",
            ]
        ),
        encoding="utf-8",
    )
    data = load_data_from_config(config, splits=("train", "val"))
    assert data["X_train"].shape == (1, 128, 128, 3)
    assert data["X_train"].dtype == np.float32
    assert data["X_train"][0, 0, 0].tolist() == [1.0, 0.0, 0.0]
    assert data["X_test"].shape == (0, 128, 128, 3)


def test_manifest_loader_rejects_hash_mismatch(tmp_path: Path):
    dataset = tmp_path / "dataset"
    protocols = tmp_path / "protocols"
    dataset.mkdir()
    image_path = dataset / "one.png"
    assert cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
    row = dict.fromkeys(MANIFEST_FIELDS, "")
    row.update(
        {
            "sample_id": "one",
            "relative_path": "one.png",
            "split": "train",
            "label": "0",
            "sha256": "wrong",
            "status": "released",
        }
    )
    write_csv(protocols / "train.csv", [row])
    registry = tmp_path / "release.json"
    registry.write_text(
        json.dumps(
            {
                "dataset_version": "v2",
                "status": "released",
                "protocols": {"train.csv": {"sha256": sha256_file(protocols / "train.csv")}},
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "data.yaml"
    config.write_text(
        "\n".join(
            [
                "version: v2",
                "status: released",
                "dataset_root: dataset",
                "release_manifest: release.json",
                "protocols:",
                "  train: protocols/train.csv",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(DatasetIntegrityError, match="checksum mismatch"):
        load_data_from_config(config, splits=("train",))
