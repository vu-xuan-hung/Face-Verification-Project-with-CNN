from pathlib import Path

import numpy as np
import pytest

from vshield.training.training_only_pad_data import (
    TrainingOnlyDataError,
    build_training_only_splits,
    prepare_training_only_face_crop,
)

def test_training_only_split_reads_collector_labels_and_is_reproducible():
    root = Path("data/DataCollect")
    if not root.is_dir():
        pytest.skip("Local consented DataCollect fixture is unavailable")

    first, summary = build_training_only_splits(root, seed=42)
    second, _ = build_training_only_splits(root, seed=42)

    assert summary["split_counts"] == {"train": 2446, "val": 431}
    assert {row["label"] for row in first["train"]} == {0, 1}
    assert {row["label"] for row in first["val"]} == {0, 1}
    assert first == second


def test_training_only_split_rejects_missing_class_folder():
    with pytest.raises(TrainingOnlyDataError, match="Missing class folder"):
        build_training_only_splits(Path("missing-training-only-data"), seed=42)


def test_training_only_crop_preserves_minifasnet_input_contract():
    result = prepare_training_only_face_crop(np.full((24, 32, 3), 120, dtype=np.uint8))

    assert result.shape == (3, 80, 80)
    assert result.dtype == np.float32
    assert float(result.mean()) == 120.0
