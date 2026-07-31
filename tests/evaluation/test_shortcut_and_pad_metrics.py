from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vshield.evaluation.pad_metrics import error_rates, select_threshold
from vshield.evaluation.shortcut import check_gates, run_pixel_probe, run_probe


def _shortcut_row(index: int, label: int, split: str) -> dict[str, str]:
    brightness = 20 + label * 200
    return {
        "sample_id": f"{split}-{index}",
        "split": split,
        "label": str(label),
        "subject_id": f"{split}-subject-{index}",
        "session_id": f"{split}-session-{index}",
        "clip_id": f"{split}-clip-{index}",
        "height": str(100 + label * 50),
        "width": str(100 + label * 50),
        "bytes": str(1000 + label * 5000),
        "brightness": str(brightness),
        "contrast": str(10 + label * 20),
        "blur": str(5 + label * 50),
    }


def test_metadata_shortcut_fails_release_gate():
    train = [_shortcut_row(index, index % 2, "train") for index in range(40)]
    test = [_shortcut_row(index, index % 2, "test") for index in range(20)]
    result = run_probe(train, test, "metadata", bootstrap_iterations=50)
    assert result["auc"] == 1.0
    assert check_gates(result, 0.60, 0.65, 0.10)


def test_low_resolution_pixel_shortcut_is_detected(tmp_path: Path):
    train = [_shortcut_row(index, index % 2, "train") for index in range(20)]
    test = [_shortcut_row(index, index % 2, "test") for index in range(10)]
    for row in [*train, *test]:
        row["relative_path"] = f"{row['sample_id']}.png"
        value = 20 if row["label"] == "0" else 230
        image = np.full((16, 16, 3), value, dtype=np.uint8)
        assert cv2.imwrite(str(tmp_path / row["relative_path"]), image)
    result = run_pixel_probe(
        train,
        test,
        tmp_path,
        "low_resolution",
        bootstrap_iterations=50,
    )
    assert result["auc"] == 1.0


def test_pad_threshold_uses_strict_greater_than():
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.5, 0.5, 0.9])
    rates = error_rates(labels, scores, threshold=0.5)
    assert rates["apcer"] == 0.0
    assert rates["bpcer"] == 0.5


def test_threshold_selection_minimizes_validation_acer():
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    threshold, rates = select_threshold(labels, scores)
    assert 0.2 <= threshold < 0.8
    assert rates["acer"] == 0.0
