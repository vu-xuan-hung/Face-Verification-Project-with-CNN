from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vshield.evaluation.promotion import build_model_release


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_model_promotion_binds_model_dataset_and_scores(tmp_path: Path):
    model = tmp_path / "model.keras"
    model.write_bytes(b"candidate-model")
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    candidate = tmp_path / "candidate.json"
    dataset = tmp_path / "dataset.json"
    evaluation = tmp_path / "evaluation.json"
    _write(
        candidate,
        {
            "status": "candidate",
            "model_version": "v2",
            "model_sha256": model_hash,
            "dataset_version": "v2",
            "config_sha256": "config-hash",
        },
    )
    _write(dataset, {"status": "released", "dataset_version": "v2"})
    _write(
        evaluation,
        {
            "threshold_source": "validation",
            "threshold_comparator": "score > threshold means real",
            "score_provenance": {
                "model_sha256": model_hash,
                "config_sha256": "config-hash",
            },
            "validation": {"threshold": 0.73},
            "test": {"apcer": 0.1, "bpcer": 0.2},
        },
    )
    release = build_model_release(model, candidate, dataset, evaluation)
    assert release["status"] == "released"
    assert release["threshold"] == 0.73


def test_model_promotion_rejects_scores_from_other_model(tmp_path: Path):
    model = tmp_path / "model.keras"
    model.write_bytes(b"candidate-model")
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    candidate = tmp_path / "candidate.json"
    dataset = tmp_path / "dataset.json"
    evaluation = tmp_path / "evaluation.json"
    _write(
        candidate,
        {
            "status": "candidate",
            "model_sha256": model_hash,
            "dataset_version": "v2",
            "config_sha256": "config-hash",
        },
    )
    _write(dataset, {"status": "released", "dataset_version": "v2"})
    _write(
        evaluation,
        {
            "threshold_source": "validation",
            "score_provenance": {
                "model_sha256": "other",
                "config_sha256": "config-hash",
            },
            "validation": {"threshold": 0.5},
            "test": {},
        },
    )
    with pytest.raises(ValueError, match="another model"):
        build_model_release(model, candidate, dataset, evaluation)
