"""Model promotion contract bound to dataset and evaluation provenance."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_model_release(
    model_path: Path,
    candidate_manifest_path: Path,
    dataset_release_path: Path,
    evaluation_path: Path,
) -> dict[str, object]:
    candidate = _read_json(candidate_manifest_path)
    dataset = _read_json(dataset_release_path)
    evaluation = _read_json(evaluation_path)
    model_hash = _sha256(model_path)

    if candidate.get("status") != "candidate":
        raise ValueError("Model manifest must have candidate status")
    if candidate.get("model_sha256") != model_hash:
        raise ValueError("Candidate model checksum mismatch")
    if dataset.get("status") != "released":
        raise ValueError("Dataset is not released")
    if str(candidate.get("dataset_version")) != str(dataset.get("dataset_version")):
        raise ValueError("Model and dataset versions do not match")
    if evaluation.get("threshold_source") != "validation":
        raise ValueError("Threshold must be selected on validation")
    if not isinstance(evaluation.get("test"), dict):
        raise ValueError("Locked test evaluation is missing")
    score_provenance = evaluation.get("score_provenance", {})
    if score_provenance.get("model_sha256") != model_hash:
        raise ValueError("Evaluation scores belong to another model")
    if score_provenance.get("config_sha256") != candidate.get("config_sha256"):
        raise ValueError("Evaluation scores use another dataset config")
    threshold = evaluation.get("validation", {}).get("threshold")
    if not isinstance(threshold, (float, int)):
        raise ValueError("Validation threshold is missing")

    return {
        "model_version": candidate.get("model_version"),
        "status": "released",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_sha256": model_hash,
        "dataset_version": dataset.get("dataset_version"),
        "dataset_release_sha256": _sha256(dataset_release_path),
        "evaluation_sha256": _sha256(evaluation_path),
        "threshold": float(threshold),
        "threshold_source": "validation",
        "threshold_comparator": evaluation.get("threshold_comparator"),
        "test_metrics": evaluation["test"],
    }
