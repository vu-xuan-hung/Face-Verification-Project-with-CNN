from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from vshield.data.deduplication import cluster_duplicates
from vshield.data.integrity import DatasetIntegrityError, build_manifest


def _write_sample(root: Path, name: str, value: int, label: int) -> None:
    image = np.full((16, 16, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(root / f"{name}.jpg"), image)
    (root / f"{name}.txt").write_text(str(label), encoding="utf-8")


def _write_metadata(path: Path, names: list[str]) -> None:
    fields = [
        "relative_path",
        "subject_id",
        "session_id",
        "clip_id",
        "device_id",
        "attack_type",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, name in enumerate(names):
            writer.writerow(
                {
                    "relative_path": f"{name}.jpg",
                    "subject_id": f"subject-{index}",
                    "session_id": f"session-{index}",
                    "clip_id": f"clip-{index}",
                    "device_id": f"device-{index}",
                    "attack_type": "bona_fide",
                }
            )


def test_manifest_quarantines_missing_provenance(tmp_path: Path):
    _write_sample(tmp_path, "one", 50, 1)
    rows = build_manifest(tmp_path)
    assert rows[0]["status"] == "quarantine"
    assert rows[0]["reason_code"].startswith("MISSING_PROVENANCE")


def test_manifest_accepts_complete_provenance(tmp_path: Path):
    _write_sample(tmp_path, "one", 50, 1)
    metadata = tmp_path / "metadata.csv"
    _write_metadata(metadata, ["one"])
    rows = build_manifest(tmp_path, metadata)
    assert rows[0]["status"] == "candidate"
    assert rows[0]["label"] == "1"
    assert len(rows[0]["sha256"]) == 64


def test_manifest_rejects_missing_label(tmp_path: Path):
    assert cv2.imwrite(str(tmp_path / "one.jpg"), np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(DatasetIntegrityError, match="Missing label"):
        build_manifest(tmp_path)


def _row(sample: str, digest: str, dhash: str, label: str = "1") -> dict[str, str]:
    return {
        "sample_id": sample,
        "relative_path": f"{sample}.jpg",
        "sha256": digest,
        "dhash": dhash,
        "label": label,
        "status": "candidate",
        "reason_code": "",
    }


def test_exact_duplicate_keeps_one_canonical():
    rows = [_row("a", "same", "0"), _row("b", "same", "0")]
    output, _ = cluster_duplicates(rows)
    assert [row["status"] for row in output].count("duplicate_excluded") == 1
    assert len({row["exact_component_id"] for row in output}) == 1


def test_exact_duplicate_with_conflicting_labels_is_quarantined():
    rows = [_row("a", "same", "0", "0"), _row("b", "same", "0", "1")]
    output, _ = cluster_duplicates(rows)
    assert {row["status"] for row in output} == {"quarantine"}
    assert {row["reason_code"] for row in output} == {"CONFLICTING_LABEL_EXACT_DUPLICATE"}


def test_near_duplicate_components_are_transitive():
    rows = [
        _row("a", "a", "0000000000000000"),
        _row("b", "b", "0000000000000001"),
        _row("c", "c", "0000000000000003"),
    ]
    output, candidates = cluster_duplicates(rows, near_threshold=1)
    assert len(candidates) == 2
    assert len({row["near_component_id"] for row in output}) == 1
