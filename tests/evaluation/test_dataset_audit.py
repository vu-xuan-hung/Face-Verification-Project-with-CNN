import csv

import cv2
import numpy as np
import pytest
from scripts.collect_evaluation_samples import SAFE_IDENTIFIER

from vshield.evaluation.datasets import (
    DatasetLeakageError,
    DatasetValidationError,
    audit_manifest,
    read_manifest,
    require_no_leakage,
)


def test_exact_gallery_test_hash_leakage_is_detected_and_aborts(tmp_path):
    image = np.full((16, 16, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "gallery.png"), image)
    (tmp_path / "test.png").write_bytes((tmp_path / "gallery.png").read_bytes())
    assert cv2.imwrite(str(tmp_path / "calibration.png"), image + 1)
    assert cv2.imwrite(str(tmp_path / "unknown.png"), image + 2)
    manifest = tmp_path / "recognition.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "image_path", "subject_id", "split", "is_enrolled"])
        writer.writeheader()
        writer.writerows([
            {"sample_id": "g", "image_path": "gallery.png", "subject_id": "a", "split": "gallery", "is_enrolled": "1"},
            {"sample_id": "c", "image_path": "calibration.png", "subject_id": "a", "split": "calibration", "is_enrolled": "1"},
            {"sample_id": "t", "image_path": "test.png", "subject_id": "a", "split": "test", "is_enrolled": "1"},
            {"sample_id": "u", "image_path": "unknown.png", "subject_id": "u", "split": "test", "is_enrolled": "0"},
        ])
    audit = audit_manifest(read_manifest(manifest, "recognition"), "recognition")
    assert audit["has_severe_leakage"] is True
    with pytest.raises(DatasetLeakageError):
        require_no_leakage(audit)
    require_no_leakage(audit, allow_leakage=True)


def test_duplicate_sample_ids_abort_even_when_leakage_is_allowed(tmp_path):
    image = np.full((16, 16, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "one.png"), image)
    assert cv2.imwrite(str(tmp_path / "two.png"), image + 1)
    manifest = tmp_path / "pad.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "image_path", "label"])
        writer.writeheader()
        writer.writerows([
            {"sample_id": "same", "image_path": "one.png", "label": "REAL"},
            {"sample_id": "same", "image_path": "two.png", "label": "SPOOF"},
        ])
    audit = audit_manifest(read_manifest(manifest, "pad"), "pad")
    with pytest.raises(ValueError, match="fatal semantic errors"):
        require_no_leakage(audit, allow_leakage=True)


def test_missing_images_and_empty_classes_abort(tmp_path):
    manifest = tmp_path / "pad.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "image_path", "label"])
        writer.writeheader()
        writer.writerow({"sample_id": "missing", "image_path": "missing.png", "label": "REAL"})
    audit = audit_manifest(read_manifest(manifest, "pad"), "pad")
    with pytest.raises(DatasetValidationError, match="EMPTY_CLASS.*MISSING_FILE"):
        require_no_leakage(audit)


def test_collector_identifiers_cannot_escape_output_directory():
    assert SAFE_IDENTIFIER.fullmatch("subject-01.session_A")
    assert not SAFE_IDENTIFIER.fullmatch("../outside")
    assert not SAFE_IDENTIFIER.fullmatch(r"folder\outside")
    assert not SAFE_IDENTIFIER.fullmatch("=FORMULA")
