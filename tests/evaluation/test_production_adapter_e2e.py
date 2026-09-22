import csv
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vshield.core.anti_spoof import PadResult, PadStatus
from vshield.core.face_preprocessor import FaceCrops
from vshield.core.identity_index import IdentityIndex
from vshield.evaluation.adapter import VShieldEvaluationAdapter
from vshield.evaluation.datasets import DatasetValidationError
from vshield.evaluation.e2e_runner import evaluate_end_to_end


def unit(index=0):
    vector = np.zeros(512, dtype=np.float32)
    vector[index] = 1
    return vector


class Preprocessor:
    def extract(self, image):
        return FaceCrops(image, image, image, (0, 0, image.shape[1], image.shape[0]))


class Pad:
    ready = True

    def __init__(self, real):
        self.real = real
        self.contract = SimpleNamespace(threshold=0.8)

    def predict(self, *, image, bbox):
        return PadResult(
            PadStatus.REAL if self.real else PadStatus.FAKE,
            score=0.9 if self.real else 0.1,
            class_index=1 if self.real else 0,
            model_version="mock",
            threshold=0.8,
            reason="PAD_REAL" if self.real else "SPOOF",
        )


class Embedder:
    def __init__(self, embedding):
        self.embedding = embedding

    def encode(self, frame):
        return self.embedding


def adapter(real, embedding):
    return VShieldEvaluationAdapter(
        ".",
        face_preprocessor=Preprocessor(),
        pad_service=Pad(real),
        face_embedder=Embedder(embedding),
    )


def test_real_known_allows_real_unknown_denies_and_spoof_denies():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    gallery = IdentityIndex({"alice": [unit(0)]}, prefer_faiss=False)
    assert adapter(True, unit(0)).full_biometric_decision(image, gallery)["final_access"] == "ALLOW"
    assert adapter(True, unit(1)).full_biometric_decision(image, gallery)["final_access"] == "DENY"
    assert adapter(False, unit(0)).full_biometric_decision(image, gallery)["final_access"] == "DENY"


def test_e2e_rejects_invalid_gallery_before_inference(tmp_path):
    for name, value in (("calibration", 20), ("test", 40), ("unknown", 60), ("real", 80), ("spoof", 100)):
        assert cv2.imwrite(
            str(tmp_path / f"{name}.png"),
            np.full((8, 8, 3), value, dtype=np.uint8),
        )

    recognition = tmp_path / "recognition.csv"
    with recognition.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["sample_id", "image_path", "subject_id", "split", "is_enrolled"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"sample_id": "g", "image_path": "missing.png", "subject_id": "a", "split": "gallery", "is_enrolled": "1"},
                {"sample_id": "c", "image_path": "calibration.png", "subject_id": "a", "split": "calibration", "is_enrolled": "1"},
                {"sample_id": "t", "image_path": "test.png", "subject_id": "a", "split": "test", "is_enrolled": "1"},
                {"sample_id": "u", "image_path": "unknown.png", "subject_id": "u", "split": "test", "is_enrolled": "0"},
            ]
        )

    e2e = tmp_path / "e2e.csv"
    with e2e.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "sample_id",
                "image_path",
                "subject_id",
                "presentation",
                "expected_identity_state",
                "expected_access",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"sample_id": "r", "image_path": "real.png", "subject_id": "a", "presentation": "REAL", "expected_identity_state": "KNOWN", "expected_access": "ALLOW"},
                {"sample_id": "s", "image_path": "spoof.png", "subject_id": "u", "presentation": "SPOOF", "expected_identity_state": "UNKNOWN", "expected_access": "DENY"},
            ]
        )

    with pytest.raises(DatasetValidationError, match="MISSING_FILE"):
        evaluate_end_to_end(
            e2e,
            recognition,
            tmp_path / "output",
            project_root=tmp_path,
            adapter=adapter(True, unit()),
        )
