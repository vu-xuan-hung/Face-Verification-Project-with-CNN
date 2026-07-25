"""Fail-closed tests for the anti-spoof authentication gate."""

import base64
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from vshield.api import app as app_module
from vshield.api import database
from vshield.api.app import create_app, decode_image
from vshield.core import anti_spoof
from vshield.core.face_preprocessor import FaceCrops, InvalidFaceCountError
from vshield.core.verifier import MatchResult
from vshield.services.authentication import (
    AuthenticationService,
)


def make_image_data() -> str:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(".png", image)
    assert encoded_ok
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def make_service(model):
    face = np.zeros((128, 128, 3), dtype=np.uint8)
    preprocessor = Mock()
    preprocessor.extract.return_value = FaceCrops(anti_spoof=face, facenet=face)
    embedder = Mock()
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    embedder.encode.return_value = embedding
    identity_index = Mock()
    identity_index.search.return_value = MatchResult("alice", 0.2, 0.8)
    service = AuthenticationService(
        face_preprocessor=preprocessor,
        anti_spoof_model=model,
        face_embedder=embedder,
        identity_index=identity_index,
    )
    return service, embedder, identity_index


def post_predict(service):
    app = create_app(service, initialize_database=False)
    with TestClient(app) as client:
        return client.post("/predict", json={"image": make_image_data()})


def test_decode_image_rejects_large_encoded_payload(monkeypatch):
    monkeypatch.setattr(app_module, "MAX_ENCODED_IMAGE_BYTES", 3)

    with pytest.raises(ValueError, match="too large"):
        decode_image("data:image/png;base64,AAAAAA==")


def test_decode_image_rejects_excessive_pixel_count(monkeypatch):
    monkeypatch.setattr(app_module, "MAX_IMAGE_PIXELS", 4)

    with pytest.raises(ValueError, match="too large"):
        decode_image(make_image_data())


def assert_recognition_skipped(embedder, identity_index):
    embedder.encode.assert_not_called()
    identity_index.search.assert_not_called()


def test_missing_model_returns_503_and_skips_recognition():
    service, embedder, identity_index = make_service(None)

    response = post_predict(service)

    assert response.status_code == 503
    assert response.json() == {
        "success": False,
        "message": "Anti-spoofing service unavailable",
    }
    service.face_preprocessor.extract.assert_not_called()
    assert_recognition_skipped(embedder, identity_index)


def test_model_load_error_returns_503_and_skips_recognition(monkeypatch):
    load_model = Mock(side_effect=ValueError("corrupt model"))
    monkeypatch.setattr(anti_spoof.tf.keras.models, "load_model", load_model)
    failed_model = anti_spoof.load_anti_spoofing_model("broken.keras")
    service, embedder, identity_index = make_service(failed_model)

    response = post_predict(service)

    assert failed_model is None
    assert response.status_code == 503
    assert_recognition_skipped(embedder, identity_index)


def test_inference_error_returns_503_and_skips_recognition():
    model = Mock()
    model.predict.side_effect = RuntimeError("inference failed")
    service, embedder, identity_index = make_service(model)

    response = post_predict(service)

    assert response.status_code == 503
    assert_recognition_skipped(embedder, identity_index)


class BadArray:
    def __array__(self, dtype=None, copy=None):
        raise RuntimeError("cannot convert prediction")


@pytest.mark.parametrize(
    "prediction",
    [
        np.array([], dtype=np.float32),
        np.array([[np.nan]], dtype=np.float32),
        np.array([[np.inf]], dtype=np.float32),
        np.array([[-0.1]], dtype=np.float32),
        np.array([[1.1]], dtype=np.float32),
        np.array([[0.2, 0.8]], dtype=np.float32),
        np.array([["0.9"]]),
        np.array([[1 + 0j]]),
        BadArray(),
    ],
    ids=[
        "empty",
        "nan",
        "infinite",
        "below-zero",
        "above-one",
        "multiple-scores",
        "non-numeric",
        "complex",
        "conversion-error",
    ],
)
def test_invalid_score_returns_503_and_skips_recognition(prediction):
    model = Mock()
    model.predict.return_value = prediction
    service, embedder, identity_index = make_service(model)

    response = post_predict(service)

    assert response.status_code == 503
    assert_recognition_skipped(embedder, identity_index)


def test_fake_returns_403_and_skips_recognition():
    model = Mock()
    model.predict.return_value = np.array([[0.5]], dtype=np.float32)
    service, embedder, identity_index = make_service(model)

    response = post_predict(service)

    assert response.status_code == 403
    assert response.json() == {
        "success": False,
        "message": "Spoofing detected",
    }
    assert_recognition_skipped(embedder, identity_index)


def test_real_calls_identity_search(monkeypatch):
    model = Mock()
    model.predict.return_value = np.array([[0.9]], dtype=np.float32)
    service, embedder, identity_index = make_service(model)
    monkeypatch.setattr(database, "get_role", Mock(return_value="user"))
    log_login = Mock()
    monkeypatch.setattr(database, "log_login", log_login)

    response = post_predict(service)

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "username": "alice",
        "role": "user",
    }
    embedder.encode.assert_called_once()
    identity_index.search.assert_called_once()
    log_login.assert_called_once_with("alice", "user")


def test_unknown_face_returns_403(monkeypatch):
    model = Mock()
    model.predict.return_value = np.array([[0.9]], dtype=np.float32)
    service, _, identity_index = make_service(model)
    identity_index.search.return_value = None
    log_login = Mock()
    monkeypatch.setattr(database, "log_login", log_login)

    response = post_predict(service)

    assert response.status_code == 403
    assert response.json()["message"] == "Unknown face, not registered"
    log_login.assert_not_called()


def test_invalid_face_returns_422_and_skips_anti_spoof():
    model = Mock()
    service, embedder, identity_index = make_service(model)
    service.face_preprocessor.extract.side_effect = InvalidFaceCountError(0)

    response = post_predict(service)

    assert response.status_code == 422
    model.predict.assert_not_called()
    assert_recognition_skipped(embedder, identity_index)
