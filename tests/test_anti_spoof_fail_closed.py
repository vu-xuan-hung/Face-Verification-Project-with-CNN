"""Fail-closed tests for the anti-spoof authentication gate (new MiniFASNet PAD API)."""

import base64
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from vshield.api import app as app_module
from vshield.api import database
from vshield.api.app import create_app, decode_image
from vshield.core.anti_spoof import (
    PadRejectedError,
    PadResult,
    PadStatus,
    UnavailablePad,
    check_pad,
    require_real,
)
from vshield.core.face_preprocessor import FaceCrops, InvalidFaceCountError
from vshield.core.identity_index import MatchResult, RecognitionDecision
from vshield.services.authentication import AuthenticationService

BBOX = (10, 10, 40, 40)


def make_image_data() -> str:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(".png", image)
    assert encoded_ok
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def make_real_pad():
    """Return a mock PAD model that reports REAL with a passing score."""
    model = Mock()
    model.ready = True
    model.predict.return_value = PadResult(
        status=PadStatus.REAL,
        score=0.95,
        class_index=1,
        model_version="test",
        threshold=0.8,
        reason="PAD_REAL",
    )
    return model


def make_fake_pad():
    """Return a mock PAD model that reports FAKE."""
    model = Mock()
    model.ready = True
    model.predict.return_value = PadResult(
        status=PadStatus.FAKE,
        score=0.1,
        class_index=0,
        model_version="test",
        threshold=0.8,
        reason="SPOOF",
    )
    return model


def make_error_pad(reason="MODEL_ERROR"):
    """Return a mock PAD model that reports ERROR."""
    model = Mock()
    model.ready = True
    model.predict.return_value = PadResult(
        status=PadStatus.ERROR,
        reason=reason,
        model_version="test",
    )
    return model


def make_service(pad_model):
    face = np.zeros((64, 64, 3), dtype=np.uint8)
    preprocessor = Mock()
    preprocessor.extract.return_value = FaceCrops(anti_spoof=face, facenet=face, bbox=BBOX)
    embedder = Mock()
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    embedder.encode.return_value = embedding
    identity_index = Mock()
    identity_index.search_decision.return_value = RecognitionDecision(
        "MATCH", MatchResult("1", 0.2, 0.8), 0.2, 0.8
    )
    service = AuthenticationService(
        face_preprocessor=preprocessor,
        anti_spoof_model=pad_model,
        face_embedder=embedder,
        identity_index=identity_index,
        identity_is_user_id=True,
    )
    return service, embedder, identity_index


def post_predict(service):
    app = create_app(service, initialize_database=False)
    with TestClient(app) as client:
        return client.post("/predict", json={"image": make_image_data()})


# ---------------------------------------------------------------------------
# Image decoding tests (unchanged contract)
# ---------------------------------------------------------------------------

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
    identity_index.search_decision.assert_not_called()


# ---------------------------------------------------------------------------
# PAD unavailable / missing model — fail closed
# ---------------------------------------------------------------------------

def test_missing_model_returns_503_and_skips_recognition():
    """None anti_spoof_model should deny with PAD_UNAVAILABLE after detection."""
    service, embedder, identity_index = make_service(None)

    response = post_predict(service)

    assert response.status_code == 503
    body = response.json()
    assert body["success"] is False
    assert "unavailable" in body["message"].lower()
    # Detection runs first (to get bbox), but recognition must be skipped.
    assert_recognition_skipped(embedder, identity_index)


def test_unavailable_pad_returns_503_and_skips_recognition():
    """UnavailablePad sentinel should deny."""
    service, embedder, identity_index = make_service(UnavailablePad())

    response = post_predict(service)

    assert response.status_code == 503
    assert_recognition_skipped(embedder, identity_index)


# ---------------------------------------------------------------------------
# PAD ERROR — fail closed, FaceNet NOT called
# ---------------------------------------------------------------------------

def test_pad_error_returns_503_and_skips_recognition():
    """A PAD model that returns ERROR must deny and not call FaceNet."""
    service, embedder, identity_index = make_service(make_error_pad())

    response = post_predict(service)

    assert response.status_code == 503
    assert_recognition_skipped(embedder, identity_index)


def test_pad_inference_exception_returns_503():
    """If PAD.predict() raises, check_pad() wraps as ERROR → deny."""
    pad = Mock()
    pad.ready = True
    pad.predict.side_effect = RuntimeError("private model path")
    service, embedder, identity_index = make_service(pad)

    response = post_predict(service)

    assert response.status_code == 503
    assert_recognition_skipped(embedder, identity_index)


# ---------------------------------------------------------------------------
# PAD FAKE — deny, FaceNet NOT called
# ---------------------------------------------------------------------------

def test_fake_returns_403_and_skips_recognition():
    service, embedder, identity_index = make_service(make_fake_pad())

    response = post_predict(service)

    assert response.status_code == 403
    body = response.json()
    assert body["success"] is False
    assert "spoof" in body["message"].lower() or body.get("code") in {"SPOOF", "PAD_UNCERTAIN"}
    assert_recognition_skipped(embedder, identity_index)


def test_pad_uncertain_returns_403_and_skips_recognition():
    pad = Mock()
    pad.ready = True
    pad.predict.return_value = PadResult(
        status=PadStatus.UNCERTAIN, score=0.5, class_index=1,
        model_version="test", threshold=0.8, reason="PAD_UNCERTAIN"
    )
    service, embedder, identity_index = make_service(pad)

    response = post_predict(service)

    assert response.status_code == 403
    assert_recognition_skipped(embedder, identity_index)


# ---------------------------------------------------------------------------
# PAD REAL — FaceNet called, identity search performed
# ---------------------------------------------------------------------------

def test_real_calls_identity_search(monkeypatch, tmp_path):
    service, embedder, identity_index = make_service(make_real_pad())
    monkeypatch.setattr(database, "get_db_path", lambda: str(tmp_path / "auth.db"))
    database.init_db()
    database.register_user("alice", "user")
    alice = database.get_account("alice")
    # Set the index to return alice's integer id as the "username" (identity_is_user_id=True).
    identity_index.search_decision.return_value = RecognitionDecision(
        "MATCH", MatchResult(str(alice["id"]), 0.2, 0.8), 0.2, 0.8
    )

    response = post_predict(service)

    # Should succeed: PAD says REAL, index returns MATCH, DB has alice with that id.
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["access_token"]
    embedder.encode.assert_called_once()
    identity_index.search_decision.assert_called_once()


def test_unknown_face_returns_403(monkeypatch, tmp_path):
    service, _, identity_index = make_service(make_real_pad())
    identity_index.search_decision.return_value = RecognitionDecision("UNKNOWN", distance=1.5)
    monkeypatch.setattr(database, "get_db_path", lambda: str(tmp_path / "auth.db"))
    database.init_db()

    response = post_predict(service)

    assert response.status_code == 403
    assert response.json()["message"] == "Unknown face, not registered"


def test_ambiguous_face_returns_403(monkeypatch, tmp_path):
    service, _, identity_index = make_service(make_real_pad())
    identity_index.search_decision.return_value = RecognitionDecision("AMBIGUOUS", distance=0.4, runner_up_distance=0.41)
    monkeypatch.setattr(database, "get_db_path", lambda: str(tmp_path / "auth.db"))
    database.init_db()

    response = post_predict(service)

    assert response.status_code == 403
    assert "ambiguous" in response.json()["message"].lower()


# ---------------------------------------------------------------------------
# Invalid face (detection failure) — anti-spoof NOT called
# ---------------------------------------------------------------------------

def test_invalid_face_returns_422_and_skips_anti_spoof():
    pad = make_real_pad()
    service, embedder, identity_index = make_service(pad)
    service.face_preprocessor.extract.side_effect = InvalidFaceCountError(0)

    response = post_predict(service)

    assert response.status_code == 422
    pad.predict.assert_not_called()
    assert_recognition_skipped(embedder, identity_index)


# ---------------------------------------------------------------------------
# check_pad() unit tests
# ---------------------------------------------------------------------------

def test_check_pad_returns_error_for_none_model():
    result = check_pad(None, np.zeros((64, 64, 3), dtype=np.uint8), (0, 0, 32, 32))
    assert result.status is PadStatus.ERROR


def test_check_pad_returns_error_for_unavailable_pad():
    result = check_pad(UnavailablePad(), np.zeros((64, 64, 3), dtype=np.uint8), (0, 0, 32, 32))
    assert result.status is PadStatus.ERROR


def test_require_real_raises_pad_rejected_on_fake():
    with pytest.raises(PadRejectedError) as exc_info:
        require_real(make_fake_pad(), np.zeros((64, 64, 3), dtype=np.uint8), (0, 0, 32, 32))
    assert exc_info.value.result.status is PadStatus.FAKE
