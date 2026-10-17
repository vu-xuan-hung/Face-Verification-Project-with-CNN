"""Multipart transport regressions; authentication decisions remain in the service."""

import io
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vshield.api import app as transport
from vshield.api import image_decoding, prediction_routes
from vshield.services.authentication import AuthenticationResult, AuthenticationStatus


def encoded_image(extension=".png"):
    frame = np.full((24, 32, 3), (20, 80, 180), dtype=np.uint8)
    ok, encoded = cv2.imencode(extension, frame)
    assert ok
    return encoded.tobytes()


@pytest.fixture
def upload_client(monkeypatch):
    service = Mock(spec=["authenticate"])
    service.authenticate.return_value = AuthenticationResult(
        AuthenticationStatus.UNKNOWN, "Unknown face", code="UNKNOWN"
    )
    audit = Mock()
    monkeypatch.setattr(prediction_routes, "safe_log_access_event", audit)
    with TestClient(transport.create_app(service, initialize_database=False)) as client:
        yield client, service, audit


@pytest.mark.parametrize(("extension", "mime"), [(".jpg", "image/jpeg"), (".png", "image/png")])
def test_valid_upload_is_bgr_uint8(upload_client, extension, mime):
    client, service, audit = upload_client
    data = encoded_image(extension)
    # Filename has no authority over the actual image format.
    response = client.post("/predict", files={"file": ("untrusted.bin", data, mime)})
    assert response.status_code == 403 and response.json()["code"] == "UNKNOWN"
    assert response.headers["cache-control"] == "no-store"
    service.authenticate.assert_called_once()
    frame = service.authenticate.call_args.args[0]
    assert frame.dtype == np.uint8 and frame.shape == (24, 32, 3)
    np.testing.assert_array_equal(frame, cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR))
    audit.assert_called_once()


@pytest.mark.parametrize(("data", "mime"), [
    (b"", "image/png"), (b"not an image", "image/jpeg"),
    (b"%PDF-1.7\n", "image/png"),
    (encoded_image(".bmp"), "image/png"),
    (encoded_image(), "image/jpeg"),
    (encoded_image(".jpg"), "image/png"),
    (encoded_image(), "application/octet-stream"),
    (encoded_image()[:40], "image/png"),
    (encoded_image(".jpg")[:-2], "image/jpeg"),
])
def test_invalid_upload_never_reaches_authentication(upload_client, data, mime):
    client, service, audit = upload_client
    response = client.post("/predict", files={"file": ("face.jpg", data, mime)},
                           headers={"x-request-id": "upload-check"})
    assert response.status_code == 400 and response.json()["code"] == "INVALID_IMAGE"
    assert response.headers["cache-control"] == "no-store"
    service.authenticate.assert_not_called()
    audit.assert_called_once_with(event_type="ACCESS_DENIED", result="DENIED",
                                 reason_code="INVALID_IMAGE", request_id="upload-check",
                                 source="testclient")


@pytest.mark.parametrize("size", [1025, 1024 + 64 * 1024 + 1])
def test_file_and_total_body_size_limits(upload_client, monkeypatch, size):
    client, service, audit = upload_client
    monkeypatch.setattr(image_decoding, "MAX_ENCODED_IMAGE_BYTES", 1024)
    response = client.post("/predict", files={"file": ("face.png", b"x" * size, "image/png")},
                           headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 400 and response.json()["code"] == "INVALID_IMAGE"
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    service.authenticate.assert_not_called()
    audit.assert_called_once()


def test_resolution_limit_precedes_opencv_decode(upload_client, monkeypatch):
    client, service, _ = upload_client
    monkeypatch.setattr(image_decoding, "MAX_IMAGE_PIXELS", 100)
    decode = Mock(side_effect=AssertionError("must reject before allocating decoded pixels"))
    monkeypatch.setattr(image_decoding.cv2, "imdecode", decode)
    response = client.post("/predict", files={"file": ("face.png", encoded_image(), "image/png")})
    assert response.status_code == 400 and response.json()["code"] == "INVALID_IMAGE"
    decode.assert_not_called()
    service.authenticate.assert_not_called()


def test_chunked_request_without_content_length_is_bounded(upload_client, monkeypatch):
    client, service, audit = upload_client
    monkeypatch.setattr(image_decoding, "MAX_ENCODED_IMAGE_BYTES", 1024)
    response = client.post("/predict", content=iter([b"x" * 32768] * 3),
                           headers={"Content-Type": "multipart/form-data; boundary=test"})
    assert "content-length" not in response.request.headers
    assert response.status_code == 400 and response.json()["code"] == "INVALID_IMAGE"
    service.authenticate.assert_not_called()
    audit.assert_called_once()


@pytest.mark.parametrize("payload", [{}, {"image": "data:image/png;base64,AAAA"}])
def test_missing_file_and_legacy_json_are_validation_errors(upload_client, payload):
    client, service, _ = upload_client
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["detail"][0]["loc"] == ["body", "file"]
    service.authenticate.assert_not_called()


@pytest.mark.parametrize("valid", [True, False])
def test_uploaded_resource_closed(upload_client, monkeypatch, valid):
    from starlette.datastructures import UploadFile

    client, _, _ = upload_client
    closed = []
    original = UploadFile.close

    async def track_close(self):
        closed.append(self.file)
        await original(self)

    monkeypatch.setattr(UploadFile, "close", track_close)
    client.post("/predict", files={"file": ("face.png", encoded_image() if valid else b"bad", "image/png")})
    assert closed and all(file.closed for file in closed)


def test_enrollment_base64_decoder_keeps_other_formats():
    import base64

    buffer = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buffer, format="WEBP")
    data_uri = "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    assert image_decoding.decode_image(data_uri).shape == (16, 16, 3)


def test_openapi_exposes_required_binary_upload(upload_client):
    client, _, _ = upload_client
    schema = client.get("/openapi.json").json()
    body = schema["paths"]["/predict"]["post"]["requestBody"]
    assert body["required"] is True
    assert set(body["content"]) == {"multipart/form-data"}
    reference = body["content"]["multipart/form-data"]["schema"]["$ref"].split("/")[-1]
    upload_schema = schema["components"]["schemas"][reference]
    assert upload_schema["required"] == ["file"]
    assert upload_schema["properties"]["file"]["type"] == "string"
    assert upload_schema["properties"]["file"]["format"] == "binary"
    assert "ImagePayload" not in schema["components"]["schemas"]
    assert "SwaggerUIBundle" in client.get("/docs").text
