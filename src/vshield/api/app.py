"""FastAPI transport layer for V-Shield authentication."""

from __future__ import annotations

import base64
import binascii
import csv
import io
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError

from vshield.api import database
from vshield.api.schemas import ImagePayload
from vshield.services.authentication import (
    AuthenticationResult,
    AuthenticationService,
    AuthenticationStatus,
    build_default_authentication_service,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAX_ENCODED_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


def decode_image(data_uri: str) -> np.ndarray:
    """Decode one base64 image data URI into a BGR uint8 image."""
    if not data_uri or "," not in data_uri:
        raise ValueError("Invalid image format")

    header, encoded = data_uri.split(",", 1)
    if not header.startswith("data:image/") or ";base64" not in header:
        raise ValueError("Invalid image format")
    if len(encoded) > (MAX_ENCODED_IMAGE_BYTES * 4 // 3) + 4:
        raise ValueError("Image payload is too large")

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image") from exc
    if len(image_bytes) > MAX_ENCODED_IMAGE_BYTES:
        raise ValueError("Image payload is too large")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image_header:
            width, height = image_header.size
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Can't decode image") from exc
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError("Decoded image is too large")

    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Can't decode image")
    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError("Decoded image is too large")
    return image


def _failure(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "message": message},
    )


def create_app(
    authentication_service: AuthenticationService | None = None,
    *,
    initialize_database: bool = True,
) -> FastAPI:
    """Create an app with injectable dependencies and startup lifecycle."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if initialize_database:
            database.init_db()
        application.state.authentication_service = (
            authentication_service
            or build_default_authentication_service(PROJECT_ROOT)
        )
        yield

    application = FastAPI(lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.post("/predict")
    def predict(data: ImagePayload):
        try:
            image = decode_image(data.image)
        except ValueError as exc:
            return _failure(400, str(exc))

        try:
            result: AuthenticationResult = (
                application.state.authentication_service.authenticate(image)
            )
        except Exception:
            return _failure(503, "Authentication service unavailable")

        if result.status is AuthenticationStatus.INVALID_FACE:
            return _failure(422, result.message)
        if result.status is AuthenticationStatus.SPOOF:
            return _failure(403, result.message)
        if result.status is AuthenticationStatus.UNKNOWN:
            return _failure(403, result.message)
        if result.status is AuthenticationStatus.UNAVAILABLE:
            return _failure(503, result.message)
        if result.status is not AuthenticationStatus.AUTHENTICATED or not result.username:
            return _failure(503, "Authentication service returned an invalid result")

        try:
            role = database.get_role(result.username)
            database.log_login(result.username, role)
        except Exception:
            return _failure(503, "Login database unavailable")

        return {
            "success": True,
            "username": result.username,
            "role": role,
        }

    @application.get("/logs")
    def get_logs(username: str | None = None, date: str | None = None):
        return database.get_logs(username_filter=username, date_filter=date)

    @application.get("/logs/export")
    def export_logs(username: str | None = None, date: str | None = None):
        logs = database.get_logs(username_filter=username, date_filter=date)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["username", "role", "time"])
        for log in logs:
            writer.writerow([log["username"], log["role"], log["timestamp"]])

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=logs_export.csv"},
        )

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
