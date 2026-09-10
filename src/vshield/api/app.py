"""FastAPI transport layer for V-Shield authentication."""

from __future__ import annotations

import base64
import binascii
import csv
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError

from vshield.api import database, sessions
from vshield.api.request_limits import EnrollmentBodyLimit
from vshield.api.schemas import ImagePayload
from vshield.api.user_routes import router as user_router
from vshield.services.authentication import (
    AuthenticationResult,
    AuthenticationService,
    AuthenticationStatus,
    build_default_authentication_service,
)
from vshield.services.user_management import UserManagementService

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
    user_management_service=None,
) -> FastAPI:
    """Create an app with injectable dependencies and startup lifecycle."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if initialize_database:
            database.init_db()
        application.state.authentication_service = (
            authentication_service or build_default_authentication_service(PROJECT_ROOT)
        )
        auth_service = application.state.authentication_service
        application.state.user_management_service = user_management_service
        if user_management_service is None and hasattr(auth_service, "face_embedder"):
            application.state.user_management_service = UserManagementService(
                PROJECT_ROOT / "data" / "authorization",
                auth_service.face_preprocessor,
                auth_service.face_embedder,
                identity_index=getattr(auth_service, "identity_index", None),
            )
        yield

    application = FastAPI(lifespan=lifespan)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        errors = [
            {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    @application.middleware("http")
    async def prevent_private_response_caching(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    application.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv(
            "VSHIELD_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(","),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )
    application.add_middleware(EnrollmentBodyLimit)
    application.include_router(user_router)

    @application.post("/predict")
    def predict(data: ImagePayload, response: Response):
        response.headers["Cache-Control"] = "no-store"
        try:
            image = decode_image(data.image)
        except ValueError as exc:
            return _failure(400, str(exc))

        try:
            result: AuthenticationResult = application.state.authentication_service.authenticate(
                image
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
        if result.status is not AuthenticationStatus.AUTHENTICATED or not (
            result.username or result.user_id
        ):
            return _failure(503, "Authentication service returned an invalid result")

        try:
            session = sessions.create_session(result.username, user_id=result.user_id)
        except PermissionError as exc:
            return _failure(403, str(exc))
        except Exception:
            return _failure(503, "Login database unavailable")

        return {
            "success": True,
            **session,
        }

    @application.get("/auth/me")
    def me(response: Response, user=Depends(sessions.current_user)):
        response.headers["Cache-Control"] = "no-store"
        return user

    @application.post("/auth/logout", status_code=204)
    def logout(user=Depends(sessions.current_user), credentials=Depends(sessions.bearer)):
        sessions.revoke_session(credentials.credentials)
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    @application.get("/logs", dependencies=[Depends(sessions.require_admin)])
    def get_logs(username: str | None = None, date: str | None = None):
        return database.get_logs(username_filter=username, date_filter=date)

    @application.get("/logs/export", dependencies=[Depends(sessions.require_admin)])
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
