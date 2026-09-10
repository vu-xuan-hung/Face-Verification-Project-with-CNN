"""FastAPI transport layer for V-Shield authentication."""

from __future__ import annotations

import base64
import binascii
import csv
import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError

from vshield.api import database, sessions
from vshield.api.readiness import readiness
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


def _failure(status_code: int, message: str, *, code=None, metadata=None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "message": message, **(metadata or {}),
                 **({"code": code} if code else {})},
    )


def safe_log_access_event(**kwargs):
    try:
        return database.log_access_event(**kwargs)
    except Exception as exc:
        logging.getLogger(__name__).error("Safe access log dispatch failed: %s", exc)
        return None


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
                pad_service=getattr(auth_service, "anti_spoof_model", None),
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
    def predict(data: ImagePayload, request: Request, response: Response):
        response.headers["Cache-Control"] = "no-store"
        source = request.client.host if request.client else None
        request_id = request.headers.get("x-request-id")

        try:
            image = decode_image(data.image)
        except ValueError as exc:
            safe_log_access_event(
                event_type="ACCESS_DENIED",
                result="DENIED",
                reason_code="INVALID_IMAGE",
                request_id=request_id,
                source=source,
            )
            return _failure(400, str(exc), code="INVALID_IMAGE")

        try:
            result: AuthenticationResult = application.state.authentication_service.authenticate(
                image
            )
        except Exception:
            safe_log_access_event(
                event_type="ACCESS_DENIED",
                result="DENIED",
                reason_code="MODEL_ERROR",
                request_id=request_id,
                source=source,
            )
            return _failure(503, "Authentication service unavailable", code="MODEL_ERROR")

        metadata = result.metadata()
        pad = result.pad
        pad_status = pad.status.value if pad else None
        pad_version = pad.model_version if pad else None
        spoof_score = pad.score if pad else None

        if result.status is AuthenticationStatus.INVALID_FACE:
            safe_log_access_event(
                event_type="ACCESS_DENIED",
                result="DENIED",
                reason_code=result.code or "INVALID_FACE",
                request_id=request_id,
                source=source,
            )
            return _failure(422, result.message, metadata=metadata)
        if result.status is AuthenticationStatus.SPOOF:
            safe_log_access_event(
                event_type="SPOOF_ATTEMPT",
                result="DENIED",
                reason_code=result.code or "SPOOF",
                recognition_distance=None,
                spoof_score=spoof_score,
                pad_status=pad_status,
                pad_model_version=pad_version,
                request_id=request_id,
                source=source,
            )
            return _failure(403, result.message, metadata=metadata)
        if result.status is AuthenticationStatus.UNKNOWN:
            safe_log_access_event(
                event_type="UNKNOWN_FACE",
                result="DENIED",
                reason_code="UNKNOWN",
                recognition_distance=result.distance,
                spoof_score=spoof_score,
                pad_status=pad_status,
                pad_model_version=pad_version,
                request_id=request_id,
                source=source,
            )
            return _failure(403, result.message, metadata=metadata)
        if result.status is AuthenticationStatus.AMBIGUOUS:
            safe_log_access_event(
                event_type="AMBIGUOUS_FACE",
                result="DENIED",
                reason_code="AMBIGUOUS",
                recognition_distance=result.distance,
                spoof_score=spoof_score,
                pad_status=pad_status,
                pad_model_version=pad_version,
                request_id=request_id,
                source=source,
            )
            return _failure(403, result.message, metadata=metadata)
        if result.status is AuthenticationStatus.UNAVAILABLE:
            if result.code == "PAD_UNAVAILABLE":
                evt = "PAD_UNAVAILABLE"
            elif (pad and pad.status.value == "error") or result.code in ("PAD_ERROR", "MODEL_ERROR"):
                evt = "PAD_ERROR"
            else:
                evt = "ACCESS_DENIED"
            safe_log_access_event(
                event_type=evt,
                result="DENIED",
                reason_code=result.code or evt,
                recognition_distance=result.distance,
                spoof_score=spoof_score,
                pad_status=pad_status,
                pad_model_version=pad_version,
                request_id=request_id,
                source=source,
            )
            return _failure(503, result.message, metadata=metadata)
        if result.status is not AuthenticationStatus.AUTHENTICATED or not (
            result.username or result.user_id
        ):
            safe_log_access_event(
                event_type="ACCESS_DENIED",
                result="DENIED",
                reason_code="INVALID_RESULT",
                request_id=request_id,
                source=source,
            )
            return _failure(503, "Authentication service returned an invalid result")

        resolved_user_id = result.user_id
        if resolved_user_id is None and result.username:
            account = database.get_account(result.username)
            if account:
                resolved_user_id = account["id"]

        try:
            session = sessions.create_session(result.username, user_id=result.user_id)
        except PermissionError as exc:
            safe_log_access_event(
                event_type="USER_DISABLED",
                result="DENIED",
                reason_code="USER_DISABLED",
                user_id=resolved_user_id,
                recognition_distance=result.distance,
                spoof_score=spoof_score,
                pad_status=pad_status,
                pad_model_version=pad_version,
                request_id=request_id,
                source=source,
            )
            return _failure(403, str(exc), code="USER_DISABLED", metadata=metadata)
        except Exception:
            safe_log_access_event(
                event_type="ACCESS_DENIED",
                result="DENIED",
                reason_code="DATABASE_ERROR",
                user_id=resolved_user_id,
                recognition_distance=result.distance,
                request_id=request_id,
                source=source,
            )
            return _failure(503, "Login database unavailable")

        safe_log_access_event(
            event_type="ACCESS_GRANTED",
            result="GRANTED",
            reason_code="SUCCESS",
            user_id=session["user_id"],
            recognition_distance=result.distance,
            spoof_score=spoof_score,
            pad_status=pad_status,
            pad_model_version=pad_version,
            request_id=request_id,
            source=source,
        )

        return {
            "success": True,
            **metadata,
            **session,
        }

    @application.get("/health/ready")
    def health_ready():
        status = readiness(application.state.authentication_service)
        return JSONResponse(status, status_code=200 if status["ready"] else 503)

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

    @application.get("/access-logs", dependencies=[Depends(sessions.require_admin)])
    def list_access_logs(
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        event_type: str | None = None,
        result: str | None = None,
        username: str | None = None,
        user_id: int | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ):
        return database.get_access_logs(
            user_id=user_id,
            username=username,
            event_type=event_type,
            result=result,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset,
        )

    @application.get("/access-logs/me")
    def my_access_logs(
        user=Depends(sessions.current_user),
        limit: int = Query(50, ge=1, le=100),
        offset: int = Query(0, ge=0),
    ):
        # user_id is derived strictly from the authenticated session, never client input
        return database.get_access_logs(
            user_id=user["id"],
            limit=limit,
            offset=offset,
        )

    @application.get("/access-logs/export", dependencies=[Depends(sessions.require_admin)])
    def export_access_logs(
        event_type: str | None = None,
        result: str | None = None,
        username: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ):
        data = database.get_access_logs(
            username=username,
            event_type=event_type,
            result=result,
            start_time=start_time,
            end_time=end_time,
            limit=500,
            offset=0,
        )
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "id", "timestamp", "user_id", "username", "event_type",
            "result", "reason_code", "recognition_distance",
            "spoof_score", "pad_status", "pad_model_version", "source"
        ])
        for row in data["items"]:
            writer.writerow([
                row["id"],
                row["timestamp"],
                row["user_id"] or "",
                row["username"] or "",
                row["event_type"],
                row["result"],
                row["reason_code"] or "",
                row["recognition_distance"] if row["recognition_distance"] is not None else "",
                row["spoof_score"] if row["spoof_score"] is not None else "",
                row["pad_status"] or "",
                row["pad_model_version"] or "",
                row["source"] or "",
            ])
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=access_logs_export.csv"},
        )

    @application.get("/dashboard/stats", dependencies=[Depends(sessions.require_admin)])
    def dashboard_stats():
        return database.get_dashboard_stats()


    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
