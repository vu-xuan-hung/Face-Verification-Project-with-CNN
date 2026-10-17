"""Image upload authentication and access-event dispatch."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from vshield.api import database, image_decoding, sessions
from vshield.services.authentication import AuthenticationResult, AuthenticationStatus

router = APIRouter()


def _failure(status_code: int, message: str, *, code=None, metadata=None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "message": message,
            **(metadata or {}),
            **({"code": code} if code else {}),
        },
    )


def safe_log_access_event(**kwargs):
    try:
        return database.log_access_event(**kwargs)
    except Exception as exc:
        logging.getLogger(__name__).error("Safe access log dispatch failed: %s", exc)
        return None


@router.post("/predict")
def predict(
    request: Request,
    response: Response,
    file: UploadFile = File(
        ...,
        description="JPEG or PNG image, at most 8 MiB and 20 million pixels",
        json_schema_extra={"format": "binary"},
    ),
):
    response.headers["Cache-Control"] = "no-store"
    source = request.client.host if request.client else None
    request_id = request.headers.get("x-request-id")

    try:
        try:
            # Sync route runs in FastAPI's worker pool; never read an unbounded file.
            image_bytes = file.file.read(image_decoding.MAX_ENCODED_IMAGE_BYTES + 1)
            image = image_decoding.decode_image_bytes(
                image_bytes, content_type=file.content_type or ""
            )
        finally:
            file.file.close()
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
        result: AuthenticationResult = request.app.state.authentication_service.authenticate(image)
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
