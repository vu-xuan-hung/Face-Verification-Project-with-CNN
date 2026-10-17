"""Bound management JSON bodies before parsing multiple biometric images."""

from starlette.responses import JSONResponse

from vshield.api import image_decoding, prediction_routes


class PredictBodyLimit:
    """Bound the entire multipart body before Starlette parses/spools any parts."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST" or scope["path"] != "/predict":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            # Allow bounded multipart headers/boundary overhead above the file limit.
            if len(body) + len(chunk) > image_decoding.MAX_ENCODED_IMAGE_BYTES + 64 * 1024:
                headers = dict(scope.get("headers", []))
                client = scope.get("client")
                prediction_routes.safe_log_access_event(
                    event_type="ACCESS_DENIED",
                    result="DENIED",
                    reason_code="INVALID_IMAGE",
                    request_id=headers.get(b"x-request-id", b"").decode("latin-1") or None,
                    source=client[0] if client else None,
                )
                return await JSONResponse(
                    {
                        "success": False,
                        "message": "Image payload is too large",
                        "code": "INVALID_IMAGE",
                    },
                    status_code=400,
                    headers={"Cache-Control": "no-store"},
                )(scope, receive, send)
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        consumed = False

        async def replay():
            nonlocal consumed
            if consumed:
                return await receive()
            consumed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)


class EnrollmentBodyLimit:
    def __init__(self, app, max_bytes=32 * 1024 * 1024):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope["path"] not in {"/users", "/admins"}
        ):
            return await self.app(scope, receive, send)
        messages, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.max_bytes:
                response = JSONResponse(
                    {"detail": "Enrollment payload exceeds 32 MiB"},
                    status_code=413,
                    headers={"Cache-Control": "no-store"},
                )
                return await response(scope, receive, send)
            messages.append(message)
            if not message.get("more_body", False):
                break
        iterator = iter(messages)

        async def replay():
            return next(iterator, {"type": "http.request", "body": b"", "more_body": False})

        await self.app(scope, replay, send)
