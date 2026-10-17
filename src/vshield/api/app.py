"""FastAPI application assembly, lifecycle, and request policies."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from vshield.api import database
from vshield.api.auth_routes import router as auth_router
from vshield.api.log_routes import router as log_router
from vshield.api.prediction_routes import router as prediction_router
from vshield.api.request_limits import EnrollmentBodyLimit, PredictBodyLimit
from vshield.api.user_routes import router as user_router
from vshield.services.authentication import (
    AuthenticationService,
    build_default_authentication_service,
)
from vshield.services.user_management import UserManagementService

PROJECT_ROOT = Path(__file__).resolve().parents[3]


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

    application.add_middleware(PredictBodyLimit)
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

    application.include_router(prediction_router)
    application.include_router(auth_router)
    application.include_router(log_router)
    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
