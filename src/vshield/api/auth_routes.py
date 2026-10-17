"""Service readiness and session endpoints."""

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from vshield.api import sessions
from vshield.api.readiness import readiness

router = APIRouter()


@router.get("/health/ready")
def health_ready(request: Request):
    status = readiness(request.app.state.authentication_service)
    return JSONResponse(status, status_code=200 if status["ready"] else 503)


@router.get("/auth/me")
def me(response: Response, user=Depends(sessions.current_user)):
    response.headers["Cache-Control"] = "no-store"
    return user


@router.post("/auth/logout", status_code=204)
def logout(user=Depends(sessions.current_user), credentials=Depends(sessions.bearer)):
    sessions.revoke_session(credentials.credentials)
    return Response(status_code=204, headers={"Cache-Control": "no-store"})
