"""
Pydantic request/response schemas for the V-Shield FastAPI server.

Keeping schemas in a separate module follows the FastAPI best-practice
of separating concerns (routes vs. data models vs. DB logic).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class ImagePayload(BaseModel):
    """Incoming base64-encoded face image from the React frontend."""

    image: str = Field(
        ...,
        max_length=11_200_000,
        description=(
            "Data URI of the face crop, e.g. "
            "'data:image/jpeg;base64,<base64-string>'"
        ),
    )


# ---------------------------------------------------------------------------
# Response bodies
# ---------------------------------------------------------------------------


class PredictResponse(BaseModel):
    """Result of a single /predict call."""

    success: bool
    user_id: int | None = None
    username: str | None = None
    role: str | None = None
    message: str | None = None
    access_token: str | None = None
    token_type: str | None = None
    expires_in: int | None = None


class LoginLog(BaseModel):
    """A single row from the login_logs table."""

    username: str
    role: str
    timestamp: str


class LogsResponse(BaseModel):
    """Paginated list of login log entries."""

    logs: list[LoginLog]
    total: int
