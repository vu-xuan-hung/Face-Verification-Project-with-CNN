"""Administrator audit exports and authenticated access-log queries."""

import csv
import io

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from vshield.api import database, sessions

router = APIRouter()


@router.get("/logs", dependencies=[Depends(sessions.require_admin)])
def get_logs(username: str | None = None, date: str | None = None):
    return database.get_logs(username_filter=username, date_filter=date)


@router.get("/logs/export", dependencies=[Depends(sessions.require_admin)])
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


@router.get("/access-logs", dependencies=[Depends(sessions.require_admin)])
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


@router.get("/access-logs/me")
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


@router.get("/access-logs/export", dependencies=[Depends(sessions.require_admin)])
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
    writer.writerow(
        [
            "id",
            "timestamp",
            "user_id",
            "username",
            "event_type",
            "result",
            "reason_code",
            "recognition_distance",
            "spoof_score",
            "pad_status",
            "pad_model_version",
            "source",
        ]
    )
    for row in data["items"]:
        writer.writerow(
            [
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
            ]
        )
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=access_logs_export.csv"},
    )


@router.get("/dashboard/stats", dependencies=[Depends(sessions.require_admin)])
def dashboard_stats():
    return database.get_dashboard_stats()
