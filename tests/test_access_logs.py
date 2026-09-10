"""Tests for persistent access logs, RBAC enforcement, filtering, and dashboard statistics."""

import base64
import sqlite3

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from vshield.api import database, sessions
from vshield.api.app import create_app
from vshield.core.anti_spoof import PadResult, PadStatus
from vshield.services.authentication import (
    AuthenticationResult,
    AuthenticationStatus,
)


class MockAuthService:
    def __init__(self, outcome_fn=None):
        self.outcome_fn = outcome_fn

    def authenticate(self, image):
        if self.outcome_fn:
            return self.outcome_fn(image)
        return AuthenticationResult(
            AuthenticationStatus.AUTHENTICATED,
            "Success",
            username="member",
            distance=0.25,
            code="SUCCESS",
            pad=PadResult(
                status=PadStatus.REAL,
                score=0.98,
                class_index=1,
                model_version="test-v2",
                threshold=0.8,
                reason="PAD_REAL",
            ),
        )


def make_dummy_image():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", img)
    assert ok
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_access.db"
    monkeypatch.setattr(database, "get_db_path", lambda: str(db_path))
    database.init_db(str(db_path))
    for name, role in [("member", "USER"), ("manager", "ADMIN"), ("owner", "SUPER_ADMIN"), ("disabled_user", "USER")]:
        database.register_user(name, role, str(db_path))
    database.change_account("disabled_user", disable=True, db_path=str(db_path))
    return str(db_path)


def auth_header(username, db_path):
    session = sessions.create_session(username, db_path=db_path)
    return {"Authorization": f"Bearer {session['access_token']}"}


def test_schema_and_migration(test_db):
    logs = database.get_access_logs(db_path=test_db)
    assert logs["total"] == 0
    assert logs["items"] == []


def test_spoof_attempt_creates_correct_log(test_db):
    def outcome(_):
        return AuthenticationResult(
            AuthenticationStatus.SPOOF,
            "Spoofing detected",
            code="SPOOF",
            pad=PadResult(
                status=PadStatus.FAKE,
                score=0.15,
                class_index=0,
                model_version="minifasnet_v2",
                threshold=0.8,
                reason="PAD_SPOOF",
            ),
        )

    auth_svc = MockAuthService(outcome)
    app = create_app(auth_svc, initialize_database=False)
    with TestClient(app) as client:
        resp = client.post("/predict", json={"image": make_dummy_image()})
        assert resp.status_code == 403

    logs = database.get_access_logs(db_path=test_db)
    assert logs["total"] == 1
    log = logs["items"][0]
    assert log["event_type"] == "SPOOF_ATTEMPT"
    assert log["result"] == "DENIED"
    assert log["reason_code"] == "SPOOF"
    assert log["recognition_distance"] is None
    assert log["spoof_score"] == pytest.approx(0.15)
    assert log["pad_status"] == "FAKE"
    assert log["pad_model_version"] == "minifasnet_v2"


def test_unknown_face_creates_correct_log(test_db):
    def outcome(_):
        return AuthenticationResult(
            AuthenticationStatus.UNKNOWN,
            "Unknown face",
            distance=1.23,
            code="UNKNOWN",
            pad=PadResult(
                status=PadStatus.REAL,
                score=0.95,
                class_index=1,
                model_version="minifasnet_v2",
                threshold=0.8,
                reason="PAD_REAL",
            ),
        )

    auth_svc = MockAuthService(outcome)
    app = create_app(auth_svc, initialize_database=False)
    with TestClient(app) as client:
        resp = client.post("/predict", json={"image": make_dummy_image()})
        assert resp.status_code == 403

    logs = database.get_access_logs(db_path=test_db)
    assert logs["total"] == 1
    log = logs["items"][0]
    assert log["event_type"] == "UNKNOWN_FACE"
    assert log["result"] == "DENIED"
    assert log["reason_code"] == "UNKNOWN"
    assert log["recognition_distance"] == pytest.approx(1.23)
    assert log["user_id"] is None


def test_ambiguous_face_creates_correct_log(test_db):
    def outcome(_):
        return AuthenticationResult(
            AuthenticationStatus.AMBIGUOUS,
            "Ambiguous match",
            distance=0.55,
            code="AMBIGUOUS",
            pad=PadResult(
                status=PadStatus.REAL,
                score=0.92,
                class_index=1,
                model_version="minifasnet_v2",
                threshold=0.8,
                reason="PAD_REAL",
            ),
        )

    auth_svc = MockAuthService(outcome)
    app = create_app(auth_svc, initialize_database=False)
    with TestClient(app) as client:
        resp = client.post("/predict", json={"image": make_dummy_image()})
        assert resp.status_code == 403

    logs = database.get_access_logs(db_path=test_db)
    assert logs["total"] == 1
    log = logs["items"][0]
    assert log["event_type"] == "AMBIGUOUS_FACE"
    assert log["result"] == "DENIED"
    assert log["reason_code"] == "AMBIGUOUS"
    assert log["recognition_distance"] == pytest.approx(0.55)


def test_disabled_recognized_user_creates_denied_log(test_db):
    def outcome(_):
        return AuthenticationResult(
            AuthenticationStatus.AUTHENTICATED,
            "Success",
            username="disabled_user",
            distance=0.31,
            code="SUCCESS",
            pad=PadResult(
                status=PadStatus.REAL,
                score=0.91,
                class_index=1,
                model_version="minifasnet_v2",
                threshold=0.8,
                reason="PAD_REAL",
            ),
        )

    auth_svc = MockAuthService(outcome)
    app = create_app(auth_svc, initialize_database=False)
    with TestClient(app) as client:
        resp = client.post("/predict", json={"image": make_dummy_image()})
        assert resp.status_code == 403

    logs = database.get_access_logs(db_path=test_db)
    assert logs["total"] == 1
    log = logs["items"][0]
    assert log["event_type"] == "USER_DISABLED"
    assert log["result"] == "DENIED"
    assert log["reason_code"] == "USER_DISABLED"
    assert log["username"] == "disabled_user"
    assert log["recognition_distance"] == pytest.approx(0.31)


def test_successful_recognition_creates_granted_log(test_db):
    auth_svc = MockAuthService()
    app = create_app(auth_svc, initialize_database=False)
    with TestClient(app) as client:
        resp = client.post("/predict", json={"image": make_dummy_image()})
        assert resp.status_code == 200

    logs = database.get_access_logs(db_path=test_db)
    assert logs["total"] == 1
    log = logs["items"][0]
    assert log["event_type"] == "ACCESS_GRANTED"
    assert log["result"] == "GRANTED"
    assert log["reason_code"] == "SUCCESS"
    assert log["username"] == "member"
    assert log["recognition_distance"] == pytest.approx(0.25)


def test_pad_unavailable_and_pad_error_logs(test_db):
    # PAD_UNAVAILABLE
    auth_svc = MockAuthService(lambda _: AuthenticationResult(
        AuthenticationStatus.UNAVAILABLE, "Anti-spoof unavailable", code="PAD_UNAVAILABLE"
    ))
    app = create_app(auth_svc, initialize_database=False)
    with TestClient(app) as client:
        resp = client.post("/predict", json={"image": make_dummy_image()})
        assert resp.status_code == 503

    # PAD_ERROR
    auth_svc2 = MockAuthService(lambda _: AuthenticationResult(
        AuthenticationStatus.UNAVAILABLE, "PAD inference error", code="PAD_ERROR",
        pad=PadResult(status=PadStatus.ERROR, score=0.0, class_index=0,
                      model_version="minifasnet_v2", threshold=0.8, reason="PAD_ERROR")
    ))
    app2 = create_app(auth_svc2, initialize_database=False)
    with TestClient(app2) as client:
        resp2 = client.post("/predict", json={"image": make_dummy_image()})
        assert resp2.status_code == 503

    logs = database.get_access_logs(db_path=test_db)
    assert logs["total"] == 2
    types = [log["event_type"] for log in logs["items"]]
    assert "PAD_UNAVAILABLE" in types
    assert "PAD_ERROR" in types


def test_access_logs_rbac_enforcement(test_db):
    auth_svc = MockAuthService()
    app = create_app(auth_svc, initialize_database=False)

    with TestClient(app) as client:
        # USER -> 403 Forbidden
        resp_user = client.get("/access-logs", headers=auth_header("member", test_db))
        assert resp_user.status_code == 403

        # ADMIN -> 200 OK
        resp_admin = client.get("/access-logs", headers=auth_header("manager", test_db))
        assert resp_admin.status_code == 200

        # SUPER_ADMIN -> 200 OK
        resp_super = client.get("/access-logs", headers=auth_header("owner", test_db))
        assert resp_super.status_code == 200


def test_access_logs_me_isolation(test_db):
    auth_svc = MockAuthService()
    app = create_app(auth_svc, initialize_database=False)

    # Perform a successful login for member
    with TestClient(app) as client:
        client.post("/predict", json={"image": make_dummy_image()})

    # Log an event for manager
    database.log_access_event(
        event_type="ACCESS_GRANTED",
        result="GRANTED",
        user_id=database.get_account("manager", test_db)["id"],
        db_path=test_db,
    )

    with TestClient(app) as client:
        # member queries /access-logs/me
        resp = client.get("/access-logs/me", headers=auth_header("member", test_db))
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["username"] == "member"

        # Even if client passes ?user_id=... it must be ignored by /me endpoint
        resp_inject = client.get("/access-logs/me?user_id=2", headers=auth_header("member", test_db))
        assert resp_inject.status_code == 200
        data_inject = resp_inject.json()
        assert data_inject["total"] == 1
        assert data_inject["items"][0]["username"] == "member"


def test_pagination_and_filtering(test_db):
    # Insert multiple distinct events
    database.log_access_event(event_type="SPOOF_ATTEMPT", result="DENIED", db_path=test_db)
    database.log_access_event(event_type="UNKNOWN_FACE", result="DENIED", db_path=test_db)
    database.log_access_event(event_type="ACCESS_GRANTED", result="GRANTED",
                              user_id=database.get_account("member", test_db)["id"], db_path=test_db)
    database.log_access_event(event_type="ACCESS_GRANTED", result="GRANTED",
                              user_id=database.get_account("manager", test_db)["id"], db_path=test_db)

    auth_svc = MockAuthService()
    app = create_app(auth_svc, initialize_database=False)
    with TestClient(app) as client:
        headers = auth_header("manager", test_db)

        # Pagination: limit=2
        resp = client.get("/access-logs?limit=2&offset=0", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 4
        assert len(data["items"]) == 2

        # Filter by result=GRANTED
        resp_g = client.get("/access-logs?result=GRANTED", headers=headers)
        assert resp_g.json()["total"] == 2

        # Filter by event_type=SPOOF_ATTEMPT
        resp_s = client.get("/access-logs?event_type=SPOOF_ATTEMPT", headers=headers)
        assert resp_s.json()["total"] == 1
        assert resp_s.json()["items"][0]["event_type"] == "SPOOF_ATTEMPT"

        # Filter by username=member
        resp_u = client.get("/access-logs?username=member", headers=headers)
        assert resp_u.json()["total"] == 1
        assert resp_u.json()["items"][0]["username"] == "member"


def test_dashboard_stats_use_real_persisted_data(test_db):
    database.log_access_event(event_type="ACCESS_GRANTED", result="GRANTED", db_path=test_db)
    database.log_access_event(event_type="SPOOF_ATTEMPT", result="DENIED", db_path=test_db)
    database.log_access_event(event_type="UNKNOWN_FACE", result="DENIED", db_path=test_db)

    auth_svc = MockAuthService()
    app = create_app(auth_svc, initialize_database=False)
    with TestClient(app) as client:
        resp = client.get("/dashboard/stats", headers=auth_header("manager", test_db))
        assert resp.status_code == 200
        stats = resp.json()
        assert stats["total_users"] == 4  # member, manager, owner, disabled_user
        assert stats["total_admins"] == 2  # manager (ADMIN), owner (SUPER_ADMIN)
        assert stats["recognitions_today"] == 3
        assert stats["granted_today"] == 1
        assert stats["denied_today"] == 2
        assert stats["spoof_attempts_today"] == 1
        assert stats["unknown_today"] == 1


def test_csv_export(test_db):
    database.log_access_event(event_type="ACCESS_GRANTED", result="GRANTED", db_path=test_db)
    auth_svc = MockAuthService()
    app = create_app(auth_svc, initialize_database=False)

    with TestClient(app) as client:
        resp_user = client.get("/access-logs/export", headers=auth_header("member", test_db))
        assert resp_user.status_code == 403

        resp_admin = client.get("/access-logs/export", headers=auth_header("manager", test_db))
        assert resp_admin.status_code == 200
        assert "text/csv" in resp_admin.headers["content-type"]
        content = resp_admin.text
        assert "event_type" in content
        assert "ACCESS_GRANTED" in content


def test_logging_failure_preserves_fail_closed_security(test_db, monkeypatch):
    """If the access log write raises an exception, the security decision must not flip to allow."""
    def broken_log(*args, **kwargs):
        raise sqlite3.OperationalError("Simulated disk error or read-only failure")

    monkeypatch.setattr(database, "log_access_event", broken_log)

    def spoof_outcome(_):
        return AuthenticationResult(
            AuthenticationStatus.SPOOF,
            "Spoofing detected",
            code="SPOOF",
            pad=PadResult(
                status=PadStatus.FAKE,
                score=0.10,
                class_index=0,
                model_version="minifasnet_v2",
                threshold=0.8,
                reason="PAD_SPOOF",
            ),
        )

    auth_svc = MockAuthService(spoof_outcome)
    app = create_app(auth_svc, initialize_database=False)
    with TestClient(app) as client:
        resp = client.post("/predict", json={"image": make_dummy_image()})
        # Security decision MUST remain 403 Forbidden fail-closed!
        assert resp.status_code == 403
