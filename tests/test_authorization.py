"""Real SQLite/session + HTTP authorization regression tests (no real face identities)."""

import base64
import sqlite3

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from vshield.api import database, sessions
from vshield.api.app import create_app
from vshield.services.authentication import AuthenticationResult, AuthenticationStatus


@pytest.fixture
def account_db(tmp_path, monkeypatch):
    path = tmp_path / "accounts.db"
    monkeypatch.setattr(database, "get_db_path", lambda: str(path))
    database.init_db()
    database.register_user("alice", "user")
    database.register_user("owner", "admin")
    return path


class RecognizedFace:
    """Isolate ML from transport tests; these are not face accuracy evidence."""

    def __init__(self, username="alice"):
        self.username = username

    def authenticate(self, image):
        return AuthenticationResult(AuthenticationStatus.AUTHENTICATED, "ok", self.username)


def image_payload():
    ok, encoded = cv2.imencode(".png", np.zeros((32, 32, 3), dtype=np.uint8))
    assert ok
    return {"image": "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")}


def headers(username):
    return {"Authorization": "Bearer " + sessions.create_session(username)["access_token"]}


def test_login_session_me_logout_and_admin_access(account_db):
    with TestClient(create_app(RecognizedFace())) as client:
        response = client.post("/predict", json=image_payload())
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        data = response.json()
        assert data["role"] == "USER" and data["expires_in"] == 3600
        auth = {"Authorization": "Bearer " + data["access_token"]}
        current = client.get("/auth/me", headers=auth).json()
        assert current == {
            "id": data["id"],
            "user_id": data["id"],
            "username": "alice",
            "name": "alice",
            "email": None,
            "status": "ACTIVE",
            "role": "USER",
        }
        assert client.get("/logs", headers=auth).status_code == 403
        assert client.get("/logs/export", headers=auth).status_code == 403
        admin = headers("owner")
        assert client.get("/logs", headers=admin).status_code == 200
        assert (
            client.get("/logs/export", headers=admin).headers["content-type"].startswith("text/csv")
        )
        assert client.post("/auth/logout", headers=auth).status_code == 204
        assert client.get("/auth/me", headers=auth).status_code == 401
        assert client.get("/logs", headers=auth).status_code == 401


@pytest.mark.parametrize("path", ["/auth/me", "/logs", "/logs/export"])
@pytest.mark.parametrize(
    "auth",
    [{}, {"Authorization": "Bearer admin"}, {"role": "admin"}, {"Authorization": "Basic alice"}],
)
def test_no_client_role_or_invalid_token_bypass(account_db, path, auth):
    with TestClient(create_app(RecognizedFace())) as client:
        assert client.get(path, headers=auth).status_code == 401


def test_unknown_and_disabled_accounts_cannot_login(account_db):
    for name in ("outsider", "alice"):
        database.change_account("alice", disable=True)
        with TestClient(create_app(RecognizedFace(name))) as client:
            assert client.post("/predict", json=image_payload()).status_code == 403
    assert database.get_logs() == []


def test_hash_storage_expiry_role_change_and_disable(account_db, monkeypatch):
    token = sessions.create_session("owner")["access_token"]
    with sqlite3.connect(account_db) as conn:
        stored = conn.execute("SELECT token_hash FROM sessions").fetchone()[0]
    assert token != stored and stored == sessions.token_hash(token)
    now = sessions.time.time()
    monkeypatch.setattr(sessions.time, "time", lambda: now + sessions.SESSION_SECONDS + 1)
    assert sessions.resolve_session(token) is None
    monkeypatch.setattr(sessions.time, "time", lambda: now)
    database.change_account("owner", role="user")
    assert sessions.resolve_session(token) is None
    token = sessions.create_session("owner")["access_token"]
    assert sessions.resolve_session(token)["role"] == "USER"
    database.change_account("owner", disable=True)
    assert sessions.resolve_session(token) is None
    with pytest.raises(PermissionError):
        sessions.create_session("owner")


def test_migration_preserves_logs_disables_implicit_accounts(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE users(id INTEGER PRIMARY KEY, username TEXT UNIQUE, role TEXT)")
        conn.execute("INSERT INTO users VALUES(1,'hung','admin')")
    database.init_db(path)
    database.init_db(path)
    assert database.get_account("hung", path)["active"] == 0
    assert database.get_role("hung", path) is None
    database.register_user("hung", "admin", path)
    assert database.get_role("hung", path) == "ADMIN"


@pytest.mark.parametrize("username", ["../alice", "Alice", "a/b", "", "con", "a.b", "nul"])
def test_unsafe_usernames_rejected(account_db, username):
    with pytest.raises(ValueError):
        database.register_user(username)


def test_invalid_role_rejected(account_db):
    with pytest.raises(ValueError):
        database.register_user("alice", "superuser")
    with pytest.raises(ValueError):
        database.change_account("alice", role="superuser")


def test_no_default_admin_created(tmp_path):
    path = tmp_path / "new.db"
    database.init_db(path)
    assert database.list_accounts(path) == []
