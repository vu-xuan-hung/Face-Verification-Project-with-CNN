"""Real HTTP/SQLite/enrollment tests; synthetic ML boundaries are not accuracy evidence."""

import base64
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from vshield.api import database, sessions
from vshield.api.app import create_app
from vshield.services.authentication import AuthenticationResult, AuthenticationStatus
from vshield.services.user_management import UserManagementService


class SyntheticFaces:
    def extract(self, frame):
        return SimpleNamespace(facenet=frame, anti_spoof=frame)

    def encode(self, frame):
        vector = np.zeros(512, dtype=np.float32)
        vector[int(frame[0, 0, 0])] = 1
        return vector


def photos(identity=10):
    images = []
    for view in (1, 2):
        frame = np.full((32, 32, 3), identity, dtype=np.uint8)
        frame[-1, -1] = view
        ok, encoded = cv2.imencode(".png", frame)
        assert ok
        images.append("data:image/png;base64," + base64.b64encode(encoded).decode("ascii"))
    return images


def profile(username="newperson", identity=10):
    return {
        "username": username,
        "name": "New Person",
        "email": username + "@example.test",
        "images": photos(identity),
        "consent": True,
    }


@pytest.fixture
def managed(tmp_path, monkeypatch):
    path = tmp_path / "accounts.db"
    monkeypatch.setattr(database, "get_db_path", lambda: str(path))
    database.init_db()
    for name, role in [("member", "USER"), ("manager", "ADMIN"), ("owner", "SUPER_ADMIN")]:
        database.register_user(name, role)
    model = SyntheticFaces()
    service = UserManagementService(tmp_path / "authorization", model, model, path)
    return service


def auth(name):
    return {"Authorization": "Bearer " + sessions.create_session(name)["access_token"]}


@pytest.mark.parametrize(
    "actor,endpoint,expected",
    [
        ("member", "/users", 403),
        ("member", "/admins", 403),
        ("manager", "/users", 201),
        ("manager", "/admins", 403),
        ("owner", "/users", 201),
        ("owner", "/admins", 201),
    ],
)
def test_creation_matrix(managed, actor, endpoint, expected):
    with TestClient(create_app(object(), user_management_service=managed)) as client:
        response = client.post(endpoint, json=profile(), headers=auth(actor))
    assert response.status_code == expected, response.text
    account = database.get_account("newperson")
    if expected == 201:
        assert account["role"] == ("ADMIN" if endpoint == "/admins" else "USER")
        assert account["created_by"] == database.get_account(actor)["id"]
        assert response.json()["id"] == account["id"]
        assert not {"password_hash", "enrollment_id", "face_embedding"} & response.json().keys()
        assert len(list((managed.root / "faces/newperson").glob("*.png"))) == 2
    else:
        assert account is None and not managed.root.exists()


@pytest.mark.parametrize("actor,expected", [("manager", 403), ("owner", 200)])
def test_promotion_requires_super_admin(managed, actor, expected):
    target = database.get_account("member")["id"]
    with TestClient(create_app(object(), user_management_service=managed)) as client:
        response = client.patch(
            f"/users/{target}/role", json={"role": "ADMIN"}, headers=auth(actor)
        )
    assert response.status_code == expected
    assert database.get_account("member")["role"] == ("ADMIN" if expected == 200 else "USER")


@pytest.mark.parametrize(
    "extra", [{"role": "SUPER_ADMIN"}, {"created_by": 3}, {"face_embedding": [1]}]
)
def test_mass_assignment_rejected(managed, extra):
    with TestClient(create_app(object(), user_management_service=managed)) as client:
        response = client.post("/users", json=profile() | extra, headers=auth("manager"))
        assert response.status_code == 422
        assert all("input" not in error for error in response.json()["detail"])
    assert database.get_account("newperson") is None


@pytest.mark.parametrize(
    "actor,target", [("manager", "manager"), ("manager", "owner"), ("owner", "owner")]
)
@pytest.mark.parametrize(
    "method,suffix,payload",
    [
        ("patch", "", {"name": "Changed"}),
        ("patch", "/status", {"status": "DISABLED"}),
        ("delete", "", None),
    ],
)
def test_privileged_account_scope(managed, actor, target, method, suffix, payload):
    original = database.get_account(target)
    with TestClient(create_app(object(), user_management_service=managed)) as client:
        response = client.request(
            method, f"/users/{original['id']}{suffix}", json=payload, headers=auth(actor)
        )
    assert response.status_code == 403
    assert database.get_account(target) == original


def test_disabled_recognized_identity_and_existing_token_denied(managed):
    target = database.get_account("member")["id"]

    class Recognized:
        def authenticate(self, image):
            return AuthenticationResult(AuthenticationStatus.AUTHENTICATED, "ok", user_id=target)

    token = auth("member")
    with TestClient(create_app(Recognized(), user_management_service=managed)) as client:
        assert client.post("/predict", json={"image": photos()[0]}).status_code == 200
        assert (
            client.patch(
                f"/users/{target}/status", json={"status": "DISABLED"}, headers=auth("manager")
            ).status_code
            == 200
        )
        assert client.post("/predict", json={"image": photos()[0]}).status_code == 403
        assert client.get("/auth/me", headers=token).status_code == 401


@pytest.mark.parametrize("disabled", [False, True])
def test_duplicate_face_rolls_back_account_and_publication(managed, disabled):
    actor = database.get_account("manager")["id"]
    decoded = [
        cv2.imdecode(np.frombuffer(base64.b64decode(p.split(",")[1]), np.uint8), 1)
        for p in photos()
    ]
    managed.create(actor, profile(), decoded, "USER")
    if disabled:
        database.change_account("newperson", disable=True)
    before = database.list_accounts()
    with TestClient(create_app(object(), user_management_service=managed)) as client:
        result = client.post("/users", json=profile("duplicate"), headers=auth("manager"))
    assert result.status_code == 409
    assert database.list_accounts() == before
    assert not (managed.root / "faces/duplicate").exists()


def test_edit_disable_reenable_and_soft_delete_user(managed):
    target = database.get_account("member")["id"]
    with TestClient(create_app(object(), user_management_service=managed)) as client:
        headers = auth("manager")
        assert (
            client.patch(
                f"/users/{target}",
                json={"name": "Updated", "email": "new@example.test"},
                headers=headers,
            ).status_code
            == 200
        )
        for status in ("DISABLED", "ACTIVE"):
            assert (
                client.patch(
                    f"/users/{target}/status", json={"status": status}, headers=headers
                ).status_code
                == 200
            )
        assert client.delete(f"/users/{target}", headers=headers).status_code == 204
    assert database.get_account("member")["status"] == "DELETED"
    with pytest.raises(PermissionError):
        sessions.create_session("member")


@pytest.mark.parametrize("count", [0, 2])
def test_invalid_face_count_is_422(managed, count):
    from vshield.core.face_preprocessor import InvalidFaceCountError

    class InvalidFaces:
        def extract(self, image):
            raise InvalidFaceCountError(count)

    managed.preprocessor = InvalidFaces()
    with TestClient(create_app(object(), user_management_service=managed)) as client:
        response = client.post("/users", json=profile(), headers=auth("manager"))
    assert response.status_code == 422
    assert database.get_account("newperson") is None
