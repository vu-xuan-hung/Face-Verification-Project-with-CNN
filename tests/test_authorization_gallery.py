"""Enrollment unit tests: real images/SQLite; deterministic ML boundary only.

Synthetic pixels and embeddings exercise storage safety, not recognition accuracy.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vshield.api import database
from vshield.core.authorization_gallery import load_authorization_gallery
from vshield.core.identity_index_support import IdentityIndexError
from vshield.services.enrollment import enroll


class DeterministicFace:
    def extract(self, frame):
        assert frame.shape == (32, 32, 3)
        return SimpleNamespace(facenet=frame)


class DeterministicEmbedding:
    def encode(self, frame):
        vector = np.zeros(512, dtype=np.float32)
        # Different photo bytes are nearby views of the same synthetic identity.
        vector[0] = 2.0
        vector[1] = float(frame[0, 0, 0]) / 100
        return vector


@pytest.fixture
def gallery(tmp_path):
    db = tmp_path / "accounts.db"
    database.init_db(db)
    photos = [tmp_path / "one.png", tmp_path / "two.png"]
    for index, photo in enumerate(photos):
        Image.new("RGB", (32, 32), (index, index, index)).save(photo)
    return tmp_path / "authorization", db, photos


def enroll_test(gallery, username="alice", **kwargs):
    root, db, photos = gallery
    options = {
        "consent": True,
        "db_path": db,
        "preprocessor": DeterministicFace(),
        "embedder": DeterministicEmbedding(),
    }
    options.update(kwargs)
    return enroll(root, username, options.pop("images", photos), **options)


def test_enrollment_publishes_consent_account_and_normalized_templates(gallery):
    root, db, _ = gallery
    result = enroll_test(gallery, role="admin")
    assert result == {
        "username": "alice",
        "role": "ADMIN",
        "templates": 2,
        "restart_required": True,
    }
    manifest = json.loads((root / "faces/alice/enrollment.json").read_text())
    assert manifest["consent"] is True and manifest["consented_at"]
    account = database.get_account("alice", db)
    assert account["enrollment_id"] == manifest["enrollment_id"]
    assert account["active"] == 1 and account["role"] == "ADMIN"
    vectors = load_authorization_gallery(root / "faces", db)
    assert list(vectors) == ["alice"] and len(vectors["alice"]) == 2
    for vector in vectors["alice"]:
        assert vector.shape == (512,) and np.linalg.norm(vector) == pytest.approx(1)
    assert len(list((root / "faces/alice").glob("*.png"))) == 2


@pytest.mark.parametrize(
    "username", ["../alice", "a/b", "a\\b", "con", "nul", "", "Alice", "a:stream", "..", "a" * 65]
)
def test_enrollment_rejects_unsafe_usernames_without_writes(gallery, username):
    with pytest.raises(ValueError):
        enroll_test(gallery, username)
    assert database.list_accounts(gallery[1]) == []
    assert not gallery[0].exists()


def test_consent_is_required_before_publication(gallery):
    with pytest.raises(ValueError, match="consent"):
        enroll_test(gallery, consent=False)
    assert database.list_accounts(gallery[1]) == []
    assert not gallery[0].exists()


def test_duplicate_photos_rejected_without_account(gallery):
    with pytest.raises(ValueError, match="Duplicate"):
        enroll_test(gallery, images=[gallery[2][0], gallery[2][0]])
    assert database.list_accounts(gallery[1]) == []
    assert not gallery[0].exists()


def test_bad_face_rejected_without_account(gallery):
    class RejectFace:
        def extract(self, frame):
            raise ValueError("No single acceptable face")

    with pytest.raises(ValueError, match="face"):
        enroll_test(gallery, preprocessor=RejectFace())
    assert database.list_accounts(gallery[1]) == []
    assert not gallery[0].exists()


def test_existing_enrollment_is_never_overwritten(gallery):
    root, db, _ = gallery
    enroll_test(gallery)
    folder = root / "faces/alice"
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    account = database.get_account("alice", db)
    with pytest.raises(ValueError, match="overwrite"):
        enroll_test(gallery, role="admin")
    assert account == database.get_account("alice", db)
    assert before == {path.name: path.read_bytes() for path in folder.iterdir()}


def test_absent_gallery_does_not_create_or_reassign_accounts(gallery):
    root, db, _ = gallery
    database.register_user("unassociated", "user", db)
    before = database.list_accounts(db)
    assert load_authorization_gallery(root / "faces", db) == {}
    assert database.list_accounts(db) == before
    assert not root.exists()
    database.register_user("missing", "user", db, enrollment_id="expected")
    before = database.list_accounts(db)
    with pytest.raises(IdentityIndexError):
        load_authorization_gallery(root / "faces", db)
    assert database.list_accounts(db) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("username", "bob"),
        ("enrollment_id", "other"),
        ("contract", "wrong-model"),
        ("consent", False),
        ("templates", []),
    ],
)
def test_manifest_association_mismatch_fails_closed(gallery, field, value):
    root, db, _ = gallery
    enroll_test(gallery)
    path = root / "faces/alice/enrollment.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(IdentityIndexError, match="Invalid active enrollment"):
        load_authorization_gallery(root / "faces", db)


@pytest.mark.parametrize(
    "vector", [[0.0] * 512, [1.0], [float("nan")] * 512, [float("inf")] * 512, "not a vector"]
)
def test_malformed_embedding_fails_closed(gallery, vector):
    root, db, _ = gallery
    enroll_test(gallery)
    path = root / "faces/alice/enrollment.json"
    manifest = json.loads(path.read_text())
    manifest["templates"][0]["embedding"] = vector
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(IdentityIndexError):
        load_authorization_gallery(root / "faces", db)


def test_tampered_image_fails_closed_but_disabled_account_is_excluded(gallery):
    root, db, _ = gallery
    enroll_test(gallery)
    Image.new("RGB", (32, 32), "red").save(root / "faces/alice/01.png")
    with pytest.raises(IdentityIndexError):
        load_authorization_gallery(root / "faces", db)
    database.change_account("alice", disable=True, db_path=db)
    assert load_authorization_gallery(root / "faces", db) == {}


def test_manifest_cannot_reference_outside_private_folder(gallery):
    root, db, _ = gallery
    enroll_test(gallery)
    path = root / "faces/alice/enrollment.json"
    manifest = json.loads(path.read_text())
    manifest["templates"][0]["image"] = "../../../one.png"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(IdentityIndexError):
        load_authorization_gallery(root / "faces", db)


def test_account_write_failure_retains_inert_draft_and_allows_retry(gallery, monkeypatch):
    root, db, _ = gallery

    def failed_registration(*args, **kwargs):
        raise RuntimeError("injected database write failure")

    with monkeypatch.context() as patch:
        patch.setattr(database, "register_user", failed_registration)
        with pytest.raises(RuntimeError, match="database write failure"):
            enroll_test(gallery)
    assert database.list_accounts(db) == [] and not (root / "faces/alice").exists()
    drafts = list((root / "drafts").iterdir())
    assert len(drafts) == 1 and len(list(drafts[0].glob("*.png"))) == 2
    assert (drafts[0] / "enrollment.json").exists()
    assert load_authorization_gallery(root / "faces", db) == {}
    assert enroll_test(gallery)["templates"] == 2
    assert database.get_account("alice", db)["active"] == 1
