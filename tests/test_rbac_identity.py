"""Live gallery integration with real SQLite/photos/index, synthetic ML outputs only."""

import base64

import cv2
import numpy as np
import pytest

from tests.test_rbac_api import managed as managed_fixture
from tests.test_rbac_api import photos, profile
from vshield.api import database
from vshield.core.identity_index import IdentityIndexUnavailableError
from vshield.core.managed_identity_index import ManagedIdentityIndex
from vshield.services.enrollment import enroll

managed = managed_fixture


def decoded(identity):
    return [
        cv2.imdecode(np.frombuffer(base64.b64decode(p.split(",")[1]), np.uint8), 1)
        for p in photos(identity)
    ]


def test_enroll_search_role_change_disable_refresh_without_training(managed):
    actor = database.get_account("owner")["id"]
    first = managed.create(actor, profile("first", 10), decoded(10), "USER")
    index = ManagedIdentityIndex(managed.root, managed.db_path)
    vector = managed.embedder.encode(decoded(10)[0])
    assert index.search(vector).username == str(first["id"])
    second = managed.create(actor, profile("second", 20), decoded(20), "ADMIN")
    assert index.search(managed.embedder.encode(decoded(20)[0])).username == str(second["id"])
    database.change_account("first", role="ADMIN")
    assert index.search(vector).username == str(first["id"])
    assert database.get_account_by_id(first["id"])["role"] == "ADMIN"
    database.change_account("first", disable=True)
    assert index.search(vector) is None
    database.change_account("second", disable=True)
    with pytest.raises(IdentityIndexUnavailableError):
        index.search(vector)


def test_mixed_identity_photos_fail_before_publication(managed):
    actor = database.get_account("manager")["id"]
    with pytest.raises(ValueError, match="consistently"):
        managed.create(actor, profile(), [decoded(10)[0], decoded(20)[1]], "USER")
    assert database.get_account("newperson") is None
    assert not managed.root.exists()


def test_duplicate_photo_fails_before_publication(managed):
    actor = database.get_account("manager")["id"]
    frame = decoded(10)[0]
    with pytest.raises(ValueError, match="Duplicate"):
        managed.create(actor, profile(), [frame, frame], "USER")
    assert database.get_account("newperson") is None


def test_commit_refreshes_managed_index_immediately(managed):
    managed.identity_index = ManagedIdentityIndex(managed.root, managed.db_path)
    actor = database.get_account("owner")["id"]
    result = managed.create(actor, profile(), decoded(10), "USER")
    assert result["vector_sync_status"] == "ready"
    assert managed.identity_index.size == 2
    assert managed.identity_index.search(managed.embedder.encode(decoded(10)[0])).username == str(
        result["id"]
    )


def test_post_commit_index_failure_reports_pending_without_losing_account(managed):
    class UnavailableIndex:
        def refresh(self):
            raise IdentityIndexUnavailableError("Index unavailable")

    managed.identity_index = UnavailableIndex()
    actor = database.get_account("owner")["id"]
    result = managed.create(actor, profile(), decoded(10), "USER")
    assert result["vector_sync_status"] == "pending"
    assert database.get_account("newperson")["id"] == result["id"]
    assert (managed.root / "faces/newperson/enrollment.json").exists()


def test_old_revision_reconcile_cannot_overwrite_new_membership(tmp_path):
    from vshield.core.chroma_identity_index import ChromaIdentityIndex

    vector = np.zeros(512, dtype=np.float32)
    vector[0] = 1
    old = ChromaIdentityIndex(tmp_path / "chroma", identity_key="user_id", collection_version=1)
    new = ChromaIdentityIndex(tmp_path / "chroma", identity_key="user_id", collection_version=2)
    old.reconcile({"1": [vector]})
    new.reconcile({"2": [vector]})
    assert new.search(vector).username == "2"
    old.reconcile({"3": [vector]})
    assert old.size == new.size == 1
    assert old.search(vector).username == "3"
    assert new.search(vector).username == "2"


def write_cli_photos(tmp_path, frames):
    paths = []
    for number, frame in enumerate(frames):
        path = tmp_path / f"cli-{number}.png"
        assert cv2.imwrite(str(path), frame)
        paths.append(path)
    return paths


def test_cli_rejects_mixed_identity_without_publication(managed, tmp_path):
    frames = [decoded(10)[0], decoded(20)[1]]
    assert (
        np.linalg.norm(managed.embedder.encode(frames[0]) - managed.embedder.encode(frames[1]))
        > 0.9
    )
    paths = write_cli_photos(tmp_path, frames)
    before = database.list_accounts()
    with pytest.raises(ValueError, match="consistently"):
        enroll(
            managed.root,
            "mixed",
            paths,
            consent=True,
            db_path=managed.db_path,
            preprocessor=managed.preprocessor,
            embedder=managed.embedder,
        )
    assert database.list_accounts() == before
    assert not managed.root.exists()


def test_cli_duplicate_against_managed_gallery_creates_no_account(managed, tmp_path):
    actor = database.get_account("manager")["id"]
    managed.create(actor, profile(), decoded(10), "USER")
    paths = write_cli_photos(tmp_path, decoded(10))
    before = database.list_accounts()
    with pytest.raises(ValueError, match="already enrolled"):
        enroll(
            managed.root,
            "duplicate",
            paths,
            consent=True,
            db_path=managed.db_path,
            preprocessor=managed.preprocessor,
            embedder=managed.embedder,
        )
    assert database.list_accounts() == before
    assert database.get_account("duplicate") is None
    assert not (managed.root / "faces/duplicate").exists()
