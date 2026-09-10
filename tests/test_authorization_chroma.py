"""Real persistent Chroma reconciliation tests; vectors are synthetic unit inputs."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vshield.core import chroma_identity_index as chroma_module
from vshield.core.chroma_identity_index import ChromaIdentityIndex, build_preferred_identity_index
from vshield.core.identity_index import IdentityIndexUnavailableError
from vshield.core.identity_index_support import IdentityIndexError


def unit_embedding(index):
    vector = np.zeros(512, dtype=np.float32)
    vector[index] = 1.0
    return vector


def test_real_reconcile_add_remove_replace_and_empty(tmp_path):
    index = ChromaIdentityIndex(tmp_path / "vectors")
    assert index.reconcile({"alice": [unit_embedding(0)], "bob": [unit_embedding(1)]}) == 2
    assert index.search(unit_embedding(0)).username == "alice"
    assert index.reconcile({"alice": [unit_embedding(2)], "carol": [unit_embedding(3)]}) == 2
    assert index.search(unit_embedding(0)) is None
    assert index.search(unit_embedding(1)) is None
    assert index.search(unit_embedding(2)).username == "alice"
    assert index.search(unit_embedding(3)).username == "carol"
    records = index._collection.get(include=["metadatas", "embeddings"])
    assert {item["username"] for item in records["metadatas"]} == {"alice", "carol"}
    assert len(records["ids"]) == 2
    assert index.reconcile({}) == 0
    assert index._collection.count() == 0 and not index.available
    assert index._names == [] and index._matrix.size == 0
    with pytest.raises(IdentityIndexUnavailableError):
        index.search(unit_embedding(2))


@pytest.mark.parametrize("empty", [False, True])
def test_reload_reconciles_authoritative_snapshot_before_fallback(tmp_path, monkeypatch, empty):
    path = tmp_path / "vectors"
    first = ChromaIdentityIndex(path)
    first.reconcile({"revoked": [unit_embedding(0)]})
    snapshot = {} if empty else {"current": [unit_embedding(1)], "other": [unit_embedding(2)]}
    reloaded = build_preferred_identity_index(path, lambda: snapshot, reconcile=True)
    assert reloaded.backend == "chroma" and reloaded.size == len(snapshot)
    assert reloaded._names == list(snapshot)

    def unavailable(*args):
        raise RuntimeError("injected query outage after real persistence")

    monkeypatch.setattr(reloaded, "_query_chroma", unavailable)
    if empty:
        with pytest.raises(IdentityIndexUnavailableError):
            reloaded.search(unit_embedding(0))
    else:
        assert reloaded.search(unit_embedding(0)) is None
        assert reloaded.search(unit_embedding(1)).username == "current"
        np.testing.assert_array_equal(reloaded._matrix, np.stack([v[0] for v in snapshot.values()]))


def test_snapshot_survives_real_python_process_restart(tmp_path):
    path = tmp_path / "vectors"
    index = ChromaIdentityIndex(path)
    index.reconcile({"alice": [unit_embedding(0)], "bob": [unit_embedding(1)]})
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    code = (
        "import sys,numpy as np; "
        "from vshield.core.chroma_identity_index import ChromaIdentityIndex; "
        "index=ChromaIdentityIndex(sys.argv[1]); "
        "v=np.zeros(512,dtype=np.float32);v[1]=1; "
        "assert index.size==2; assert index.search(v).username=='bob'; "
        "print('persisted snapshot verified')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "persisted snapshot verified" in result.stdout


def test_invalid_gallery_never_reuses_existing_persisted_identity(tmp_path):
    path = tmp_path / "vectors"
    ChromaIdentityIndex(path).reconcile({"stale": [unit_embedding(0)]})

    def invalid_gallery():
        raise IdentityIndexError("Invalid active enrollment")

    with pytest.raises(IdentityIndexError, match="Invalid active enrollment"):
        build_preferred_identity_index(path, invalid_gallery, reconcile=True)


@pytest.mark.parametrize("empty", [False, True])
def test_initialization_outage_falls_back_to_exact_current_snapshot(tmp_path, monkeypatch, empty):
    path = tmp_path / "vectors"
    ChromaIdentityIndex(path).reconcile({"stale": [unit_embedding(0)]})
    snapshot = {} if empty else {"current": [unit_embedding(1)]}
    loads = []

    def loader():
        loads.append(True)
        return snapshot

    def fail(*args):
        raise RuntimeError("injected storage outage")

    monkeypatch.setattr(chroma_module, "ChromaIdentityIndex", fail)
    fallback = build_preferred_identity_index(path, loader, reconcile=True)
    assert fallback.backend in {"numpy", "faiss"} and len(loads) == 1
    assert fallback.size == len(snapshot)
    if empty:
        with pytest.raises(IdentityIndexUnavailableError):
            fallback.search(unit_embedding(0))
    else:
        assert fallback.search(unit_embedding(0)) is None
        assert fallback.search(unit_embedding(1)).username == "current"
