"""Tests for the local persistent face-embedding store."""

import importlib.util
from unittest.mock import Mock

import numpy as np
import pytest

from vshield.core import chroma_identity_index as chroma_module
from vshield.core.chroma_identity_index import (
    ChromaIdentityIndex,
    build_preferred_identity_index,
)
from vshield.core.identity_index_support import IdentityIndexError


def unit_embedding(index: int) -> np.ndarray:
    embedding = np.zeros(512, dtype=np.float32)
    embedding[index] = 1.0
    return embedding


class FakeCollection:
    def __init__(self):
        self.records = {}
        self.query_error = None
        self.last_n_results = None

    def count(self):
        return len(self.records)

    def upsert(self, *, ids, embeddings, metadatas):
        for record_id, embedding, metadata in zip(ids, embeddings, metadatas, strict=True):
            self.records[record_id] = (np.asarray(embedding), metadata)

    def query(self, *, query_embeddings, n_results, include):
        if self.query_error:
            raise self.query_error
        self.last_n_results = n_results
        assert include == ["metadatas", "distances"]
        query = np.asarray(query_embeddings[0])
        ranked = sorted(
            (
                (float(np.sum((embedding - query) ** 2)), metadata)
                for embedding, metadata in self.records.values()
            ),
            key=lambda item: item[0],
        )[:n_results]
        return {
            "metadatas": [[metadata for _, metadata in ranked]],
            "distances": [[distance for distance, _ in ranked]],
        }


class FakeClient:
    def __init__(self, collection=None):
        self.collection = collection or FakeCollection()

    def get_or_create_collection(self, **_kwargs):
        return self.collection


def test_chroma_queries_face_embeddings_and_converts_squared_l2(tmp_path):
    index = ChromaIdentityIndex(tmp_path, client=FakeClient())
    index.seed({"alice": [unit_embedding(0)], "bob": [unit_embedding(1)]})

    match = index.search(unit_embedding(0))

    assert index.backend == "chroma"
    assert match is not None
    assert match.username == "alice"
    assert match.distance == pytest.approx(0.0)


def test_chroma_keeps_existing_store_as_primary(monkeypatch, tmp_path):
    index = ChromaIdentityIndex(tmp_path, client=FakeClient())
    index.seed({"alice": [unit_embedding(0)]})
    enrollment_loader = Mock(side_effect=AssertionError("must not re-encode gallery"))
    monkeypatch.setattr(chroma_module, "ChromaIdentityIndex", lambda _path: index)

    selected = build_preferred_identity_index(tmp_path, enrollment_loader)

    assert selected is index
    enrollment_loader.assert_not_called()


def test_builder_falls_back_when_chroma_is_unavailable(monkeypatch, tmp_path):
    def fail_to_start(_path):
        raise RuntimeError("chroma unavailable")

    monkeypatch.setattr(chroma_module, "ChromaIdentityIndex", fail_to_start)
    enrollment_loader = Mock(return_value={"alice": [unit_embedding(0)]})

    index = build_preferred_identity_index(tmp_path, enrollment_loader)

    assert index.backend in {"faiss", "numpy"}
    assert index.search(unit_embedding(0)).username == "alice"
    enrollment_loader.assert_called_once_with()


def test_seeded_index_uses_memory_if_chroma_query_fails(tmp_path):
    collection = FakeCollection()
    index = ChromaIdentityIndex(tmp_path, client=FakeClient(collection))
    index.seed({"alice": [unit_embedding(0)]})
    collection.query_error = RuntimeError("query unavailable")

    assert index.search(unit_embedding(0)).username == "alice"


def test_reloaded_index_fails_closed_if_chroma_query_fails(tmp_path):
    collection = FakeCollection()
    collection.records["alice-template"] = (
        unit_embedding(0),
        {"username": "alice"},
    )
    collection.query_error = RuntimeError("query unavailable")
    index = ChromaIdentityIndex(tmp_path, client=FakeClient(collection))

    with pytest.raises(IdentityIndexError, match="Chroma identity query failed"):
        index.search(unit_embedding(0))


def test_chroma_rejects_ambiguous_identity(tmp_path):
    query = unit_embedding(0)
    nearly_same = query.copy()
    nearly_same[1] = 0.01
    index = ChromaIdentityIndex(tmp_path, min_margin=0.05, client=FakeClient())
    index.seed({"alice": [query], "bob": [nearly_same]})

    assert index.search(query) is None


def test_chroma_searches_complete_collection_for_security_margin(tmp_path):
    collection = FakeCollection()
    index = ChromaIdentityIndex(tmp_path, client=FakeClient(collection))
    index.seed(
        {
            "alice": [unit_embedding(i) for i in range(6)],
            "bob": [unit_embedding(10)],
        }
    )

    index.search(unit_embedding(0))

    assert collection.last_n_results == index.size


@pytest.mark.skipif(
    importlib.util.find_spec("chromadb") is None,
    reason="chromadb wheel is not installed",
)
def test_real_chroma_persists_embeddings_across_clients(tmp_path):
    first = ChromaIdentityIndex(tmp_path)
    first.seed({"alice": [unit_embedding(0)], "bob": [unit_embedding(1)]})

    reloaded = ChromaIdentityIndex(tmp_path)
    match = reloaded.search(unit_embedding(1))

    assert reloaded.size == 2
    assert match is not None
    assert match.username == "bob"
