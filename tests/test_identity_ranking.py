"""Tests for unthresholded identity-level nearest-neighbour ranking."""

import numpy as np
import pytest

from vshield.core.identity_index import (
    IdentityIndex,
    IdentityIndexError,
    IdentityIndexUnavailableError,
)


def unit_embedding(index: int) -> np.ndarray:
    embedding = np.zeros(512, dtype=np.float32)
    embedding[index] = 1.0
    return embedding


def test_ranking_returns_unique_identities_and_nearest_template():
    index = IdentityIndex(
        {
            "alice": [unit_embedding(9), unit_embedding(0)],
            "bob": [unit_embedding(1), unit_embedding(2)],
            "carol": [unit_embedding(3)],
        },
        prefer_faiss=False,
    )

    ranked = index.ranked_identities(unit_embedding(0), k=5)

    assert [candidate.username for candidate in ranked] == ["alice", "bob", "carol"]
    assert ranked[0].distance == pytest.approx(0.0)
    assert len({candidate.username for candidate in ranked}) == len(ranked)


def test_ranking_uses_deterministic_username_tie_break():
    index = IdentityIndex(
        {"carol": [unit_embedding(3)], "bob": [unit_embedding(2)]},
        prefer_faiss=False,
    )

    ranked = index.ranked_identities(unit_embedding(0), k=2)

    assert [candidate.username for candidate in ranked] == ["bob", "carol"]


def test_ranking_does_not_change_thresholded_search_result():
    index = IdentityIndex(
        {"alice": [unit_embedding(0)], "bob": [unit_embedding(1)]},
        prefer_faiss=False,
    )

    before = index.search(unit_embedding(0))
    index.ranked_identities(unit_embedding(0), k=2)
    after = index.search(unit_embedding(0))

    assert before == after


def test_ranking_rejects_invalid_k_and_query():
    index = IdentityIndex({"alice": [unit_embedding(0)]}, prefer_faiss=False)

    with pytest.raises(ValueError, match="k must be at least 1"):
        index.ranked_identities(unit_embedding(0), k=0)
    with pytest.raises(IdentityIndexError, match="Query embedding is invalid"):
        index.ranked_identities(np.zeros(512, dtype=np.float32))


def test_ranking_rejects_empty_gallery():
    index = IdentityIndex({}, prefer_faiss=False)

    with pytest.raises(IdentityIndexUnavailableError):
        index.ranked_identities(unit_embedding(0))


def test_faiss_and_numpy_rankings_are_equivalent():
    gallery = {
        "alice": [unit_embedding(0), unit_embedding(4)],
        "bob": [unit_embedding(1)],
        "carol": [unit_embedding(2)],
    }
    faiss_index = IdentityIndex(gallery, prefer_faiss=True)
    if faiss_index.backend != "faiss":
        pytest.skip("FAISS is not installed in this test environment")
    numpy_index = IdentityIndex(gallery, prefer_faiss=False)

    faiss_ranked = faiss_index.ranked_identities(unit_embedding(0), k=3)
    numpy_ranked = numpy_index.ranked_identities(unit_embedding(0), k=3)

    assert [candidate.username for candidate in faiss_ranked] == [
        candidate.username for candidate in numpy_ranked
    ]
    assert [candidate.distance for candidate in faiss_ranked] == pytest.approx(
        [candidate.distance for candidate in numpy_ranked]
    )
