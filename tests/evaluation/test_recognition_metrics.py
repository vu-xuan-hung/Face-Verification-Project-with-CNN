"""Tests for closed-set identity retrieval metrics."""

import numpy as np
import pytest

from vshield.core.identity_index import (
    IdentityIndex,
    IdentityIndexError,
    IdentityIndexUnavailableError,
)
from vshield.evaluation.recognition_metrics import (
    IdentityProbe,
    evaluate_identity_retrieval,
)


def unit_embedding(index: int) -> np.ndarray:
    embedding = np.zeros(512, dtype=np.float32)
    embedding[index] = 1.0
    return embedding


def test_precision_recall_and_hit_rate_at_five_are_macro_averaged():
    index = IdentityIndex(
        {
            "alice": [unit_embedding(0), unit_embedding(10)],
            "bob": [unit_embedding(1)],
            "carol": [unit_embedding(2)],
            "dave": [unit_embedding(3)],
            "erin": [unit_embedding(4)],
            "frank": [unit_embedding(5)],
        },
        prefer_faiss=False,
    )
    probes = [
        IdentityProbe("alice", unit_embedding(0)),
        IdentityProbe("not-enrolled", unit_embedding(1)),
    ]

    metrics = evaluate_identity_retrieval(index, probes, k=5)

    assert metrics.k == 5
    assert metrics.query_count == 2
    assert metrics.precision_at_k == pytest.approx(0.1)
    assert metrics.recall_at_k == pytest.approx(0.5)
    assert metrics.hit_rate_at_k == metrics.recall_at_k


def test_template_count_does_not_weight_query_metrics():
    index = IdentityIndex(
        {
            "alice": [unit_embedding(0), unit_embedding(2), unit_embedding(3)],
            "bob": [unit_embedding(1)],
        },
        prefer_faiss=False,
    )

    metrics = evaluate_identity_retrieval(
        index,
        [IdentityProbe("alice", unit_embedding(0)), IdentityProbe("bob", unit_embedding(1))],
        k=5,
    )

    assert metrics.precision_at_k == pytest.approx(0.2)
    assert metrics.recall_at_k == pytest.approx(1.0)


def test_empty_probe_set_returns_zero_metrics():
    index = IdentityIndex({"alice": [unit_embedding(0)]}, prefer_faiss=False)

    metrics = evaluate_identity_retrieval(index, [], k=5)

    assert metrics.query_count == 0
    assert metrics.precision_at_k == 0.0
    assert metrics.recall_at_k == 0.0
    assert metrics.hit_rate_at_k == 0.0


def test_empty_gallery_is_not_reported_as_zero_performance():
    index = IdentityIndex({}, prefer_faiss=False)

    with pytest.raises(IdentityIndexUnavailableError):
        evaluate_identity_retrieval(
            index,
            [IdentityProbe("alice", unit_embedding(0))],
        )


def test_evaluator_rejects_invalid_cutoff_and_expected_identity():
    index = IdentityIndex({"alice": [unit_embedding(0)]}, prefer_faiss=False)

    with pytest.raises(ValueError, match="k must be at least 1"):
        evaluate_identity_retrieval(index, [], k=0)
    with pytest.raises(ValueError, match="expected_identity cannot be empty"):
        evaluate_identity_retrieval(index, [IdentityProbe("", unit_embedding(0))])


def test_evaluator_propagates_invalid_probe_embedding():
    index = IdentityIndex({"alice": [unit_embedding(0)]}, prefer_faiss=False)

    with pytest.raises(IdentityIndexError, match="Query embedding is invalid"):
        evaluate_identity_retrieval(
            index,
            [IdentityProbe("alice", np.zeros(512, dtype=np.float32))],
        )
