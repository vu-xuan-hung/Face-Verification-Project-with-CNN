"""Closed-set identity retrieval metrics for an external probe set."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from vshield.core.identity_index import IdentityIndex, IdentityIndexUnavailableError


@dataclass(frozen=True)
class IdentityProbe:
    """One labeled probe embedding that is independent from the gallery."""

    expected_identity: str
    embedding: np.ndarray


@dataclass(frozen=True)
class RetrievalMetrics:
    """Macro-averaged identity retrieval results at a fixed cutoff."""

    k: int
    query_count: int
    precision_at_k: float
    recall_at_k: float
    hit_rate_at_k: float


def evaluate_identity_retrieval(
    identity_index: IdentityIndex,
    probes: Sequence[IdentityProbe],
    *,
    k: int = 5,
) -> RetrievalMetrics:
    """Evaluate unique-identity retrieval without applying auth thresholds."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not identity_index.available:
        raise IdentityIndexUnavailableError("No enrolled face embeddings are available")

    query_count = len(probes)
    if query_count == 0:
        return RetrievalMetrics(k, 0, 0.0, 0.0, 0.0)

    hits = 0
    for probe in probes:
        if not isinstance(probe.expected_identity, str) or not probe.expected_identity:
            raise ValueError("Probe expected_identity cannot be empty")
        ranked = identity_index.ranked_identities(probe.embedding, k=k)
        hits += int(
            probe.expected_identity in {candidate.username for candidate in ranked}
        )

    hit_rate = hits / query_count
    return RetrievalMetrics(
        k=k,
        query_count=query_count,
        precision_at_k=hits / (query_count * k),
        recall_at_k=hit_rate,
        hit_rate_at_k=hit_rate,
    )
