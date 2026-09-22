"""Closed-set identity retrieval metrics for an external probe set."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

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


def open_set_metrics(rows: Sequence[dict]) -> dict[str, object]:
    """Compute open-set identification metrics with all probes in denominators."""
    known = [row for row in rows if _enrolled(row["is_enrolled"])]
    unknown = [row for row in rows if not _enrolled(row["is_enrolled"])]
    correct = [row for row in known if row["decision"] == "KNOWN" and row["identity"] == row["subject_id"]]
    wrong = [row for row in known if row["decision"] == "KNOWN" and row["identity"] != row["subject_id"]]
    known_unknown = [row for row in known if row["decision"] == "UNKNOWN"]
    ambiguous = [row for row in known if row["decision"] == "AMBIGUOUS"]
    unknown_accepted = [row for row in unknown if row["decision"] == "KNOWN"]
    unknown_rejected = [row for row in unknown if row["decision"] in {"UNKNOWN", "AMBIGUOUS"}]
    return {
        "n_probes": len(rows),
        "n_known": len(known),
        "n_unknown": len(unknown),
        "known_identification_accuracy": _rate(len(correct), len(known)),
        "known_acceptance_rate": _rate(len(correct) + len(wrong), len(known)),
        "known_rejection_rate": _rate(len(known_unknown) + len(ambiguous), len(known)),
        "wrong_identity_rate": _rate(len(wrong), len(known)),
        "ambiguous_rejection_rate": _rate(len(ambiguous), len(known)),
        "rank_1_identification_rate": _rate(
            sum(
                not row.get("error") and row.get("best_identity") == row["subject_id"]
                for row in known
            ),
            len(known),
        ),
        "unknown_rejection_rate": _rate(len(unknown_rejected), len(unknown)),
        "unknown_false_acceptance_rate": _rate(len(unknown_accepted), len(unknown)),
        "inference_errors": sum(bool(row.get("error")) for row in rows),
    }


def verification_metrics(scores: Sequence[dict], threshold: float) -> dict[str, object]:
    genuine = np.asarray([row["distance"] for row in scores if row["pair_type"] == "genuine"])
    impostor = np.asarray([row["distance"] for row in scores if row["pair_type"] == "impostor"])
    metrics: dict[str, object] = {
        "threshold": threshold,
        "n_genuine": len(genuine),
        "n_impostor": len(impostor),
        "far_fmr": float(np.mean(impostor <= threshold)) if len(impostor) else None,
        "frr_fnmr": float(np.mean(genuine > threshold)) if len(genuine) else None,
    }
    if len(genuine) and len(impostor):
        labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
        similarities = -np.concatenate([genuine, impostor])
        fpr, tpr, cutoffs = roc_curve(labels, similarities)
        fnr = 1 - tpr
        index = int(np.argmin(np.abs(fpr - fnr)))
        metrics.update(
            roc_auc=float(roc_auc_score(labels, similarities)),
            eer=float((fpr[index] + fnr[index]) / 2),
            eer_distance_threshold=float(-cutoffs[index]),
        )
    else:
        metrics.update(roc_auc=None, eer=None, eer_distance_threshold=None)
    return metrics


def apply_open_set_thresholds(
    rows: Sequence[dict], distance_threshold: float, ambiguity_threshold: float
) -> list[dict]:
    """Apply production's inclusive distance and margin boundaries to cached rankings."""
    result = []
    for source in rows:
        row = dict(source)
        best = row.get("best_score")
        margin = row.get("margin")
        if row.get("error") or best is None:
            row["decision"], row["identity"] = "ERROR", None
        elif best > distance_threshold:
            row["decision"], row["identity"] = "UNKNOWN", None
        elif margin is not None and margin < ambiguity_threshold:
            row["decision"], row["identity"] = "AMBIGUOUS", None
        else:
            row["decision"], row["identity"] = "KNOWN", row.get("best_identity")
        result.append(row)
    return result


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _enrolled(value) -> bool:
    return value is True or str(value).lower() in {"1", "true"}
