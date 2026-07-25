"""Validated identity-vector index with FAISS and exact NumPy backends."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from vshield.core.embedder import EmbeddingError, normalize_embedding

logger = logging.getLogger(__name__)

DEFAULT_DISTANCE_THRESHOLD = 0.9
DEFAULT_MIN_MARGIN = 0.05


class IdentityIndexError(RuntimeError):
    """Base error for identity index failures."""


class IdentityIndexUnavailableError(IdentityIndexError):
    """Raised when no searchable enrollment index is available."""


@dataclass(frozen=True)
class MatchResult:
    """A calibrated identity match returned by the vector index."""

    username: str
    distance: float
    runner_up_distance: float | None = None


def flatten_database_embeddings(
    database_faces: dict[str, list[np.ndarray]],
) -> tuple[np.ndarray, list[str]]:
    vectors: list[np.ndarray] = []
    names: list[str] = []

    for username, embeddings in database_faces.items():
        if not username:
            raise IdentityIndexError("Enrollment username cannot be empty")
        for embedding in embeddings:
            try:
                vectors.append(normalize_embedding(embedding))
            except EmbeddingError as exc:
                raise IdentityIndexError(
                    f"Invalid enrollment embedding for {username}"
                ) from exc
            names.append(username)

    if not vectors:
        return np.empty((0, 512), dtype=np.float32), []
    return np.ascontiguousarray(np.stack(vectors), dtype=np.float32), names


def import_faiss():
    try:
        import faiss

        return faiss
    except ImportError:
        return None


class IdentityIndex:
    """Immutable identity snapshot backed by FAISS with exact NumPy fallback."""

    def __init__(
        self,
        database_faces: dict[str, list[np.ndarray]],
        *,
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
        min_margin: float = DEFAULT_MIN_MARGIN,
        search_k: int = 5,
        prefer_faiss: bool = True,
    ):
        if not np.isfinite(distance_threshold) or not 0 < distance_threshold <= 2:
            raise ValueError(
                "distance_threshold must be finite and within unit-vector L2 range (0, 2]"
            )
        if not np.isfinite(min_margin) or not 0 <= min_margin <= 2:
            raise ValueError(
                "min_margin must be finite and within unit-vector L2 range [0, 2]"
            )
        if search_k < 1:
            raise ValueError("search_k must be at least 1")

        self.distance_threshold = float(distance_threshold)
        self.min_margin = float(min_margin)
        self.search_k = int(search_k)
        self._matrix, self._names = flatten_database_embeddings(database_faces)
        self._faiss_index = None

        if prefer_faiss and self._matrix.size:
            faiss = import_faiss()
            if faiss is not None:
                try:
                    self._faiss_index = faiss.IndexFlatL2(self._matrix.shape[1])
                    self._faiss_index.add(self._matrix)
                except Exception as exc:
                    self._faiss_index = None
                    logger.warning(
                        "FAISS index initialization failed; using NumPy fallback: %s",
                        exc,
                    )

    @property
    def available(self) -> bool:
        return bool(self._names)

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss_index is not None else "numpy"

    @property
    def size(self) -> int:
        return len(self._names)

    def search(self, embedding: np.ndarray) -> MatchResult | None:
        if not self.available:
            raise IdentityIndexUnavailableError("No enrolled face embeddings are available")

        try:
            query = normalize_embedding(embedding)
        except EmbeddingError as exc:
            raise IdentityIndexError("Query embedding is invalid") from exc

        candidate_count = min(
            self.size,
            self.search_k,
        )
        candidates = self._search_candidates(query, candidate_count)
        return self._decide(candidates)

    def _search_candidates(
        self,
        query: np.ndarray,
        candidate_count: int,
    ) -> list[tuple[str, float]]:
        if self._faiss_index is not None:
            try:
                while True:
                    squared_distances, indices = self._faiss_index.search(
                        query.reshape(1, -1),
                        candidate_count,
                    )
                    candidates = [
                        (self._names[int(index)], float(distance) ** 0.5)
                        for distance, index in zip(
                            squared_distances[0],
                            indices[0],
                            strict=False,
                        )
                        if int(index) >= 0 and np.isfinite(distance)
                    ]
                    enough_identities = len({name for name, _ in candidates}) >= min(
                        2,
                        len(set(self._names)),
                    )
                    if enough_identities or candidate_count == self.size:
                        return candidates
                    candidate_count = min(self.size, candidate_count * 2)
            except Exception as exc:
                logger.warning("FAISS search failed; using NumPy fallback: %s", exc)

        distances = np.linalg.norm(self._matrix - query, axis=1)
        return [
            (username, float(distance))
            for username, distance in zip(self._names, distances, strict=True)
        ]

    def _decide(self, candidates: list[tuple[str, float]]) -> MatchResult | None:
        per_identity: dict[str, float] = {}
        for username, distance in candidates:
            current = per_identity.get(username)
            if current is None or distance < current:
                per_identity[username] = distance

        ranked = sorted(per_identity.items(), key=lambda item: item[1])
        if not ranked:
            raise IdentityIndexError("Vector search returned no candidates")

        username, best_distance = ranked[0]
        if best_distance > self.distance_threshold:
            return None

        runner_up = ranked[1][1] if len(ranked) > 1 else None
        if runner_up is not None and runner_up - best_distance < self.min_margin:
            return None

        return MatchResult(username, best_distance, runner_up)
