"""Enrollment-vector preparation and optional FAISS loading."""

from __future__ import annotations

import numpy as np

from vshield.core.embedder import EmbeddingError, normalize_embedding


class IdentityIndexError(RuntimeError):
    """Base error for identity index preparation and search failures."""


def flatten_database_embeddings(
    database_faces: dict[str, list[np.ndarray]],
) -> tuple[np.ndarray, list[str]]:
    """Validate enrollment vectors and flatten them for exact search."""
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
    """Return FAISS when installed, otherwise select the NumPy fallback."""
    try:
        import faiss

        return faiss
    except ImportError:
        return None
