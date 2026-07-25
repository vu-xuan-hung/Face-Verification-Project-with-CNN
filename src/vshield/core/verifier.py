"""Enrollment loading and backward-compatible identity helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from vshield.core.embedder import EmbeddingError, img_to_encoding_file, normalize_embedding
from vshield.core.identity_index import (
    DEFAULT_DISTANCE_THRESHOLD,
    IdentityIndex,
    IdentityIndexError,
    IdentityIndexUnavailableError,
    MatchResult,
    flatten_database_embeddings,
    import_faiss,
)

logger = logging.getLogger(__name__)

UNKNOWN_IDENTITY = "Unknown"


def load_database(
    faces_dir: str | Path = "data/faces",
    encode_file: Callable[[str | Path], np.ndarray] | None = None,
) -> dict[str, list[np.ndarray]]:
    """Load ``faces_dir/<username>/*`` into validated unit embeddings."""
    faces_path = Path(faces_dir)
    encoder = encode_file or img_to_encoding_file
    database_faces: dict[str, list[np.ndarray]] = {}

    if not faces_path.is_dir():
        logger.warning("Face enrollment directory does not exist: %s", faces_path)
        return database_faces

    root_images = [
        path
        for path in faces_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if root_images:
        logger.warning(
            "Ignoring %d root-level face images; expected faces/<username>/*",
            len(root_images),
        )

    for user_folder in sorted(path for path in faces_path.iterdir() if path.is_dir()):
        embeddings: list[np.ndarray] = []
        for image_path in sorted(user_folder.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                embeddings.append(normalize_embedding(encoder(image_path)))
            except Exception as exc:
                logger.warning("Skipping enrollment image %s: %s", image_path, exc)
        if embeddings:
            database_faces[user_folder.name] = embeddings

    return database_faces


def build_faiss_index(database_faces):
    """Backward-compatible FAISS builder with enforced vector normalization."""
    matrix, names = flatten_database_embeddings(database_faces)
    faiss = import_faiss()
    if faiss is None or matrix.size == 0:
        return None, None

    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    return index, dict(enumerate(names))


def who_is_it(
    encoding,
    database_faces,
    threshold=DEFAULT_DISTANCE_THRESHOLD,
    faiss_index=None,
    id_to_name=None,
):
    """Return the closest identity name or ``Unknown``."""
    try:
        query = normalize_embedding(encoding)
    except EmbeddingError:
        return UNKNOWN_IDENTITY

    if faiss_index is not None and id_to_name is not None:
        try:
            squared_distances, indices = faiss_index.search(query.reshape(1, -1), 1)
            index = int(indices[0][0])
            distance = float(squared_distances[0][0]) ** 0.5
            identity = id_to_name.get(index)
            if identity is not None and np.isfinite(distance) and distance <= threshold:
                return identity
            return UNKNOWN_IDENTITY
        except Exception as exc:
            logger.warning("FAISS search failed; using exact NumPy fallback: %s", exc)

    best_identity = UNKNOWN_IDENTITY
    best_distance = float("inf")
    for username, embeddings in database_faces.items():
        for embedding in embeddings:
            try:
                candidate = normalize_embedding(embedding)
            except EmbeddingError:
                continue
            distance = float(np.linalg.norm(query - candidate))
            if distance < best_distance:
                best_distance = distance
                best_identity = username

    return best_identity if best_distance <= threshold else UNKNOWN_IDENTITY


__all__ = [
    "IdentityIndex",
    "IdentityIndexError",
    "IdentityIndexUnavailableError",
    "MatchResult",
    "UNKNOWN_IDENTITY",
    "build_faiss_index",
    "load_database",
    "who_is_it",
]
