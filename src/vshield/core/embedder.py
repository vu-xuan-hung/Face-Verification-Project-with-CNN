"""FaceNet embedding with validation and enforced L2 normalization."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np

EMBEDDING_DIMENSION = 512
_NORM_EPSILON = 1e-12


class EmbeddingError(RuntimeError):
    """Raised when FaceNet cannot produce a valid embedding."""


def normalize_embedding(embedding: Any) -> np.ndarray:
    """Return one finite, contiguous, unit-length float32 embedding."""
    try:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    except Exception as exc:
        raise EmbeddingError("Embedding is not a numeric vector") from exc

    if vector.shape != (EMBEDDING_DIMENSION,):
        raise EmbeddingError(
            f"Expected a {EMBEDDING_DIMENSION}-D embedding, got shape {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise EmbeddingError("Embedding contains non-finite values")

    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= _NORM_EPSILON:
        raise EmbeddingError("Embedding norm is zero or invalid")

    return np.ascontiguousarray(vector / norm, dtype=np.float32)


class FaceEmbedder:
    """Lazy FaceNet adapter so importing the API does not load model weights."""

    def __init__(self, model=None):
        self._model = model
        self._model_lock = Lock()
        self._inference_lock = Lock()

    def _get_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    try:
                        from keras_facenet import FaceNet

                        self._model = FaceNet()
                    except Exception as exc:
                        raise EmbeddingError("Cannot load FaceNet model") from exc
        return self._model

    @property
    def ready(self):
        """Readiness observes lazy loading; it never downloads weights itself."""
        return self._model is not None

    def encode(self, frame: np.ndarray) -> np.ndarray:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise EmbeddingError("Expected a BGR face crop with three channels")

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._inference_lock:
                embeddings = self._get_model().embeddings([rgb])
            embedding = np.asarray(embeddings)[0]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("FaceNet inference failed") from exc

        return normalize_embedding(embedding)

    def encode_file(self, path: str | Path, face_preprocessor=None) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise EmbeddingError(f"Could not read image: {path}")
        if face_preprocessor is not None:
            image = face_preprocessor.extract(image).facenet
        return self.encode(image)


_default_embedder: FaceEmbedder | None = None
_default_embedder_lock = Lock()


def _get_default_embedder() -> FaceEmbedder:
    global _default_embedder
    if _default_embedder is None:
        with _default_embedder_lock:
            if _default_embedder is None:
                _default_embedder = FaceEmbedder()
    return _default_embedder


def img_to_encoding_frame(frame: np.ndarray) -> np.ndarray:
    """Backward-compatible wrapper around the default FaceNet adapter."""
    return _get_default_embedder().encode(frame)


def img_to_encoding_file(path: str | Path) -> np.ndarray:
    """Backward-compatible file embedding helper."""
    return _get_default_embedder().encode_file(path)
