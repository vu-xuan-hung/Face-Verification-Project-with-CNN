"""Local persistent Chroma backend for FaceNet identity embeddings."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path
from threading import Lock

import numpy as np

from vshield.core.identity_index import IdentityIndex
from vshield.core.identity_index_support import (
    IdentityIndexError,
    flatten_database_embeddings,
)

logger = logging.getLogger(__name__)
COLLECTION_NAME = "vshield_facenet_512_l2_v1"


class ChromaIdentityIndex(IdentityIndex):
    """Identity index backed by a local persistent Chroma collection."""

    def __init__(
        self,
        persist_path: str | Path,
        *,
        distance_threshold: float = 0.9,
        min_margin: float = 0.05,
        search_k: int = 5,
        client=None,
    ) -> None:
        super().__init__(
            {},
            distance_threshold=distance_threshold,
            min_margin=min_margin,
            search_k=search_k,
            prefer_faiss=False,
        )
        self.persist_path = Path(persist_path).resolve()
        self._store_lock = Lock()
        try:
            self._client = client or self._persistent_client(self.persist_path)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=None,
                configuration={"hnsw": {"space": "l2"}},
            )
            self._count = int(self._collection.count())
        except Exception as exc:
            raise IdentityIndexError("Cannot initialize local Chroma identity store") from exc

    @staticmethod
    def _persistent_client(path: Path):
        import chromadb
        from chromadb.config import Settings

        return chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False),
        )

    @property
    def available(self) -> bool:
        return self._count > 0

    @property
    def backend(self) -> str:
        return "chroma"

    @property
    def size(self) -> int:
        return self._count

    def seed(self, database_faces: dict[str, list[np.ndarray]]) -> int:
        """Persist a validated enrollment snapshot when the collection is empty."""
        if self.available:
            return 0
        matrix, names = flatten_database_embeddings(database_faces)
        if not names:
            return 0

        ids = [
            self._record_id(name, index, vector)
            for index, (name, vector) in enumerate(zip(names, matrix, strict=True))
        ]
        try:
            with self._store_lock:
                self._collection.upsert(
                    ids=ids,
                    embeddings=matrix.tolist(),
                    metadatas=[{"username": name} for name in names],
                )
        except Exception as exc:
            raise IdentityIndexError("Cannot persist enrollment embeddings in Chroma") from exc

        self._matrix = matrix
        self._names = names
        self._count = len(names)
        return self._count

    @staticmethod
    def _record_id(username: str, index: int, vector: np.ndarray) -> str:
        digest = hashlib.sha256()
        digest.update(username.encode("utf-8"))
        digest.update(index.to_bytes(8, "big"))
        digest.update(vector.tobytes())
        return digest.hexdigest()

    def _search_candidates(
        self,
        query: np.ndarray,
        candidate_count: int,
    ) -> list[tuple[str, float]]:
        try:
            # Authentication uses the true runner-up distance as a rejection gate.
            # Requesting every local template avoids accepting a query because an
            # approximate top-k result omitted a competing identity.
            candidates = self._query_chroma(query, self.size)
            if len(candidates) != self.size:
                raise IdentityIndexError("Chroma did not return the complete enrollment collection")
            return candidates
        except Exception as exc:
            if self._names:
                logger.warning("Chroma query failed; using seeded memory fallback: %s", exc)
                return super()._search_candidates(query, candidate_count)
            raise IdentityIndexError("Chroma identity query failed") from exc

    def _query_chroma(
        self,
        query: np.ndarray,
        candidate_count: int,
    ) -> list[tuple[str, float]]:
        with self._store_lock:
            result = self._collection.query(
                query_embeddings=[query.tolist()],
                n_results=candidate_count,
                include=["metadatas", "distances"],
            )
        metadata_batch = (result.get("metadatas") or [[]])[0]
        distance_batch = (result.get("distances") or [[]])[0]
        candidates: list[tuple[str, float]] = []
        for metadata, squared_distance in zip(metadata_batch, distance_batch, strict=True):
            username = metadata.get("username") if isinstance(metadata, dict) else None
            distance = float(squared_distance)
            if not username or not np.isfinite(distance) or distance < 0:
                continue
            candidates.append((username, distance**0.5))
        if not candidates:
            raise IdentityIndexError("Chroma returned no valid identity candidates")
        return candidates


def build_preferred_identity_index(
    persist_path: str | Path,
    enrollment_loader: Callable[[], dict[str, list[np.ndarray]]],
) -> IdentityIndex:
    """Prefer persisted Chroma and fall back to the existing in-memory index."""
    database_faces: dict[str, list[np.ndarray]] | None = None
    try:
        index = ChromaIdentityIndex(persist_path)
        if not index.available:
            database_faces = enrollment_loader()
            index.seed(database_faces)
        return index
    except Exception as exc:
        logger.warning("Chroma unavailable; using FAISS/NumPy identity index: %s", exc)
        database_faces = database_faces if database_faces is not None else enrollment_loader()
        return IdentityIndex(database_faces)
