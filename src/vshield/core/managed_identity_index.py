"""Revision-aware identity index; account IDs, never roles, are vector identifiers."""

from pathlib import Path
from threading import RLock

from vshield.api import database
from vshield.core.authorization_gallery import load_authorization_gallery
from vshield.core.chroma_identity_index import build_preferred_identity_index
from vshield.core.identity_index import IdentityIndex, IdentityIndexUnavailableError


class ManagedIdentityIndex:
    def __init__(self, root, db_path=None):
        self.root = Path(root)
        self.db_path = db_path
        self._lock = RLock()
        self._revision = -1
        self._index = IdentityIndex({})

    def refresh(self):
        with self._lock:
            revision = database.gallery_revision(self.db_path)
            if revision != self._revision:
                # Build fully before replacement; failed refresh never advances
                # revision, so callers cannot continue using a stale snapshot.
                candidate = build_preferred_identity_index(
                    self.root / "chroma",
                    lambda: load_authorization_gallery(
                        self.root / "faces", self.db_path, by_user_id=True
                    ),
                    reconcile=True,
                    identity_key="user_id",
                    collection_version=revision,
                )
                self._index, self._revision = candidate, revision
            return self._index.size

    @property
    def available(self):
        return self._index.available

    @property
    def backend(self):
        return self._index.backend

    @property
    def size(self):
        return self._index.size

    def search(self, embedding):
        with self._lock:
            for _ in range(2):
                self.refresh()
                result = self._index.search(embedding)
                if database.gallery_revision(self.db_path) == self._revision:
                    return result
            raise IdentityIndexUnavailableError("Gallery changed during recognition; retry")
