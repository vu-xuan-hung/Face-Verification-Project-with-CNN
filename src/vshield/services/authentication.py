"""Authentication orchestration from face detection through identity search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Lock

import numpy as np

from vshield.core.anti_spoof import (
    AntiSpoofingError,
    load_anti_spoofing_model,
    predict_is_real,
)
from vshield.core.embedder import EmbeddingError, FaceEmbedder
from vshield.core.face_preprocessor import (
    FacePreprocessingError,
    FacePreprocessor,
    InvalidFaceCountError,
)
from vshield.core.verifier import (
    IdentityIndex,
    IdentityIndexError,
    IdentityIndexUnavailableError,
    load_database,
)


class AuthenticationStatus(str, Enum):
    AUTHENTICATED = "authenticated"
    INVALID_FACE = "invalid_face"
    SPOOF = "spoof"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class AuthenticationResult:
    status: AuthenticationStatus
    message: str
    username: str | None = None
    distance: float | None = None


class AuthenticationService:
    """Enforce anti-spoof before FaceNet and identity search."""

    def __init__(
        self,
        *,
        face_preprocessor: FacePreprocessor,
        anti_spoof_model,
        face_embedder: FaceEmbedder,
        identity_index: IdentityIndex,
    ):
        self.face_preprocessor = face_preprocessor
        self.anti_spoof_model = anti_spoof_model
        self.face_embedder = face_embedder
        self.identity_index = identity_index
        self._anti_spoof_lock = Lock()

    def authenticate(self, image: np.ndarray) -> AuthenticationResult:
        if self.anti_spoof_model is None:
            return AuthenticationResult(
                status=AuthenticationStatus.UNAVAILABLE,
                message="Anti-spoofing service unavailable",
            )

        try:
            crops = self.face_preprocessor.extract(image)
        except InvalidFaceCountError as exc:
            return AuthenticationResult(
                status=AuthenticationStatus.INVALID_FACE,
                message=str(exc),
            )
        except FacePreprocessingError as exc:
            return AuthenticationResult(
                status=AuthenticationStatus.INVALID_FACE,
                message=str(exc),
            )
        except Exception:
            return AuthenticationResult(
                status=AuthenticationStatus.UNAVAILABLE,
                message="Face preprocessing service unavailable",
            )

        face_batch = np.expand_dims(crops.anti_spoof, axis=0).astype(np.float32) / 255.0
        try:
            with self._anti_spoof_lock:
                is_real = predict_is_real(self.anti_spoof_model, face_batch)
        except AntiSpoofingError:
            return AuthenticationResult(
                status=AuthenticationStatus.UNAVAILABLE,
                message="Anti-spoofing service unavailable",
            )

        if not is_real:
            return AuthenticationResult(
                status=AuthenticationStatus.SPOOF,
                message="Spoofing detected",
            )

        try:
            embedding = self.face_embedder.encode(crops.facenet)
            match = self.identity_index.search(embedding)
        except (EmbeddingError, IdentityIndexUnavailableError, IdentityIndexError):
            return AuthenticationResult(
                status=AuthenticationStatus.UNAVAILABLE,
                message="Face recognition service unavailable",
            )
        except Exception:
            return AuthenticationResult(
                status=AuthenticationStatus.UNAVAILABLE,
                message="Face recognition service unavailable",
            )

        if match is None:
            return AuthenticationResult(
                status=AuthenticationStatus.UNKNOWN,
                message="Unknown face, not registered",
            )

        return AuthenticationResult(
            status=AuthenticationStatus.AUTHENTICATED,
            message="Authentication successful",
            username=match.username,
            distance=match.distance,
        )


def build_default_authentication_service(project_root: str | Path) -> AuthenticationService:
    """Build one immutable authentication snapshot for application startup."""
    root = Path(project_root).resolve()
    face_preprocessor = FacePreprocessor()
    face_embedder = FaceEmbedder()
    anti_spoof_model = load_anti_spoofing_model(
        root / "artifacts" / "models" / "face_verify_v1.keras"
    )
    if anti_spoof_model is None:
        return AuthenticationService(
            face_preprocessor=face_preprocessor,
            anti_spoof_model=None,
            face_embedder=face_embedder,
            identity_index=IdentityIndex({}),
        )

    def encode_enrollment(path: str | Path) -> np.ndarray:
        return face_embedder.encode_file(path, face_preprocessor=face_preprocessor)

    database_faces = load_database(root / "data" / "faces", encode_file=encode_enrollment)
    identity_index = IdentityIndex(database_faces)
    return AuthenticationService(
        face_preprocessor=face_preprocessor,
        anti_spoof_model=anti_spoof_model,
        face_embedder=face_embedder,
        identity_index=identity_index,
    )
