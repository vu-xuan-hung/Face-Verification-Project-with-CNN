"""PAD-gated identification; SQLite authorization remains in the API layer."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from vshield.core.anti_spoof import PadResult, PadStatus, build_pad_service, check_pad
from vshield.core.embedder import FaceEmbedder
from vshield.core.face_preprocessor import (
    FacePreprocessingError,
    FacePreprocessor,
    InvalidFaceCountError,
)
from vshield.core.managed_identity_index import ManagedIdentityIndex


class AuthenticationStatus(str, Enum):
    AUTHENTICATED = "authenticated"
    INVALID_FACE = "invalid_face"
    SPOOF = "spoof"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class AuthenticationResult:
    status: AuthenticationStatus
    message: str
    username: str | None = None
    distance: float | None = None
    user_id: int | None = None
    code: str | None = None
    pad: PadResult | None = None
    recognition: dict | None = None

    def metadata(self):
        return {"code": self.code or self.status.value.upper(),
                "pad": self.pad.to_dict() if self.pad else None,
                "recognition": self.recognition,
                "user_id": self.user_id}


class AuthenticationService:
    def __init__(self, *, face_preprocessor, anti_spoof_model, face_embedder,
                 identity_index, identity_is_user_id=False):
        self.face_preprocessor = face_preprocessor
        self.anti_spoof_model = anti_spoof_model
        self.face_embedder = face_embedder
        self.identity_index = identity_index
        self.identity_is_user_id = identity_is_user_id

    def authenticate(self, image: np.ndarray) -> AuthenticationResult:
        # Detection must establish exactly one face before PAD receives a bbox.
        try:
            crops = self.face_preprocessor.extract(image)
        except InvalidFaceCountError as exc:
            code = "NO_FACE" if exc.count == 0 else "MULTIPLE_FACES"
            return AuthenticationResult(AuthenticationStatus.INVALID_FACE, str(exc), code=code)
        except FacePreprocessingError:
            return AuthenticationResult(AuthenticationStatus.INVALID_FACE, "Invalid face image",
                                        code="INVALID_IMAGE")
        except Exception:
            return AuthenticationResult(AuthenticationStatus.UNAVAILABLE, "Face detection unavailable",
                                        code="MODEL_ERROR")

        pad = check_pad(self.anti_spoof_model, image, crops.bbox)
        if pad.status is PadStatus.ERROR:
            return AuthenticationResult(AuthenticationStatus.UNAVAILABLE,
                                        "Anti-spoofing service unavailable",
                                        code=pad.reason, pad=pad)
        if not pad.is_real:
            return AuthenticationResult(AuthenticationStatus.SPOOF,
                                        "Spoofing detected" if pad.status is PadStatus.FAKE else "Liveness uncertain",
                                        code="SPOOF" if pad.status is PadStatus.FAKE else "PAD_UNCERTAIN", pad=pad)
        try:
            embedding = self.face_embedder.encode(crops.facenet)
            decision = self.identity_index.search_decision(embedding)
            details = decision.to_dict()
        except Exception:
            return AuthenticationResult(AuthenticationStatus.UNAVAILABLE,
                                        "Face recognition service unavailable", code="MODEL_ERROR", pad=pad)
        if decision.decision == "GALLERY_UNAVAILABLE":
            return AuthenticationResult(AuthenticationStatus.UNAVAILABLE, "Identity gallery unavailable",
                                        code="GALLERY_UNAVAILABLE", pad=pad, recognition=details)
        if decision.decision != "MATCH":
            ambiguous = decision.decision == "AMBIGUOUS"
            return AuthenticationResult(
                AuthenticationStatus.AMBIGUOUS if ambiguous else AuthenticationStatus.UNKNOWN,
                "Ambiguous face match" if ambiguous else "Unknown face, not registered",
                distance=decision.distance, code=decision.decision, pad=pad, recognition=details)
        match = decision.match
        return AuthenticationResult(
            AuthenticationStatus.AUTHENTICATED, "Authentication successful",
            username=None if self.identity_is_user_id else match.username,
            user_id=int(match.username) if self.identity_is_user_id else None,
            distance=match.distance, code="SUCCESS", pad=pad, recognition=details)


def build_default_authentication_service(project_root: str | Path) -> AuthenticationService:
    root = Path(project_root).resolve()
    preprocessor = FacePreprocessor()
    embedder = FaceEmbedder()
    pad = build_pad_service(root / "configs/ai-models.yaml")
    index = ManagedIdentityIndex(root / "data/authorization", root / "login_logs.db")
    if pad.ready:
        try:
            index.refresh()
        except Exception:
            # Preserve server/readiness diagnostics; login refresh still fails closed.
            pass
    return AuthenticationService(face_preprocessor=preprocessor, anti_spoof_model=pad,
                                 face_embedder=embedder, identity_index=index,
                                 identity_is_user_id=True)
