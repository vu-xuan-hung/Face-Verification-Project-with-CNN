"""Evaluation adapter around the production biometric services."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from vshield.core.anti_spoof import build_pad_service, check_pad
from vshield.core.embedder import FaceEmbedder, normalize_embedding
from vshield.core.face_preprocessor import FacePreprocessor
from vshield.core.identity_index import (
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_MIN_MARGIN,
    IdentityIndex,
)
from vshield.services.authentication import AuthenticationService, AuthenticationStatus


def _milliseconds(start: float) -> float:
    return (perf_counter() - start) * 1000.0


class VShieldEvaluationAdapter:
    """Reuse production detection, PAD, embedding, matching, and orchestration."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        pad_threshold: float | None = None,
        recognition_threshold: float | None = None,
        ambiguity_threshold: float | None = None,
        face_preprocessor=None,
        pad_service=None,
        face_embedder=None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.face_preprocessor = face_preprocessor or FacePreprocessor()
        self.pad_service = pad_service or build_pad_service(
            self.project_root / "configs" / "ai-models.yaml"
        )
        self.face_embedder = face_embedder or FaceEmbedder()
        self.recognition_threshold = (
            DEFAULT_DISTANCE_THRESHOLD
            if recognition_threshold is None
            else float(recognition_threshold)
        )
        self.ambiguity_threshold = (
            DEFAULT_MIN_MARGIN if ambiguity_threshold is None else float(ambiguity_threshold)
        )
        if pad_threshold is not None:
            if not 0.5 < float(pad_threshold) < 1:
                raise ValueError("pad_threshold must satisfy the runnable production contract (0.5, 1)")
            contract = getattr(self.pad_service, "contract", None)
            if contract is None:
                raise ValueError("PAD threshold override requires an available production model")
            self.pad_service.contract = replace(contract, threshold=float(pad_threshold))

    @property
    def production_thresholds(self) -> dict[str, float | None]:
        contract = getattr(self.pad_service, "contract", None)
        return {
            "pad": getattr(contract, "threshold", None),
            "recognition": self.recognition_threshold,
            "ambiguity": self.ambiguity_threshold,
        }

    def build_gallery(self, embeddings: dict[str, list[np.ndarray]]) -> IdentityIndex:
        return IdentityIndex(
            embeddings,
            distance_threshold=self.recognition_threshold,
            min_margin=self.ambiguity_threshold,
        )

    def detect_face(self, image: np.ndarray) -> dict:
        started = perf_counter()
        crops = self.face_preprocessor.extract(image)
        return {"crops": crops, "face_detection_ms": _milliseconds(started)}

    def pad_predict(self, image: np.ndarray, *, detection: dict | None = None) -> dict:
        detection = detection or self.detect_face(image)
        started = perf_counter()
        result = check_pad(self.pad_service, image, detection["crops"].bbox)
        elapsed = _milliseconds(started)
        decision = "REAL" if result.is_real else "SPOOF" if result.score is not None else "ERROR"
        return {
            "decision": decision,
            "status": result.status.value,
            "real_score": result.score,
            "spoof_score": None if result.score is None else 1.0 - result.score,
            "class_index": result.class_index,
            "threshold": result.threshold,
            "reason": result.reason,
            "raw_output": result.to_dict(),
            "face_detection_ms": detection["face_detection_ms"],
            "pad_total_ms": elapsed,
        }

    def extract_embedding(self, image: np.ndarray, *, detection: dict | None = None) -> dict:
        detection = detection or self.detect_face(image)
        started = perf_counter()
        embedding = self.face_embedder.encode(detection["crops"].facenet)
        return {
            "embedding": normalize_embedding(embedding),
            "face_detection_ms": detection["face_detection_ms"],
            "embedding_ms": _milliseconds(started),
        }

    def compare_embedding(self, embedding: np.ndarray, gallery: IdentityIndex) -> dict:
        started = perf_counter()
        decision = gallery.search_decision(embedding)
        matching_ms = _milliseconds(started)
        return {
            "decision": "KNOWN" if decision.decision == "MATCH" else decision.decision,
            "identity": decision.match.username if decision.match else None,
            "best_identity": decision.best_identity,
            "best_score": decision.distance,
            "second_identity": decision.runner_up_identity,
            "second_best_score": decision.runner_up_distance,
            "margin": None
            if decision.runner_up_distance is None
            else decision.runner_up_distance - decision.distance,
            "metric": "normalized_l2",
            "matching_ms": matching_ms,
            "raw_output": decision.to_dict(),
        }

    def recognize(self, image: np.ndarray, gallery: IdentityIndex) -> dict:
        started = perf_counter()
        detection = self.detect_face(image)
        encoded = self.extract_embedding(image, detection=detection)
        result = self.compare_embedding(encoded["embedding"], gallery)
        result.update(
            face_detection_ms=detection["face_detection_ms"],
            embedding_ms=encoded["embedding_ms"],
            recognition_total_ms=_milliseconds(started),
        )
        return result

    def full_biometric_decision(self, image: np.ndarray, gallery: IdentityIndex) -> dict:
        service = AuthenticationService(
            face_preprocessor=self.face_preprocessor,
            anti_spoof_model=self.pad_service,
            face_embedder=self.face_embedder,
            identity_index=gallery,
        )
        started = perf_counter()
        result = service.authenticate(image)
        total_ms = _milliseconds(started)
        recognition = result.recognition or {}
        return {
            "status": result.status.value,
            "final_access": "ALLOW"
            if result.status is AuthenticationStatus.AUTHENTICATED
            else "DENY",
            "identity": result.username,
            "pad": result.pad.to_dict() if result.pad else None,
            "recognition": recognition,
            "failure_code": result.code,
            "total_biometric_pipeline_ms": total_ms,
        }

    def warmup(self, image: np.ndarray, gallery: IdentityIndex | None = None, runs: int = 1) -> None:
        for _ in range(max(0, runs)):
            if gallery is None:
                detection = self.detect_face(image)
                self.pad_predict(image, detection=detection)
            else:
                self.full_biometric_decision(image, gallery)
