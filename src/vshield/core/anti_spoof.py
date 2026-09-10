"""Structured pretrained PAD boundary. No production fallback to legacy Keras."""

import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path


class PadStatus(str, Enum):
    REAL = "REAL"
    FAKE = "FAKE"
    UNCERTAIN = "UNCERTAIN"
    ERROR = "ERROR"


@dataclass(frozen=True)
class PadResult:
    status: PadStatus
    score: float | None = None
    class_index: int | None = None
    model_version: str = "unavailable"
    threshold: float | None = None
    reason: str = "PAD_UNAVAILABLE"

    @property
    def is_real(self):
        return self.status is PadStatus.REAL

    def to_dict(self):
        return {**asdict(self), "status": self.status.value, "is_real": self.is_real}


class PadRejectedError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(result.reason)


class UnavailablePad:
    ready = False

    def __init__(self, reason="PAD_UNAVAILABLE"):
        self.reason = reason
        self.model_version = "unavailable"

    def predict(self, *, image, bbox):
        return PadResult(PadStatus.ERROR, reason=self.reason)


def build_pad_service(config_path=None):
    from vshield.core.minifasnet_onnx import MiniFASNetONNX

    config_path = config_path or Path(__file__).resolve().parents[3] / "configs/ai-models.yaml"
    try:
        return MiniFASNetONNX.from_config(config_path)
    except Exception:
        # Readiness and API expose only a safe code, never source paths/stack traces.
        return UnavailablePad()


def check_pad(pad, image, bbox):
    if pad is None or getattr(pad, "ready", False) is not True:
        return PadResult(PadStatus.ERROR, reason="PAD_UNAVAILABLE")
    try:
        result = pad.predict(image=image, bbox=bbox)
        if not isinstance(result, PadResult) or not isinstance(result.status, PadStatus):
            raise ValueError("Invalid PAD result")
        if result.status is not PadStatus.ERROR:
            if (type(result.class_index) is not int or result.class_index not in {0, 1, 2}
                    or result.score is None or not math.isfinite(result.score)
                    or not 0 <= result.score <= 1
                    or result.threshold is None or not math.isfinite(result.threshold)
                    or not 0 < result.threshold < 1):
                raise ValueError("Invalid PAD score")
        if result.is_real and (result.class_index != 1 or result.score < result.threshold):
            raise ValueError("Inconsistent PAD acceptance")
        return result
    except Exception:
        return PadResult(PadStatus.ERROR, reason="MODEL_ERROR")


def require_real(pad, image, bbox):
    result = check_pad(pad, image, bbox)
    if not result.is_real:
        raise PadRejectedError(result)
    return result


def load_anti_spoofing_model(model_path):
    """Explicit offline compatibility for historical training/evaluation tools only."""
    from vshield.core.legacy_keras_pad import load_anti_spoofing_model as legacy_load

    return legacy_load(model_path)
