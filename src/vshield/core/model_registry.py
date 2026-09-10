"""Strict, local model contract loading; unverified artifacts are never runnable."""
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/ai-models.yaml"
SOURCE_COMMIT = "b6d5f04ad78778917853b25c778acef6d5626d15"


class ModelContractError(RuntimeError):
    """Public-safe model configuration error."""


@dataclass(frozen=True)
class PadContract:
    model_name: str
    model_version: str
    model_path: Path
    checksum: str
    threshold: float
    crop_scale: float
    real_class_index: int
    source_commit: str
    source_checkpoint_sha256: str


def _artifact_path(value, root):
    path = (root / value).resolve()
    if not path.is_relative_to((root / "artifacts/models").resolve()):
        raise ValueError("artifact location")
    return path


def load_pad_contract(config_path=DEFAULT_CONFIG, *, project_root=PROJECT_ROOT):
    """Require a pinned reference, checksum, and matching passing parity evidence."""
    try:
        root = Path(project_root).resolve()
        data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["anti_spoof"]
        required = {
            "model_name": "MiniFASNetV2", "runtime": "onnxruntime_cpu",
            "input_size": [80, 80], "input_layout": "NCHW",
            "input_color_space": "BGR", "input_dtype": "float32",
            "normalization": "none", "output_type": "logits",
            "output_classes": ["spoof_0", "real", "spoof_2"],
            "real_class_index": 1, "crop_scale": 2.7,
            "source_commit": SOURCE_COMMIT,
        }
        if any(data.get(key) != value for key, value in required.items()):
            raise ValueError("unsupported contract")
        if data.get("verified_contract") is not True:
            raise ValueError("unverified contract")
        threshold = data["threshold"]
        if isinstance(threshold, bool) or not 0.5 < float(threshold) <= 1.0:
            raise ValueError("threshold")
        for field in ("checksum", "source_checkpoint_sha256"):
            if not isinstance(data[field], str) or not re.fullmatch(r"[0-9a-f]{64}", data[field]):
                raise ValueError("checksum")
        if not isinstance(data["model_version"], str) or not data["model_version"].strip():
            raise ValueError("version")
        model_path = _artifact_path(data["model_path"], root)
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != data["checksum"]:
            raise ValueError("artifact mismatch")
        report = json.loads(_artifact_path(data["parity_report"], root).read_text(encoding="utf-8"))
        if report.get("passed") is not True or report.get("cases", 0) < 3:
            raise ValueError("parity not established")
        for key, expected in {
            "model_sha256": data["checksum"], "source_commit": SOURCE_COMMIT,
            "source_checkpoint_sha256": data["source_checkpoint_sha256"],
        }.items():
            if report.get(key) != expected:
                raise ValueError("parity artifact mismatch")
        for key in ("atol", "rtol"):
            if not 0 < float(report[key]) <= 0.0001:
                raise ValueError("parity tolerance")
        error = float(report["max_absolute_error"])
        if not 0 <= error < float("inf"):
            raise ValueError("invalid parity error")
        return PadContract(
            data["model_name"], data["model_version"], model_path, data["checksum"],
            float(threshold), 2.7, 1, SOURCE_COMMIT, data["source_checkpoint_sha256"],
        )
    except Exception as exc:
        raise ModelContractError("PAD model contract unavailable or invalid") from exc
