"""Reproducible evaluation I/O helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields and not key.startswith("_"))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_clean(value), sort_keys=True)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_audit(output: str | Path, audit: dict) -> None:
    output = Path(output)
    write_json(output / "dataset_audit.json", audit)
    rows = []
    for issue in audit.get("issues", []):
        rows.append(dict(issue))
    write_csv(output / "dataset_audit.csv", rows)


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def save_yaml(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_clean(value), sort_keys=False), encoding="utf-8")


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment_info(project_root: str | Path) -> dict:
    root = Path(project_root).resolve()
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "packages": {
            name: package_version(name)
            for name in ("numpy", "opencv-python", "onnxruntime", "tensorflow", "keras", "keras-facenet", "chromadb", "faiss-cpu")
        },
        "git_commit": _git_commit(root),
    }
    try:
        import onnxruntime as ort

        info["onnxruntime_providers"] = ort.get_available_providers()
    except Exception:
        info["onnxruntime_providers"] = None
    try:
        import tensorflow as tf

        info["gpus"] = [device.name for device in tf.config.list_physical_devices("GPU")]
        info["cuda_built"] = bool(tf.test.is_built_with_cuda())
    except Exception:
        info["gpus"] = None
        info["cuda_built"] = None
    config_path = root / "configs/ai-models.yaml"
    if config_path.is_file():
        pad = load_yaml(config_path).get("anti_spoof", {})
        model_path = root / pad.get("model_path", "")
        info["models"] = {
            "pad_name": pad.get("model_name"),
            "pad_version": pad.get("model_version"),
            "pad_path": str(model_path),
            "pad_sha256": _sha256(model_path) if model_path.is_file() else None,
            "facenet": "keras-facenet default 20180402-114759",
            "facenet_weights_sha256": _facenet_hash(),
        }
    return info


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def timestamped_run_dir(root: str | Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(root) / "runs" / stamp
    suffix = 1
    while output.exists():
        output = Path(root) / "runs" / f"{stamp}_{suffix:02d}"
        suffix += 1
    output.mkdir(parents=True)
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _facenet_hash() -> str | None:
    path = Path.home() / ".keras-facenet/20180402-114759/20180402-114759-weights.h5"
    return _sha256(path) if path.is_file() else None
