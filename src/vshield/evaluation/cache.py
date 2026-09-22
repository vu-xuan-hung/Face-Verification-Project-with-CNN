"""Embedding cache bound to manifests and production implementation fingerprints."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embedding_fingerprint(
    project_root: str | Path, manifest: str | Path, sample_hashes: list[str] | None = None
) -> dict:
    root = Path(project_root)
    sources = [
        root / "src/vshield/core/face_preprocessor.py",
        root / "src/vshield/core/embedder.py",
        root / "src/vshield/core/identity_index.py",
    ]
    try:
        package = version("keras-facenet")
    except PackageNotFoundError:
        package = None
    payload = {
        "manifest_sha256": file_hash(manifest),
        "source_sha256": {str(path.relative_to(root)): file_hash(path) for path in sources},
        "keras_facenet_version": package,
        "contract": "facenet-512-l2-bgr-v1",
        "sample_hashes": sorted(sample_hashes or []),
    }
    weights = Path.home() / ".keras-facenet/20180402-114759/20180402-114759-weights.h5"
    payload["facenet_weights_sha256"] = file_hash(weights) if weights.is_file() else None
    encoded = json.dumps(payload, sort_keys=True).encode()
    return {**payload, "fingerprint": hashlib.sha256(encoded).hexdigest()}


def load_embeddings(path: str | Path, fingerprint: dict) -> dict[str, np.ndarray] | None:
    path = Path(path)
    metadata = path.with_suffix(".json")
    if not path.is_file() or not metadata.is_file():
        return None
    stored = json.loads(metadata.read_text(encoding="utf-8"))
    if stored.get("fingerprint") != fingerprint["fingerprint"]:
        return None
    with np.load(path, allow_pickle=False) as data:
        identifiers = data["sample_ids"].astype(str).tolist()
        matrix = data["embeddings"].astype(np.float32)
    return dict(zip(identifiers, matrix, strict=True))


def save_embeddings(path: str | Path, embeddings: dict[str, np.ndarray], fingerprint: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    identifiers = list(embeddings)
    matrix = (
        np.stack([embeddings[key] for key in identifiers])
        if identifiers
        else np.empty((0, 512), dtype=np.float32)
    )
    np.savez_compressed(path, sample_ids=np.asarray(identifiers), embeddings=matrix)
    path.with_suffix(".json").write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
