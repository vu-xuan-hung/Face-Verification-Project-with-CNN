"""Dataset manifest primitives for leakage-safe anti-spoof protocols."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROVENANCE_FIELDS = ("subject_id", "session_id", "clip_id", "device_id")
MANIFEST_FIELDS = (
    "sample_id",
    "relative_path",
    "split",
    "label",
    "subject_id",
    "session_id",
    "clip_id",
    "device_id",
    "attack_type",
    "attack_instrument_id",
    "capture_time",
    "sha256",
    "dhash",
    "height",
    "width",
    "bytes",
    "brightness",
    "contrast",
    "blur",
    "exact_component_id",
    "near_component_id",
    "status",
    "reason_code",
)


class DatasetIntegrityError(ValueError):
    """Raised when data cannot be released without guessing."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dhash(gray: np.ndarray) -> str:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = sum(int(bit) << index for index, bit in enumerate(bits.flat))
    return f"{value:016x}"


def image_features(path: Path) -> dict[str, str]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise DatasetIntegrityError(f"Cannot decode image: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "height": str(image.shape[0]),
        "width": str(image.shape[1]),
        "bytes": str(path.stat().st_size),
        "brightness": f"{float(gray.mean()):.6f}",
        "contrast": f"{float(gray.std()):.6f}",
        "blur": f"{float(cv2.Laplacian(gray, cv2.CV_64F).var()):.6f}",
        "dhash": _dhash(gray),
    }


def _label_path(image_path: Path) -> Path:
    if image_path.parent.name == "images":
        return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    return image_path.with_suffix(".txt")


def read_label(image_path: Path) -> int:
    path = _label_path(image_path)
    if not path.is_file():
        raise DatasetIntegrityError(f"Missing label: {path}")
    content = path.read_text(encoding="utf-8").strip().splitlines()
    if not content:
        raise DatasetIntegrityError(f"Empty label: {path}")
    try:
        label = int(content[0].split()[0])
    except (IndexError, ValueError) as exc:
        raise DatasetIntegrityError(f"Invalid label: {path}") from exc
    if label not in {0, 1}:
        raise DatasetIntegrityError(f"Label must be 0 or 1: {path}")
    return label


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields=MANIFEST_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _metadata_index(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    rows = read_csv(path)
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row.get("relative_path") or row.get("sample_id")
        if not key or key in index:
            raise DatasetIntegrityError("Metadata needs unique relative_path or sample_id")
        index[key.replace("\\", "/")] = row
    return index


def _source_split(relative: Path) -> str:
    parts = {part.lower() for part in relative.parts}
    return next((name for name in ("train", "val", "test") if name in parts), "")


def build_manifest(root: Path, metadata_path: Path | None = None) -> list[dict[str, str]]:
    root = root.resolve()
    metadata = _metadata_index(metadata_path)
    images = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise DatasetIntegrityError(f"No images found below {root}")

    rows: list[dict[str, str]] = []
    for image_path in images:
        relative = image_path.relative_to(root).as_posix()
        supplied = metadata.get(relative, metadata.get(image_path.stem, {}))
        missing = [field for field in PROVENANCE_FIELDS if not supplied.get(field)]
        reason = "MISSING_PROVENANCE:" + ",".join(missing) if missing else ""
        digest = sha256_file(image_path)
        row = dict.fromkeys(MANIFEST_FIELDS, "")
        row.update(supplied)
        row.update(image_features(image_path))
        row.update(
            {
                "sample_id": supplied.get("sample_id")
                or hashlib.sha256(relative.encode()).hexdigest()[:20],
                "relative_path": relative,
                "split": supplied.get("split") or _source_split(Path(relative)),
                "label": str(read_label(image_path)),
                "sha256": digest,
                "status": "quarantine" if missing else "candidate",
                "reason_code": reason,
            }
        )
        rows.append(row)
    return rows
