"""Strict CelebA-Spoof annotations and explicitly evidenced provenance."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath

from vshield.data.integrity import PROVENANCE_FIELDS, DatasetIntegrityError, read_csv

SOURCE = "celeba-spoof"
SOURCE_URL = "https://github.com/ZhangYuanhan-AI/CelebA-Spoof"
ATTACK_TYPES = ("live", "photo", "poster", "a4", "face-mask", "upper-body-mask",
                "region-mask", "pc", "pad", "phone", "3d-mask")


def safe_relative(value: str) -> PurePosixPath:
    """Reject traversal, Windows drives/ADS, aliases and reserved path segments."""
    path = PurePosixPath(value)
    if (not value or "\\" in value or ":" in value or path.is_absolute()
            or path.as_posix() != value or any(part in {".", ".."} for part in path.parts)
            or any(part.rstrip(" .") != part for part in path.parts)
            or any(re.match(r"^(CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])(?:\.|$)", part, re.I)
                   for part in path.parts)
            or any(any(ord(char) < 32 or char in '<>"|?*' for char in part) for part in path.parts)):
        raise DatasetIntegrityError(f"Unsafe source path: {value!r}")
    return path


def source_file(root: Path, relative: str) -> Path:
    path = root.joinpath(*safe_relative(relative).parts).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise DatasetIntegrityError(f"Missing or escaping source file: {relative}")
    return path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DatasetIntegrityError(f"Duplicate annotation key: {key}")
        result[key] = value
    return result


def read_annotations(path: Path, official_split: str) -> list[dict]:
    if official_split not in {"train", "val", "test"}:
        raise DatasetIntegrityError("Official split must be train, val or test")
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(payload, dict) or not payload:
        raise DatasetIntegrityError("Expected non-empty official path -> 44 attributes JSON")
    rows = []
    for relative, attributes in sorted(payload.items()):
        parts = safe_relative(relative).parts
        if (not isinstance(attributes, list) or len(attributes) != 44
                or type(attributes[43]) is not int or attributes[43] not in {0, 1}
                or type(attributes[40]) is not int or not 0 <= attributes[40] <= 10):
            raise DatasetIntegrityError(f"Invalid CelebA-Spoof attributes: {relative}")
        source_label = attributes[43]  # Author client.py: 0 live, 1 spoof.
        if (attributes[40] == 0) != (source_label == 0):
            raise DatasetIntegrityError(f"Conflicting binary/attack labels: {relative}")
        if len(parts) != 5 or parts[0] != "Data" or not parts[2].isdigit():
            raise DatasetIntegrityError(f"Expected Data/split/subject/live-or-spoof/file: {relative}")
        if parts[1] != official_split or parts[3] != ("live" if source_label == 0 else "spoof"):
            raise DatasetIntegrityError(f"Conflicting official split/path/label: {relative}")
        rows.append({"source_path": relative, "official_split": official_split,
                     "source_label": source_label, "label": 1 - source_label,
                     "subject_id": f"{SOURCE}:{parts[2]}",
                     "attack_type": ATTACK_TYPES[attributes[40]]})
    return rows


def read_provenance(path: Path | None) -> dict[str, dict]:
    """Never infer capture device/session/clip from individual image filenames."""
    result = {}
    for row in read_csv(path) if path else []:
        key = row.get("source_path", "")
        safe_relative(key)
        if key in result or not row.get("provenance_source", "").strip():
            raise DatasetIntegrityError("Sidecar requires unique source_path and provenance_source")
        result[key] = {field: row.get(field, "").strip()
                       for field in (*PROVENANCE_FIELDS, "attack_instrument_id", "capture_time")}
        result[key]["provenance_source"] = row["provenance_source"].strip()
    return result
