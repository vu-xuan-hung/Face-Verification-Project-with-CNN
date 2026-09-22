"""Strict manifest loading and exact-leakage auditing for biometric evaluation."""

from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from PIL import Image, UnidentifiedImageError

PAD_FIELDS = {"sample_id", "image_path", "label"}
RECOGNITION_FIELDS = {"sample_id", "image_path", "subject_id", "split", "is_enrolled"}
E2E_FIELDS = {
    "sample_id",
    "image_path",
    "subject_id",
    "presentation",
    "expected_identity_state",
    "expected_access",
}


class DatasetValidationError(ValueError):
    pass


class DatasetLeakageError(DatasetValidationError):
    pass


def read_manifest(path: str | Path, kind: str) -> list[dict[str, str]]:
    path = Path(path).resolve()
    required = {"pad": PAD_FIELDS, "recognition": RECOGNITION_FIELDS, "e2e": E2E_FIELDS}[kind]
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise DatasetValidationError(f"Missing {kind} manifest columns: {sorted(missing)}")
        rows = []
        for line, raw in enumerate(reader, start=2):
            row = {key: (value or "").strip() for key, value in raw.items()}
            row["_manifest"] = str(path)
            row["_line"] = str(line)
            image = Path(row["image_path"])
            row["_resolved_path"] = str(
                image.resolve() if image.is_absolute() else (path.parent / image).resolve()
            )
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_error(path: Path) -> str | None:
    try:
        if path.stat().st_size > 8 * 1024 * 1024:
            return "FILE_TOO_LARGE"
        with Image.open(path) as header:
            if header.width * header.height > 20_000_000:
                return "IMAGE_TOO_LARGE"
            header.verify()
        if cv2.imread(str(path), cv2.IMREAD_COLOR) is None:
            return "CORRUPTED_IMAGE"
    except (OSError, UnidentifiedImageError):
        return "CORRUPTED_IMAGE"
    return None


def audit_manifest(rows: list[dict[str, str]], kind: str) -> dict:
    issues: list[dict] = []
    path_groups: dict[str, list[dict]] = defaultdict(list)
    hash_groups: dict[str, list[dict]] = defaultdict(list)
    sample_counts = Counter(row.get("sample_id", "") for row in rows)
    allowed = {
        "pad": {"label": {"REAL", "SPOOF"}},
        "recognition": {"split": {"GALLERY", "CALIBRATION", "TEST"}, "is_enrolled": {"0", "1"}},
        "e2e": {
            "presentation": {"REAL", "SPOOF"},
            "expected_identity_state": {"KNOWN", "UNKNOWN"},
            "expected_access": {"ALLOW", "DENY"},
        },
    }[kind]
    audited = []
    for row in rows:
        item = dict(row)
        item_issues = []
        path = Path(row["_resolved_path"])
        path_groups[str(path).lower()].append(row)
        if not path.is_file():
            item_issues.append("MISSING_FILE")
        else:
            error = _image_error(path)
            if error:
                item_issues.append(error)
            try:
                item["sha256"] = _sha256(path)
                hash_groups[item["sha256"]].append(row)
            except OSError:
                item_issues.append("UNREADABLE_FILE")
        if not row.get("sample_id") or sample_counts[row.get("sample_id", "")] > 1:
            item_issues.append("DUPLICATE_OR_EMPTY_SAMPLE_ID")
        for field, values in allowed.items():
            if row.get(field, "").upper() not in values:
                item_issues.append(f"INVALID_{field.upper()}")
        item["issues"] = sorted(set(item_issues))
        audited.append(item)
        issues.extend({"code": code, "sample_id": row.get("sample_id"), "path": str(path)} for code in item["issues"])

    for group in path_groups.values():
        if len(group) > 1:
            for row in group:
                issues.append({"code": "DUPLICATE_PATH", "sample_id": row["sample_id"], "path": row["_resolved_path"]})
    leakage = []
    for digest, group in hash_groups.items():
        if len(group) < 2:
            continue
        splits = {row.get("split", "test").lower() for row in group}
        exact_leak = bool({"gallery", "test"} <= splits or {"calibration", "test"} <= splits)
        record = {
            "sha256": digest,
            "sample_ids": [row["sample_id"] for row in group],
            "splits": sorted(splits),
            "severe": exact_leak,
        }
        leakage.append(record)
        issues.append({"code": "EXACT_HASH_LEAKAGE" if exact_leak else "DUPLICATE_HASH", **record})

    _append_semantic_issues(rows, kind, issues)
    class_field = "label" if kind == "pad" else "presentation" if kind == "e2e" else "is_enrolled"
    classes = Counter(row.get(class_field, "").upper() for row in rows)
    return {
        "kind": kind,
        "n_rows": len(rows),
        "class_counts": dict(classes),
        "issues": issues,
        "leakage": leakage,
        "has_severe_leakage": any(item["severe"] for item in leakage),
        "rows": audited,
    }


def _append_semantic_issues(rows: list[dict[str, str]], kind: str, issues: list[dict]) -> None:
    if kind == "pad":
        labels = {row.get("label", "").upper() for row in rows}
        for missing in {"REAL", "SPOOF"} - labels:
            issues.append({"code": "EMPTY_CLASS", "class": missing})
    elif kind == "recognition":
        gallery_ids = {row["subject_id"] for row in rows if row.get("split", "").lower() == "gallery"}
        splits = Counter(row.get("split", "").lower() for row in rows)
        for split in ("gallery", "calibration", "test"):
            if not splits[split]:
                issues.append({"code": "EMPTY_SPLIT", "split": split})
        for enrolled in ("0", "1"):
            if not any(row.get("is_enrolled") == enrolled and row.get("split", "").lower() != "gallery" for row in rows):
                issues.append({"code": "EMPTY_PROBE_CLASS", "is_enrolled": enrolled})
        for row in rows:
            enrolled = row.get("is_enrolled") == "1"
            if row.get("split", "").lower() == "gallery" and not enrolled:
                issues.append({"code": "UNKNOWN_IN_GALLERY", "sample_id": row["sample_id"]})
            if row.get("split", "").lower() != "gallery" and enrolled and row["subject_id"] not in gallery_ids:
                issues.append({"code": "KNOWN_IDENTITY_MISSING_FROM_GALLERY", "sample_id": row["sample_id"]})
            if not enrolled and row["subject_id"] in gallery_ids:
                issues.append({"code": "UNKNOWN_IDENTITY_PRESENT_IN_GALLERY", "sample_id": row["sample_id"]})
    else:
        presentations = {row.get("presentation", "").upper() for row in rows}
        identity_states = {
            row.get("expected_identity_state", "").upper() for row in rows
        }
        for missing in {"REAL", "SPOOF"} - presentations:
            issues.append({"code": "EMPTY_CLASS", "field": "presentation", "class": missing})
        for missing in {"KNOWN", "UNKNOWN"} - identity_states:
            issues.append(
                {"code": "EMPTY_CLASS", "field": "expected_identity_state", "class": missing}
            )
        for row in rows:
            expected = "ALLOW" if row.get("presentation", "").upper() == "REAL" and row.get("expected_identity_state", "").upper() == "KNOWN" else "DENY"
            if row.get("expected_access", "").upper() != expected:
                issues.append({"code": "INCONSISTENT_EXPECTED_ACCESS", "sample_id": row["sample_id"], "expected_for_category": expected})


def require_no_leakage(audit: dict, allow_leakage: bool = False) -> None:
    fatal = {
        "CORRUPTED_IMAGE",
        "DUPLICATE_HASH",
        "DUPLICATE_OR_EMPTY_SAMPLE_ID",
        "DUPLICATE_PATH",
        "EMPTY_CLASS",
        "EMPTY_PROBE_CLASS",
        "EMPTY_SPLIT",
        "FILE_TOO_LARGE",
        "IMAGE_TOO_LARGE",
        "INVALID_LABEL",
        "INVALID_SPLIT",
        "INVALID_IS_ENROLLED",
        "INVALID_PRESENTATION",
        "INVALID_EXPECTED_IDENTITY_STATE",
        "INVALID_EXPECTED_ACCESS",
        "INCONSISTENT_EXPECTED_ACCESS",
        "UNKNOWN_IN_GALLERY",
        "KNOWN_IDENTITY_MISSING_FROM_GALLERY",
        "MISSING_FILE",
        "UNREADABLE_FILE",
        "UNKNOWN_IDENTITY_PRESENT_IN_GALLERY",
    }
    invalid = sorted({issue.get("code") for issue in audit.get("issues", []) if issue.get("code") in fatal})
    if invalid:
        raise DatasetValidationError(f"Manifest contains fatal semantic errors: {invalid}")
    if audit["has_severe_leakage"] and not allow_leakage:
        raise DatasetLeakageError(
            "Exact gallery/test or calibration/test hash leakage detected; rerun only with --allow-leakage"
        )
    if audit["has_severe_leakage"]:
        print("WARNING: EXACT DATA LEAKAGE IS PRESENT; RESULTS ARE NOT A CLEAN FINAL EVALUATION", file=sys.stderr)


def cross_manifest_leakage(first: dict, second: dict, first_name: str, second_name: str) -> list[dict]:
    first_hashes = {row.get("sha256"): row for row in first.get("rows", []) if row.get("sha256")}
    second_hashes = {row.get("sha256"): row for row in second.get("rows", []) if row.get("sha256")}
    return [
        {
            "code": "EXACT_CROSS_MANIFEST_LEAKAGE",
            "sha256": digest,
            "first_manifest": first_name,
            "second_manifest": second_name,
            "first_sample_id": first_hashes[digest]["sample_id"],
            "second_sample_id": second_hashes[digest]["sample_id"],
        }
        for digest in sorted(first_hashes.keys() & second_hashes.keys())
    ]
