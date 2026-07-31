"""Release gates for immutable anti-spoof dataset protocols."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from vshield.data.integrity import PROVENANCE_FIELDS


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cross_split_values(rows: list[dict[str, str]], field: str) -> dict[str, list[str]]:
    splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        value = row.get(field, "")
        if value:
            splits[value].add(row["split"])
    return {value: sorted(names) for value, names in splits.items() if len(names) > 1}


def audit_release(
    rows: list[dict[str, str]],
    shortcut_report: dict[str, object] | None = None,
    expected_manifest_sha256: str | None = None,
    gates: dict[str, object] | None = None,
) -> dict[str, object]:
    gates = gates or {}
    released = [row for row in rows if row.get("status") == "released"]
    failures: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []

    required = (
        "sample_id",
        "relative_path",
        "split",
        "label",
        "sha256",
        "dhash",
        "exact_component_id",
        "near_component_id",
        *gates.get("required_provenance", PROVENANCE_FIELDS),
    )
    missing = [
        row.get("sample_id", f"row-{index}")
        for index, row in enumerate(released)
        if any(not row.get(field) for field in required)
    ]
    if missing:
        failures.append({"gate": "PROVENANCE", "count": len(missing), "examples": missing[:10]})

    split_names = tuple(gates.get("split_names", ("train", "val", "test")))
    invalid_splits = sorted({row.get("split", "") for row in released} - set(split_names))
    if invalid_splits:
        failures.append({"gate": "SPLIT_NAMES", "values": invalid_splits})
    counts = {split: sum(row.get("split") == split for row in released) for split in split_names}
    if gates.get("require_non_empty_splits", True) and not all(counts.values()):
        failures.append({"gate": "NON_EMPTY_SPLITS", "counts": counts})

    for field, gate, limit_key in (
        ("sha256", "EXACT_CONTENT_LEAKAGE", "max_cross_split_exact_hashes"),
        ("exact_component_id", "EXACT_COMPONENT_LEAKAGE", "max_cross_split_exact_components"),
        ("near_component_id", "NEAR_COMPONENT_LEAKAGE", "max_cross_split_near_components"),
        ("subject_id", "SUBJECT_LEAKAGE", "max_cross_split_subjects"),
        ("session_id", "SESSION_LEAKAGE", "max_cross_split_sessions"),
        ("clip_id", "CLIP_LEAKAGE", "max_cross_split_clips"),
        ("device_id", "DEVICE_LEAKAGE", "max_cross_split_devices"),
    ):
        overlap = _cross_split_values(released, field)
        if len(overlap) > int(gates.get(limit_key, 0)):
            failures.append(
                {"gate": gate, "count": len(overlap), "examples": dict(list(overlap.items())[:10])}
            )

    for split in split_names:
        labels = {row["label"] for row in released if row["split"] == split}
        if labels != {"0", "1"}:
            failures.append({"gate": "LABEL_COVERAGE", "split": split, "labels": sorted(labels)})

    attack_types = {
        row.get("attack_type", "")
        for row in released
        if row["split"] == "train" and row["label"] == "0" and row.get("attack_type")
    }
    for split in (name for name in ("val", "test") if name in split_names):
        available = {
            row.get("attack_type", "")
            for row in released
            if row["split"] == split and row["label"] == "0"
        }
        missing_attacks = sorted(attack_types - available)
        if missing_attacks:
            failures.append({"gate": "ATTACK_COVERAGE", "split": split, "missing": missing_attacks})

    identifiers: dict[str, str] = {}
    for row in released:
        sample_id = row["sample_id"]
        if sample_id in identifiers:
            failures.append({"gate": "DUPLICATE_SAMPLE_ID", "sample_id": sample_id})
            break
        identifiers[sample_id] = row["relative_path"]

    if shortcut_report is None and gates.get("require_shortcut_report", True):
        failures.append({"gate": "SHORTCUT_REPORT_MISSING"})
    elif not shortcut_report.get("passed"):
        failures.append(
            {"gate": "SHORTCUT_PROBES", "failures": shortcut_report.get("failures", [])}
        )
    if shortcut_report is not None and expected_manifest_sha256:
        if shortcut_report.get("manifest_sha256") != expected_manifest_sha256:
            failures.append({"gate": "SHORTCUT_REPORT_MANIFEST_MISMATCH"})

    quarantined = sum(row.get("status") == "quarantine" for row in rows)
    if quarantined:
        warnings.append({"gate": "QUARANTINED_ROWS", "count": quarantined})
    return {
        "passed": not failures,
        "released_samples": len(released),
        "split_counts": counts,
        "failures": failures,
        "warnings": warnings,
    }


def registry_manifest(
    protocol_paths: list[Path],
    audit_report: dict[str, object],
    dataset_version: str,
) -> dict[str, object]:
    if not audit_report.get("passed"):
        raise ValueError("Cannot create registry manifest for failed release")
    return {
        "dataset_version": dataset_version,
        "status": "released",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocols": {
            path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in protocol_paths
        },
        "audit_sha256": hashlib.sha256(
            json.dumps(audit_report, sort_keys=True).encode()
        ).hexdigest(),
    }
