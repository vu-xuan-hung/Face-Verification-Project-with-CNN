from __future__ import annotations

from vshield.data.grouped_split import grouped_split
from vshield.data.release_audit import audit_release


def _row(index: int, label: int) -> dict[str, str]:
    return {
        "sample_id": f"sample-{index}",
        "relative_path": f"{index}.jpg",
        "split": "",
        "label": str(label),
        "subject_id": f"subject-{index}",
        "session_id": f"session-{index}",
        "clip_id": f"clip-{index}",
        "device_id": f"device-{index}",
        "attack_instrument_id": "",
        "sha256": f"hash-{index}",
        "dhash": f"{index:016x}",
        "exact_component_id": f"exact-{index}",
        "near_component_id": f"near-{index}",
        "status": "candidate",
        "reason_code": "",
    }


def test_grouped_split_is_deterministic_and_non_empty():
    rows = [_row(index, index % 2) for index in range(30)]
    first, quarantine = grouped_split([dict(row) for row in rows])
    second, _ = grouped_split([dict(row) for row in rows])
    assert not quarantine
    assert all(first[name] for name in ("train", "val", "test"))
    assert {row["sample_id"]: row["split"] for split in first.values() for row in split} == {
        row["sample_id"]: row["split"] for split in second.values() for row in split
    }


def test_near_component_never_crosses_splits():
    rows = [_row(index, index % 2) for index in range(30)]
    rows[0]["near_component_id"] = "shared-near"
    rows[1]["near_component_id"] = "shared-near"
    splits, _ = grouped_split(rows)
    selected = {
        row["split"]
        for members in splits.values()
        for row in members
        if row["near_component_id"] == "shared-near"
    }
    assert len(selected) == 1


def test_subject_never_crosses_splits_across_sessions():
    rows = [_row(index, index % 2) for index in range(30)]
    rows[0]["subject_id"] = "same-subject"
    rows[1]["subject_id"] = "same-subject"
    splits, _ = grouped_split(rows)
    selected = {
        row["split"]
        for members in splits.values()
        for row in members
        if row["subject_id"] == "same-subject"
    }
    assert len(selected) == 1


def test_missing_provenance_is_quarantined():
    row = _row(1, 1)
    row["subject_id"] = ""
    splits, quarantine = grouped_split([row])
    assert not any(splits.values())
    assert quarantine[0]["reason_code"] == "MISSING_PROVENANCE"


def test_release_audit_detects_cross_split_exact_content():
    rows = [_row(index, index % 2) for index in range(6)]
    for index, row in enumerate(rows):
        row["status"] = "released"
        row["split"] = ("train", "val", "test")[index % 3]
    rows[1]["sha256"] = rows[0]["sha256"]
    report = audit_release(rows, {"passed": True})
    assert not report["passed"]
    assert "EXACT_CONTENT_LEAKAGE" in {failure["gate"] for failure in report["failures"]}


def test_release_requires_shortcut_report():
    rows = [_row(index, index % 2) for index in range(6)]
    for index, row in enumerate(rows):
        row["status"] = "released"
        row["split"] = ("train", "val", "test")[index % 3]
    report = audit_release(rows)
    assert "SHORTCUT_REPORT_MISSING" in {failure["gate"] for failure in report["failures"]}


def test_release_binds_shortcut_report_to_manifest():
    rows = [_row(index, index % 2) for index in range(6)]
    for index, row in enumerate(rows):
        row["status"] = "released"
        row["split"] = ("train", "val", "test")[index % 3]
    report = audit_release(
        rows,
        {"passed": True, "manifest_sha256": "other"},
        expected_manifest_sha256="expected",
    )
    assert "SHORTCUT_REPORT_MANIFEST_MISMATCH" in {
        failure["gate"] for failure in report["failures"]
    }
