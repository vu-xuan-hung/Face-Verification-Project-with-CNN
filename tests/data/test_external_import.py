"""External source contracts; generated test fixtures are not research samples."""

import json

import cv2
import numpy as np
import pytest

from vshield.data.external_import import import_celeba
from vshield.data.external_sources import read_annotations, read_provenance, safe_relative
from vshield.data.integrity import DatasetIntegrityError, read_csv


def fixture_source(tmp_path):
    source = tmp_path / "source"
    annotations = {}
    for index, kind in enumerate(("live", "spoof")):
        relative = f"Data/train/{index + 1}/{kind}/001.png"
        path = source / relative
        path.parent.mkdir(parents=True)
        image = np.random.default_rng(index).integers(0, 256, (48, 60, 3), dtype=np.uint8)
        assert cv2.imwrite(str(path), image)
        attrs = [0] * 44
        attrs[40] = attrs[43] = index
        annotations[relative] = attrs
    labels = tmp_path / "train.json"
    labels.write_text(json.dumps(annotations))
    return source, labels


def test_official_labels_are_inverted_and_unknown_provenance_quarantined(tmp_path):
    source, labels = fixture_source(tmp_path)
    output = tmp_path / "candidate"
    report = import_celeba(source, {"train": labels}, output)
    assert report["imported_samples"] == report["quarantined_samples"] == 2
    assert report["released_samples"] == 0
    assert report["proposed_split_counts"] == {"train": 0, "val": 0, "test": 0}
    assert report["label_counts"] == {"1": 1, "0": 1}
    for row in read_csv(output / "manifest.csv"):
        assert row["status"] == "quarantine"
        assert "session_id" in row["reason_code"]
        image = cv2.imread(str(output / "normalized" / row["relative_path"]))
        assert image.shape == (128, 128, 3)
    assert len(list((output / "originals").rglob("*.png"))) == 2
    assert not json.loads((output / "release-audit.json").read_text())["passed"]


@pytest.mark.parametrize("value", ["../a.jpg", "/a.jpg", "C:/a.jpg", "a\\b.jpg", "a//b.jpg", "a/../b.jpg", "a./b.jpg", "a:stream", "CON.png", "a/LPT1.jpg", "a/b?.png"])
def test_unsafe_paths_rejected(value):
    with pytest.raises(DatasetIntegrityError):
        safe_relative(value)


def test_annotation_conflict_and_split_rejected(tmp_path):
    _, labels = fixture_source(tmp_path)
    with pytest.raises(DatasetIntegrityError, match="Conflicting official split"):
        read_annotations(labels, "test")
    payload = json.loads(labels.read_text())
    next(iter(payload.values()))[43] = 1
    labels.write_text(json.dumps(payload))
    with pytest.raises(DatasetIntegrityError, match="Conflicting binary"):
        read_annotations(labels, "train")


def test_existing_output_and_bad_limits_rejected(tmp_path):
    source, labels = fixture_source(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        import_celeba(source, {"train": labels}, output)
    with pytest.raises(DatasetIntegrityError):
        import_celeba(source, {"train": labels}, tmp_path / "new", limit=0)
    with pytest.raises(DatasetIntegrityError):
        import_celeba(source, {"train": labels}, tmp_path / "new", ratios={"train": float("nan"), "val": .1, "test": .1})


def test_sidecar_needs_evidence(tmp_path):
    metadata = tmp_path / "meta.csv"
    metadata.write_text("source_path,session_id\nData/train/1/live/001.png,guessed\n")
    with pytest.raises(DatasetIntegrityError, match="provenance_source"):
        read_provenance(metadata)


def test_source_missing_and_nested_outputs_fail_without_writes(tmp_path):
    source, labels = fixture_source(tmp_path)
    with pytest.raises(DatasetIntegrityError, match="overlap"):
        import_celeba(source, {"train": labels}, source / "new")
    with pytest.raises(DatasetIntegrityError, match="Missing"):
        import_celeba(tmp_path / "missing", {"train": labels}, tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_duplicate_json_keys_fail(tmp_path):
    labels = tmp_path / "duplicate.json"
    labels.write_text('{"x": [], "x": []}')
    with pytest.raises(DatasetIntegrityError, match="Duplicate annotation"):
        read_annotations(labels, "train")


def test_provenance_proposals_never_automatically_released(tmp_path):
    source, labels = fixture_source(tmp_path)
    meta = tmp_path / "meta.csv"
    meta.write_text("source_path,session_id,clip_id,device_id,provenance_source\n"
                    "Data/train/1/live/001.png,s1,c1,d1,test-fixture-only\n"
                    "Data/train/2/spoof/001.png,s2,c2,d2,test-fixture-only\n")
    output = tmp_path / "candidate"
    report = import_celeba(source, {"train": labels}, output, metadata=meta)
    rows = read_csv(output / "manifest.csv")
    assert all(row["status"] == "candidate" for row in rows)
    assert report["released_samples"] == 0
    audit = json.loads((output / "release-audit.json").read_text())
    assert audit["released_samples"] == 0
    assert audit["proposed_samples_evaluated"] == 2
    assert not audit["passed"]
    assert "SHORTCUT_REPORT_MISSING" in {failure["gate"] for failure in audit["failures"]}


def test_normalized_copy_uses_existing_bgr_area_contract(tmp_path):
    source, labels = fixture_source(tmp_path)
    output = tmp_path / "candidate"
    import_celeba(source, {"train": labels}, output, limit=1)
    ledger = read_csv(output / "source-ledger.csv")[0]
    original = cv2.imread(str(source / ledger["source_path"]))
    expected = cv2.resize(original, (128, 128), interpolation=cv2.INTER_AREA)
    actual = cv2.imread(str(output / "normalized" / ledger["relative_path"]))
    np.testing.assert_array_equal(actual, expected)
