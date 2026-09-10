"""Isolated external PAD import; grouping is not approval for model training."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image

from vshield.data.deduplication import cluster_duplicates, write_candidates
from vshield.data.external_sources import (
    SOURCE,
    SOURCE_URL,
    read_annotations,
    read_provenance,
    source_file,
)
from vshield.data.grouped_split import grouped_split, write_protocols
from vshield.data.integrity import (
    MANIFEST_FIELDS,
    DatasetIntegrityError,
    build_manifest,
    sha256_file,
    write_csv,
)
from vshield.data.release_audit import audit_release


def _json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def import_celeba(source_root: Path, annotations: dict[str, Path], output: Path,
                  metadata: Path | None = None, limit: int = 200,
                  ratios: dict[str, float] | None = None) -> dict:
    """Write a fresh candidate version; existing releases are never touched."""
    ratios = ratios or {"train": .7, "val": .15, "test": .15}
    if (set(ratios) != {"train", "val", "test"}
            or any(not math.isfinite(v) or v <= 0 for v in ratios.values())
            or abs(sum(ratios.values()) - 1) > 1e-9):
        raise DatasetIntegrityError("Positive finite train/val/test ratios must sum to one")
    if not 1 <= limit <= 10000:
        raise DatasetIntegrityError("Limit must be 1..10000; current duplicate clustering is quadratic")
    source_root, output = source_root.resolve(), output.resolve()
    if output == source_root or source_root.is_relative_to(output) or output.is_relative_to(source_root):
        raise DatasetIntegrityError("Source and output trees must not overlap")
    source_rows = [row for split, path in sorted(annotations.items())
                   for row in read_annotations(path, split)]
    if not source_rows:
        raise DatasetIntegrityError("Provide at least one official annotation JSON")
    keys = [row["source_path"] for row in source_rows]
    if len({key.casefold() for key in keys}) != len(keys):
        raise DatasetIntegrityError("Image repeated or case-aliased across official annotation lists")
    sidecar = read_provenance(metadata)
    if set(sidecar) - set(keys):
        raise DatasetIntegrityError("Sidecar contains paths absent from supplied annotations")
    # Round-robin official split/label buckets makes bounded imports less one-sided.
    buckets = {}
    for row in source_rows:
        buckets.setdefault((row["official_split"], row["label"]), []).append(row)
    selected = []
    for index in range(max(map(len, buckets.values()))):
        for bucket in buckets.values():
            if index < len(bucket) and len(selected) < limit:
                selected.append(bucket[index])
        if len(selected) == limit:
            break
    files = [source_file(source_root, row["source_path"]) for row in selected]
    output.mkdir(parents=True, exist_ok=False)
    (output / "annotations").mkdir()
    annotation_hashes = {}
    for split, path in annotations.items():
        shutil.copyfile(path, output / "annotations" / f"{split}-label.json")
        annotation_hashes[split] = sha256_file(path)
    if metadata:
        shutil.copyfile(metadata, output / "annotations" / "provenance.csv")
    prepared, ledger = [], []
    for row, original in zip(selected, files, strict=True):
        token = hashlib.sha256(f"{SOURCE}:{row['source_path']}".encode()).hexdigest()[:24]
        if original.stat().st_size > 32_000_000:
            raise DatasetIntegrityError("Source image exceeds 32MB compressed size limit")
        with Image.open(original) as header:
            if header.width * header.height > 25_000_000 or min(header.size) <= 0:
                raise DatasetIntegrityError("Source image exceeds 25 megapixel decode limit")
        original_copy = output / "originals" / row["source_path"]
        original_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, original_copy)
        image = cv2.imread(str(original_copy), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise DatasetIntegrityError(f"Cannot decode {row['source_path']}; partial import retained")
        normalized = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
        relative = f"images/{token}.png"
        destination = output / "normalized" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), normalized):
            raise DatasetIntegrityError(f"Cannot write {destination}")
        labels = destination.parent.parent / "labels"
        labels.mkdir(exist_ok=True)
        (labels / f"{token}.txt").write_text(f"{row['label']}\n", encoding="utf-8")
        supplied = sidecar.get(row["source_path"], {})
        if supplied.get("subject_id") and supplied["subject_id"] != row["subject_id"]:
            raise DatasetIntegrityError("Sidecar subject must match namespaced official subject")
        record = {key: value for key, value in supplied.items() if key in MANIFEST_FIELDS}
        record.update(sample_id=token, relative_path=relative, subject_id=row["subject_id"],
                      attack_type=row["attack_type"], split=row["official_split"])
        prepared.append(record)
        ledger.append({**row, "sample_id": token, "relative_path": relative,
                       "original_sha256": sha256_file(original_copy),
                       "normalized_sha256": sha256_file(destination),
                       "provenance_source": supplied.get("provenance_source", ""),
                       "subject_provenance": "official Data/split/subject path", "source_url": SOURCE_URL})
    write_csv(output / "metadata.csv", prepared)
    write_csv(output / "source-ledger.csv", ledger, tuple(ledger[0]))
    rows = build_manifest(output / "normalized", output / "metadata.csv")
    # Preserve official membership as evidence, never label it strict-v2 released.
    write_csv(output / "official-membership.csv", rows)
    rows, duplicates = cluster_duplicates(rows)
    write_candidates(output / "duplicate-candidates.csv", duplicates)
    splits, quarantine = grouped_split(rows, ratios)
    proposed = [row for split in splits.values() for row in split]
    audit = audit_release([*proposed, *quarantine])  # No fabricated shortcut probe.
    audit["proposed_samples_evaluated"] = audit["released_samples"]
    audit["released_samples"] = 0
    audit["note"] = "Proposed custom grouped split only; requires real shortcut audit and release approval"
    _json(output / "release-audit.json", audit)
    # Existing grouped_split marks proposals released; do not expose that as approval.
    for row in proposed:
        row["status"] = "candidate"
        row["reason_code"] = "RELEASE_AUDIT_REQUIRED"
    write_protocols(output / "proposed-protocols", splits)
    write_csv(output / "quarantine.csv", quarantine)
    write_csv(output / "manifest.csv", [*proposed, *quarantine])
    report = {"source": SOURCE, "source_url": SOURCE_URL,
              "license": "non-commercial research only; no redistribution",
              "annotation_sha256": annotation_hashes, "annotation_samples": len(source_rows),
              "imported_samples": len(rows), "limit": limit, "released_samples": 0,
              "quarantined_samples": len(quarantine), "proposed_split_counts": {k: len(v) for k, v in splits.items()},
              "official_split_counts": dict(Counter(row["official_split"] for row in ledger)),
              "label_counts": dict(Counter(str(row["label"]) for row in ledger)),
              "target_ratios": ratios, "preprocessing": "BGR uint8 128x128 PNG; loader float32 /255",
              "passed": False, "status": "candidate_only", "source_protocol_modified": False}
    _json(output / "import-report.json", report)
    return report
