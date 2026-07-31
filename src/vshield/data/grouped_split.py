"""Deterministic group-disjoint splitting for anti-spoof data."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

from vshield.data.deduplication import UnionFind
from vshield.data.integrity import MANIFEST_FIELDS, PROVENANCE_FIELDS, write_csv

SPLITS = ("train", "val", "test")


def _cost(
    counts: dict[str, list[int]],
    split: str,
    addition: list[int],
    targets: dict[str, list[float]],
) -> float:
    current = counts[split]
    projected = [counts[split][index] + addition[index] for index in range(2)]
    return sum(
        (projected[index] - targets[split][index]) ** 2
        - (current[index] - targets[split][index]) ** 2
        for index in range(2)
    )


def grouped_split(
    rows: list[dict[str, str]],
    ratios: dict[str, float] | None = None,
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    ratios = ratios or {"train": 0.7, "val": 0.15, "test": 0.15}
    if set(ratios) != set(SPLITS) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Ratios must define train/val/test and sum to 1")

    quarantine = [row for row in rows if row["status"] != "candidate"]
    eligible = [row for row in rows if row["status"] == "candidate"]
    for row in list(eligible):
        if any(not row.get(field) for field in PROVENANCE_FIELDS):
            row["status"] = "quarantine"
            row["reason_code"] = "MISSING_PROVENANCE"
            quarantine.append(row)
            eligible.remove(row)

    union = UnionFind(len(eligible))
    indexes_by_key: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(eligible):
        for field in (*PROVENANCE_FIELDS, "attack_instrument_id"):
            if row.get(field):
                indexes_by_key[f"{field}:{row[field]}"].append(index)
        indexes_by_key[f"near:{row['near_component_id']}"].append(index)
    for members in indexes_by_key.values():
        for member in members[1:]:
            union.union(members[0], member)

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for index, row in enumerate(eligible):
        groups[str(union.find(index))].append(row)

    totals = [sum(int(row["label"]) == label for row in eligible) for label in range(2)]
    targets = {split: [total * ratios[split] for total in totals] for split in SPLITS}
    counts = {split: [0, 0] for split in SPLITS}
    result: dict[str, list[dict[str, str]]] = {split: [] for split in SPLITS}

    ordered = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            hashlib.sha256(item[0].encode()).hexdigest(),
        ),
    )
    for _, members in ordered:
        addition = [sum(int(row["label"]) == label for row in members) for label in range(2)]
        split = min(
            SPLITS,
            key=lambda name: (
                _cost(counts, name, addition, targets),
                len(result[name]) / max(ratios[name], 1e-9),
                name,
            ),
        )
        for row in members:
            row["split"] = split
            row["status"] = "released"
        result[split].extend(members)
        counts[split] = [counts[split][index] + addition[index] for index in range(2)]
    return result, quarantine


def write_protocols(output_dir: Path, splits: dict[str, list[dict[str, str]]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        write_csv(output_dir / f"{split}-v2.csv", rows, MANIFEST_FIELDS)
