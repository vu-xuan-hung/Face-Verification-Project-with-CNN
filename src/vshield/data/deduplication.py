"""Exact and conservative perceptual duplicate clustering."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from vshield.data.integrity import MANIFEST_FIELDS, write_csv


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _component_ids(groups: dict[int, list[int]], prefix: str) -> dict[int, str]:
    result: dict[int, str] = {}
    for members in groups.values():
        token = ",".join(map(str, sorted(members)))
        component_id = f"{prefix}-{hashlib.sha256(token.encode()).hexdigest()[:12]}"
        for member in members:
            result[member] = component_id
    return result


def cluster_duplicates(
    rows: list[dict[str, str]], near_threshold: int = 4
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    exact = UnionFind(len(rows))
    by_hash: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_hash[row["sha256"]].append(index)
    for members in by_hash.values():
        for member in members[1:]:
            exact.union(members[0], member)

    exact_groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        exact_groups[exact.find(index)].append(index)
    exact_ids = _component_ids(exact_groups, "exact")

    near = UnionFind(len(rows))
    candidates: list[dict[str, object]] = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if rows[left]["sha256"] == rows[right]["sha256"]:
                near.union(left, right)
                continue
            distance = hamming_hex(rows[left]["dhash"], rows[right]["dhash"])
            if distance <= near_threshold:
                near.union(left, right)
                candidates.append(
                    {
                        "left_sample_id": rows[left]["sample_id"],
                        "right_sample_id": rows[right]["sample_id"],
                        "hamming_distance": distance,
                        "label_conflict": rows[left]["label"] != rows[right]["label"],
                        "review_status": "pending",
                    }
                )

    near_groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        near_groups[near.find(index)].append(index)
    near_ids = _component_ids(near_groups, "near")

    for index, row in enumerate(rows):
        row["exact_component_id"] = exact_ids[index]
        row["near_component_id"] = near_ids[index]

    for members in exact_groups.values():
        if len(members) < 2:
            continue
        labels = {rows[index]["label"] for index in members}
        if len(labels) > 1:
            for index in members:
                rows[index]["status"] = "quarantine"
                rows[index]["reason_code"] = "CONFLICTING_LABEL_EXACT_DUPLICATE"
            continue
        canonical = min(members, key=lambda index: rows[index]["relative_path"])
        for index in members:
            if index != canonical and rows[index]["status"] != "quarantine":
                rows[index]["status"] = "duplicate_excluded"
                rows[index]["reason_code"] = "EXACT_DUPLICATE"

    conflict_components = {
        near.find(index)
        for index, row in enumerate(rows)
        if any(
            pair["label_conflict"]
            and row["sample_id"] in {pair["left_sample_id"], pair["right_sample_id"]}
            for pair in candidates
        )
    }
    for index, row in enumerate(rows):
        if near.find(index) in conflict_components:
            row["status"] = "quarantine"
            row["reason_code"] = "NEAR_DUPLICATE_LABEL_CONFLICT"
    return rows, candidates


def write_candidates(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "left_sample_id",
        "right_sample_id",
        "hamming_distance",
        "label_conflict",
        "review_status",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_deduplicated_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    write_csv(path, rows, MANIFEST_FIELDS)
