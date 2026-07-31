"""Cluster exact and perceptual duplicate candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

from vshield.data.deduplication import (
    cluster_duplicates,
    write_candidates,
    write_deduplicated_manifest,
)
from vshield.data.integrity import read_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--near-threshold", type=int, default=4)
    args = parser.parse_args()

    rows = read_csv(args.manifest)
    rows, candidates = cluster_duplicates(rows, args.near_threshold)
    write_deduplicated_manifest(args.output, rows)
    write_candidates(args.candidates, candidates)
    excluded = sum(row["status"] == "duplicate_excluded" for row in rows)
    print(f"manifest={args.output} exact_excluded={excluded} near_candidates={len(candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
