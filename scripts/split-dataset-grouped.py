"""Create deterministic group-disjoint train/val/test protocols."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vshield.data.grouped_split import grouped_split, write_protocols
from vshield.data.integrity import read_csv, write_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    splits, quarantine = grouped_split(read_csv(args.manifest), ratios)
    write_protocols(args.output_dir, splits)
    write_csv(args.quarantine, quarantine)
    output_manifest = args.output_manifest or (
        args.output_dir.parent / "manifests" / "dataset-v2-split.csv"
    )
    released = [row for rows in splits.values() for row in rows]
    write_csv(output_manifest, [*released, *quarantine])
    counts = " ".join(f"{name}={len(rows)}" for name, rows in splits.items())
    print(f"{counts} quarantine={len(quarantine)} manifest={output_manifest}")
    if not all(splits.values()):
        print("release blocked: every split must be non-empty", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
