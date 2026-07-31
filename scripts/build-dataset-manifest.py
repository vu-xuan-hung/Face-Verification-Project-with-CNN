"""Build a deterministic image manifest; incomplete provenance is quarantined."""

from __future__ import annotations

import argparse
from pathlib import Path

from vshield.data.integrity import build_manifest, write_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = build_manifest(args.input, args.metadata)
    write_csv(args.output, rows)
    quarantine = sum(row["status"] == "quarantine" for row in rows)
    print(f"manifest={args.output} samples={len(rows)} quarantine={quarantine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
