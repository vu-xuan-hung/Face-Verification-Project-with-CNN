"""Acquire official CelebA-Spoof evidence and probe archive within a strict budget."""

import argparse
import json
from pathlib import Path

from vshield.data.external_download import acquire_source_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New private acquisition directory")
    parser.add_argument("--max-bytes", type=int, default=200_000_000)
    args = parser.parse_args()
    try:
        report = acquire_source_evidence(args.output, args.max_bytes)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Acquisition blocked: {exc}\n")
    print(json.dumps(report, indent=2))
    return 2 if report["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
