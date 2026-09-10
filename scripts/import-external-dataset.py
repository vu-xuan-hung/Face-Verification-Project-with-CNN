"""Import legally acquired CelebA-Spoof into isolated, audited candidate data."""

import argparse
import json
from pathlib import Path

from vshield.data.external_import import import_celeba


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path)
    parser.add_argument("--val-annotations", type=Path)
    parser.add_argument("--test-annotations", type=Path)
    parser.add_argument("--metadata", type=Path, help="Evidenced source_path provenance CSV")
    parser.add_argument("--output", type=Path, required=True, help="Fresh version directory; must not exist")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--train-ratio", type=float, default=.7)
    parser.add_argument("--val-ratio", type=float, default=.15)
    parser.add_argument("--test-ratio", type=float, default=.15)
    args = parser.parse_args()
    annotations = {split: getattr(args, f"{split}_annotations") for split in ("train", "val", "test")
                   if getattr(args, f"{split}_annotations")}
    try:
        report = import_celeba(args.source_root, annotations, args.output, args.metadata, args.limit,
                              {split: getattr(args, f"{split}_ratio") for split in ("train", "val", "test")})
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Import blocked: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0  # Import success is not release success; report always states candidate_only.


if __name__ == "__main__":
    raise SystemExit(main())
