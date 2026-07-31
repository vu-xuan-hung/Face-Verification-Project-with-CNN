"""Promote a candidate only when dataset and evaluation provenance agree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vshield.evaluation.promotion import build_model_release


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--dataset-release", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    release = build_model_release(
        args.model,
        args.candidate_manifest,
        args.dataset_release,
        args.evaluation,
    )
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite model release: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(release, indent=2), encoding="utf-8")
    print(json.dumps(release, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
