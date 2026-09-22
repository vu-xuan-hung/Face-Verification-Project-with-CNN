"""Evaluate VShield PAD from a manifest using the production pipeline."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vshield.evaluation.io import save_yaml  # noqa: E402
from vshield.evaluation.pad_runner import evaluate_pad  # noqa: E402
from vshield.evaluation.seed import set_seed  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pad-threshold", type=float)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--allow-leakage", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    metrics = evaluate_pad(
        args.manifest,
        args.output,
        project_root=ROOT,
        threshold=args.pad_threshold,
        warmup_runs=args.warmup_runs,
        allow_leakage=args.allow_leakage,
        save_plots=args.save_plots,
    )
    save_yaml(
        args.output / "config.yaml",
        {
            "manifest": str(args.manifest.resolve()),
            "seed": args.seed,
            "pad_threshold": metrics.get("threshold"),
            "warmup_runs": args.warmup_runs,
            "allow_leakage": args.allow_leakage,
            "save_plots": args.save_plots,
        },
    )


if __name__ == "__main__":
    main()
