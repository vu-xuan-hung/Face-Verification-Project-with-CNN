"""Evaluate VShield open-set recognition from a gallery/probe manifest."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vshield.evaluation.io import save_yaml  # noqa: E402
from vshield.evaluation.recognition_runner import evaluate_recognition  # noqa: E402
from vshield.evaluation.seed import set_seed  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["calibration", "test"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recognition-threshold", type=float)
    parser.add_argument("--ambiguity-threshold", type=float)
    parser.add_argument("--allow-leakage", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--no-latency", action="store_true", help="Permit cached probes; latency fields may be unavailable")
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    metrics = evaluate_recognition(
        args.manifest,
        args.output,
        project_root=ROOT,
        split=args.split,
        recognition_threshold=args.recognition_threshold,
        ambiguity_threshold=args.ambiguity_threshold,
        allow_leakage=args.allow_leakage,
        force_recompute=args.force_recompute,
        measure_latency=not args.no_latency,
        save_plots=args.save_plots,
    )
    save_yaml(
        args.output / "config.yaml",
        {
            "manifest": str(args.manifest.resolve()),
            "split": args.split,
            "seed": args.seed,
            "thresholds": metrics.get("thresholds"),
            "allow_leakage": args.allow_leakage,
            "force_recompute": args.force_recompute,
            "measure_latency": not args.no_latency,
            "save_plots": args.save_plots,
        },
    )


if __name__ == "__main__":
    main()
