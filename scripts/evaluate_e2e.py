"""Evaluate the complete PAD-gated VShield biometric decision."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vshield.evaluation.e2e_runner import evaluate_end_to_end  # noqa: E402
from vshield.evaluation.io import save_yaml  # noqa: E402
from vshield.evaluation.seed import set_seed  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gallery-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pad-threshold", type=float)
    parser.add_argument("--recognition-threshold", type=float)
    parser.add_argument("--ambiguity-threshold", type=float)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--allow-leakage", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    metrics = evaluate_end_to_end(
        args.manifest,
        args.gallery_manifest,
        args.output,
        project_root=ROOT,
        pad_threshold=args.pad_threshold,
        recognition_threshold=args.recognition_threshold,
        ambiguity_threshold=args.ambiguity_threshold,
        warmup_runs=args.warmup_runs,
        allow_leakage=args.allow_leakage,
        save_plots=args.save_plots,
    )
    save_yaml(
        args.output / "config.yaml",
        {
            "manifest": str(args.manifest.resolve()),
            "gallery_manifest": str(args.gallery_manifest.resolve()),
            "seed": args.seed,
            "thresholds": metrics.get("thresholds"),
            "warmup_runs": args.warmup_runs,
            "allow_leakage": args.allow_leakage,
            "save_plots": args.save_plots,
        },
    )


if __name__ == "__main__":
    main()
