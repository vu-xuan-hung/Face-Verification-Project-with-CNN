"""Grid-calibrate recognition thresholds on the calibration split only."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vshield.core.identity_index import DEFAULT_DISTANCE_THRESHOLD, DEFAULT_MIN_MARGIN  # noqa: E402
from vshield.evaluation.calibration import calibrate_recognition_predictions  # noqa: E402
from vshield.evaluation.recognition_runner import evaluate_recognition  # noqa: E402


def _grid(value):
    return [float(item) for item in value.split(",")] if value else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distance-grid", help="Comma-separated normalized-L2 thresholds")
    parser.add_argument("--ambiguity-grid", help="Comma-separated runner-up margins")
    parser.add_argument("--objective", choices=["min-open-set-error", "target-unknown-far"], required=True)
    parser.add_argument("--target", type=float)
    parser.add_argument("--allow-leakage", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()
    evaluate_recognition(
        args.manifest,
        args.output,
        project_root=ROOT,
        split="calibration",
        allow_leakage=args.allow_leakage,
        force_recompute=args.force_recompute,
    )
    calibrate_recognition_predictions(
        args.output / "predictions.csv",
        args.output,
        current_distance=DEFAULT_DISTANCE_THRESHOLD,
        current_ambiguity=DEFAULT_MIN_MARGIN,
        distance_grid=_grid(args.distance_grid),
        ambiguity_grid=_grid(args.ambiguity_grid),
        objective=args.objective,
        target=args.target,
    )


if __name__ == "__main__":
    main()
