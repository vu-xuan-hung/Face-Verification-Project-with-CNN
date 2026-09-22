"""Calibrate PAD on a dedicated calibration manifest without changing production."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vshield.evaluation.adapter import VShieldEvaluationAdapter  # noqa: E402
from vshield.evaluation.calibration import calibrate_pad_predictions  # noqa: E402
from vshield.evaluation.pad_runner import evaluate_pad  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objective", choices=["min-acer", "target-apcer", "target-bpcer"], default="min-acer")
    parser.add_argument("--target-apcer", type=float)
    parser.add_argument("--target-bpcer", type=float)
    parser.add_argument("--allow-leakage", action="store_true")
    args = parser.parse_args()
    adapter = VShieldEvaluationAdapter(ROOT)
    evaluate_pad(args.manifest, args.output, project_root=ROOT, adapter=adapter, allow_leakage=args.allow_leakage)
    target = args.target_apcer if args.objective == "target-apcer" else args.target_bpcer
    calibrate_pad_predictions(
        args.output / "predictions.csv",
        args.output,
        current_threshold=adapter.production_thresholds["pad"],
        objective=args.objective,
        target=target,
    )


if __name__ == "__main__":
    main()
