"""Run non-visual baselines and block shortcut-contaminated releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

from vshield.data.integrity import read_csv
from vshield.evaluation.shortcut import (
    FEATURE_SETS,
    category_label_purity,
    check_gates,
    run_pixel_probe,
    run_probe,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/shortcut-gates.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Audit a quarantined historical protocol; never permits release.",
    )
    args = parser.parse_args()

    all_rows = read_csv(args.manifest)
    rows = (
        all_rows
        if args.allow_invalid
        else [row for row in all_rows if row.get("status") == "released"]
    )
    train = [row for row in rows if row.get("split") == "train"]
    test = [row for row in rows if row.get("split") == "test"]
    if not train or not test:
        print("shortcut audit requires non-empty released train and test", file=sys.stderr)
        return 2

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gate = config["metadata_probe"]
    results = [
        run_probe(train, test, name, int(gate["bootstrap_iterations"])) for name in FEATURE_SETS
    ]
    failures = check_gates(
        results[0],
        float(gate["max_auc"]),
        float(gate["max_auc_ci_high"]),
        float(gate["max_accuracy_above_majority"]),
    )
    purity = [category_label_purity(rows, field) for field in config["category_probe"]["fields"]]
    max_purity = float(config["category_probe"]["max_weighted_purity"])
    failures.extend(
        f"{result['field'].upper()}_PURITY_ABOVE_LIMIT"
        for result in purity
        if result["coverage"] and result["weighted_purity"] > max_purity
    )
    pixel_results = []
    if args.dataset_root:
        pixel_results = [
            run_pixel_probe(
                train,
                test,
                args.dataset_root,
                mode,
                int(gate["bootstrap_iterations"]),
            )
            for mode in (
                "low_resolution",
                "color_histogram",
                "background_only",
                "center_only",
            )
        ]
        for result in pixel_results[:3]:
            failures.extend(
                f"{result['feature_set'].upper()}_{failure}"
                for failure in check_gates(
                    result,
                    float(gate["max_auc"]),
                    float(gate["max_auc_ci_high"]),
                    float(gate["max_accuracy_above_majority"]),
                )
            )
    if args.allow_invalid:
        failures.append("HISTORICAL_INVALID_PROTOCOL")
    report = {
        "passed": not failures,
        "protocol_status": "invalid_historical" if args.allow_invalid else "release_candidate",
        "manifest_sha256": _sha256(args.manifest),
        "failures": failures,
        "probes": results,
        "pixel_probes": pixel_results,
        "purity": purity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
