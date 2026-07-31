"""Apply leakage, provenance and shortcut release gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from vshield.data.integrity import read_csv
from vshield.data.release_audit import audit_release, file_sha256, registry_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset-release-gates.yaml"),
    )
    parser.add_argument("--shortcut-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry-output", type=Path)
    parser.add_argument("--dataset-version", default="v2")
    parser.add_argument("--protocol", type=Path, action="append", default=[])
    args = parser.parse_args()

    shortcut = None
    if args.shortcut_report and args.shortcut_report.is_file():
        shortcut = json.loads(args.shortcut_report.read_text(encoding="utf-8"))
    gates = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = audit_release(
        read_csv(args.manifest),
        shortcut,
        expected_manifest_sha256=file_sha256(args.manifest),
        gates=gates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["passed"] and args.registry_output:
        names = {path.name for path in args.protocol}
        if names != {"train-v2.csv", "val-v2.csv", "test-v2.csv"}:
            raise ValueError("Registry requires train-v2.csv, val-v2.csv and test-v2.csv")
        registry = registry_manifest(args.protocol, report, args.dataset_version)
        args.registry_output.parent.mkdir(parents=True, exist_ok=True)
        args.registry_output.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
