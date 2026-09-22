"""Run configured VShield evaluations into one reproducible timestamped directory."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vshield.core.identity_index import (  # noqa: E402
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_MIN_MARGIN,
)
from vshield.evaluation.adapter import VShieldEvaluationAdapter  # noqa: E402
from vshield.evaluation.calibration import (  # noqa: E402
    calibrate_pad_predictions,
    calibrate_recognition_predictions,
)
from vshield.evaluation.datasets import (  # noqa: E402
    audit_manifest,
    cross_manifest_leakage,
    read_manifest,
    require_no_leakage,
)
from vshield.evaluation.e2e_runner import evaluate_end_to_end  # noqa: E402
from vshield.evaluation.io import (  # noqa: E402
    environment_info,
    load_yaml,
    save_yaml,
    timestamped_run_dir,
    write_audit,
    write_json,
)
from vshield.evaluation.pad_runner import evaluate_pad  # noqa: E402
from vshield.evaluation.recognition_runner import evaluate_recognition  # noqa: E402
from vshield.evaluation.reports import combined_report, write_markdown  # noqa: E402
from vshield.evaluation.seed import set_seed  # noqa: E402


def _threshold(value):
    return None if value in (None, "production") else float(value)


def _path(value):
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _incomplete_run_error(errors, calibration_summary):
    failures = [f"{name}: {errors[name]}" for name in sorted(errors)]
    failures.extend(
        f"{name} calibration: {value['explanation']}"
        for name, value in sorted(calibration_summary.items())
        if isinstance(value, dict) and value.get("available") is False
    )
    return f"Evaluation incomplete: {'; '.join(failures)}" if failures else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--allow-leakage", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    seed = int(config.get("evaluation", {}).get("seed", 42))
    set_seed(seed)
    output_root = _path(config.get("output", {}).get("root", "outputs/evaluation"))
    run = timestamped_run_dir(output_root)
    save_yaml(run / "config.yaml", config)
    environment = environment_info(ROOT)
    write_json(run / "environment.json", environment)
    thresholds = {
        "pad": _threshold(config.get("pad", {}).get("threshold")),
        "recognition": _threshold(config.get("recognition", {}).get("threshold")),
        "ambiguity": _threshold(config.get("recognition", {}).get("ambiguity_threshold")),
    }
    data = config.get("data", {})
    options = config.get("output", {})
    audits = {}
    for name, kind in (
        ("pad_test_manifest", "pad"),
        ("pad_calibration_manifest", "pad"),
        ("recognition_manifest", "recognition"),
        ("e2e_manifest", "e2e"),
    ):
        path = _path(data.get(name))
        if path and path.is_file():
            audits[name] = audit_manifest(read_manifest(path, kind), kind)
    cross = cross_manifest_leakage(
        audits.get("pad_calibration_manifest", {}),
        audits.get("pad_test_manifest", {}),
        "pad_calibration",
        "pad_test",
    )
    all_issues = [
        {"manifest": name, **issue}
        for name, audit in audits.items()
        for issue in audit.get("issues", [])
    ]
    combined_audit = {
        "manifests": audits,
        "issues": [*all_issues, *cross],
        "has_severe_leakage": bool(cross)
        or any(audit.get("has_severe_leakage") for audit in audits.values()),
    }
    write_audit(run, combined_audit)
    write_audit(output_root, combined_audit)
    require_no_leakage(combined_audit, args.allow_leakage)
    summary, errors = {}, {}
    calibration = config.get("calibration", {})
    calibration_summary = {}
    current_distance = DEFAULT_DISTANCE_THRESHOLD if thresholds["recognition"] is None else thresholds["recognition"]
    current_ambiguity = DEFAULT_MIN_MARGIN if thresholds["ambiguity"] is None else thresholds["ambiguity"]
    try:
        pad_adapter = VShieldEvaluationAdapter(ROOT, pad_threshold=thresholds["pad"])
        evaluate_pad(
            _path(data.get("pad_calibration_manifest")),
            run / "pad/calibration-scores",
            project_root=ROOT,
            adapter=pad_adapter,
            allow_leakage=args.allow_leakage,
            save_plots=False,
        )
        calibration_summary["pad"] = calibrate_pad_predictions(
            run / "pad/calibration-scores/predictions.csv",
            run / "pad",
            current_threshold=pad_adapter.production_thresholds["pad"],
            objective=calibration.get("pad_objective", "min-acer"),
            target=calibration.get("pad_target"),
        )
    except Exception as exc:
        calibration_summary["pad"] = {"available": False, "explanation": f"{type(exc).__name__}: {exc}"}
    try:
        evaluate_recognition(
            _path(data.get("recognition_manifest")),
            run / "recognition/calibration-scores",
            project_root=ROOT,
            split="calibration",
            recognition_threshold=thresholds["recognition"],
            ambiguity_threshold=thresholds["ambiguity"],
            allow_leakage=args.allow_leakage,
            force_recompute=args.force_recompute,
            save_plots=False,
        )
        calibration_summary["recognition"] = calibrate_recognition_predictions(
            run / "recognition/calibration-scores/predictions.csv",
            run / "recognition",
            current_distance=current_distance,
            current_ambiguity=current_ambiguity,
            objective=calibration.get("recognition_objective", "min-open-set-error"),
            target=calibration.get("recognition_target"),
        )
    except Exception as exc:
        calibration_summary["recognition"] = {"available": False, "explanation": f"{type(exc).__name__}: {exc}"}
    summary["calibration"] = calibration_summary
    calls = [
        ("pad", lambda: evaluate_pad(
            _path(data.get("pad_test_manifest")), run / "pad", project_root=ROOT,
            threshold=thresholds["pad"], allow_leakage=args.allow_leakage,
            save_plots=bool(options.get("save_plots", True)),
            warmup_runs=int(config.get("performance", {}).get("warmup_runs", 1)),
        )),
        ("recognition", lambda: evaluate_recognition(
            _path(data.get("recognition_manifest")), run / "recognition", project_root=ROOT,
            recognition_threshold=thresholds["recognition"], ambiguity_threshold=thresholds["ambiguity"],
            allow_leakage=args.allow_leakage, force_recompute=args.force_recompute,
            measure_latency=bool(config.get("performance", {}).get("measure_latency", True)),
            save_plots=bool(options.get("save_plots", True)),
        )),
        ("e2e", lambda: evaluate_end_to_end(
            _path(data.get("e2e_manifest")), _path(data.get("recognition_manifest")), run / "e2e",
            project_root=ROOT, pad_threshold=thresholds["pad"],
            recognition_threshold=thresholds["recognition"], ambiguity_threshold=thresholds["ambiguity"],
            allow_leakage=args.allow_leakage, save_plots=bool(options.get("save_plots", True)),
            warmup_runs=int(config.get("performance", {}).get("warmup_runs", 1)),
        )),
    ]
    for name, call in calls:
        try:
            summary[name] = call()
        except Exception as exc:
            summary[name] = None
            errors[name] = f"{type(exc).__name__}: {exc}"
    summary["latency"] = {
        name: metrics.get("latency", {})
        for name, metrics in summary.items()
        if isinstance(metrics, dict) and metrics.get("latency")
    }
    summary["unavailable"] = errors
    write_json(run / "summary.json", summary)
    calibration_errors = [value["explanation"] for value in calibration_summary.values()
                          if isinstance(value, dict) and value.get("available") is False]
    limitations = list(dict.fromkeys([*[errors[key] for key in sorted(errors)], *calibration_errors]))
    pad_result = summary.get("pad")
    if isinstance(pad_result, dict) and pad_result.get("n_inference_errors"):
        limitations.append(
            f"PAD scored-only rates exclude {pad_result['n_inference_errors']} failed attempts; "
            "consult the full predictions and do not extrapolate to every manifest sample."
        )
    recognition_result = summary.get("recognition")
    if isinstance(recognition_result, dict) and (
        recognition_result.get("gallery_errors") or recognition_result.get("inference_errors")
    ):
        limitations.append(
            "Recognition has gallery/probe inference failures; missing gallery identities and "
            "all failed probes require separate coverage analysis."
        )
    e2e_result = summary.get("e2e")
    if isinstance(e2e_result, dict):
        if not e2e_result.get("n_legitimate"):
            limitations.append("No REAL+KNOWN E2E attempts: legitimate access success is unavailable.")
        if not e2e_result.get("n_real_unknown"):
            limitations.append("No REAL+UNKNOWN E2E attempts: bona-fide unknown blocking is unavailable.")
        if e2e_result.get("inference_errors"):
            limitations.append(
                "Some E2E attempts stop before PAD/recognition due to face or inference errors; "
                "stage counts must be reported with spoof blocking."
            )
    if isinstance(recognition_result, dict) and recognition_result.get("verification_protocol"):
        limitations.append(recognition_result["verification_protocol"])
    if audits.get("e2e_manifest", {}).get("class_counts", {}).get("SPOOF") and not audits.get(
        "e2e_manifest", {}
    ).get("class_counts", {}).get("REAL"):
        limitations.append("Spoof-only E2E results cannot establish full deployed access-control performance.")
    limitations = list(dict.fromkeys(limitations))
    report = combined_report(
        summary,
        {"run_directory": run, "seed": seed, "git_commit": environment.get("git_commit")},
        limitations or ["No additional limitations recorded."],
    )
    write_markdown(run / "summary.md", report)
    write_json(output_root / "summary.json", summary)
    write_markdown(output_root / "VSHIELD_EVALUATION_REPORT.md", report)
    print(run)
    if failure := _incomplete_run_error(errors, calibration_summary):
        raise RuntimeError(failure)


if __name__ == "__main__":
    main()
