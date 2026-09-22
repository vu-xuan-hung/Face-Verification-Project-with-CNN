"""Calibration-only threshold selection; never writes production configuration."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from vshield.evaluation.io import write_csv, write_json
from vshield.evaluation.pad_metrics import evaluate_scores, select_threshold, threshold_sweep
from vshield.evaluation.recognition_metrics import apply_open_set_thresholds, open_set_metrics


def calibrate_pad_predictions(
    predictions_path: str | Path,
    output: str | Path,
    *,
    current_threshold: float,
    objective: str = "min-acer",
    target: float | None = None,
) -> dict:
    all_rows = _read_csv(predictions_path)
    rows = [
        row
        for row in all_rows
        if row.get("pad_score") not in (None, "") and not row.get("error")
    ]
    labels = np.asarray([1 if row["ground_truth"] == "REAL" else 0 for row in rows])
    scores = np.asarray([float(row["pad_score"]) for row in rows])
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("PAD calibration requires scored REAL and SPOOF samples")
    if objective == "target-apcer":
        if target is None:
            raise ValueError("target-apcer requires a target")
        threshold, _ = select_threshold(labels, scores, max_apcer=target)
    elif objective == "target-bpcer":
        if target is None:
            raise ValueError("target-bpcer requires a target")
        threshold, _ = select_threshold(labels, scores, max_bpcer=target)
    elif objective == "min-acer":
        threshold, _ = select_threshold(labels, scores)
    else:
        raise ValueError(f"Unknown PAD calibration objective: {objective}")
    sweep = threshold_sweep(labels, scores)
    recommendation = {
        "current_threshold": current_threshold,
        "recommended_threshold": threshold,
        "selection_method": objective,
        "target": target,
        "calibration_metrics": evaluate_scores(labels, scores, threshold),
        "n_manifest_samples": len(all_rows),
        "n_inference_errors": len(all_rows) - len(rows),
        "warning": "Error rows were retained but excluded from scored PAD rates; check coverage. Validate once on untouched test data; production configuration was not changed.",
    }
    output = Path(output)
    write_csv(output / "threshold_sweep.csv", sweep)
    write_json(output / "pad_threshold_recommendation.json", recommendation)
    return recommendation


def calibrate_recognition_predictions(
    predictions_path: str | Path,
    output: str | Path,
    *,
    current_distance: float,
    current_ambiguity: float,
    distance_grid: list[float] | None = None,
    ambiguity_grid: list[float] | None = None,
    objective: str,
    target: float | None = None,
) -> dict:
    rows = _coerce_recognition(_read_csv(predictions_path))
    valid = [row for row in rows if row.get("best_score") is not None and not row.get("error")]
    if not any(row["is_enrolled"] for row in rows) or not any(not row["is_enrolled"] for row in rows):
        raise ValueError("Recognition calibration requires known and unknown probes")
    if not valid:
        raise ValueError("Recognition calibration requires at least one scored probe")
    observed_distances = [float(row["best_score"]) for row in valid]
    observed_margins = [float(row["margin"]) for row in valid if row.get("margin") is not None]
    distances = distance_grid or sorted(
        {max(0.0, np.nextafter(min(observed_distances), 0.0)), *observed_distances}
    )
    margins = ambiguity_grid or sorted(
        {0.0, *observed_margins, min(2.0, np.nextafter(max(observed_margins, default=0.0), 2.0))}
    )
    grid = []
    for distance in distances:
        for ambiguity in margins:
            metrics = open_set_metrics(apply_open_set_thresholds(rows, distance, ambiguity))
            grid.append({"distance_threshold": distance, "ambiguity_threshold": ambiguity, **metrics})
    best = _select_recognition(grid, objective, target)
    recommendation = {
        "current_distance_threshold": current_distance,
        "current_ambiguity_threshold": current_ambiguity,
        "recommended_distance_threshold": best["distance_threshold"],
        "recommended_ambiguity_threshold": best["ambiguity_threshold"],
        "selection_method": objective,
        "target": target,
        "calibration_metrics": best,
        "warning": "All calibration probes, including inference errors, remain in metric denominators. Validate once on untouched test data; production configuration was not changed.",
    }
    output = Path(output)
    write_csv(output / "calibration_grid.csv", grid)
    write_json(output / "threshold_recommendation.json", recommendation)
    return recommendation


def _select_recognition(grid, objective, target):
    if objective == "min-open-set-error":
        return min(
            grid,
            key=lambda row: (
                (1 - row["known_identification_accuracy"] + row["unknown_false_acceptance_rate"]) / 2,
                row["unknown_false_acceptance_rate"],
                row["wrong_identity_rate"],
            ),
        )
    if objective == "target-unknown-far":
        if target is None:
            raise ValueError("target-unknown-far requires --target")
        feasible = [row for row in grid if row["unknown_false_acceptance_rate"] <= target]
        if not feasible:
            raise ValueError("Calibration cannot satisfy target unknown false acceptance rate")
        return max(feasible, key=lambda row: (row["known_identification_accuracy"], -row["wrong_identity_rate"]))
    raise ValueError(f"Unknown recognition calibration objective: {objective}")


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _coerce_recognition(rows):
    for row in rows:
        row["is_enrolled"] = str(row["is_enrolled"]).lower() in {"1", "true"}
        for field in ("best_score", "second_best_score", "margin"):
            row[field] = float(row[field]) if row.get(field) not in (None, "") else None
    return rows
