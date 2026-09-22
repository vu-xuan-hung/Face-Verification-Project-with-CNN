"""Manifest-driven evaluation of the deployed VShield PAD path."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from vshield.evaluation.adapter import VShieldEvaluationAdapter
from vshield.evaluation.datasets import audit_manifest, read_manifest, require_no_leakage
from vshield.evaluation.images import load_image
from vshield.evaluation.io import environment_info, write_audit, write_csv, write_json
from vshield.evaluation.latency import latency_summary
from vshield.evaluation.pad_metrics import evaluate_scores, per_attack_apcer, threshold_sweep
from vshield.evaluation.plots import confusion, error_curve, roc_plot, score_histogram
from vshield.evaluation.reports import metric_markdown, write_markdown


def evaluate_pad(
    manifest: str | Path,
    output: str | Path,
    *,
    project_root: str | Path,
    threshold: float | None = None,
    allow_leakage: bool = False,
    save_plots: bool = False,
    warmup_runs: int = 1,
    adapter: VShieldEvaluationAdapter | None = None,
) -> dict:
    output = Path(output)
    rows = read_manifest(manifest, "pad")
    audit = audit_manifest(rows, "pad")
    write_audit(output, audit)
    require_no_leakage(audit, allow_leakage)
    adapter = adapter or VShieldEvaluationAdapter(project_root, pad_threshold=threshold)
    _warmup(adapter, rows, warmup_runs)
    predictions = [_predict(row, adapter) for row in rows]
    scored = [row for row in predictions if row["pad_score"] is not None and not row["error"]]
    effective_threshold = adapter.production_thresholds["pad"]
    labels = np.asarray([1 if row["ground_truth"] == "REAL" else 0 for row in scored])
    scores = np.asarray([float(row["pad_score"]) for row in scored])
    metrics = evaluate_scores(labels, scores, effective_threshold) if len(scored) else _empty_metrics()
    metrics.update(
        n_samples=len(scored),
        n_manifest_samples=len(rows),
        n_inference_errors=len(rows) - len(scored),
        apcer_by_attack_type=per_attack_apcer(scored, effective_threshold),
        latency=latency_summary(predictions, ["face_detection_ms", "pad_total_ms", "inference_time_ms"]),
        score_meaning="MiniFASNet real-class softmax probability",
        threshold_comparator="score >= threshold means REAL",
    )
    if not scored:
        metrics["unavailable_explanation"] = "No successfully scored PAD samples were present."
    sweep = threshold_sweep(labels, scores) if len(scored) else []
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "threshold_sweep.csv", sweep)
    write_json(output / "metrics.json", metrics)
    write_json(output / "environment.json", environment_info(project_root))
    write_markdown(
        output / "metrics.md",
        metric_markdown(
            "VShield PAD Evaluation",
            metrics,
            [
                "APCER = attacks incorrectly classified as bona fide / scored attacks.",
                "BPCER = bona-fide presentations incorrectly classified as attacks / scored bona fide.",
                "ACER = (APCER + BPCER) / 2 when both classes exist.",
            ],
        ),
    )
    if save_plots and scored:
        _plots(output, predictions, labels, scores, metrics, sweep)
    return metrics


def _predict(row: dict, adapter: VShieldEvaluationAdapter) -> dict:
    result = {
        "sample_id": row["sample_id"],
        "image_path": row["image_path"],
        "ground_truth": row["label"].upper(),
        "attack_type": row.get("attack_type", ""),
        "pad_score": None,
        "predicted_label": "ERROR",
        "correct": False,
        "inference_time_ms": None,
        "face_detection_ms": None,
        "pad_total_ms": None,
        "raw_output": None,
        "error": "",
    }
    started = perf_counter()
    try:
        image = load_image(row["_resolved_path"])
        prediction = adapter.pad_predict(image)
        result.update(
            pad_score=prediction["real_score"],
            predicted_label=prediction["decision"],
            face_detection_ms=prediction["face_detection_ms"],
            pad_total_ms=prediction["pad_total_ms"],
            raw_output=prediction["raw_output"],
            error="" if prediction["decision"] != "ERROR" else prediction["reason"],
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["inference_time_ms"] = (perf_counter() - started) * 1000
    result["correct"] = result["predicted_label"] == result["ground_truth"]
    return result


def _warmup(adapter, rows, runs):
    for row in rows:
        try:
            adapter.warmup(load_image(row["_resolved_path"]), runs=runs)
            return
        except Exception:
            continue


def _plots(output, rows, labels, scores, metrics, sweep):
    confusion(output / "confusion_matrix.png", metrics["confusion_matrix"], ["SPOOF", "REAL"], "PAD confusion matrix")
    score_histogram(
        output / "score_distribution.png",
        {label: [row["pad_score"] for row in rows if row["ground_truth"] == label and row["pad_score"] is not None] for label in ("REAL", "SPOOF")},
        "Real-class softmax score",
        "PAD score distribution",
    )
    score_histogram(
        output / "score_by_attack_type.png",
        {
            row.get("attack_type") or "other": [
                item["pad_score"]
                for item in rows
                if (item.get("attack_type") or "other") == (row.get("attack_type") or "other")
                and item["pad_score"] is not None
            ]
            for row in rows
            if row["ground_truth"] == "SPOOF"
        },
        "Real-class softmax score",
        "PAD attack score distribution",
    )
    roc_plot(output / "roc_curve.png", labels, scores, title="PAD ROC curve")
    error_curve(output / "threshold_errors.png", sweep, "threshold", "apcer", "bpcer", "PAD error rates by threshold")


def _empty_metrics():
    return dict.fromkeys(("sample_count", "accuracy", "precision", "recall", "f1", "apcer", "bpcer", "acer", "roc_auc", "eer"))
