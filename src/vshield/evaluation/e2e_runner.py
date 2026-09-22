"""End-to-end evaluation through production face, PAD, and recognition orchestration."""

from __future__ import annotations

from pathlib import Path

from vshield.evaluation.adapter import VShieldEvaluationAdapter
from vshield.evaluation.datasets import (
    audit_manifest,
    cross_manifest_leakage,
    read_manifest,
    require_no_leakage,
)
from vshield.evaluation.e2e_metrics import evaluate_e2e, failure_reason, spoof_failure_stage
from vshield.evaluation.images import load_image
from vshield.evaluation.io import environment_info, write_audit, write_csv, write_json
from vshield.evaluation.latency import latency_summary
from vshield.evaluation.plots import category_counts, confusion
from vshield.evaluation.reports import metric_markdown, write_markdown


def evaluate_end_to_end(
    manifest: str | Path,
    gallery_manifest: str | Path,
    output: str | Path,
    *,
    project_root: str | Path,
    pad_threshold: float | None = None,
    recognition_threshold: float | None = None,
    ambiguity_threshold: float | None = None,
    allow_leakage: bool = False,
    save_plots: bool = False,
    warmup_runs: int = 1,
    adapter: VShieldEvaluationAdapter | None = None,
) -> dict:
    output = Path(output)
    rows = read_manifest(manifest, "e2e")
    gallery_rows_all = read_manifest(gallery_manifest, "recognition")
    gallery_rows = [row for row in gallery_rows_all if row["split"].lower() == "gallery"]
    audit = audit_manifest(rows, "e2e")
    gallery_audit = audit_manifest(gallery_rows_all, "recognition")
    require_no_leakage(gallery_audit, allow_leakage)
    gallery_only_audit = {
        "rows": [row for row in gallery_audit["rows"] if row["split"].lower() == "gallery"]
    }
    cross = cross_manifest_leakage(gallery_only_audit, audit, "gallery", "e2e_test")
    audit["issues"].extend(cross)
    audit["has_severe_leakage"] = audit["has_severe_leakage"] or bool(cross)
    audit["gallery_manifest"] = gallery_audit
    write_audit(output, audit)
    require_no_leakage(audit, allow_leakage)
    adapter = adapter or VShieldEvaluationAdapter(
        project_root,
        pad_threshold=pad_threshold,
        recognition_threshold=recognition_threshold,
        ambiguity_threshold=ambiguity_threshold,
    )
    grouped, gallery_errors = _gallery(gallery_rows, adapter)
    if gallery_errors:
        write_csv(output / "gallery_errors.csv", gallery_errors)
    if not grouped:
        raise ValueError("No valid gallery embeddings are available")
    gallery = adapter.build_gallery(grouped)
    _warmup(rows, adapter, gallery, warmup_runs)
    predictions = [_predict(row, adapter, gallery) for row in rows]
    for row in predictions:
        row["failure_reason"] = failure_reason(row)
        row["spoof_failure_stage"] = spoof_failure_stage(row)
    failures = [row for row in predictions if row["failure_reason"]]
    metrics = evaluate_e2e(predictions)
    metrics["thresholds"] = adapter.production_thresholds
    metrics["latency"] = latency_summary(predictions, ["total_biometric_pipeline_ms"])
    metrics["spoof_failure_stages"] = _counts(predictions, "spoof_failure_stage")
    metrics["failure_reasons"] = _counts(failures, "failure_reason")
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "failures.csv", failures)
    write_json(output / "metrics.json", metrics)
    write_json(output / "environment.json", environment_info(project_root))
    write_markdown(output / "metrics.md", metric_markdown("VShield End-to-End Evaluation", metrics))
    if save_plots:
        category_counts(output / "failure_reason_counts.png", metrics["failure_reasons"], "End-to-end failure reasons")
        matrix = [
            [sum(row["expected_access"] == "DENY" and row["final_access"] == "DENY" for row in predictions), sum(row["expected_access"] == "DENY" and row["final_access"] == "ALLOW" for row in predictions)],
            [sum(row["expected_access"] == "ALLOW" and row["final_access"] == "DENY" for row in predictions), sum(row["expected_access"] == "ALLOW" and row["final_access"] == "ALLOW" for row in predictions)],
        ]
        confusion(output / "final_outcome_confusion_matrix.png", matrix, ["DENY", "ALLOW"], "Final access outcome")
    return metrics


def _gallery(rows, adapter):
    grouped, errors = {}, []
    for row in rows:
        try:
            encoded = adapter.extract_embedding(load_image(row["_resolved_path"]))
            grouped.setdefault(row["subject_id"], []).append(encoded["embedding"])
        except Exception as exc:
            errors.append({"sample_id": row["sample_id"], "image_path": row["image_path"], "error": f"{type(exc).__name__}: {exc}"})
    return grouped, errors


def _warmup(rows, adapter, gallery, runs):
    for row in rows:
        try:
            image = load_image(row["_resolved_path"])
            for _ in range(max(0, runs)):
                adapter.full_biometric_decision(image, gallery)
            return
        except Exception:
            continue


def _predict(row, adapter, gallery):
    result = {
        "sample_id": row["sample_id"],
        "path": row["image_path"],
        "presentation": row["presentation"].upper(),
        "subject_id": row["subject_id"],
        "expected_identity_state": row["expected_identity_state"].upper(),
        "expected_access": row["expected_access"].upper(),
        "attack_type": row.get("attack_type", ""),
        "pad_score": None,
        "pad_decision": "ERROR",
        "recognition_decision": None,
        "predicted_identity": None,
        "best_distance": None,
        "second_best_distance": None,
        "margin": None,
        "final_access": "DENY",
        "failure_code": "INVALID_IMAGE",
        "total_biometric_pipeline_ms": None,
        "error": "",
    }
    try:
        outcome = adapter.full_biometric_decision(load_image(row["_resolved_path"]), gallery)
        pad = outcome.get("pad") or {}
        recognition = outcome.get("recognition") or {}
        best = recognition.get("recognition_distance")
        second = recognition.get("runner_up_distance")
        pad_status = pad.get("status")
        pad_decision = "REAL" if pad_status == "REAL" else "SPOOF" if pad_status in {"FAKE", "UNCERTAIN"} else "ERROR"
        result.update(
            pad_score=pad.get("score"),
            pad_decision=pad_decision,
            recognition_decision="KNOWN" if recognition.get("decision") == "MATCH" else recognition.get("decision"),
            predicted_identity=outcome.get("identity"),
            best_identity=recognition.get("best_identity"),
            second_identity=recognition.get("runner_up_identity"),
            best_distance=best,
            second_best_distance=second,
            margin=None if best is None or second is None else second - best,
            final_access=outcome["final_access"],
            failure_code=outcome.get("failure_code"),
            total_biometric_pipeline_ms=outcome["total_biometric_pipeline_ms"],
        )
        if outcome["status"] in {"unavailable", "invalid_face"}:
            result["error"] = outcome.get("failure_code") or outcome["status"]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _counts(rows, field):
    return {
        value: sum(row.get(field) == value for row in rows)
        for value in sorted({row.get(field) for row in rows if row.get(field)})
    }
