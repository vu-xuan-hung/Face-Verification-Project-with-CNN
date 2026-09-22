"""Open-set recognition evaluation using production FaceNet and identity search."""

from __future__ import annotations

from pathlib import Path

from vshield.evaluation.adapter import VShieldEvaluationAdapter
from vshield.evaluation.cache import (
    embedding_fingerprint,
    load_embeddings,
    save_embeddings,
)
from vshield.evaluation.datasets import audit_manifest, read_manifest, require_no_leakage
from vshield.evaluation.images import load_image
from vshield.evaluation.io import environment_info, write_audit, write_csv, write_json
from vshield.evaluation.latency import latency_summary
from vshield.evaluation.plots import error_curve, roc_plot, score_histogram
from vshield.evaluation.recognition_metrics import open_set_metrics, verification_metrics
from vshield.evaluation.reports import metric_markdown, write_markdown


def evaluate_recognition(
    manifest: str | Path,
    output: str | Path,
    *,
    project_root: str | Path,
    split: str = "test",
    recognition_threshold: float | None = None,
    ambiguity_threshold: float | None = None,
    allow_leakage: bool = False,
    force_recompute: bool = False,
    measure_latency: bool = True,
    save_plots: bool = False,
    adapter: VShieldEvaluationAdapter | None = None,
) -> dict:
    output = Path(output)
    rows = read_manifest(manifest, "recognition")
    audit = audit_manifest(rows, "recognition")
    write_audit(output, audit)
    require_no_leakage(audit, allow_leakage)
    adapter = adapter or VShieldEvaluationAdapter(
        project_root,
        recognition_threshold=recognition_threshold,
        ambiguity_threshold=ambiguity_threshold,
    )
    fingerprint = embedding_fingerprint(
        project_root, manifest, [row.get("sha256", "") for row in audit["rows"]]
    )
    gallery_rows = [row for row in rows if row["split"].lower() == "gallery"]
    probe_rows = [row for row in rows if row["split"].lower() == split]
    gallery_vectors, gallery_errors, _ = _embeddings(
        gallery_rows, adapter, output.parent / "cache/gallery_embeddings.npz", fingerprint, force_recompute
    )
    probe_vectors, probe_errors, probe_timings = _embeddings(
        probe_rows,
        adapter,
        output.parent / "cache/probe_embeddings.npz",
        fingerprint,
        force_recompute or measure_latency,
    )
    grouped = {}
    for row in gallery_rows:
        if row["sample_id"] in gallery_vectors:
            grouped.setdefault(row["subject_id"], []).append(gallery_vectors[row["sample_id"]])
    if not grouped:
        raise ValueError("No valid gallery embeddings are available")
    gallery = adapter.build_gallery(grouped)
    predictions = []
    pair_scores = []
    errors_by_id = {row["sample_id"]: row["error"] for row in probe_errors}
    for row in probe_rows:
        predictions.append(
            _recognize_probe(
                row,
                probe_vectors.get(row["sample_id"]),
                errors_by_id,
                probe_timings.get(row["sample_id"]),
                adapter,
                gallery,
            )
        )
        if row["sample_id"] in probe_vectors:
            pair_scores.extend(_pair_scores(row, probe_vectors[row["sample_id"]], gallery, len(grouped)))
    thresholds = adapter.production_thresholds
    metrics = open_set_metrics(predictions)
    metrics["verification"] = verification_metrics(pair_scores, thresholds["recognition"])
    metrics["verification_protocol"] = (
        "Probe-to-identity minimum gallery-template distance; these are not independent "
        "all-template verification trials."
    )
    metrics["thresholds"] = thresholds
    metrics["gallery_identities"] = len(grouped)
    metrics["gallery_templates"] = sum(map(len, grouped.values()))
    metrics["gallery_errors"] = len(gallery_errors)
    metrics["latency"] = latency_summary(
        predictions, ["embedding_ms", "matching_ms", "recognition_total_ms"]
    )
    sweep = _threshold_sweep(pair_scores)
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "known_predictions.csv", [row for row in predictions if row["is_enrolled"]])
    write_csv(output / "unknown_predictions.csv", [row for row in predictions if not row["is_enrolled"]])
    write_csv(output / "genuine_impostor_scores.csv", pair_scores)
    write_csv(output / "threshold_sweep.csv", sweep)
    write_csv(output / "gallery_errors.csv", gallery_errors)
    write_json(output / "metrics.json", metrics)
    write_json(output / "cache_fingerprint.json", fingerprint)
    write_json(output / "environment.json", environment_info(project_root))
    write_markdown(output / "metrics.md", metric_markdown("VShield Open-Set Recognition Evaluation", metrics))
    if save_plots:
        _plots(output, predictions, pair_scores, sweep)
    return metrics


def _embeddings(rows, adapter, cache_path, fingerprint, force):
    cached = None if force else load_embeddings(cache_path, fingerprint)
    embeddings = cached or {}
    errors, timings = [], {}
    for row in rows:
        if row["sample_id"] in embeddings:
            continue
        try:
            image = load_image(row["_resolved_path"])
            encoded = adapter.extract_embedding(image)
            embeddings[row["sample_id"]] = encoded["embedding"]
            timings[row["sample_id"]] = encoded
        except Exception as exc:
            errors.append({"sample_id": row["sample_id"], "image_path": row["image_path"], "error": f"{type(exc).__name__}: {exc}"})
    save_embeddings(cache_path, embeddings, fingerprint)
    return embeddings, errors, timings


def _recognize_probe(row, embedding, errors, timing, adapter, gallery):
    base = {
        "sample_id": row["sample_id"],
        "image_path": row["image_path"],
        "subject_id": row["subject_id"],
        "is_enrolled": row["is_enrolled"] == "1",
        "decision": "ERROR",
        "identity": None,
        "best_identity": None,
        "best_score": None,
        "second_identity": None,
        "second_best_score": None,
        "margin": None,
        "error": errors.get(row["sample_id"], ""),
        "face_detection_ms": timing.get("face_detection_ms") if timing else None,
        "embedding_ms": timing.get("embedding_ms") if timing else None,
        "recognition_total_ms": None,
    }
    if embedding is None:
        return base
    try:
        base.update(adapter.compare_embedding(embedding, gallery))
        if base["embedding_ms"] is not None:
            base["recognition_total_ms"] = (
                base["face_detection_ms"] + base["embedding_ms"] + base["matching_ms"]
            )
    except Exception as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
    return base


def _pair_scores(row, embedding, gallery, identity_count):
    ranked = gallery.ranked_identities(embedding, k=identity_count)
    return [
        {
            "sample_id": row["sample_id"],
            "probe_identity": row["subject_id"],
            "gallery_identity": candidate.username,
            "pair_type": "genuine" if row["is_enrolled"] == "1" and candidate.username == row["subject_id"] else "impostor",
            "distance": candidate.distance,
        }
        for candidate in ranked
    ]


def _threshold_sweep(scores):
    candidates = sorted({0.0, 2.0, *[float(row["distance"]) for row in scores]})
    return [verification_metrics(scores, threshold) for threshold in candidates]


def _plots(output, predictions, scores, sweep):
    score_histogram(
        output / "distance_distribution.png",
        {kind: [row["distance"] for row in scores if row["pair_type"] == kind] for kind in ("genuine", "impostor")},
        "Normalized L2 distance (lower is better)",
        "Genuine and impostor distances",
    )
    labels = [1 if row["pair_type"] == "genuine" else 0 for row in scores]
    roc_plot(output / "roc_curve.png", labels, [row["distance"] for row in scores], higher_is_positive=False, title="Verification ROC")
    error_curve(output / "far_frr_curve.png", sweep, "threshold", "far_fmr", "frr_fnmr", "Verification errors by threshold")
    score_histogram(
        output / "ambiguity_distribution.png",
        {
            "correct known": [row["margin"] for row in predictions if row["is_enrolled"] and row.get("identity") == row["subject_id"] and row.get("margin") is not None],
            "wrong known": [row["margin"] for row in predictions if row["is_enrolled"] and row.get("identity") not in (None, row["subject_id"]) and row.get("margin") is not None],
            "unknown": [row["margin"] for row in predictions if not row["is_enrolled"] and row.get("margin") is not None],
        },
        "Runner-up distance minus best distance",
        "Ambiguity margin distribution",
    )
