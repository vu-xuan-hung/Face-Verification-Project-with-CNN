"""Select PAD threshold on validation and evaluate the locked test scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vshield.data.integrity import read_csv
from vshield.evaluation.pad_metrics import (
    bootstrap_metric_ci,
    evaluate_scores,
    select_threshold,
)


def _arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([int(row["label"]) for row in rows])
    scores = np.asarray([float(row["score"]) for row in rows])
    if set(labels) != {0, 1}:
        raise ValueError("Each evaluation split must contain fake and real")
    if np.any(~np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be finite probabilities in [0, 1]")
    return labels, scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-apcer", type=float)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    args = parser.parse_args()

    rows = read_csv(args.scores)
    score_manifest_path = args.scores.with_suffix(".manifest.json")
    if not score_manifest_path.is_file():
        raise ValueError(f"Missing score provenance: {score_manifest_path}")
    score_provenance = json.loads(score_manifest_path.read_text(encoding="utf-8"))
    validation = [row for row in rows if row.get("split") == "val"]
    test = [row for row in rows if row.get("split") == "test"]
    if not validation or not test:
        raise ValueError("Score file requires non-empty val and test rows")
    val_labels, val_scores = _arrays(validation)
    test_labels, test_scores = _arrays(test)
    threshold, validation_rates = select_threshold(val_labels, val_scores, args.max_apcer)
    report = {
        "threshold_source": "validation",
        "threshold_comparator": "score > threshold means real",
        "score_provenance": score_provenance,
        "validation": {**evaluate_scores(val_labels, val_scores, threshold), **validation_rates},
        "test": evaluate_scores(test_labels, test_scores, threshold),
        "test_ci_95": bootstrap_metric_ci(test, threshold, args.bootstrap_iterations),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
