"""Latency summaries for successful, measured stages."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def latency_summary(rows: Sequence[dict], fields: Sequence[str]) -> dict[str, dict]:
    result = {}
    for field in fields:
        values = np.asarray(
            [float(row[field]) for row in rows if row.get(field) is not None], dtype=float
        )
        if values.size:
            result[field] = {
                "n": int(values.size),
                "mean_ms": float(np.mean(values)),
                "median_ms": float(np.median(values)),
                "p95_ms": float(np.percentile(values, 95)),
            }
    return result
