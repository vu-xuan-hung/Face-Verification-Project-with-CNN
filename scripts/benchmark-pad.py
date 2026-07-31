"""Warm single-sample PAD benchmark with native-process RSS reporting."""

from __future__ import annotations

import argparse
import ctypes
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from vshield.data.loader import load_data_from_config


def _rss_bytes() -> int:
    if os.name != "nt":
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value * (1 if sys.platform == "darwin" else 1024))

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    process = ctypes.windll.kernel32.GetCurrentProcess()
    ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
    return int(counters.WorkingSetSize)


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values), percentile))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    data = load_data_from_config(args.config, splits=("test",))
    model = tf.keras.models.load_model(args.model, compile=False)
    samples = data["X_test"]
    model.predict(samples[:1], verbose=0)
    rss_before = _rss_bytes()
    latencies = []
    for index in range(args.iterations):
        started = time.perf_counter()
        model.predict(samples[index % len(samples) : index % len(samples) + 1], verbose=0)
        latencies.append((time.perf_counter() - started) * 1000)
    rss_after = _rss_bytes()
    print(
        {
            "iterations": args.iterations,
            "latency_ms_mean": statistics.mean(latencies),
            "latency_ms_p50": _percentile(latencies, 50),
            "latency_ms_p95": _percentile(latencies, 95),
            "latency_ms_p99": _percentile(latencies, 99),
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
