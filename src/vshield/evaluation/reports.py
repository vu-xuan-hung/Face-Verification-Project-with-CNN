"""Markdown reports containing only computed metrics and explicit limitations."""

from __future__ import annotations

from pathlib import Path


def metric_markdown(title: str, metrics: dict, definitions: list[str] | None = None) -> str:
    lines = [f"# {title}", "", "| Metric | Value |", "|---|---:|"]
    for key, value in metrics.items():
        if isinstance(value, (dict, list)):
            continue
        rendered = "N/A" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
        lines.append(f"| {key.replace('_', ' ').title()} | {rendered} |")
    if definitions:
        lines.extend(["", "## Definitions", "", *[f"- {item}" for item in definitions]])
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "Metrics exclude no samples from the result files. Biometric success metrics treat "
            "inference failures as unsuccessful; fail-closed access metrics may count the resulting "
            "DENY as a security block and report inference failures separately.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_markdown(path: str | Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def combined_report(summary: dict, setup: dict, limitations: list[str]) -> str:
    lines = ["# VShield Biometric Evaluation", "", "## 1. Evaluation setup", ""]
    lines.extend(f"- {key}: {value}" for key, value in setup.items())
    lines.extend(["", "## 2. Dataset", "", "See the run's `dataset_audit.json` and `dataset_audit.csv`."])
    sections = [
        ("3. Phase 1 - PAD Results", "pad"),
        ("4. Phase 2 - Face Recognition Results", "recognition"),
        ("5. End-to-End Results", "e2e"),
        ("6. Latency", "latency"),
    ]
    for heading, key in sections:
        lines.extend(["", f"## {heading}", ""])
        values = summary.get(key)
        if not values:
            lines.append("Not computed.")
            continue
        for metric, value in _flat_metrics(values):
            lines.append(f"- {metric.replace('_', ' ')}: {'N/A' if value is None else value}")
    lines.extend(["", "## 7. Failure Analysis", "", "See component `failures.csv` files."])
    lines.extend(["", "## 8. Limitations", "", *[f"- {item}" for item in limitations]])
    lines.extend(
        [
            "",
            "## 9. Threshold Analysis",
            "",
            "Recommendations, when present, were selected only on calibration data and do not modify production configuration.",
        ]
    )
    calibration = summary.get("calibration")
    if calibration:
        for metric, value in _flat_metrics(calibration):
            lines.append(f"- {metric.replace('_', ' ')}: {'N/A' if value is None else value}")
    lines.extend(["", "## 10. Recommended Future Evaluation", "", "Collect independent, session-separated gallery, calibration, and final-test data following `evaluation_data/README.md`."])
    return "\n".join(lines) + "\n"


def _flat_metrics(values: dict, prefix: str = ""):
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _flat_metrics(value, name)
        elif not isinstance(value, list):
            yield name, value
