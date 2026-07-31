"""Generate immutable validation/test score rows from a frozen candidate model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import tensorflow as tf
import yaml

from vshield.data.integrity import MANIFEST_FIELDS, read_csv
from vshield.data.loader import load_data_from_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = load_data_from_config(args.config, splits=("val", "test"))
    model = tf.keras.models.load_model(args.model, compile=False)
    fields = (*MANIFEST_FIELDS, "score")
    scored: list[dict[str, object]] = []
    for split in ("val", "test"):
        protocol = (args.config.parent / config["protocols"][split]).resolve()
        rows = read_csv(protocol)
        scores = model.predict(data[f"X_{split}"], verbose=0).reshape(-1)
        if len(rows) != len(scores):
            raise ValueError(f"Protocol/model count mismatch for {split}")
        scored.extend(
            {**row, "score": float(score)} for row, score in zip(rows, scores, strict=True)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scored)
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "model_sha256": _sha256(args.model),
                "config_sha256": _sha256(args.config),
                "rows": len(scored),
                "splits": ["val", "test"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
