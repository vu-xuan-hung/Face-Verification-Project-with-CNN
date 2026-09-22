"""Create non-targeted SPOOF-only E2E probes from the held-out PAD clips."""

import argparse
import csv
import os
from pathlib import Path

FIELDS = ["sample_id", "image_path", "subject_id", "presentation", "expected_identity_state", "expected_access", "attack_type", "notes"]


def _relative_image_path(image_path: str, source_dir: Path, output_dir: Path) -> str:
    source = Path(image_path)
    resolved = source.resolve() if source.is_absolute() else (source_dir / source).resolve()
    return Path(os.path.relpath(resolved, output_dir.resolve())).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pad-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pad_manifest = args.pad_manifest.resolve()
    output = args.output.resolve()
    with pad_manifest.open("r", encoding="utf-8", newline="") as stream:
        source = list(csv.DictReader(stream))
    rows = [
        {
            "sample_id": f"e2e-{row['sample_id']}",
            "image_path": _relative_image_path(
                row["image_path"], pad_manifest.parent, output.parent
            ),
            "subject_id": f"unknown-clip-{row['session_id']}",
            "presentation": "SPOOF",
            "expected_identity_state": "UNKNOWN",
            "expected_access": "DENY",
            "attack_type": row["attack_type"],
            "notes": "Non-targeted attack against unrelated LFW gallery; NOT a spoof of an enrolled person",
        }
        for row in source if row["label"] == "SPOOF"
    ]
    if len({row["image_path"] for row in rows}) != len(rows):
        raise ValueError("Duplicate PAD sample path")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} non-targeted spoof-only E2E probes")


if __name__ == "__main__":
    main()
