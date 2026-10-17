"""Explicitly download and verify the YOLO face model; never downloads at runtime."""

import hashlib
import urllib.request
from pathlib import Path

from vshield.core.yolo_face_detector import DEFAULT_MODEL_PATH, MODEL_SHA256, MODEL_URL


def main():
    destination = Path(DEFAULT_MODEL_PATH)
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == MODEL_SHA256:
        print(f"Verified existing model: {destination}")
        return
    with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
        artifact = response.read(20_000_001)
    if len(artifact) > 20_000_000 or hashlib.sha256(artifact).hexdigest() != MODEL_SHA256:
        raise RuntimeError("YOLO model download failed size/checksum verification")
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidate = destination.with_suffix(".onnx.download")
    try:
        candidate.write_bytes(artifact)
        candidate.replace(destination)
    finally:
        candidate.unlink(missing_ok=True)
    print(f"Installed verified model: {destination}")


if __name__ == "__main__":
    main()
