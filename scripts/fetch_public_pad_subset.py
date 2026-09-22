"""Download a bounded, clip-disjoint research PAD subset with source labels."""

import argparse
import csv
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

import cv2

REPO = "UniqueData/web-camera-face-liveness-detection"
REVISION = "7fe129eabc69aee1a6cd28b9a3c58d98e19d0249"
ATTACKS = {
    "monitor": "screen", "print": "print", "mask": "mask",
    "silicone": "silicone", "outline": "mask", "print_cut": "print",
}
FIELDS = ["sample_id", "image_path", "label", "attack_type", "subject_id", "session_id", "device", "notes"]


def _download(url: str, path: Path, max_bytes: int) -> int:
    if path.is_file():
        return path.stat().st_size
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "VShield-research-evaluation/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if int(response.headers.get("Content-Length", "0")) > max_bytes:
            raise ValueError(f"File too large: {url}")
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"File too large: {url}")
    path.write_bytes(data)
    return len(data)


def _middle_frame(path: Path):
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        capture.release()
        raise ValueError(f"Video has no frames: {path}")
    try:
        position = total // 2
        capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Cannot decode middle frame {position}: {path}")
        return position, frame
    finally:
        capture.release()


def _write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-total-mb", type=int, default=350)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    test, calibration, provenance = [], [], []
    total_bytes = 0
    classes = {"real": "none", **ATTACKS}
    for category, attack_type in classes.items():
        # Entire source clips are assigned to exactly one split.
        calibration_clips = range(0, 2)
        test_clips = range(2, 10)
        for split, clips in (("calibration", calibration_clips), ("test", test_clips)):
            for clip in clips:
                source_path = f"files/{category}/{clip}.mp4"
                encoded = urllib.parse.quote(source_path, safe="/")
                url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{encoded}"
                video = root / "source_videos" / category / f"{clip}.mp4"
                size = _download(url, video, 25 * 1024 * 1024)
                total_bytes += size
                if total_bytes > args.max_total_mb * 1024 * 1024:
                    raise ValueError("Acquisition exceeds --max-total-mb")
                position, frame = _middle_frame(video)
                image = root / "middle_frames" / split / category / f"{clip}.png"
                image.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(image), frame):
                    raise RuntimeError(f"Cannot write frame: {image}")
                row = {
                    "sample_id": f"{split}-{category}-{clip}",
                    "image_path": str(image.relative_to(root)),
                    "label": "REAL" if category == "real" else "SPOOF",
                    "attack_type": attack_type,
                    "subject_id": "",  # Source does not identify participants.
                    "session_id": f"{category}-{clip}",
                    "device": "webcam (source card; model unreported)",
                    "notes": f"source_clip={source_path}; middle_frame={position}",
                }
                (test if split == "test" else calibration).append(row)
                provenance.append({"path": source_path, "split": split, "sha256": hashlib.sha256(video.read_bytes()).hexdigest(), "size": size})
                print(f"{split}: {source_path} ({size} bytes)", flush=True)
    _write_manifest(root / "pad_calibration.csv", calibration)
    _write_manifest(root / "pad_test.csv", test)
    (root / "source_provenance.json").write_text(
        json.dumps({"repo": REPO, "revision": REVISION, "license": "cc-by-nc-nd-4.0", "source_split": "public preview/train", "test_frames": len(test), "calibration_frames": len(calibration), "independent_test_clips": len(test), "limitations": "One middle frame per video. Subject IDs and original train/model overlap are unverified; research-only, not deployed-system validation. Do not redistribute extracted frames.", "videos": provenance}, indent=2), encoding="utf-8"
    )
    print(f"PAD: {len(test)} test frames and {len(calibration)} calibration frames", flush=True)


if __name__ == "__main__":
    main()
