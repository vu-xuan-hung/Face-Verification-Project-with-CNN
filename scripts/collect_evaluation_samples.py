"""Optional human-labelled webcam collector for evaluation manifests."""

import argparse
import csv
import os
import re
import time
from pathlib import Path

import cv2

SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["pad", "recognition"], required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--label", choices=["REAL", "SPOOF"], help="Human PAD ground truth")
    parser.add_argument("--attack-type", default="none")
    parser.add_argument("--split", choices=["gallery", "calibration", "test"])
    parser.add_argument("--is-enrolled", choices=["0", "1"])
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    _validate(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = (
        ["sample_id", "image_path", "label", "attack_type", "subject_id", "session_id", "device", "notes"]
        if args.mode == "pad"
        else ["sample_id", "image_path", "subject_id", "split", "is_enrolled", "session_id", "notes"]
    )
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError("Cannot open camera")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera capture failed")
            cv2.putText(frame, "SPACE capture | Q quit", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("VShield evaluation collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32:
                _save(frame, args, fields)
    finally:
        camera.release()
        cv2.destroyAllWindows()


def _validate(args):
    if args.mode == "pad" and not args.label:
        raise ValueError("PAD collection requires --label chosen by the evaluator")
    if args.mode == "recognition" and (not args.split or args.is_enrolled is None):
        raise ValueError("Recognition collection requires --split and --is-enrolled")
    for name in ("subject_id", "session_id"):
        value = getattr(args, name)
        if not SAFE_IDENTIFIER.fullmatch(value):
            raise ValueError(
                f"--{name.replace('_', '-')} must contain only letters, digits, '.', '_', or '-'"
            )


def _save(frame, args, fields):
    sample_id = f"{args.subject_id}-{args.session_id}-{time.time_ns()}"
    output_dir = args.output_dir.resolve()
    image_path = (output_dir / f"{sample_id}.png").resolve()
    if not image_path.is_relative_to(output_dir):
        raise ValueError("Capture path escapes --output-dir")
    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError("Cannot save capture")
    relative = Path(os.path.relpath(image_path, args.manifest.resolve().parent))
    row = {"sample_id": sample_id, "image_path": str(relative), "subject_id": args.subject_id, "session_id": args.session_id, "notes": ""}
    if args.mode == "pad":
        row.update(label=args.label, attack_type=args.attack_type, device=str(args.camera))
    else:
        row.update(split=args.split, is_enrolled=args.is_enrolled)
    exists = args.manifest.exists() and args.manifest.stat().st_size > 0
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"saved {sample_id}")


if __name__ == "__main__":
    main()
