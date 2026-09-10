"""Evaluation workflow for MiniFASNetV2 Anti-Spoofing on local dataset.

Expected layout:
evaluation/
├── real/
├── fake_print/
└── fake_screen/

If directories are missing or empty, prints:
NO EVALUATION DATASET PROVIDED
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vshield.core.anti_spoof import build_pad_service, check_pad  # noqa: E402
from vshield.core.face_preprocessor import FacePreprocessor  # noqa: E402

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]


def find_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]


def evaluate_dataset(eval_dir: Path, thresholds: list[float] | None = None) -> int:
    thresholds = thresholds or DEFAULT_THRESHOLDS

    real_dir = eval_dir / "real"
    fake_print_dir = eval_dir / "fake_print"
    fake_screen_dir = eval_dir / "fake_screen"
    legacy_fake_dir = eval_dir / "fake"

    real_images = find_images(real_dir)
    fake_images = find_images(fake_print_dir) + find_images(fake_screen_dir)
    if not fake_images and legacy_fake_dir.is_dir():
        fake_images = find_images(legacy_fake_dir)

    total_images = len(real_images) + len(fake_images)

    if total_images == 0:
        print("=" * 60)
        print("NO EVALUATION DATASET PROVIDED")
        print("=" * 60)
        print(f"Looked in: {eval_dir.resolve()}")
        print("Expected directory layout:")
        print("  evaluation/")
        print("  |-- real/         (genuine live face photos)")
        print("  |-- fake_print/   (printed photo attacks)")
        print("  \\-- fake_screen/  (screen / replay attacks)")
        print()

        print("Please place test samples in the folders above to run empirical PAD evaluation.")
        print("Current production threshold 0.80 remains PROVISIONAL and must not be")
        print("assumed optimal without empirical validation on target sensors.")
        print("=" * 60)
        return 0

    print(f"Found {len(real_images)} REAL samples and {len(fake_images)} FAKE samples.")
    print("Initializing FacePreprocessor and MiniFASNet PAD service...")
    preprocessor = FacePreprocessor()
    pad_service = build_pad_service(PROJECT_ROOT)

    labels = []  # 1 = real, 0 = fake
    scores = []

    for path in real_images:
        img = cv2.imread(str(path))
        if img is None:
            continue
        try:
            crops = preprocessor.extract(img)
            bbox = crops.bbox
        except Exception:
            bbox = (0, 0, img.shape[1], img.shape[0])
        pad = check_pad(pad_service, img, bbox)
        if pad.score is not None:
            labels.append(1)
            scores.append(pad.score)

    for path in fake_images:
        img = cv2.imread(str(path))
        if img is None:
            continue
        try:
            crops = preprocessor.extract(img)
            bbox = crops.bbox
        except Exception:
            bbox = (0, 0, img.shape[1], img.shape[0])
        pad = check_pad(pad_service, img, bbox)
        if pad.score is not None:
            labels.append(0)
            scores.append(pad.score)

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    total_real = int(np.sum(labels == 1))
    total_fake = int(np.sum(labels == 0))

    print("\n" + "=" * 80)
    print("MINIFASNET V2 EMPIRICAL PAD EVALUATION REPORT")
    print("=" * 80)
    print(f"Total Evaluated: {len(labels)} (Real: {total_real}, Fake: {total_fake})")
    print("-" * 80)
    print(f"{'Threshold':>9} | {'Real Ok':>7} | {'Fake Ok':>7} | {'APCER (FAR)':>11} | {'BPCER (FRR)':>11} | {'ACER':>7} | {'Accuracy':>8}")
    print("-" * 80)

    for th in thresholds:
        preds = scores > th  # True = classified as Real (1)

        # Real stats
        if total_real > 0:
            correct_real = int(np.sum(preds[labels == 1]))
            false_reject_count = total_real - correct_real
            bpcer = false_reject_count / total_real
        else:
            correct_real = 0
            bpcer = float("nan")

        # Fake stats
        if total_fake > 0:
            correct_fake = int(np.sum(~preds[labels == 0]))
            false_accept_count = total_fake - correct_fake
            apcer = false_accept_count / total_fake
        else:
            correct_fake = 0
            apcer = float("nan")

        acer = (apcer + bpcer) / 2 if (not np.isnan(apcer) and not np.isnan(bpcer)) else float("nan")
        acc = (correct_real + correct_fake) / len(labels) if len(labels) > 0 else 0.0

        marker = " *" if abs(th - 0.80) < 1e-4 else "  "
        print(f"{th:>8.2f}{marker} | {correct_real:>7}/{total_real} | {correct_fake:>7}/{total_fake} | {apcer:>10.2%} | {bpcer:>10.2%} | {acer:>6.2%} | {acc:>7.2%}")

    print("-" * 80)
    print(" * Note: Threshold 0.80 is PROVISIONAL.")
    print("   APCER = Attack Presentation Classification Error Rate (False Accept Rate for spoof)")
    print("   BPCER = Bona Fide Presentation Classification Error Rate (False Reject Rate for real)")
    print("   ACER  = (APCER + BPCER) / 2")
    print("=" * 80)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate MiniFASNet PAD dataset.")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation",
        help="Path to evaluation directory containing real/ and fake_print/ / fake_screen/ folders",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Thresholds to evaluate (default: 0.50 0.60 0.70 0.80 0.90)",
    )
    args = parser.parse_args()
    return evaluate_dataset(args.eval_dir, args.thresholds)


if __name__ == "__main__":
    sys.exit(main())
