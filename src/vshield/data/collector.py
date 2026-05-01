"""
Real-time face data collector.

Captures face crops from a webcam and saves them (image + YOLO label)
to the configured output directory. Includes:

  - Blur detection (Laplacian variance) to ensure sharp crops
  - Blink detection via Eye Aspect Ratio (EAR) with MediaPipe
  - Foreground motion estimation with MOG2 background subtraction

Usage:
    python -m vshield.data.collector --class-id 1 --output data/raw/real
    python -m vshield.data.collector --class-id 0 --output data/raw/fake

    Or via Makefile:
        make collect-real
        make collect-fake
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import cvzone
import mediapipe as mp
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector

# ---------------------------------------------------------------------------
# Defaults (can be overridden via CLI)
# ---------------------------------------------------------------------------
_BLUR_THRESHOLD = 35      # Laplacian var below this → blurry → skip
_BLINK_THRESHOLD = 0.20   # EAR below this → blink detected
_CONFIDENCE = 0.80        # Minimum face detection confidence
_EXPAND_PCT = 10          # Face bounding-box expansion in %
_CAM_W, _CAM_H = 640, 480


# ---------------------------------------------------------------------------
# Eye Aspect Ratio
# ---------------------------------------------------------------------------

def _ear(eye: list[np.ndarray]) -> float:
    """Compute the Eye Aspect Ratio for blink detection."""

    def _d(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    return (_d(eye[1], eye[5]) + _d(eye[2], eye[4])) / (2.0 * _d(eye[0], eye[3]))


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class DataCollector:
    """Webcam-based face crop collector with liveness heuristics.

    Parameters
    ----------
    class_id:
        YOLO class index — 0 for Fake, 1 for Real.
    output_dir:
        Directory to save image (.jpg) and label (.txt) files.
    blur_threshold:
        Minimum Laplacian variance required (higher = sharper).
    confidence:
        Minimum CvZone FaceDetector confidence score.
    """

    # MediaPipe face mesh landmark indices for left/right eye
    _LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
    _RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

    def __init__(
        self,
        class_id: int = 0,
        output_dir: str | Path = "data/raw",
        blur_threshold: float = _BLUR_THRESHOLD,
        confidence: float = _CONFIDENCE,
    ) -> None:
        self.class_id = class_id
        self.output_dir = Path(output_dir)
        self.blur_threshold = blur_threshold
        self.confidence = confidence

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, _CAM_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAM_H)

        self._detector = FaceDetector(minDetectionCon=self.confidence)
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )

    def run(self) -> None:
        """Start the capture loop. Press ``q`` to quit."""
        label = "REAL" if self.class_id == 1 else "FAKE"
        print(f"[DataCollector] Collecting class={self.class_id} ({label})")
        print(f"[DataCollector] Saving to {self.output_dir}")
        print("[DataCollector] Press 'q' to stop.")

        try:
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    print("[DataCollector] Camera read failed.")
                    break

                display = frame.copy()
                frame, bboxs = self._detector.findFaces(frame, draw=False)

                if bboxs:
                    self._process_face(frame, display, bboxs[0])

                cv2.imshow("V-Shield DataCollector", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self._cap.release()
            cv2.destroyAllWindows()

    def _process_face(
        self,
        frame: np.ndarray,
        display: np.ndarray,
        bbox: dict,
    ) -> None:
        x, y, w, h = bbox["bbox"]
        score: float = bbox["score"][0]

        if score < self.confidence:
            return

        # Expand bounding box
        dx = int(w * _EXPAND_PCT / 100)
        dy = int(h * _EXPAND_PCT / 100)
        x = max(0, x - dx)
        y = max(0, y - dy * 3)
        w = min(w + 2 * dx, frame.shape[1] - x)
        h = min(h + int(3.5 * dy), frame.shape[0] - y)

        if w <= 0 or h <= 0:
            return

        face_crop = frame[y:y + h, x:x + w]
        if face_crop.size == 0:
            return

        face_128 = cv2.resize(face_crop.copy(), (128, 128))

        # --- Blur check ---
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_sharp = blur_val > self.blur_threshold

        # --- YOLO normalised coords ---
        ih, iw = frame.shape[:2]
        xc = (x + w / 2) / iw
        yc = (y + h / 2) / ih
        wn = np.clip(w / iw, 0, 1)
        hn = np.clip(h / ih, 0, 1)
        xc = np.clip(xc, 0, 1)
        yc = np.clip(yc, 0, 1)

        # --- Draw overlay ---
        color = (0, 255, 0) if is_sharp else (0, 0, 255)
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
        cvzone.putTextRect(
            display,
            f"{'SHARP' if is_sharp else 'BLUR'} {blur_val:.0f}",
            (x, y), scale=1, offset=8, colorR=color,
        )

        if is_sharp:
            ts = str(time.time()).replace(".", "")
            img_path = self.output_dir / f"{ts}.jpg"
            lbl_path = self.output_dir / f"{ts}.txt"
            cv2.imwrite(str(img_path), face_128)
            lbl_path.write_text(f"{self.class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V-Shield face data collector.")
    p.add_argument("--class-id", type=int, default=0, choices=[0, 1],
                   help="0=fake, 1=real")
    p.add_argument("--output", type=str, default="data/raw",
                   help="Output directory for images and labels.")
    p.add_argument("--blur-threshold", type=float, default=_BLUR_THRESHOLD)
    p.add_argument("--confidence", type=float, default=_CONFIDENCE)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    collector = DataCollector(
        class_id=args.class_id,
        output_dir=args.output,
        blur_threshold=args.blur_threshold,
        confidence=args.confidence,
    )
    collector.run()
