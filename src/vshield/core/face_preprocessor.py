"""Face detection, alignment, and model-specific crop preparation."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import cv2
import numpy as np

ANTI_SPOOF_SIZE = (128, 128)


class FacePreprocessingError(ValueError):
    """Raised when an image cannot be converted into a valid face crop."""


class InvalidFaceCountError(FacePreprocessingError):
    """Raised when an image does not contain exactly one detectable face."""

    def __init__(self, count: int):
        super().__init__(f"Exactly one face is required; detected {count}")
        self.count = count


@dataclass(frozen=True)
class FaceCrops:
    """Model-specific crops derived from one detected face."""

    anti_spoof: np.ndarray
    facenet: np.ndarray


class FacePreprocessor:
    """Detect one face and prepare anti-spoof and FaceNet crops."""

    def __init__(self, face_detector=None, eye_detector=None):
        self.face_detector = face_detector or self._load_cascade(
            "haarcascade_frontalface_default.xml"
        )
        self.eye_detector = eye_detector or self._load_cascade("haarcascade_eye.xml")
        self._detector_lock = Lock()

    @staticmethod
    def _load_cascade(filename: str):
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + filename)
        if detector.empty():
            raise RuntimeError(f"Cannot load OpenCV cascade: {filename}")
        return detector

    def extract(self, image: np.ndarray) -> FaceCrops:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise FacePreprocessingError("Expected a BGR image with three channels")
        if image.size == 0:
            raise FacePreprocessingError("Image is empty")

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        with self._detector_lock:
            detected = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(30, 30),
            )
        faces = list(detected)
        if len(faces) != 1:
            raise InvalidFaceCountError(len(faces))

        face_crop = self._expanded_crop(image, faces[0])
        if face_crop.size == 0:
            raise FacePreprocessingError("Detected face crop is empty")

        anti_spoof = cv2.resize(face_crop, ANTI_SPOOF_SIZE)
        facenet = self._align_by_eyes(face_crop)
        return FaceCrops(anti_spoof=anti_spoof, facenet=facenet)

    @staticmethod
    def _expanded_crop(image: np.ndarray, box) -> np.ndarray:
        x, y, width, height = (int(value) for value in box)
        image_height, image_width = image.shape[:2]

        left = max(0, x - int(width * 0.10))
        top = max(0, y - int(height * 0.30))
        right = min(image_width, x + width + int(width * 0.10))
        bottom = min(image_height, y + height + int(height * 0.05))
        return image[top:bottom, left:right].copy()

    def _align_by_eyes(self, face_crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        upper_height = max(1, int(gray.shape[0] * 0.65))
        upper_face = gray[:upper_height]
        with self._detector_lock:
            detected = self.eye_detector.detectMultiScale(
                upper_face,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(8, 8),
            )
        eyes = sorted(detected, key=lambda eye: int(eye[2]) * int(eye[3]), reverse=True)[:2]
        if len(eyes) != 2:
            return face_crop

        centers = [
            (float(x + width / 2), float(y + height / 2))
            for x, y, width, height in eyes
        ]
        left_eye, right_eye = sorted(centers, key=lambda center: center[0])
        delta_x = right_eye[0] - left_eye[0]
        if delta_x <= 0:
            return face_crop

        delta_y = right_eye[1] - left_eye[1]
        angle = float(np.degrees(np.arctan2(delta_y, delta_x)))
        midpoint = (
            (left_eye[0] + right_eye[0]) / 2,
            (left_eye[1] + right_eye[1]) / 2,
        )
        rotation = cv2.getRotationMatrix2D(midpoint, angle, 1.0)
        return cv2.warpAffine(
            face_crop,
            rotation,
            (face_crop.shape[1], face_crop.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
