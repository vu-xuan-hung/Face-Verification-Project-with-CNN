"""Bounded image decoding shared by prediction and enrollment."""

from __future__ import annotations

import base64
import binascii
import io

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_ENCODED_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


def decode_image(data_uri: str) -> np.ndarray:
    """Decode one base64 image data URI into a BGR uint8 image."""
    if not data_uri or "," not in data_uri:
        raise ValueError("Invalid image format")

    header, encoded = data_uri.split(",", 1)
    if not header.startswith("data:image/") or ";base64" not in header:
        raise ValueError("Invalid image format")
    if len(encoded) > (MAX_ENCODED_IMAGE_BYTES * 4 // 3) + 4:
        raise ValueError("Image payload is too large")

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image") from exc
    return decode_image_bytes(image_bytes)


def decode_image_bytes(image_bytes: bytes, *, content_type: str | None = None) -> np.ndarray:
    """Validate bounded encoded bytes before allocating an OpenCV BGR image.

    Uploads require JPEG/PNG and matching MIME; enrollment keeps its data-URI contract.
    """
    if not image_bytes:
        raise ValueError("Empty image")
    if len(image_bytes) > MAX_ENCODED_IMAGE_BYTES:
        raise ValueError("Image payload is too large")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image_header:
            width, height = image_header.size
            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError("Decoded image is too large")
            if content_type is not None:
                expected_mime = {"JPEG": "image/jpeg", "PNG": "image/png"}.get(image_header.format)
                if expected_mime is None:
                    raise ValueError("Only JPEG and PNG images are supported")
                if content_type != expected_mime:
                    raise ValueError("Image content does not match Content-Type")
                # Pillow's JPEG verify() is a no-op; load rejects truncated scans
                # that OpenCV can otherwise decode permissively. Close releases
                # these bounded pixels before allocating the BGR frame below.
                if image_header.format == "JPEG":
                    image_header.load()
                else:
                    image_header.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError("Can't decode image") from exc
    try:
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError("Can't decode image") from exc
    if image is None:
        raise ValueError("Can't decode image")
    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError("Decoded image is too large")
    return image
