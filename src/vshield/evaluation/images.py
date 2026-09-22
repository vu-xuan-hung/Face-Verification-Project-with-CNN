"""Bounded local image decoding for evaluation manifests."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Image exceeds production 8 MiB decoded-payload limit")
    with Image.open(path) as header:
        if header.width * header.height > 20_000_000:
            raise ValueError("Image exceeds production 20 megapixel limit")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image")
    return image
