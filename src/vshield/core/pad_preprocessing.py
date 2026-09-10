"""MiniVision scale-2.7 contextual crop, pinned to reference b6d5f04."""
import cv2
import numpy as np


def prepare_pad_input(image, bbox, scale=2.7):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("invalid image")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2:
        raise ValueError("invalid image")
    if max(image.shape[:2]) > 16384 or image.shape[0] * image.shape[1] > 40_000_000:
        raise ValueError("image too large")
    box = np.asarray(bbox)
    if box.shape != (4,) or not np.issubdtype(box.dtype, np.integer):
        raise ValueError("invalid bounding box")
    x, y, width, height = map(int, box)
    src_h, src_w = image.shape[:2]
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > src_w or y + height > src_h:
        raise ValueError("bounding box outside image")
    if scale != 2.7:
        raise ValueError("unsupported crop scale")
    # Preserve reference clamping, truncation and inclusive bottom/right pixels.
    scale = min((src_h - 1) / height, (src_w - 1) / width, scale)
    center_x, center_y = width / 2 + x, height / 2 + y
    left, top = center_x - width * scale / 2, center_y - height * scale / 2
    right, bottom = center_x + width * scale / 2, center_y + height * scale / 2
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > src_w - 1:
        left -= right - src_w + 1
        right = src_w - 1
    if bottom > src_h - 1:
        top -= bottom - src_h + 1
        bottom = src_h - 1
    crop = image[int(top):int(bottom) + 1, int(left):int(right) + 1]
    resized = cv2.resize(crop, (80, 80), interpolation=cv2.INTER_LINEAR)
    # Reference ToTensor does NOT divide NumPy BGR images by 255.
    return np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.float32)
