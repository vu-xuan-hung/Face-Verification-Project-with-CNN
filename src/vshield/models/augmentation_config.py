"""Canonical, JSON-safe configuration for training-time augmentation."""

from __future__ import annotations

import hashlib
import json

AUGMENTATION_VERSION = "appearance-v1"


def training_augmentation_spec(seed: int = 42) -> dict[str, object]:
    """Return the stable augmentation contract recorded with trained models."""

    return {
        "version": AUGMENTATION_VERSION,
        "scope": "train_only",
        "label_policy": "symmetric",
        "input_contract": {"color_order": "BGR", "value_range": [0.0, 1.0]},
        "seed": seed,
        "rng_resume_semantics": "deterministic_uninterrupted_run",
        "operations": [
            {
                "name": "horizontal_flip",
                "type": "RandomFlip",
                "mode": "horizontal",
                "seed": seed,
            },
            {
                "name": "rotation",
                "type": "RandomRotation",
                "factor": 0.05,
                "fill_mode": "reflect",
                "seed": seed + 1,
            },
            {
                "name": "zoom_resize",
                "type": "RandomZoom",
                "height_factor": [-0.12, 0.12],
                "width_factor": [-0.12, 0.12],
                "fill_mode": "reflect",
                "seed": seed + 2,
            },
            {
                "name": "brightness",
                "type": "RandomBrightness",
                "factor": 0.15,
                "seed": seed + 3,
            },
            {
                "name": "contrast",
                "type": "RandomContrast",
                "factor": 0.15,
                "seed": seed + 4,
            },
            {
                "name": "gamma",
                "type": "RandomGamma",
                "factor": 0.15,
                "probability": 0.30,
                "seed": seed + 5,
            },
            {
                "name": "histogram_normalization",
                "type": "RandomHistogramNormalization",
                "probability": 0.10,
                "strength": [0.25, 0.50],
                "seed": seed + 6,
            },
            {
                "name": "color_and_grayscale",
                "type": "BgrColorAugmentation",
                "saturation_factor": 0.15,
                "hue_factor": 0.03,
                "grayscale_probability": 0.05,
                "seed": seed + 7,
            },
            {
                "name": "blur",
                "type": "RandomGaussianBlur",
                "factor": 0.15,
                "kernel_size": 3,
                "sigma": [0.1, 0.8],
                "seed": seed + 8,
            },
            {
                "name": "jpeg_compression",
                "type": "RandomJpegCompression",
                "quality": [70, 100],
                "probability": 0.20,
                "seed": seed + 9,
            },
            {
                "name": "gaussian_noise",
                "type": "GaussianNoise",
                "standard_deviation": 0.015,
                "seed": seed + 10,
            },
            {
                "name": "clip_to_unit_range",
                "type": "ReLU",
                "max_value": 1.0,
            },
        ],
    }


def training_augmentation_spec_sha256(seed: int = 42) -> str:
    """Hash the canonical augmentation contract for audit comparisons."""

    canonical = json.dumps(
        training_augmentation_spec(seed),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
