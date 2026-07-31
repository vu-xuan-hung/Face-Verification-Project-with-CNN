"""Tests for seeded, train-only anti-spoof augmentation."""

import subprocess
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras

from vshield.models.augmentation import build_training_augmentation
from vshield.models.augmentation_config import (
    training_augmentation_spec,
    training_augmentation_spec_sha256,
)
from vshield.models.color_augmentation import BgrColorAugmentation
from vshield.models.lighting_normalization import RandomHistogramNormalization


def _sample_batch() -> tf.Tensor:
    values = tf.linspace(0.05, 0.95, 2 * 32 * 32 * 3)
    return tf.reshape(values, (2, 32, 32, 3))


def test_augmentation_is_disabled_for_validation_and_inference():
    images = _sample_batch()
    augmentation = build_training_augmentation(seed=17)

    output = augmentation(images, training=False)

    np.testing.assert_allclose(output.numpy(), images.numpy(), atol=0.0)


def test_augmentation_preserves_shape_dtype_and_unit_range():
    images = _sample_batch()
    output = build_training_augmentation(seed=23)(images, training=True)

    assert output.shape == images.shape
    assert output.dtype == images.dtype
    assert float(tf.reduce_min(output)) >= 0.0
    assert float(tf.reduce_max(output)) <= 1.0
    assert not np.allclose(output.numpy(), images.numpy())


def test_same_seed_reproduces_first_augmented_batch():
    images = _sample_batch()

    first = build_training_augmentation(seed=31)(images, training=True)
    second = build_training_augmentation(seed=31)(images, training=True)

    np.testing.assert_allclose(first.numpy(), second.numpy(), atol=1e-6)


def test_augmentation_spec_is_stable_and_seed_sensitive():
    first = training_augmentation_spec(seed=31)
    second = training_augmentation_spec(seed=31)

    assert first == second
    assert training_augmentation_spec_sha256(31) == (
        training_augmentation_spec_sha256(31)
    )
    assert training_augmentation_spec_sha256(31) != (
        training_augmentation_spec_sha256(32)
    )
    assert len(training_augmentation_spec_sha256(31)) == 64


def test_histogram_normalization_is_mild_and_train_only():
    images = tf.pow(_sample_batch(), 3)
    layer = RandomHistogramNormalization(
        probability=1.0,
        min_strength=0.4,
        max_strength=0.4,
        seed=37,
    )

    normalized = layer(images, training=True)
    inference = layer(images, training=False)

    assert normalized.shape == images.shape
    assert float(tf.reduce_min(normalized)) >= 0.0
    assert float(tf.reduce_max(normalized)) <= 1.0
    assert not np.allclose(normalized.numpy(), images.numpy())
    np.testing.assert_allclose(inference.numpy(), images.numpy(), atol=0.0)


def test_color_augmentation_respects_bgr_channel_order():
    bgr_blue = tf.constant([[[[1.0, 0.0, 0.0]]]], dtype=tf.float32)
    layer = BgrColorAugmentation(
        saturation_factor=0.0,
        hue_factor=0.0,
        grayscale_probability=1.0,
        seed=39,
    )

    grayscale = layer(bgr_blue, training=True)

    np.testing.assert_allclose(grayscale.numpy(), 0.114, atol=1e-4)


def test_augmentation_pipeline_round_trips_through_keras_format(tmp_path):
    model = keras.Sequential(
        [
            keras.layers.Input((32, 32, 3)),
            build_training_augmentation(seed=41),
        ]
    )
    model_path = tmp_path / "augmentation.keras"
    model.save(model_path)

    restored = keras.models.load_model(model_path)
    images = _sample_batch()
    output = restored(images, training=False)

    np.testing.assert_allclose(output.numpy(), images.numpy(), atol=0.0)


def test_production_loader_restores_augmentation_in_fresh_process(tmp_path):
    model = keras.Sequential(
        [
            keras.layers.Input((16, 16, 3)),
            build_training_augmentation(seed=43),
        ]
    )
    model_path = tmp_path / "production-load.keras"
    model.save(model_path)
    script = (
        "import numpy as np\n"
        "from vshield.core.anti_spoof import load_anti_spoofing_model\n"
        f"model = load_anti_spoofing_model({str(model_path)!r})\n"
        "if model is None:\n"
        "    raise SystemExit(1)\n"
        "output = model(np.zeros((1, 16, 16, 3), dtype=np.float32), training=False)\n"
        "raise SystemExit(tuple(output.shape) != (1, 16, 16, 3))\n"
    )

    completed = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", script],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
