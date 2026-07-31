"""Seeded, train-only appearance augmentation for anti-spoof images."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

from vshield.models.color_augmentation import BgrColorAugmentation
from vshield.models.lighting_normalization import RandomHistogramNormalization


def _during_training(inputs, training, augment):
    if training is None:
        return inputs
    if isinstance(training, bool):
        return augment() if training else inputs
    return tf.cond(tf.cast(training, tf.bool), augment, lambda: tf.identity(inputs))


@keras.utils.register_keras_serializable(package="vshield")
class RandomGamma(keras.layers.Layer):
    """Randomly adjust gamma per image while preserving the unit value range."""

    def __init__(
        self,
        factor: float = 0.15,
        probability: float = 0.30,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not 0 <= factor < 1:
            raise ValueError("factor must be in [0, 1)")
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        self.factor = factor
        self.probability = probability
        self.seed = seed
        self._seed_generator = keras.random.SeedGenerator(seed)

    def call(self, inputs, training=None):
        images = tf.cast(inputs, self.compute_dtype)

        def augment():
            batch_shape = tf.stack([tf.shape(images)[0], 1, 1, 1])
            gamma = keras.random.uniform(
                batch_shape,
                minval=1.0 - self.factor,
                maxval=1.0 + self.factor,
                seed=self._seed_generator,
            )
            selected = keras.random.uniform(
                batch_shape,
                seed=self._seed_generator,
            ) < self.probability
            adjusted = tf.pow(tf.clip_by_value(images, 0.0, 1.0), gamma)
            return tf.where(selected, adjusted, images)

        return _during_training(images, training, augment)

    def get_config(self):
        return {
            **super().get_config(),
            "factor": self.factor,
            "probability": self.probability,
            "seed": self.seed,
        }


@keras.utils.register_keras_serializable(package="vshield")
class RandomJpegCompression(keras.layers.Layer):
    """Randomly round-trip images through JPEG to model codec variation."""

    def __init__(
        self,
        min_quality: int = 70,
        max_quality: int = 100,
        probability: float = 0.20,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not 1 <= min_quality <= max_quality <= 100:
            raise ValueError("JPEG quality must satisfy 1 <= min <= max <= 100")
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.probability = probability
        self.seed = seed
        self._seed_generator = keras.random.SeedGenerator(seed)

    def call(self, inputs, training=None):
        images = tf.cast(inputs, self.compute_dtype)

        def augment():
            batch_size = tf.shape(images)[0]
            qualities = tf.cast(
                keras.random.uniform(
                    (batch_size,),
                    minval=float(self.min_quality),
                    maxval=float(self.max_quality + 1),
                    seed=self._seed_generator,
                ),
                tf.int32,
            )
            selected = keras.random.uniform(
                (batch_size,),
                seed=self._seed_generator,
            ) < self.probability

            def compress(values):
                image, quality, should_compress = values

                def round_trip():
                    encoded = tf.image.convert_image_dtype(
                        tf.clip_by_value(image, 0.0, 1.0),
                        tf.uint8,
                        saturate=True,
                    )
                    decoded = tf.image.adjust_jpeg_quality(encoded, quality)
                    return tf.image.convert_image_dtype(decoded, images.dtype)

                return tf.cond(should_compress, round_trip, lambda: image)

            return tf.map_fn(
                compress,
                (images, qualities, selected),
                fn_output_signature=tf.TensorSpec(images.shape[1:], images.dtype),
            )

        return _during_training(images, training, augment)

    def get_config(self):
        return {
            **super().get_config(),
            "min_quality": self.min_quality,
            "max_quality": self.max_quality,
            "probability": self.probability,
            "seed": self.seed,
        }


def build_training_augmentation(seed: int = 42) -> keras.Sequential:
    """Build mild label-independent augmentation used only during training."""

    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=seed),
            keras.layers.RandomRotation(0.05, fill_mode="reflect", seed=seed + 1),
            keras.layers.RandomZoom(
                height_factor=(-0.12, 0.12),
                width_factor=(-0.12, 0.12),
                fill_mode="reflect",
                seed=seed + 2,
            ),
            keras.layers.RandomBrightness(
                0.15,
                value_range=(0.0, 1.0),
                seed=seed + 3,
            ),
            keras.layers.RandomContrast(
                0.15,
                value_range=(0.0, 1.0),
                seed=seed + 4,
            ),
            RandomGamma(seed=seed + 5),
            RandomHistogramNormalization(seed=seed + 6),
            BgrColorAugmentation(seed=seed + 7),
            keras.layers.RandomGaussianBlur(
                factor=0.15,
                kernel_size=3,
                sigma=(0.1, 0.8),
                value_range=(0.0, 1.0),
                seed=seed + 8,
            ),
            RandomJpegCompression(seed=seed + 9),
            keras.layers.GaussianNoise(0.015, seed=seed + 10),
            keras.layers.ReLU(max_value=1.0, name="clip_to_unit_range"),
        ],
        name="training_augmentation",
    )
