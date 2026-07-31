"""Mild train-only lighting normalization layers."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="vshield")
class RandomHistogramNormalization(keras.layers.Layer):
    """Blend global luminance equalization into a small fraction of train images."""

    def __init__(
        self,
        probability: float = 0.10,
        min_strength: float = 0.25,
        max_strength: float = 0.50,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        if not 0 <= min_strength <= max_strength <= 1:
            raise ValueError("strength must satisfy 0 <= min <= max <= 1")
        self.probability = probability
        self.min_strength = min_strength
        self.max_strength = max_strength
        self.seed = seed
        self._seed_generator = keras.random.SeedGenerator(seed)

    @staticmethod
    def _equalize(image, strength):
        bgr_luminance = tf.cast((0.114, 0.587, 0.299), image.dtype)
        intensity = tf.tensordot(image, bgr_luminance, axes=((-1,), (0,)))
        bins = tf.cast(
            tf.round(tf.clip_by_value(intensity, 0.0, 1.0) * 255.0),
            tf.int32,
        )
        histogram = tf.math.bincount(
            tf.reshape(bins, (-1,)),
            minlength=256,
            maxlength=256,
            dtype=tf.int32,
        )
        cumulative = tf.cumsum(histogram)
        nonzero = tf.boolean_mask(cumulative, histogram > 0)
        minimum = tf.reduce_min(nonzero)
        denominator = tf.size(bins, out_type=tf.int32) - minimum

        def normalized():
            lookup = tf.clip_by_value(
                tf.cast(cumulative - minimum, image.dtype)
                / tf.cast(denominator, image.dtype),
                0.0,
                1.0,
            )
            equalized = tf.gather(lookup, bins)
            scale = equalized / tf.maximum(intensity, tf.cast(1.0 / 255.0, image.dtype))
            corrected = tf.clip_by_value(image * scale[..., tf.newaxis], 0.0, 1.0)
            return image + (corrected - image) * strength

        return tf.cond(denominator > 0, normalized, lambda: image)

    def call(self, inputs, training=None):
        images = tf.cast(inputs, self.compute_dtype)
        if training is None or training is False:
            return images

        def augment():
            batch_size = tf.shape(images)[0]
            selected = keras.random.uniform(
                (batch_size,),
                seed=self._seed_generator,
            ) < self.probability
            strengths = keras.random.uniform(
                (batch_size,),
                minval=self.min_strength,
                maxval=self.max_strength,
                seed=self._seed_generator,
            )

            def normalize(values):
                image, should_normalize, strength = values
                return tf.cond(
                    should_normalize,
                    lambda: self._equalize(image, strength),
                    lambda: image,
                )

            return tf.map_fn(
                normalize,
                (images, selected, strengths),
                fn_output_signature=tf.TensorSpec(images.shape[1:], images.dtype),
            )

        if isinstance(training, bool):
            return augment()
        return tf.cond(tf.cast(training, tf.bool), augment, lambda: tf.identity(images))

    def get_config(self):
        return {
            **super().get_config(),
            "probability": self.probability,
            "min_strength": self.min_strength,
            "max_strength": self.max_strength,
            "seed": self.seed,
        }
