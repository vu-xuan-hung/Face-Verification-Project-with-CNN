"""Color augmentation that preserves the model's BGR tensor contract."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="vshield")
class BgrColorAugmentation(keras.layers.Layer):
    """Apply RGB-semantic color operations to BGR input tensors."""

    def __init__(
        self,
        saturation_factor: float = 0.15,
        hue_factor: float = 0.03,
        grayscale_probability: float = 0.05,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not 0 <= saturation_factor <= 1:
            raise ValueError("saturation_factor must be in [0, 1]")
        if not 0 <= hue_factor <= 0.5:
            raise ValueError("hue_factor must be in [0, 0.5]")
        if not 0 <= grayscale_probability <= 1:
            raise ValueError("grayscale_probability must be in [0, 1]")
        self.saturation_factor = saturation_factor
        self.hue_factor = hue_factor
        self.grayscale_probability = grayscale_probability
        self.seed = seed
        self._color_seed_generator = keras.random.SeedGenerator(seed)
        grayscale_seed = None if seed is None else seed + 1
        self._grayscale_seed_generator = keras.random.SeedGenerator(grayscale_seed)

    def call(self, inputs, training=None):
        images = tf.cast(inputs, self.compute_dtype)

        def augment():
            rgb = tf.reverse(images, axis=(-1,))
            batch_size = tf.shape(images)[0]
            saturation = keras.random.uniform(
                (batch_size,),
                minval=1.0 - self.saturation_factor,
                maxval=1.0 + self.saturation_factor,
                seed=self._color_seed_generator,
            )
            hue = keras.random.uniform(
                (batch_size,),
                minval=-self.hue_factor,
                maxval=self.hue_factor,
                seed=self._color_seed_generator,
            )

            def jitter(values):
                image, saturation_factor, hue_delta = values
                saturated = tf.image.adjust_saturation(image, saturation_factor)
                return tf.image.adjust_hue(saturated, hue_delta)

            jittered_rgb = tf.map_fn(
                jitter,
                (rgb, saturation, hue),
                fn_output_signature=tf.TensorSpec(images.shape[1:], images.dtype),
            )
            jittered_bgr = tf.reverse(
                tf.clip_by_value(jittered_rgb, 0.0, 1.0),
                axis=(-1,),
            )
            luminance_weights = tf.cast((0.114, 0.587, 0.299), images.dtype)
            luminance = tf.tensordot(
                jittered_bgr,
                luminance_weights,
                axes=((-1,), (0,)),
            )
            grayscale = tf.repeat(luminance[..., tf.newaxis], repeats=3, axis=-1)
            selection_shape = tf.stack([tf.shape(images)[0], 1, 1, 1])
            selected = keras.random.uniform(
                selection_shape,
                seed=self._grayscale_seed_generator,
            ) < self.grayscale_probability
            return tf.where(selected, grayscale, jittered_bgr)

        if training is None or training is False:
            return images
        if isinstance(training, bool):
            return augment()
        return tf.cond(tf.cast(training, tf.bool), augment, lambda: tf.identity(images))

    def get_config(self):
        return {
            **super().get_config(),
            "saturation_factor": self.saturation_factor,
            "hue_factor": self.hue_factor,
            "grayscale_probability": self.grayscale_probability,
            "seed": self.seed,
        }
