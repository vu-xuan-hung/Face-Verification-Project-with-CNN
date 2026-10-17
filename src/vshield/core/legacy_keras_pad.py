"""Load historical Keras PAD artifacts for offline serialization checks."""

import tensorflow as tf
from tensorflow.keras import layers

from vshield.models.augmentation import RandomGamma, RandomJpegCompression
from vshield.models.color_augmentation import BgrColorAugmentation
from vshield.models.lighting_normalization import RandomHistogramNormalization

CUSTOM_AUGMENTATION_LAYERS = {
    "RandomGamma": RandomGamma,
    "vshield>RandomGamma": RandomGamma,
    "RandomJpegCompression": RandomJpegCompression,
    "vshield>RandomJpegCompression": RandomJpegCompression,
    "BgrColorAugmentation": BgrColorAugmentation,
    "vshield>BgrColorAugmentation": BgrColorAugmentation,
    "RandomHistogramNormalization": RandomHistogramNormalization,
    "vshield>RandomHistogramNormalization": RandomHistogramNormalization,
}


# --- Patch các lớp để loại bỏ quantization_config ---
def patch_layer(cls):
    """Allow historical layer configs that include quantization metadata."""
    original_from_config = getattr(cls, "from_config", None)
    if not original_from_config:
        return cls

    def new_from_config(cls, config):
        config.pop("quantization_config", None)
        return original_from_config(config)

    cls.from_config = classmethod(new_from_config)
    return cls


layer_classes = [
    layers.Dense,
    layers.Conv2D,
    layers.DepthwiseConv2D,
    layers.BatchNormalization,
    layers.Add,
    layers.Multiply,
    layers.Reshape,
    layers.Activation,
    layers.GlobalAveragePooling2D,
    layers.GlobalMaxPooling2D,
    layers.Flatten,
    layers.Dropout,
    layers.InputLayer,
    layers.ZeroPadding2D,
    layers.MaxPooling2D,
    layers.AveragePooling2D,
]

for cls in layer_classes:
    try:
        patch_layer(cls)
    except AttributeError:
        pass


def load_anti_spoofing_model(model_path="artifacts/models/face_verify_v1.keras"):
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=CUSTOM_AUGMENTATION_LAYERS,
        )
        print("Anti-spoofing model loaded.")
        return model
    except Exception as e:
        print("Cannot load model:", e)
        return None
