import numpy as np
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


class AntiSpoofingError(RuntimeError):
    """Raised when anti-spoofing cannot produce a trustworthy result."""

# --- Patch các lớp để loại bỏ quantization_config ---
def patch_layer(cls):
    original_from_config = getattr(cls, "from_config", None)
    if not original_from_config:
        return cls

    def new_from_config(cls, config):
        config.pop("quantization_config", None)
        return original_from_config(config)

    cls.from_config = classmethod(new_from_config)
    return cls

layer_classes = [
    layers.Dense, layers.Conv2D, layers.DepthwiseConv2D, layers.BatchNormalization,
    layers.Add, layers.Multiply, layers.Reshape, layers.Activation,
    layers.GlobalAveragePooling2D, layers.GlobalMaxPooling2D, layers.Flatten,
    layers.Dropout, layers.InputLayer, layers.ZeroPadding2D, layers.MaxPooling2D, layers.AveragePooling2D,
]

for cls in layer_classes:
    try:
        patch_layer(cls)
    except AttributeError:
        pass
# ------------------------------------------------

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

def predict_is_real(model, face_batch):
    if model is None:
        raise AntiSpoofingError("Anti-spoofing model is unavailable")

    try:
        prediction = model.predict(face_batch, verbose=0)
    except Exception as exc:
        raise AntiSpoofingError("Anti-spoofing inference failed") from exc

    try:
        scores = np.asarray(prediction)
        is_numeric = np.issubdtype(scores.dtype, np.number)
        is_complex = np.issubdtype(scores.dtype, np.complexfloating)
        if scores.size != 1 or not is_numeric or is_complex:
            raise ValueError("expected exactly one score")
        score = float(scores.reshape(-1)[0])
    except Exception as exc:
        raise AntiSpoofingError("Anti-spoofing returned an invalid score") from exc

    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise AntiSpoofingError("Anti-spoofing returned an invalid score")

    return score > 0.5
