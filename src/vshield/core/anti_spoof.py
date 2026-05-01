import os
import tensorflow as tf
from tensorflow.keras import layers

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
        model = tf.keras.models.load_model(model_path)
        print("Anti-spoofing model loaded.")
        return model
    except Exception as e:
        print("Cannot load model:", e)
        return None

def predict_is_real(model, face_batch):
    if model is None:
        return True # Fallback nếu không có model
    
    prediction = model.predict(face_batch, verbose=0)
    prob = prediction[0][0]
    return prob > 0.5
