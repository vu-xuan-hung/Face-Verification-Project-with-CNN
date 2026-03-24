import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras import layers
import cv2
import cvzone
import numpy as np


# --- Patch các lớp để loại bỏ quantization_config ---
def patch_layer(cls):
    original_from_config = cls.from_config

    def new_from_config(cls, config):
        config.pop("quantization_config", None)
        return original_from_config(config)

    cls.from_config = classmethod(new_from_config)
    return cls


# Danh sách các lớp cần patch
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
        pass  # Một số lớp có thể không có from_config
# ------------------------------------------------

# Load model
model = tf.keras.models.load_model("./models/face_verify_v1.keras")

# Cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
cap = cv2.VideoCapture(0)
classNames = ["fake", "real"]

while True:
    success, img = cap.read()
    if not success:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 10)

    for x, y, w, h in faces:
        face_img = img[y : y + h, x : x + w]
        face_resized = cv2.resize(face_img, (128, 128))
        face_batch = np.expand_dims(face_resized, axis=0) / 255.0

        prediction = model.predict(face_batch, verbose=0)
        prob = prediction[0][0]
        cls = 1 if prob > 0.5 else 0
        conf = prob if cls == 1 else 1 - prob

        color = (0, 255, 0) if cls == 1 else (0, 0, 255)
        cvzone.cornerRect(img, (x, y, w, h), colorC=color, colorR=color)
        cvzone.putTextRect(
            img,
            f"{classNames[cls].upper()} {int(conf * 100)}%",
            (x, y - 10),
            scale=1.5,
            offset=10,
            colorR=color,
        )

    cv2.imshow("Face Anti-Spoofing", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
