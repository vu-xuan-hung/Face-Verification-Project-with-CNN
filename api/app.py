import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras import layers
import cv2
import cvzone
import numpy as np

from keras_facenet import FaceNet
import tkinter as tk
from PIL import Image, ImageTk


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
embedder = FaceNet()


def img_to_encoding_file(path):
    img = cv2.imread(path)
    return img_to_encoding_frame(img)


def img_to_encoding_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    embeddings = embedder.embeddings([rgb])
    emb = embeddings[0]
    emb = emb / np.linalg.norm(emb)
    return emb


def who_is_it(encoding, database):
    min_dist = 100
    identity = "Unknown"
    for name, embeddings_list in database.items():
        for db_emb in embeddings_list:
            dist = np.linalg.norm(encoding - db_emb)
            if dist < min_dist:
                min_dist = dist
                identity = name
    print("Distance:", min_dist)
    if min_dist > 0.9:
        return "Unknown"
    else:
        return identity


database = {
    "hung": [
        img_to_encoding_file("./faceverifi/images/1.jpg"),
        img_to_encoding_file("./faceverifi/images/2.jpg"),
        img_to_encoding_file("./faceverifi/images/3.jpg"),
        img_to_encoding_file("./faceverifi/images/4.jpg"),
        img_to_encoding_file("./faceverifi/images/5.jpg"),
        img_to_encoding_file("./faceverifi/images/6.jpg"),
        img_to_encoding_file("./faceverifi/images/7.jpg"),
        img_to_encoding_file("./faceverifi/images/8.jpg"),
        img_to_encoding_file("./faceverifi/images/9.jpg"),
        img_to_encoding_file("./faceverifi/images/10.jpg"),
    ],
    "hoang": [
        img_to_encoding_file("./faceverifi/images/hoa1.jpg"),
        img_to_encoding_file("./faceverifi/images/hoa2.jpg"),
        img_to_encoding_file("./faceverifi/images/hoa3.jpg"),
        img_to_encoding_file("./faceverifi/images/hoa4.jpg"),
        img_to_encoding_file("./faceverifi/images/hoa5.jpg"),
    ],
    "younes": [img_to_encoding_file("./faceverifi/images/younes.jpg")],
    "tian": [img_to_encoding_file("./faceverifi/images/tian.jpg")],
}

root = tk.Tk()
root.title("Face Anti-Spoofing & Verification")
root.geometry("800x600")

# Video display label
video_label = tk.Label(root)
video_label.pack(pady=10)

# Info labels
anti_spoof_label = tk.Label(root, text="Anti-Spoofing: Waiting...", font=("Arial", 16))
anti_spoof_label.pack()

verify_label = tk.Label(root, text="Verification: Unknown", font=("Arial", 16))
verify_label.pack()

# Video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()


def process_frame():
    ret, img = cap.read()
    if not ret:
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 10)
    anti_spoof_text = "No face detected"
    verify_text = "Unknown"
    color = (0, 0, 255)
    for x, y, w, h in faces:
        # extract face and norm
        face_img = img[y : y + h, x : x + w]
        face_resized = cv2.resize(face_img, (128, 128))
        face_batch = np.expand_dims(face_resized, axis=0) / 255.0
        # Anti-spoofing prediction
        prediction = model.predict(face_batch, verbose=0)
        prob = prediction[0][0]
        is_real = prob > 0.5
        conf = prob if is_real else 1 - prob
        cls = "REAL" if is_real else "FAKE"
        anti_spoof_text = f"{cls} ({int(conf * 100)}%)"
        # face verifi
        if is_real:
            encoding = img_to_encoding_frame(face_img)
            identity = who_is_it(encoding, database)
            verify_text = identity
            color = (0, 255, 0)  # green for real
        else:
            verify_text = "Unknown"
            color = (0, 0, 255)

        cvzone.cornerRect(img, (x, y, w, h), colorC=color, colorR=color)
        cvzone.putTextRect(
            img, anti_spoof_text, (x, y - 10), scale=1, offset=10, colorR=color
        )
        cvzone.putTextRect(
            img, verify_text, (x, y + h + 30), scale=1, offset=10, colorR=color
        )

        break  # Remove this line to handle multiple faces

    # Update labels
    anti_spoof_label.config(text=f"Anti-Spoofing: {anti_spoof_text}")
    verify_label.config(text=f"Verification: {verify_text}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    # Resize to fit GUI
    img_pil = img_pil.resize((640, 480), Image.Resampling.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Schedule next frame
    root.after(10, process_frame)


process_frame()


def on_closing():
    cap.release()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
