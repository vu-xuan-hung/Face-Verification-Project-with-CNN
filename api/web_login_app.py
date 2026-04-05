import os
import io
import base64
import csv

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras import layers
import cv2
import numpy as np
from keras_facenet import FaceNet
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import sys
from typing import Optional

import database  # Import the new database module
database.init_db() # Create tables if not exists

# --- Patch các lớp để loại bỏ quantization_config (giữ nguyên từ app.py) ---
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

tf.get_logger().setLevel('ERROR')

app = FastAPI()

# Enable CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model anti-spoofing
try:
    model = tf.keras.models.load_model("./models/face_verify_v1.keras")
    print("Anti-spoofing model loaded.")
except Exception as e:
    print("Cannot load model:", e)
    model = None

# Cascade classifier (nếu tương lai cần detect lại)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
embedder = FaceNet()

def img_to_encoding_file(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img_to_encoding_frame(img)

def img_to_encoding_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    embeddings = embedder.embeddings([rgb])
    emb = embeddings[0]
    emb = emb / np.linalg.norm(emb)
    return emb

def who_is_it(encoding, database_faces):
    min_dist = 100
    identity = "Unknown"
    for name, embeddings_list in database_faces.items():
        for db_emb in embeddings_list:
            dist = np.linalg.norm(encoding - db_emb)
            if dist < min_dist:
                min_dist = dist
                identity = name
    if min_dist > 0.9: 
        return "Unknown"
    else:
        return identity

# Load database (giống y app gốc)
print("Loading face dataset...")
try:
    database_faces = {
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
        "younes": [img_to_encoding_file("./faceverifi/images/younes.jpg")],
        "tian": [img_to_encoding_file("./faceverifi/images/tian.jpg")],
    }
    print("Dataset loaded successfully!")
except Exception as e:
    print("Warning: Failed to load dataset images:", e)
    database_faces = {}

class ImageData(BaseModel):
    image: str # e.g. "data:image/jpeg;base64,......."

@app.post("/predict")
async def predict(data: ImageData):
    try:
        if not data.image or not "," in data.image:
            return {"success": False, "message": "Invalid image format"}
            
        # Parse base64
        header, encoded = data.image.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"success": False, "message": "Can't decode image"}

        # Resize to exactly 128x128 as specified by requirements
        if img.shape[:2] != (128, 128):
            img = cv2.resize(img, (128, 128))

        # Anti-spoofing checks (Data requires norm by 255.0)
        face_batch = np.expand_dims(img, axis=0) / 255.0

        if model:
            prediction = model.predict(face_batch, verbose=0)
            prob = prediction[0][0]
            is_real = prob > 0.5
        else:
            is_real = True # Fallback for local testing without valid model
        
        if is_real:
            # Face verification
            encoding = img_to_encoding_frame(img)
            identity = who_is_it(encoding, database_faces)
            
            if identity != "Unknown":
                # Determine role
                role = "admin" if identity == "hung" else "user"
                
                # Save to login_logs DB
                database.log_login(identity, role)
                
                return {"success": True, "username": identity, "role": role}
            else:
                return {"success": False, "message": "Unknown face, not registered"}
        else:
            return {"success": False, "message": "Spoofing detected"}

    except Exception as e:
        print("Prediction Error:", e)
        return {"success": False, "message": str(e)}


@app.get("/logs")
def get_logs(username: Optional[str] = None, date: Optional[str] = None):
    # Fetch logs from DB
    logs = database.get_logs(username_filter=username, date_filter=date)
    return logs


@app.get("/logs/export")
def export_logs(username: Optional[str] = None, date: Optional[str] = None):
    logs = database.get_logs(username_filter=username, date_filter=date)
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["username", "role", "time"])
    for log in logs:
        writer.writerow([log["username"], log["role"], log["timestamp"]])
        
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=logs_export.csv"}
    )

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI Server at http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
