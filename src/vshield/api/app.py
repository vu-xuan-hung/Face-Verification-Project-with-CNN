import os
import io
import base64
import csv
import cv2
import numpy as np

import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from vshield.api import database
from vshield.core.anti_spoof import load_anti_spoofing_model, predict_is_real
from vshield.core.embedder import img_to_encoding_frame
from vshield.core.verifier import load_database, who_is_it, build_faiss_index

# Tắt warning tf
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
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

# Khởi tạo DB, model và face data
database.init_db()
model = load_anti_spoofing_model()
database_faces = load_database()

# Build FAISS index để tìm kiếm nhanh hơn (tự fallback về for-loop nếu chưa cài faiss)
faiss_index, id_to_name = build_faiss_index(database_faces)

class ImageData(BaseModel):
    image: str

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

        # Resize to exactly 128x128
        if img.shape[:2] != (128, 128):
            img = cv2.resize(img, (128, 128))

        # Anti-spoofing checks (Data requires norm by 255.0)
        face_batch = np.expand_dims(img, axis=0) / 255.0

        is_real = predict_is_real(model, face_batch)
        
        if is_real:
            # Face verification
            encoding = img_to_encoding_frame(img)
            identity = who_is_it(encoding, database_faces, faiss_index=faiss_index, id_to_name=id_to_name)
            
            if identity != "Unknown":
                # Lấy role từ database thay vì hardcode if/else (CẢI TIẾN RBAC)
                role = database.get_role(identity)
                
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
