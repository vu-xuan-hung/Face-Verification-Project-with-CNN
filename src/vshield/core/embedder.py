import cv2
import numpy as np
from keras_facenet import FaceNet


embedder = FaceNet()

def img_to_encoding_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    embeddings = embedder.embeddings([rgb])
    emb = embeddings[0]
    emb = emb / np.linalg.norm(emb)
    return emb

def img_to_encoding_file(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img_to_encoding_frame(img)
