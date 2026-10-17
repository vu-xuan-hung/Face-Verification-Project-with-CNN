"""Hash-pinned YOLOv8n-Face inference using the author's ONNX input contract."""

import hashlib
from pathlib import Path

import cv2
import numpy as np

MODEL_FILENAME = "yolov8n-face-lindevs.onnx"
MODEL_SHA256 = "8d0bfb0c3383c5bd7a78dd24ef79a21e2aa456619b6ab5e53867092d1c7dc414"
MODEL_URL = f"https://github.com/lindevs/yolov8-face/releases/latest/download/{MODEL_FILENAME}"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "artifacts/models" / MODEL_FILENAME
CONFIDENCE_THRESHOLD = 0.75
NMS_IOU_THRESHOLD = 0.7


class YoloFaceDetector:
    """Detect faces in BGR frames. No network access or Haar fallback at runtime.

    FacePreprocessor serializes calls with its detector lock. Load lazily so a
    missing model produces MODEL_ERROR during authentication, rather than taking
    down the HTTP server. Only a successfully verified session is cached.
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self._session = None

    def _load_session(self):
        import onnxruntime as ort

        artifact = self.model_path.read_bytes()
        if hashlib.sha256(artifact).hexdigest() != MODEL_SHA256:
            raise RuntimeError("Face detector model checksum mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(
            artifact, sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if (
            len(inputs) != 1
            or inputs[0].shape != [1, 3, 640, 640]
            or inputs[0].type != "tensor(float)"
            or len(outputs) != 1
            or outputs[0].shape != [1, 5, 8400]
            or outputs[0].type != "tensor(float)"
        ):
            raise RuntimeError("Unexpected face detector model signature")
        return session

    def prepare(self):
        """Verify/load the local model once; caller serializes initialization."""
        if self._session is None:
            self._session = self._load_session()

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Return all retained (x, y, width, height) boxes in original pixels."""
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.size == 0
        ):
            raise ValueError("Expected nonempty uint8 BGR image")
        self.prepare()
        height, width = image.shape[:2]
        length = max(height, width)
        # Resize before padding: original-resolution square padding can exhaust
        # memory for long narrow frames even when their pixel count is bounded.
        # Keep top-left placement, black padding and the author's RGB /255 input.
        resized_width = max(1, round(width * 640 / length))
        resized_height = max(1, round(height * 640 / length))
        padded = np.zeros((640, 640, 3), dtype=np.uint8)
        padded[:resized_height, :resized_width] = cv2.resize(image, (resized_width, resized_height))
        # YOLO trả về tọa độ tâm bbox, còn OpenCV NMS cần tọa độ góc trên bên trái và kích thước.
        blob = cv2.dnn.blobFromImage(padded, 1 / 255, (640, 640), swapRB=True)
        output = self._session.run(None, {self._session.get_inputs()[0].name: blob})[0]
        if (
            output.shape != (1, 5, 8400)
            or not np.isfinite(output).all()
            or np.any(output[:, 4, :] < 0)
            or np.any(output[:, 4, :] > 1)
        ):
            raise RuntimeError("Invalid face detector output")
        predictions = output[0].T
        candidates = predictions[predictions[:, 4] >= CONFIDENCE_THRESHOLD]
        boxes, scores = [], []
        scale_x, scale_y = width / resized_width, height / resized_height
        for cx, cy, box_width, box_height, confidence in candidates:
            if box_width <= 0 or box_height <= 0:
                raise RuntimeError("Invalid face detector bounding box")
            boxes.append(
                [
                    (float(cx) - float(box_width) / 2) * scale_x,
                    (float(cy) - float(box_height) / 2) * scale_y,
                    float(box_width) * scale_x,
                    float(box_height) * scale_y,
                ]
            )
            scores.append(float(confidence))
        retained = []
        indices = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD)
        for index in np.asarray(indices, dtype=int).reshape(-1):
            x, y, box_width, box_height = boxes[index]
            left, top = max(0, round(x)), max(0, round(y))
            right = min(width, round(x + box_width))
            bottom = min(height, round(y + box_height))
            if right > left and bottom > top:
                retained.append((left, top, right - left, bottom - top))
        return np.asarray(retained, dtype=np.int32).reshape(-1, 4)
