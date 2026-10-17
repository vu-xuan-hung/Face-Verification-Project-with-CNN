# YOLO face detection

Login, managed enrollment, local enrollment and evaluation now use YOLOv8n-Face
through `FacePreprocessor`. Haar is retained only for optional eye alignment,
not face detection. The standalone legacy data collector is a separate pipeline.

## Model and setup

- Source: [Lindevs YOLOv8-Face](https://github.com/lindevs/yolov8-face).
- File: `artifacts/models/yolov8n-face-lindevs.onnx` (ignored by Git).
- SHA-256: `8d0bfb0c3383c5bd7a78dd24ef79a21e2aa456619b6ab5e53867092d1c7dc414`.
- Runtime: existing ONNX Runtime CPU dependency; no Ultralytics/PyTorch needed.
- Input: float32 RGB /255, `[1,3,640,640]`; output: `[1,5,8400]`.
- Confidence threshold: 0.75; NMS IoU threshold: 0.7. These are provisional.

Run from the project root on a new installation:

```powershell
.\.venv\Scripts\python.exe scripts/setup-yolo-face.py
```

The setup command downloads only model weights, with timeout, size limit and
checksum verification. Runtime never downloads weights and never falls back to
Haar. Restart the backend after updating code. The current workspace already
contains the verified model.

Frames are resized with preserved aspect ratio, then padded at the bottom/right
to 640 square with zeros. Resizing before padding bounds allocation for narrow
images; rounded resized dimensions are used to map boxes back to original pixels.
This introduces small interpolation differences versus the author's example,
which pads at full image resolution before resizing.

## Behavior

The model loads once on the first detection or readiness request. Initialization
and inference are serialized by the preprocessor lock. Checksum, graph signature,
finite outputs and bbox sizes are checked. Missing/corrupt weights or inference
errors deny authentication with `MODEL_ERROR`; `/health/ready` includes `face_detector`.

No face gives `NO_FACE`; multiple retained faces give `MULTIPLE_FACES`. There is
no largest-box bypass. PAD receives the original BGR frame and selected original
pixel bbox; FaceNet keeps its separate expanded/aligned crop. PAD must still
return REAL before recognition. Enrollment uses the same preprocessor.

## Validation and migration limits

The production preprocessor was exercised on seven supplied photos and their
seven approximate login resize/JPEG variants: each yielded a foreground face
crop and a valid `[1,3,80,80]` PAD input. This verifies detection/crop plumbing,
not successful liveness, recognition or camera authentication. In photo 01 a
partly occluded background face is missed at the current threshold: one detected
face does not prove there is only one person.

Stored enrollment vectors are preserved and may have been produced using Haar
crops. Validate real-camera matching and PAD behavior after the change; existing
users may need fresh enrollment. No retraining, automatic re-enrollment or
threshold recalibration is performed. Evaluation cache fingerprints now include
the YOLO implementation and model bytes.
