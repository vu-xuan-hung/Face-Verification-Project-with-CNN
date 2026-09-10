# MiniFASNetV2 pretrained PAD

VShield keeps FaceNet recognition and uses CPU ONNX Runtime for passive single-frame PAD. No training or recognition-gallery re-embedding.

## Provenance

Official [source repository](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/tree/b6d5f04ad78778917853b25c778acef6d5626d15).

- Commit: `b6d5f04ad78778917853b25c778acef6d5626d15`.
- Checkpoint: `resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth`.
- Checkpoint SHA-256: `a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0`.
- Architecture SHA-256: `e498c4ec5e1ddfaba62b941a126c19d65aa564999f3309661fe43ee8bf38acd7`.
- ONNX SHA-256: `04e8890346498741e655adfcb67b48ee2620fc2d251321a8f0ad3262bb842067`.
- Export: CPU eval, opset 17, conv6 kernel `(5,5)`, strict checkpoint load with `weights_only=True`.

Only the reviewed, hash-pinned upstream architecture is imported. Cached source/checkpoint are verified again before import/load. Review upstream code and weight terms for the intended deployment; this project does not assert unrestricted commercial rights.

## Exact input/output contract

Input uses original OpenCV BGR uint8 image plus `(x,y,width,height)` bbox. Follow pinned `generate_patches.py`: crop scale 2.7 reduced to fit image, shift box at borders, integer coordinates, inclusive right/bottom. Resize with OpenCV linear interpolation to 80x80. Transpose HWC to NCHW, batch 1, float32 values 0-255. **No RGB conversion, /255, or mean/std normalization.** Recognition retains its separate crop/alignment.

Evidence: pinned `src/data_io/functional.py` NumPy `to_tensor` returns `img.float()`; `src/anti_spoof_predict.py` applies softmax outside the network; `test.py` identifies index 1 as real.

Graph input `input` is float32 `[1,3,80,80]`; output `logits` is float32 `[1,3]`. Export checks there is no ONNX Softmax. Runtime applies stable softmax once. Index 1 is real; 0/2 are spoof classes, with no unsupported subtype names.

PAD score is the real-class softmax value, not calibrated real-world liveness probability. REAL requires real argmax and threshold. Initial `0.8` is provisional, not optimal or calibrated. All other outcomes deny before FaceNet. This is a single-model policy, not the upstream multi-model ensemble.

## Setup and verification

Production requires CPU `onnxruntime` and the verified artifact/config/evidence, not PyTorch. For export only, install CPU PyTorch and ONNX into the project environment:

```powershell
uv pip install torch==2.14.0+cpu --index https://download.pytorch.org/whl/cpu
uv pip install onnx
uv run --no-sync python scripts/setup-minifasnet.py
uv run --no-sync python scripts/validate-minifasnet.py
```

Setup downloads fixed HTTPS URLs with a body size limit and timeout, checks hardcoded SHA-256, exports a candidate, checks graph, verifies CPU numerical parity, then publishes artifact/evidence. It does not automatically trust a different runtime checksum. If export tool versions produce different bytes, review new evidence and deliberately update config; otherwise runtime fails closed.

Outputs: ignored `artifacts/models/minifasnet-v2.onnx`; numerical evidence `artifacts/models/minifasnet-v2-parity.json`; ignored pinned cache `tmp/minifasnet-export/<commit>/`. Runtime contract is `configs/ai-models.yaml`.

Actually verified 2026-09-10: PyTorch `2.14.0+cpu` versus ONNX Runtime `1.29.0`, four tensor cases (zeros, 255, ramp, seeded noise), logit atol/rtol `1e-4`. Maximum absolute logit error `2.7418136596679688e-6`; maximum softmax error `9.5367431640625e-7`. These are actual checkpoint/ONNX inferences, not mocks. Synthetic tensors establish export mathematics, **not camera accuracy or spoof resistance**.

## Limitations and safety

- Passive RGB PAD does not authenticate camera origin or eliminate replay/deepfake risks.
- Haar is intentionally retained; its bbox distribution differs from reference RetinaFace. Camera/domain validation remains required.
- No real-camera accuracy benchmark or threshold calibration is claimed.
- Legacy enrollments remain unverified, never retroactively marked as PAD-passed.
- Missing/corrupt model, invalid contract/output, load/inference error must deny. Never bypass PAD to restore login.
- Do not commit tracked `login_logs.db` or account/session/biometric data.

Unresolved: deployment camera calibration and weight-license review for intended use.
