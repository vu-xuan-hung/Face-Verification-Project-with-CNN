"""Download hash-pinned official weights, export logits ONNX, and prove CPU parity.

No training. Only the reviewed, SHA-256 pinned upstream architecture is imported.
Export dependencies are optional setup tools, not production inference dependencies.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "b6d5f04ad78778917853b25c778acef6d5626d15"
BASE = f"https://raw.githubusercontent.com/minivision-ai/Silent-Face-Anti-Spoofing/{COMMIT}/"
SOURCE = "src/model_lib/MiniFASNet.py"
SOURCE_SHA = "e498c4ec5e1ddfaba62b941a126c19d65aa564999f3309661fe43ee8bf38acd7"
WEIGHTS = "resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
WEIGHTS_SHA = "a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0"
CACHE = ROOT / "tmp/minifasnet-export" / COMMIT
MODEL = ROOT / "artifacts/models/minifasnet-v2.onnx"
REPORT = ROOT / "artifacts/models/minifasnet-v2-parity.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_verified(relative: str, expected: str, destination: Path) -> None:
    if destination.exists():
        if sha256(destination) != expected:
            raise ValueError(f"Cached {destination.name} checksum mismatch; refusing overwrite")
        return
    # Fixed HTTPS host + immutable commit + hash; bounded body and timeout.
    with urllib.request.urlopen(BASE + relative, timeout=60) as response:
        if not response.url.startswith(BASE):
            raise ValueError("Unexpected model download redirect")
        content = response.read(5_000_001)
    if len(content) > 5_000_000 or hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("Official source/checkpoint checksum mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


def reference_model(*, download: bool = False):
    import torch

    source = CACHE / "MiniFASNet.py"
    weights = CACHE / "2.7_80x80_MiniFASNetV2.pth"
    if download:
        fetch_verified(SOURCE, SOURCE_SHA, source)
        fetch_verified(WEIGHTS, WEIGHTS_SHA, weights)
    if sha256(source) != SOURCE_SHA or sha256(weights) != WEIGHTS_SHA:
        raise ValueError("Reference source/weights checksum mismatch")
    spec = importlib.util.spec_from_file_location("vshield_verified_minifasnet", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load verified reference architecture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.MiniFASNetV2(conv6_kernel=(5, 5)).cpu().eval()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def validate(model_path: Path = MODEL) -> dict:
    import numpy as np
    import onnxruntime as ort
    import torch

    torch.set_num_threads(1)
    reference = reference_model()
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(20260910)
    inputs = [
        np.zeros((1, 3, 80, 80), dtype=np.float32),
        np.full((1, 3, 80, 80), 255, dtype=np.float32),
        np.arange(19200, dtype=np.float32).reshape(1, 3, 80, 80) % 256,
        rng.integers(0, 256, (1, 3, 80, 80)).astype(np.float32),
    ]
    errors = []
    probability_errors = []
    with torch.inference_mode():
        for value in inputs:
            expected = reference(torch.from_numpy(value)).numpy()
            actual = session.run(["logits"], {"input": value})[0]
            if actual.shape != (1, 3) or not np.isfinite(actual).all():
                raise ValueError("Invalid ONNX output")
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
            expected_probability = torch.softmax(torch.from_numpy(expected), dim=1).numpy()
            actual_probability = torch.softmax(torch.from_numpy(actual), dim=1).numpy()
            np.testing.assert_allclose(actual_probability, expected_probability, atol=1e-5, rtol=1e-4)
            errors.append(float(np.max(np.abs(actual - expected))))
            probability_errors.append(float(np.max(np.abs(actual_probability - expected_probability))))
    return {
        "passed": True, "model_sha256": sha256(model_path),
        "source_checkpoint_sha256": WEIGHTS_SHA, "source_commit": COMMIT,
        "source_architecture_sha256": SOURCE_SHA,
        "cases": len(inputs), "input_cases": ["zeros", "255", "ramp", "seeded_noise"],
        "max_absolute_error": max(errors), "max_probability_error": max(probability_errors),
        "atol": 1e-4, "rtol": 1e-4,
        "runtime_versions": {"torch": torch.__version__, "onnxruntime": ort.__version__},
        "scope": "numerical export parity only; not camera accuracy or threshold calibration",
    }


def main() -> None:
    import onnx
    import torch

    torch.set_num_threads(1)
    model = reference_model(download=True)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    candidate = MODEL.with_suffix(".candidate.onnx")
    torch.onnx.export(
        model, torch.zeros(1, 3, 80, 80), str(candidate),
        input_names=["input"], output_names=["logits"],
        opset_version=17, dynamo=False,
    )
    graph = onnx.load(str(candidate))
    onnx.checker.check_model(graph)
    if any(node.op_type == "Softmax" for node in graph.graph.node):
        raise ValueError("Expected raw logits, found graph Softmax")
    report = validate(candidate)
    os.replace(candidate, MODEL)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Export verified. Ensure configs/ai-models.yaml checksum matches model_sha256.")


if __name__ == "__main__":
    main()
