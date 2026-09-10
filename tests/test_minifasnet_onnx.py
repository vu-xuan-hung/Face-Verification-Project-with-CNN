"""Unit contracts and explicit real-artifact smoke test (never a mocked integration)."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from vshield.core.minifasnet_onnx import MiniFASNetONNX
from vshield.core.model_registry import (
    DEFAULT_CONFIG,
    SOURCE_COMMIT,
    ModelContractError,
    load_pad_contract,
)
from vshield.core.pad_preprocessing import prepare_pad_input


@pytest.fixture
def contract_files(tmp_path):
    folder = tmp_path / "artifacts/models"
    folder.mkdir(parents=True)
    model = folder / "pad.onnx"
    model.write_bytes(b"unit-test-artifact-not-production")
    checksum = hashlib.sha256(model.read_bytes()).hexdigest()
    report = {
        "passed": True,
        "model_sha256": checksum,
        "source_commit": SOURCE_COMMIT,
        "source_checkpoint_sha256": "a" * 64,
        "cases": 3,
        "atol": 1e-4,
        "rtol": 1e-4,
        "max_absolute_error": 1e-6,
    }
    (folder / "parity.json").write_text(json.dumps(report))
    config = {
        "model_name": "MiniFASNetV2",
        "model_version": "unit-test",
        "runtime": "onnxruntime_cpu",
        "input_size": [80, 80],
        "input_layout": "NCHW",
        "input_color_space": "BGR",
        "input_dtype": "float32",
        "normalization": "none",
        "output_type": "logits",
        "output_classes": ["spoof_0", "real", "spoof_2"],
        "real_class_index": 1,
        "crop_scale": 2.7,
        "threshold": 0.8,
        "source_commit": SOURCE_COMMIT,
        "checksum": checksum,
        "source_checkpoint_sha256": "a" * 64,
        "verified_contract": True,
        "model_path": "artifacts/models/pad.onnx",
        "parity_report": "artifacts/models/parity.json",
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"anti_spoof": config}))
    return tmp_path, path, config


def test_verified_contract(contract_files):
    root, path, _ = contract_files
    assert load_pad_contract(path, project_root=root).threshold == 0.8


@pytest.mark.parametrize("field,value", [
    ("normalization", "divide_255"), ("real_class_index", 4),
    ("verified_contract", False), ("checksum", "0" * 64),
    ("input_color_space", "RGB"), ("output_type", "probabilities"),
    ("threshold", float("nan")), ("threshold", 0.5),
    ("model_path", "../outside.onnx"), ("runtime", "onnxruntime-gpu"),
])
def test_invalid_contract_closed(contract_files, field, value):
    root, path, config = contract_files
    config[field] = value
    path.write_text(yaml.safe_dump({"anti_spoof": config}))
    with pytest.raises(ModelContractError, match="unavailable or invalid"):
        load_pad_contract(path, project_root=root)


def test_missing_model_closed(contract_files):
    root, path, _ = contract_files
    (root / "artifacts/models/pad.onnx").unlink()
    with pytest.raises(ModelContractError):
        load_pad_contract(path, project_root=root)


def test_parity_evidence_required(contract_files):
    root, path, _ = contract_files
    (root / "artifacts/models/parity.json").write_text('{"passed": false}')
    with pytest.raises(ModelContractError):
        load_pad_contract(path, project_root=root)


def test_crop_bgr_nchw_without_normalization():
    image = np.full((200, 240, 3), [17, 83, 255], dtype=np.uint8)
    batch = prepare_pad_input(image, (80, 60, 50, 60))
    assert batch.shape == (1, 3, 80, 80)
    assert batch.dtype == np.float32
    np.testing.assert_array_equal(batch[0, :, 0, 0], [17, 83, 255])


@pytest.mark.parametrize("bbox", [(0, 0, 0, 2), (-1, 0, 2, 2), (0, 0, 1000, 2),
                                  (0., 0., 2., 2.), (0, 0, 2), (0, 0, np.nan, 2)])
def test_invalid_bbox(bbox):
    with pytest.raises(ValueError):
        prepare_pad_input(np.zeros((100, 100, 3), dtype=np.uint8), bbox)


def fake_adapter(monkeypatch, contract_files, output):
    import onnxruntime as ort
    root, path, _ = contract_files
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="input", type="tensor(float)", shape=[1, 3, 80, 80])],
        get_outputs=lambda: [SimpleNamespace(name="output", type="tensor(float)", shape=[1, 3])],
        get_providers=lambda: ["CPUExecutionProvider"],
        run=lambda *args: [output],
    )
    calls = []
    def factory(*args, **kwargs):
        calls.append(kwargs)
        return session
    monkeypatch.setattr(ort, "InferenceSession", factory)
    return MiniFASNetONNX(load_pad_contract(path, project_root=root)), calls


@pytest.mark.parametrize("logits,status", [([0, 6, 0], "REAL"), ([6, 0, 0], "FAKE"),
                                             ([0, 0.2, 0], "UNCERTAIN")])
def test_decisions_and_session_reuse(monkeypatch, contract_files, logits, status):
    adapter, calls = fake_adapter(monkeypatch, contract_files, np.array([logits], dtype=np.float32))
    for _ in range(2):
        result = adapter.predict(image=np.zeros((100, 100, 3), dtype=np.uint8), bbox=(20, 20, 40, 40))
        assert result.status.value == status
        assert result.is_real == (status == "REAL")
        assert 0 <= result.score <= 1
    assert calls == [{"providers": ["CPUExecutionProvider"]}]


@pytest.mark.parametrize("output", [np.zeros((1, 2), dtype=np.float32),
    np.array([[0, np.nan, 0]], dtype=np.float32), np.array([[0, np.inf, 0]], dtype=np.float32),
    np.array([[0, 1, 0]], dtype=np.int64), np.array([[0, 1, 0]], dtype=np.complex64)])
def test_invalid_outputs_closed(monkeypatch, contract_files, output):
    adapter, _ = fake_adapter(monkeypatch, contract_files, output)
    result = adapter.predict(image=np.zeros((100, 100, 3), dtype=np.uint8), bbox=(20, 20, 40, 40))
    assert result.status.value == "ERROR"
    assert result.reason == "MODEL_ERROR"


def test_inference_exception_closed(monkeypatch, contract_files):
    adapter, _ = fake_adapter(monkeypatch, contract_files, np.zeros((1, 3), dtype=np.float32))
    def broken(*args):
        raise RuntimeError("private model path and stack")
    adapter._session.run = broken
    result = adapter.predict(image=np.zeros((100, 100, 3), dtype=np.uint8), bbox=(20, 20, 40, 40))
    assert result.status.value == "ERROR"
    assert result.reason == "MODEL_ERROR"
    assert result.score is None


def test_invalid_input_never_runs_session(monkeypatch, contract_files):
    adapter, _ = fake_adapter(monkeypatch, contract_files, np.zeros((1, 3), dtype=np.float32))
    def forbidden(*args):
        pytest.fail("invalid input must not run inference")
    adapter._session.run = forbidden
    result = adapter.predict(image=np.zeros((100, 100, 3), dtype=np.float32), bbox=(20, 20, 40, 40))
    assert result.reason == "INVALID_INPUT"


def test_invalid_model_signature_closed(monkeypatch, contract_files):
    import onnxruntime as ort
    root, path, _ = contract_files
    monkeypatch.setattr(ort, "InferenceSession", lambda *a, **k: SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(type="tensor(float)", shape=[1, 80, 80, 3])],
        get_outputs=lambda: [SimpleNamespace(type="tensor(float)", shape=[1, 3])]))
    with pytest.raises(ModelContractError, match="runtime unavailable"):
        MiniFASNetONNX(load_pad_contract(path, project_root=root))


def test_real_artifact_inference():
    if not DEFAULT_CONFIG.exists():
        pytest.skip("MiniFASNet configuration/artifact not installed")
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())["anti_spoof"]
    from vshield.core.model_registry import PROJECT_ROOT
    if not (PROJECT_ROOT / config["model_path"]).exists():
        pytest.skip("Trusted ONNX artifact not installed; real inference not executable")
    adapter = MiniFASNetONNX.from_config()
    result = adapter.predict(image=np.full((180, 180, 3), 127, dtype=np.uint8), bbox=(50, 50, 60, 60))
    assert result.status.value in {"REAL", "FAKE", "UNCERTAIN"}
    assert np.isfinite(result.score)
