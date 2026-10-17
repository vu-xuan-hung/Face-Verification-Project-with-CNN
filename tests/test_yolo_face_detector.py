"""YOLO image contract, bbox decoding, integrity, and preprocessing wiring."""

from unittest.mock import Mock

import numpy as np
import pytest

from vshield.core.face_preprocessor import FacePreprocessor, InvalidFaceCountError
from vshield.core.yolo_face_detector import YoloFaceDetector


def detector_with_predictions(*predictions):
    output = np.zeros((1, 5, 8400), dtype=np.float32)
    for index, prediction in enumerate(predictions):
        output[0, :, index] = prediction
    detector = YoloFaceDetector()
    detector._session = Mock()
    detector._session.get_inputs.return_value = [Mock()]
    detector._session.get_inputs.return_value[0].name = 'images'
    detector._session.run.return_value = [output]
    return detector


def test_rgb_normalization_and_top_left_zero_padding():
    detector = detector_with_predictions()
    image = np.full((320, 640, 3), [10, 20, 30], dtype=np.uint8)
    assert detector.detect(image).shape == (0, 4)
    blob = detector._session.run.call_args.args[1]['images']
    assert blob.dtype == np.float32
    assert blob.shape == (1, 3, 640, 640)
    np.testing.assert_allclose(blob[0, :, 0, 0], np.array([30, 20, 10]) / 255)
    assert not blob[:, :, 320:, :].any()


def test_decode_scale_and_suppress_duplicates():
    detector = detector_with_predictions([320, 200, 200, 100, .9],
                                        [321, 201, 200, 100, .85], [50, 50, 20, 20, .4])
    np.testing.assert_array_equal(detector.detect(np.zeros((640, 1280, 3), dtype=np.uint8)),
                                  [[440, 300, 400, 200]])


def test_clip_and_ignore_padding_detections():
    detector = detector_with_predictions([10, 20, 100, 100, .9], [500, 500, 50, 50, .9])
    np.testing.assert_array_equal(detector.detect(np.zeros((320, 640, 3), dtype=np.uint8)),
                                  [[0, 0, 60, 70]])


def test_two_faces_are_retained_and_rejected():
    detector = detector_with_predictions([100, 100, 80, 80, .9], [400, 100, 80, 80, .95])
    with pytest.raises(InvalidFaceCountError) as error:
        FacePreprocessor(face_detector=detector).extract(np.zeros((640, 640, 3), dtype=np.uint8))
    assert error.value.count == 2


def test_bbox_and_original_image_preserved_for_pad():
    detector = detector_with_predictions([100, 100, 80, 80, .9])
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    crops = FacePreprocessor(face_detector=detector).extract(image)
    assert crops.bbox == (60, 60, 80, 80)
    assert crops.original_image is image
    assert crops.anti_spoof.shape == (128, 128, 3)
    assert crops.facenet.size > 0


@pytest.mark.parametrize('kind', ['nan', 'shape', 'confidence', 'negative_width'])
def test_malformed_output_fails_closed(kind):
    detector = detector_with_predictions([100, 100, 80, 80, .9])
    output = detector._session.run.return_value[0]
    if kind == 'nan':
        output[0, 0, 0] = np.nan
    elif kind == 'shape':
        detector._session.run.return_value = [np.zeros((1, 6, 8400))]
    elif kind == 'confidence':
        output[0, 4, 0] = 1.1
    else:
        output[0, 2, 0] = -10
    with pytest.raises(RuntimeError):
        detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))


def test_default_preprocessor_uses_yolo():
    assert isinstance(FacePreprocessor().face_detector, YoloFaceDetector)


def test_missing_model_has_no_fallback(tmp_path):
    preprocessor = FacePreprocessor(model_path=tmp_path / 'missing.onnx')
    with pytest.raises(FileNotFoundError):
        preprocessor.extract(np.zeros((64, 64, 3), dtype=np.uint8))
    assert preprocessor.face_detector._session is None


def test_corrupt_model_rejected(tmp_path):
    path = tmp_path / 'corrupt.onnx'
    path.write_bytes(b'corrupt model')
    detector = YoloFaceDetector(path)
    with pytest.raises(RuntimeError, match='checksum'):
        detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))
    assert detector._session is None


def test_runtime_session_reused():
    detector = YoloFaceDetector()
    detector._load_session = Mock(return_value=detector_with_predictions()._session)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    detector.detect(frame)
    detector.detect(frame)
    detector._load_session.assert_called_once_with()


def test_narrow_image_uses_bounded_model_input():
    detector = detector_with_predictions()
    detector.detect(np.zeros((32, 32768, 3), dtype=np.uint8))
    blob = detector._session.run.call_args.args[1]['images']
    assert blob.shape == (1, 3, 640, 640)


def test_missing_model_reports_not_ready(tmp_path):
    assert FacePreprocessor(model_path=tmp_path / 'missing.onnx').ready is False


def test_readiness_includes_unavailable_face_detector(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from vshield.api import readiness as readiness_module

    connection = Mock()
    connection.execute.return_value.fetchone.return_value = [7]
    monkeypatch.setattr(readiness_module.sqlite3, 'connect', Mock(return_value=connection))
    service = SimpleNamespace(
        face_preprocessor=FacePreprocessor(model_path=tmp_path / 'missing.onnx'),
        anti_spoof_model=SimpleNamespace(ready=True), face_embedder=SimpleNamespace(ready=True),
        identity_index=SimpleNamespace(available=True, _revision=7),
    )
    status = readiness_module.readiness(service)
    assert status['components']['face_detector'] is False
    assert all(value for key, value in status['components'].items() if key != 'face_detector')
    assert status['ready'] is False


@pytest.mark.parametrize('image', [np.zeros((10, 10)), np.zeros((0, 10, 3), dtype=np.uint8),
                                   np.zeros((10, 10, 3), dtype=np.float32)])
def test_invalid_input_rejected(image):
    with pytest.raises(ValueError):
        YoloFaceDetector().detect(image)
