"""Unit tests for preprocessing, embedding validation, and identity search."""

from unittest.mock import Mock

import numpy as np
import pytest

from vshield.core import identity_index as identity_index_module
from vshield.core.embedder import EmbeddingError, FaceEmbedder, normalize_embedding
from vshield.core.face_preprocessor import FacePreprocessor, InvalidFaceCountError
from vshield.core.verifier import (
    UNKNOWN_IDENTITY,
    IdentityIndex,
    IdentityIndexError,
    IdentityIndexUnavailableError,
    load_database,
    who_is_it,
)
from vshield.services import authentication


class DetectorStub:
    def __init__(self, boxes):
        self.boxes = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)

    def detectMultiScale(self, *_args, **_kwargs):  # noqa: N802 - OpenCV API
        return self.boxes


def unit_embedding(index: int) -> np.ndarray:
    embedding = np.zeros(512, dtype=np.float32)
    embedding[index] = 1.0
    return embedding


def test_face_preprocessor_requires_exactly_one_face():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    preprocessor = FacePreprocessor(
        face_detector=DetectorStub([]),
        eye_detector=DetectorStub([]),
    )

    with pytest.raises(InvalidFaceCountError):
        preprocessor.extract(image)


def test_face_preprocessor_rejects_multiple_faces():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    preprocessor = FacePreprocessor(
        face_detector=DetectorStub([[10, 10, 50, 50], [100, 10, 50, 50]]),
        eye_detector=DetectorStub([]),
    )

    with pytest.raises(InvalidFaceCountError):
        preprocessor.extract(image)


def test_face_preprocessor_creates_model_specific_crops():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    preprocessor = FacePreprocessor(
        face_detector=DetectorStub([[50, 50, 80, 80]]),
        eye_detector=DetectorStub([]),
    )

    crops = preprocessor.extract(image)

    assert crops.anti_spoof.shape == (128, 128, 3)
    assert crops.facenet.ndim == 3
    assert crops.facenet.size > 0


def test_normalize_embedding_enforces_unit_length():
    vector = unit_embedding(0) * 7.0

    normalized = normalize_embedding(vector)

    assert normalized.dtype == np.float32
    assert np.linalg.norm(normalized) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "embedding",
    [
        np.zeros(512, dtype=np.float32),
        np.zeros(511, dtype=np.float32),
        np.full(512, np.nan, dtype=np.float32),
    ],
)
def test_normalize_embedding_rejects_invalid_vectors(embedding):
    with pytest.raises(EmbeddingError):
        normalize_embedding(embedding)


def test_face_embedder_validates_and_normalizes_model_output():
    model = Mock()
    model.embeddings.return_value = np.array([unit_embedding(1) * 4.0])
    embedder = FaceEmbedder(model=model)

    result = embedder.encode(np.zeros((128, 128, 3), dtype=np.uint8))

    assert np.linalg.norm(result) == pytest.approx(1.0)
    model.embeddings.assert_called_once()


def test_identity_index_finds_exact_match_with_numpy_fallback():
    index = IdentityIndex(
        {"alice": [unit_embedding(0)], "bob": [unit_embedding(1)]},
        prefer_faiss=False,
    )

    match = index.search(unit_embedding(0) * 3.0)

    assert index.backend == "numpy"
    assert match is not None
    assert match.username == "alice"
    assert match.distance == pytest.approx(0.0)


def test_faiss_and_numpy_backends_return_equivalent_match():
    database_faces = {
        "alice": [unit_embedding(0)],
        "bob": [unit_embedding(1)],
    }
    faiss_index = IdentityIndex(database_faces, prefer_faiss=True)
    if faiss_index.backend != "faiss":
        pytest.skip("FAISS is not installed in this test environment")
    numpy_index = IdentityIndex(database_faces, prefer_faiss=False)

    faiss_match = faiss_index.search(unit_embedding(0))
    numpy_match = numpy_index.search(unit_embedding(0))

    assert faiss_match is not None
    assert numpy_match is not None
    assert faiss_match.username == numpy_match.username
    assert faiss_match.distance == pytest.approx(numpy_match.distance)


def test_identity_index_rejects_far_query():
    index = IdentityIndex(
        {"alice": [unit_embedding(0)]},
        distance_threshold=0.5,
        prefer_faiss=False,
    )

    assert index.search(unit_embedding(2)) is None


def test_identity_index_rejects_ambiguous_match():
    query = normalize_embedding(unit_embedding(0) + unit_embedding(1))
    index = IdentityIndex(
        {"alice": [unit_embedding(0)], "bob": [unit_embedding(1)]},
        distance_threshold=1.0,
        min_margin=0.1,
        prefer_faiss=False,
    )

    assert index.search(query) is None


def test_identity_index_checks_runner_up_when_one_identity_crowds_top_k():
    query = unit_embedding(0)
    alice_embeddings = [
        normalize_embedding(query + unit_embedding(index) * (index / 1000))
        for index in range(1, 10)
    ]
    bob_embedding = normalize_embedding(query + unit_embedding(20) * 0.02)
    index = IdentityIndex(
        {"alice": alice_embeddings, "bob": [bob_embedding]},
        distance_threshold=1.0,
        min_margin=0.05,
        search_k=2,
        prefer_faiss=False,
    )

    assert index.search(query) is None


def test_identity_index_falls_back_when_faiss_search_fails():
    index = IdentityIndex({"alice": [unit_embedding(0)]}, prefer_faiss=False)
    failing_faiss = Mock()
    failing_faiss.search.side_effect = RuntimeError("faiss failed")
    index._faiss_index = failing_faiss

    match = index.search(unit_embedding(0))

    assert match is not None
    assert match.username == "alice"


def test_identity_index_falls_back_when_faiss_initialization_fails(monkeypatch):
    broken_faiss = Mock()
    broken_faiss.IndexFlatL2.side_effect = OSError("incompatible FAISS DLL")
    monkeypatch.setattr(identity_index_module, "import_faiss", lambda: broken_faiss)

    index = IdentityIndex({"alice": [unit_embedding(0)]})

    assert index.backend == "numpy"
    assert index.search(unit_embedding(0)) is not None


def test_identity_index_rejects_invalid_enrollment_embedding():
    with pytest.raises(IdentityIndexError):
        IdentityIndex({"alice": [np.zeros(511, dtype=np.float32)]})


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("distance_threshold", np.nan),
        ("distance_threshold", np.inf),
        ("distance_threshold", 2.1),
        ("min_margin", np.nan),
        ("min_margin", np.inf),
        ("min_margin", 2.1),
    ],
)
def test_identity_index_rejects_invalid_match_configuration(argument, value):
    with pytest.raises(ValueError):
        IdentityIndex(
            {"alice": [unit_embedding(0)]},
            prefer_faiss=False,
            **{argument: value},
        )


def test_empty_identity_index_is_unavailable():
    index = IdentityIndex({}, prefer_faiss=False)

    with pytest.raises(IdentityIndexUnavailableError):
        index.search(unit_embedding(0))


def test_legacy_who_is_it_keeps_exact_numpy_fallback():
    query = unit_embedding(0)

    assert who_is_it(query, {"alice": [query]}) == "alice"
    assert who_is_it(query, {}) == UNKNOWN_IDENTITY


def test_load_database_requires_username_directories(tmp_path):
    root_image = tmp_path / "ignored.jpg"
    root_image.write_bytes(b"not-used")
    user_dir = tmp_path / "alice"
    user_dir.mkdir()
    user_image = user_dir / "face.jpg"
    user_image.write_bytes(b"encoded-by-stub")
    encoder = Mock(return_value=unit_embedding(0) * 4.0)

    database_faces = load_database(tmp_path, encode_file=encoder)

    assert list(database_faces) == ["alice"]
    assert np.linalg.norm(database_faces["alice"][0]) == pytest.approx(1.0)
    encoder.assert_called_once_with(user_image)


def test_default_service_skips_enrollment_when_anti_spoof_model_is_missing(
    tmp_path,
    monkeypatch,
):
    user_dir = tmp_path / "data" / "faces" / "alice"
    user_dir.mkdir(parents=True)
    (user_dir / "face.jpg").write_bytes(b"not-read")
    embedder = Mock()

    monkeypatch.setattr(authentication, "load_anti_spoofing_model", Mock(return_value=None))
    monkeypatch.setattr(authentication, "FacePreprocessor", Mock(return_value=Mock()))
    monkeypatch.setattr(authentication, "FaceEmbedder", Mock(return_value=embedder))

    service = authentication.build_default_authentication_service(tmp_path)

    assert service.anti_spoof_model is None
    assert not service.identity_index.available
    embedder.encode_file.assert_not_called()
