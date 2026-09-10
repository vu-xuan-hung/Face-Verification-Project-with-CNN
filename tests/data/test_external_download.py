"""Bounded network behavior without live-network dependency in tests."""

import pytest

from vshield.data.external_download import acquire_source_evidence, bounded_get


class Response:
    is_redirect = False
    status_code = 200

    def __init__(self, chunks, headers=None):
        self.data = b"".join(chunks)
        self.headers = headers or {}
        self.raw = self
        self.read_sizes = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        return None

    def read(self, size, decode_content=True):
        self.read_sizes.append(size)
        result, self.data = self.data[:size], self.data[size:]
        return result


def test_streaming_cap_applies_without_content_length(monkeypatch):
    monkeypatch.setattr("vshield.data.external_download.requests.get", lambda *a, **k: Response([b"123", b"456"]))
    with pytest.raises(ValueError, match="exhausted"):
        bounded_get("https://drive.google.com/file", 5)


def test_content_length_limit_precedes_body(monkeypatch):
    monkeypatch.setattr("vshield.data.external_download.requests.get", lambda *a, **k: Response([], {"Content-Length": "999"}))
    with pytest.raises(ValueError, match="exceeds"):
        bounded_get("https://drive.google.com/file", 5)


@pytest.mark.parametrize("url", ["http://drive.google.com/a", "https://evil.test/a", "https://user@drive.google.com/a"])
def test_only_official_https(url):
    with pytest.raises(ValueError, match="official HTTPS"):
        bounded_get(url, 10)


def test_exact_byte_budget(monkeypatch):
    monkeypatch.setattr("vshield.data.external_download.requests.get", lambda *a, **k: Response([b"12345"], {"Content-Length": "5"}))
    data, details = bounded_get("https://drive.google.com/file", 5)
    assert data == b"12345"
    assert details["status"] == 200


def test_cross_host_redirect_blocked(monkeypatch):
    response = Response([], {"Location": "https://evil.test/a"})
    response.is_redirect = True
    monkeypatch.setattr("vshield.data.external_download.requests.get", lambda *a, **k: response)
    with pytest.raises(ValueError, match="official HTTPS"):
        bounded_get("https://drive.google.com/file", 10)


def test_failure_consumption_is_charged_without_overread(monkeypatch):
    response = Response([b"x" * 100])
    consumed = []
    monkeypatch.setattr("vshield.data.external_download.requests.get", lambda *a, **k: response)
    with pytest.raises(ValueError, match="exhausted"):
        bounded_get("https://drive.google.com/file", 7, consume=consumed.append)
    assert sum(consumed) == 7
    assert response.read_sizes == [7]
    assert len(response.data) == 93


def test_failed_resource_exhausts_shared_acquisition_budget(monkeypatch, tmp_path):
    responses = []

    def request(*args, **kwargs):
        response = Response([b"x" * 5000])
        responses.append(response)
        return response

    monkeypatch.setattr("vshield.data.external_download.requests.get", request)
    report = acquire_source_evidence(tmp_path / "acquisition", max_bytes=4096)
    assert report["downloaded_bytes"] == 4096
    assert report["saved_bytes"] == 0
    assert len(responses) == 1  # Failed first payload leaves zero budget for retries/resources.
    assert responses[0].read_sizes == [4096]
    assert report["status"] == "blocked"


def test_interrupted_read_conservatively_reserves_consumed_budget(monkeypatch, tmp_path):
    from urllib3.exceptions import ProtocolError

    response = Response([])

    def broken_read(size, decode_content=True):
        raise ProtocolError("connection interrupted during body read")

    response.read = broken_read
    requests_made = []

    def request(*args, **kwargs):
        requests_made.append(args)
        return response

    monkeypatch.setattr("vshield.data.external_download.requests.get", request)
    report = acquire_source_evidence(tmp_path / "acquisition", max_bytes=4096)
    assert report["downloaded_bytes"] == 4096  # Conservative reservation, not saved data.
    assert report["saved_bytes"] == 0
    assert len(requests_made) == 1
