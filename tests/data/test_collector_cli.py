"""Tests for real/fake data-collector CLI routing."""

from pathlib import Path

import pytest

from vshield.data import collector


@pytest.mark.parametrize(
    ("class_id", "folder"),
    [(0, "fake"), (1, "real")],
)
def test_main_routes_class_and_output(monkeypatch, class_id, folder):
    captured = {}

    class FakeCollector:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr(collector, "DataCollector", FakeCollector)
    output_dir = Path("data/DataCollect") / folder

    collector.main(["--class-id", str(class_id), "--output", str(output_dir)])

    assert captured == {
        "class_id": class_id,
        "output_dir": str(output_dir),
        "blur_threshold": collector._BLUR_THRESHOLD,
        "confidence": collector._CONFIDENCE,
        "ran": True,
    }


def test_parse_args_rejects_unknown_class():
    with pytest.raises(SystemExit):
        collector._parse_args(["--class-id", "2", "--output", "data/DataCollect/invalid"])


@pytest.mark.parametrize("argv", [[], ["--class-id", "0"], ["--output", "fake"]])
def test_parse_args_requires_class_and_output(argv):
    with pytest.raises(SystemExit):
        collector._parse_args(argv)
