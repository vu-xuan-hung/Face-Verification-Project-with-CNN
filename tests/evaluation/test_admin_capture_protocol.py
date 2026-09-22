import hashlib

import pytest
from scripts.build_admin_capture_manifests import select_per_second


def test_admin_capture_selection_uses_one_per_second_and_unique_hashes(tmp_path):
    photos = []
    for second, content in ((1785426717, b"a"), (1785426717, b"a"),
                            (1785426718, b"b"), (1785426719, b"c")):
        path = tmp_path / f"{second}{len(photos):04d}.jpg"
        path.write_bytes(content)
        photos.append(path)
    used = {hashlib.sha256(b"a").hexdigest()}
    selected = select_per_second(photos, 1, used)
    assert [_seconds.stem[:10] for _seconds in selected] == ["1785426718", "1785426719"]
    assert len(used) == 3


def test_admin_capture_selection_rejects_invalid_stride(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        select_per_second([], 0, set())
