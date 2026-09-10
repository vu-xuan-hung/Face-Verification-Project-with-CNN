"""
Tests cơ bản cho V-Shield.

Chạy: make test  hoặc  uv run pytest tests/ -v
"""

import os
import sys

import numpy as np

# Thêm src/ vào path để import được vshield
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))


# -------------------------------------------------------------------
# Test verifier.py — who_is_it()
# -------------------------------------------------------------------


def make_fake_embedding(seed=0):
    """Tạo embedding giả ngẫu nhiên để test (không cần model thật)."""
    rng = np.random.RandomState(seed)
    emb = rng.randn(512).astype(np.float32)
    return emb / np.linalg.norm(emb)


def test_who_is_it_finds_correct_user():
    """Nếu query gần giống embedding trong DB thì phải trả đúng tên."""
    from vshield.core.verifier import who_is_it

    hung_emb = make_fake_embedding(seed=1)
    alice_emb = make_fake_embedding(seed=99)  # rất xa hung

    database_faces = {
        "hung": [hung_emb],
        "alice": [alice_emb],
    }

    # Query gần giống hung (thêm noise nhỏ)
    query = hung_emb + np.random.RandomState(42).randn(512).astype(np.float32) * 0.01
    query = query / np.linalg.norm(query)

    result = who_is_it(query, database_faces, threshold=0.9)
    assert result == "hung", f"Expected 'hung', got '{result}'"


def test_who_is_it_returns_unknown_when_far():
    """Query rất xa tất cả embeddings → trả về Unknown."""
    from vshield.core.verifier import who_is_it

    hung_emb = make_fake_embedding(seed=1)
    database_faces = {"hung": [hung_emb]}

    # Query hoàn toàn khác (seed khác xa)
    query = make_fake_embedding(seed=999)

    result = who_is_it(query, database_faces, threshold=0.05)  # threshold nhỏ → dễ reject
    assert result == "Unknown"


def test_who_is_it_empty_database():
    """Database rỗng → luôn trả Unknown."""
    from vshield.core.verifier import who_is_it

    query = make_fake_embedding(seed=1)
    result = who_is_it(query, {})
    assert result == "Unknown"


# -------------------------------------------------------------------
# Test database.py — get_role()
# -------------------------------------------------------------------


def test_get_role_returns_correct_role(tmp_path):
    """Sau khi register_user, get_role() phải trả đúng role."""
    from vshield.api.database import get_role, init_db, register_user

    db_file = str(tmp_path / "test.db")
    init_db(db_path=db_file)

    register_user("hung", role="admin", db_path=db_file)
    register_user("alice", role="user", db_path=db_file)

    assert get_role("hung", db_path=db_file) == "ADMIN"
    assert get_role("alice", db_path=db_file) == "USER"


def test_get_role_denies_unknown(tmp_path):
    """Unknown accounts must never receive implicit access."""
    from vshield.api.database import get_role, init_db

    db_file = str(tmp_path / "test.db")
    init_db(db_path=db_file)

    result = get_role("nguoi_la", db_path=db_file)
    assert result is None


def test_log_and_get_logs(tmp_path):
    """log_login() ghi vào DB, get_logs() đọc được."""
    from vshield.api.database import get_logs, init_db, log_login

    db_file = str(tmp_path / "test.db")
    init_db(db_path=db_file)

    log_login("hung", "admin", db_path=db_file)
    log_login("alice", "user", db_path=db_file)

    logs = get_logs(db_path=db_file)
    assert len(logs) == 2

    # Kiểm tra filter theo username
    filtered = get_logs(username_filter="hung", db_path=db_file)
    assert len(filtered) == 1
    assert filtered[0]["username"] == "hung"
