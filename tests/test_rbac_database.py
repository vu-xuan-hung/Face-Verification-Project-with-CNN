"""Real SQLite migrations and the role/target authorization matrix."""

import sqlite3
from contextlib import closing

import pytest

from vshield.api import database, user_store
from vshield.api.bootstrap import bootstrap_super_admin


@pytest.fixture
def accounts(tmp_path):
    path = tmp_path / "roles.db"
    database.init_db(path)
    ids = {}
    with closing(database.connect(path)) as conn, conn:
        for username, role in (("root", "SUPER_ADMIN"), ("admin", "ADMIN"), ("user", "USER")):
            ids[role] = user_store.insert_user(
                conn,
                username=username,
                name=username,
                email=None,
                role=role,
                created_by=None,
                enrollment_id=username,
            )["id"]
    return path, ids


def test_additive_migration_preserves_ids_and_revokes_only_once(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE users(id INTEGER PRIMARY KEY, username TEXT UNIQUE, role TEXT,active INTEGER,enrollment_id TEXT)"
        )
        conn.execute("INSERT INTO users VALUES(15,'alice','admin',1,'face-15')")
        conn.execute(
            "CREATE TABLE sessions(token_hash TEXT PRIMARY KEY,username TEXT,expires_at INTEGER)"
        )
        conn.execute("INSERT INTO sessions VALUES('old','alice',9999999999)")
    database.init_db(path)
    user = database.get_account_by_id(15, path)
    assert (user["role"], user["status"], user["enrollment_id"]) == ("ADMIN", "ACTIVE", "face-15")
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
        conn.execute("INSERT INTO sessions VALUES('new','alice',9999999999)")
    database.init_db(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM sessions").fetchone()[0] == 1
    assert not any(row["role"] == "SUPER_ADMIN" for row in database.list_accounts(path))


@pytest.mark.parametrize(
    "actor,role,success",
    [
        ("USER", "USER", False),
        ("USER", "ADMIN", False),
        ("ADMIN", "USER", True),
        ("ADMIN", "ADMIN", False),
        ("SUPER_ADMIN", "USER", True),
        ("SUPER_ADMIN", "ADMIN", True),
        ("ADMIN", "SUPER_ADMIN", False),
        ("SUPER_ADMIN", "SUPER_ADMIN", False),
    ],
)
def test_create_matrix(accounts, actor, role, success):
    path, ids = accounts
    with closing(database.connect(path)) as conn, conn:
        arguments = {
            "username": "newperson",
            "name": "New Person",
            "email": None,
            "role": role,
            "created_by": ids[actor],
            "enrollment_id": "newface",
        }
        if success:
            assert user_store.insert_user(conn, **arguments)["role"] == role
        else:
            with pytest.raises(PermissionError):
                user_store.insert_user(conn, **arguments)


@pytest.mark.parametrize("actor,success", [("ADMIN", False), ("SUPER_ADMIN", True)])
def test_promote_matrix(accounts, actor, success):
    path, ids = accounts
    revision = database.gallery_revision(path)
    if success:
        assert (
            user_store.mutate_user(ids[actor], ids["USER"], role="ADMIN", db_path=path)["role"]
            == "ADMIN"
        )
    else:
        with pytest.raises(PermissionError):
            user_store.mutate_user(ids[actor], ids["USER"], role="ADMIN", db_path=path)
    assert database.gallery_revision(path) == revision


def test_disable_actor_reload_revocation_and_revision(accounts):
    path, ids = accounts
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO sessions VALUES('token','admin',9999999999)")
    revision = database.gallery_revision(path)
    user_store.mutate_user(ids["SUPER_ADMIN"], ids["ADMIN"], status="DISABLED", db_path=path)
    assert database.gallery_revision(path) == revision + 1
    assert database.get_role("admin", path) is None
    with closing(database.connect(path)) as conn:
        assert conn.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
        with pytest.raises(PermissionError):
            user_store.authorize(conn, ids["ADMIN"], create_role="USER")


def test_protected_super_admin_and_scoped_listing(accounts):
    path, ids = accounts
    for actor in ("ADMIN", "SUPER_ADMIN"):
        with pytest.raises(PermissionError):
            user_store.mutate_user(ids[actor], ids["SUPER_ADMIN"], delete=True, db_path=path)
    assert [row["role"] for row in user_store.list_users(ids["ADMIN"], path)] == ["USER"]
    assert len(user_store.list_users(ids["SUPER_ADMIN"], path)) == 3
    with pytest.raises(PermissionError):
        user_store.list_users(ids["USER"], path)


def test_duplicate_deleted_identity_and_email(accounts):
    path, ids = accounts
    user_store.mutate_user(ids["ADMIN"], ids["USER"], email="Some@Example.com", db_path=path)
    user_store.mutate_user(ids["ADMIN"], ids["USER"], delete=True, db_path=path)
    with closing(database.connect(path)) as conn, conn:
        for username, email in (("user", None), ("another", "some@example.com")):
            with pytest.raises(ValueError):
                user_store.insert_user(
                    conn,
                    username=username,
                    name=username,
                    email=email,
                    role="USER",
                    created_by=ids["ADMIN"],
                    enrollment_id="new",
                )
    with pytest.raises(LookupError):
        user_store.mutate_user(ids["SUPER_ADMIN"], ids["USER"], status="ACTIVE", db_path=path)


def test_role_change_revokes_sessions_and_public_record_hides_secrets(accounts):
    path, ids = accounts
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO sessions VALUES('token','user',9999999999)")
    row = user_store.mutate_user(ids["SUPER_ADMIN"], ids["USER"], role="ADMIN", db_path=path)
    assert "password_hash" not in row and "enrollment_id" not in row
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
        assert conn.execute("SELECT old_role,new_role FROM user_management_audit").fetchone() == (
            "USER",
            "ADMIN",
        )


def test_explicit_bootstrap_once_revokes_session_without_gallery_rebuild(tmp_path):
    path = tmp_path / "bootstrap.db"
    database.init_db(path)
    database.register_user("owner", "ADMIN", path, enrollment_id="enrolled")
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO sessions VALUES('token','owner',9999999999)")
    revision = database.gallery_revision(path)
    assert bootstrap_super_admin("owner", path)["role"] == "SUPER_ADMIN"
    assert database.gallery_revision(path) == revision
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
        assert conn.execute("SELECT action FROM user_management_audit").fetchone()[0] == "bootstrap"
        conn.execute("UPDATE users SET status='DISABLED',active=0 WHERE username='owner'")
    with pytest.raises(PermissionError):
        bootstrap_super_admin("owner", path)


@pytest.mark.parametrize("state", ["missing", "unenrolled", "disabled"])
def test_bootstrap_requires_existing_active_enrollment(tmp_path, state):
    path = tmp_path / "bootstrap.db"
    database.init_db(path)
    if state != "missing":
        database.register_user(
            "owner", "ADMIN", path, enrollment_id="" if state == "unenrolled" else "face"
        )
    if state == "disabled":
        database.change_account("owner", disable=True, db_path=path)
    with pytest.raises(LookupError if state == "missing" else ValueError):
        bootstrap_super_admin("owner", path)


@pytest.mark.parametrize(
    "assignment",
    [
        "role='admin'",
        "role='EVIL'",
        "status='UNKNOWN'",
        "active=0",
        "status='DISABLED'",
        "active=2",
    ],
)
def test_database_constraints_reject_invalid_state(accounts, assignment):
    path, _ = accounts
    database.init_db(path)
    with sqlite3.connect(path) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(f"UPDATE users SET {assignment} WHERE username='user'")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO users(username,role,status,active) VALUES('invalid','USER','ACTIVE',0)"
            )
