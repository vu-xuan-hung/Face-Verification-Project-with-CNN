"""SQLite accounts and login audit; permissions never come from the client."""

import re
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from vshield.api.migrations import migrate_users
from vshield.api.roles import ROLES, normalize_role

_DB_DEFAULT = Path(__file__).resolve().parents[3] / "login_logs.db"


def get_db_path():
    return str(_DB_DEFAULT)


def connect(db_path=None):
    connection = sqlite3.connect(db_path or get_db_path(), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def validate_username(username):
    if not isinstance(username, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", username):
        raise ValueError("Username must be lowercase ASCII, start with a letter, max 64 characters")
    if username in {
        "con",
        "prn",
        "aux",
        "nul",
        *[f"com{i}" for i in range(10)],
        *[f"lpt{i}" for i in range(10)],
    }:
        raise ValueError("Reserved username")
    return username


def init_db(db_path=None):
    with closing(connect(db_path)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("""CREATE TABLE IF NOT EXISTS login_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL,
            role TEXT NOT NULL, timestamp TEXT NOT NULL)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
            role TEXT NOT NULL DEFAULT 'user', active INTEGER NOT NULL DEFAULT 0,
            enrollment_id TEXT NOT NULL DEFAULT '')""")
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
        # Legacy implicit accounts must be explicitly re-enrolled by the owner.
        if "active" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN active INTEGER NOT NULL DEFAULT 0")
        if "enrollment_id" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN enrollment_id TEXT NOT NULL DEFAULT ''")
        conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY, username TEXT NOT NULL,
            expires_at INTEGER NOT NULL,
            FOREIGN KEY(username) REFERENCES users(username) ON DELETE CASCADE)""")
        conn.execute("CREATE INDEX IF NOT EXISTS sessions_user ON sessions(username)")
        migrate_users(conn)


def get_account(username, db_path=None):
    with closing(connect(db_path)) as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username=?",
            (username,),
        ).fetchone()
    return dict(row) if row else None


def get_account_by_id(user_id, db_path=None):
    with closing(connect(db_path)) as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return dict(row) if row else None


def gallery_revision(db_path=None):
    with closing(connect(db_path)) as conn:
        return conn.execute("SELECT revision FROM identity_revision WHERE id=1").fetchone()[0]


def bump_gallery_revision(conn):
    conn.execute("UPDATE identity_revision SET revision=revision+1 WHERE id=1")


def get_role(username, db_path=None):
    account = get_account(username, db_path)
    if (
        account
        and account["active"] == 1
        and account["status"] == "ACTIVE"
        and account["role"] in ROLES
    ):
        return account["role"]
    return None


def list_accounts(db_path=None):
    with closing(connect(db_path)) as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM users ORDER BY username")]


def register_user(
    username, role="user", db_path=None, *, enrollment_id="", validate_enrollment=None
):
    """Trusted local operation, never exposed as anonymous HTTP enrollment."""
    validate_username(username)
    role = normalize_role(role)
    with closing(connect(db_path)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        if validate_enrollment is not None:
            validate_enrollment()
        existing = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if existing and (
            existing["enrollment_id"]
            or existing["active"]
            or existing["status"] == "DELETED"
            or existing["role"] == "SUPER_ADMIN"
        ):
            raise ValueError("Account already exists")
        conn.execute(
            """INSERT INTO users(username,role,active,enrollment_id,status,name,created_at,updated_at)
            VALUES(?,?,1,?,'ACTIVE',?,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)
            ON CONFLICT(username) DO UPDATE SET role=excluded.role, active=1,status='ACTIVE',
            enrollment_id=excluded.enrollment_id,updated_at=CURRENT_TIMESTAMP""",
            (username, role, enrollment_id, username),
        )
        conn.execute("DELETE FROM sessions WHERE username=?", (username,))
        bump_gallery_revision(conn)


def change_account(username, *, role=None, disable=False, db_path=None):
    validate_username(username)
    if role is not None:
        role = normalize_role(role)
        if role == "SUPER_ADMIN":
            raise ValueError("Use explicit offline super-admin provisioning")
    with closing(connect(db_path)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        account = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if not account:
            raise ValueError("Unknown account")
        if account["role"] == "SUPER_ADMIN" or account["status"] == "DELETED":
            raise PermissionError("Protected account")
        if role is not None:
            conn.execute("UPDATE users SET role=? WHERE username=?", (role, username))
        if disable:
            conn.execute(
                "UPDATE users SET active=0,status='DISABLED' WHERE username=?", (username,)
            )
            bump_gallery_revision(conn)
        conn.execute("UPDATE users SET updated_at=CURRENT_TIMESTAMP WHERE username=?", (username,))
        conn.execute("DELETE FROM sessions WHERE username=?", (username,))


def log_login(username, role, db_path=None):
    with closing(connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT INTO login_logs(username,role,timestamp) VALUES(?,?,?)",
            (username, role, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )


def get_logs(username_filter=None, date_filter=None, db_path=None):
    query = "SELECT username, role, timestamp FROM login_logs WHERE 1=1"
    params = []
    if username_filter:
        query += " AND username LIKE ?"
        params.append(f"%{username_filter}%")
    if date_filter:
        query += " AND timestamp LIKE ?"
        params.append(f"{date_filter}%")
    with closing(connect(db_path)) as conn:
        return [dict(row) for row in conn.execute(query + " ORDER BY id DESC", params)]
