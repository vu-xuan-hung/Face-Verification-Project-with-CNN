"""Revocable, hashed opaque bearer sessions and server-side RBAC dependencies."""

import hashlib
import secrets
import sqlite3
import time
from contextlib import closing
from datetime import datetime

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from vshield.api import database

SESSION_SECONDS = 3600
bearer = HTTPBearer(auto_error=False)


def token_hash(token):
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def create_session(username=None, db_path=None, *, user_id=None):
    token = secrets.token_urlsafe(32)
    now = int(time.time())
    with closing(database.connect(db_path)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        if user_id is not None:
            row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        else:
            row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if (
            row is None
            or row["active"] != 1
            or row["status"] != "ACTIVE"
            or row["role"] not in database.ROLES
        ):
            raise PermissionError("Account is not registered or is disabled")
        username = row["username"]
        conn.execute("DELETE FROM sessions WHERE expires_at<=?", (now,))
        conn.execute(
            "INSERT INTO sessions VALUES(?,?,?)",
            (token_hash(token), username, now + SESSION_SECONDS),
        )
        conn.execute(
            "INSERT INTO login_logs(username,role,timestamp) VALUES(?,?,?)",
            (username, row["role"], datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        return {
            "id": row["id"],
            "user_id": row["id"],
            "username": username,
            "name": row["name"],
            "email": row["email"],
            "status": row["status"],
            "role": row["role"],
            "access_token": token,
            "token_type": "bearer",
            "expires_in": SESSION_SECONDS,
        }


def resolve_session(token, db_path=None):
    if not isinstance(token, str) or len(token) != 43 or not token.isascii():
        return None
    with closing(database.connect(db_path)) as conn:
        row = conn.execute(
            """SELECT u.id,u.id AS user_id,u.username,u.name,u.email,u.role,u.status FROM sessions s
            JOIN users u ON u.username=s.username WHERE s.token_hash=? AND s.expires_at>?
            AND u.active=1 AND u.status='ACTIVE'
            AND u.role IN ('USER','ADMIN','SUPER_ADMIN')""",
            (token_hash(token), int(time.time())),
        ).fetchone()
    return dict(row) if row else None


def revoke_session(token, db_path=None):
    with closing(database.connect(db_path)) as conn, conn:
        conn.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash(token),))


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    try:
        user = resolve_session(credentials.credentials) if credentials else None
    except sqlite3.Error as exc:
        raise HTTPException(503, "Session database unavailable") from exc
    if user is None:
        raise HTTPException(401, "Authentication required", headers={"WWW-Authenticate": "Bearer"})
    return user


def require_admin(user=Depends(current_user)):
    if user["role"] not in {"ADMIN", "SUPER_ADMIN"}:
        raise HTTPException(403, "Admin access required")
    return user


def require_super_admin(user=Depends(current_user)):
    if user["role"] != "SUPER_ADMIN":
        raise HTTPException(403, "Super-admin access required")
    return user
