"""Transactional user management; role decisions always reload the actor from SQLite."""

import sqlite3
from contextlib import closing

from vshield.api import database


def public_user(row):
    fields = (
        "id",
        "username",
        "name",
        "email",
        "role",
        "status",
        "created_at",
        "updated_at",
        "created_by",
    )
    return {key: row[key] for key in fields}


def authorize(conn, actor_id, target_id=None, create_role=None, role_change=False):
    actor = conn.execute("SELECT * FROM users WHERE id=?", (actor_id,)).fetchone()
    if (
        not actor
        or actor["active"] != 1
        or actor["status"] != "ACTIVE"
        or actor["role"] not in {"ADMIN", "SUPER_ADMIN"}
    ):
        raise PermissionError("User management access required")
    if role_change and actor["role"] != "SUPER_ADMIN":
        raise PermissionError("Super-admin access required")
    allowed = {"USER", "ADMIN"} if actor["role"] == "SUPER_ADMIN" else {"USER"}
    if create_role is not None and create_role not in allowed:
        raise PermissionError("Cannot create or assign this role")
    if target_id is None:
        return dict(actor)
    target = conn.execute("SELECT * FROM users WHERE id=?", (target_id,)).fetchone()
    if target is None:
        raise LookupError("Unknown user")
    if target["role"] not in allowed:
        raise PermissionError("Cannot manage this account")
    if target["status"] == "DELETED":
        raise LookupError("User is deleted")
    return dict(target)


def _profile(name, email):
    if name is not None and (not isinstance(name, str) or not name.strip() or len(name) > 200):
        raise ValueError("Name must contain 1-200 characters")
    if email is not None:
        if (
            not isinstance(email, str)
            or len(email) > 254
            or email.count("@") != 1
            or any(c.isspace() for c in email)
        ):
            raise ValueError("Invalid email")
        local, domain = email.split("@")
        if not local or not domain:
            raise ValueError("Invalid email")
        email = email.lower()
    return name.strip() if name is not None else None, email


def insert_user(conn, *, username, name, email, role, created_by, enrollment_id):
    database.validate_username(username)
    if role not in database.ROLES:
        raise ValueError("Invalid role")
    name, email = _profile(name, email)
    # A null actor is reserved for explicitly authorized offline provisioning.
    if created_by is not None:
        authorize(conn, created_by, create_role=role)
    try:
        cursor = conn.execute(
            """INSERT INTO users(username,name,email,role,active,status,enrollment_id,
                created_by,created_at,updated_at)
                VALUES(?,?,?,?,1,'ACTIVE',?,?,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)""",
            (username, name or username, email, role, enrollment_id, created_by),
        )
    except sqlite3.IntegrityError as exc:
        raise ValueError("Username or email already exists, or creator is invalid") from exc
    database.bump_gallery_revision(conn)
    if created_by is not None:
        conn.execute(
            """INSERT INTO user_management_audit(actor_id,target_id,action,new_role)
            VALUES(?,?,'create',?)""",
            (created_by, cursor.lastrowid, role),
        )
    return dict(conn.execute("SELECT * FROM users WHERE id=?", (cursor.lastrowid,)).fetchone())


def list_users(actor_id, db_path=None):
    with closing(database.connect(db_path)) as conn, conn:
        conn.execute("BEGIN")
        actor = authorize(conn, actor_id)
        query = "SELECT * FROM users"
        if actor["role"] == "ADMIN":
            query += " WHERE role='USER'"
        return [public_user(row) for row in conn.execute(query + " ORDER BY id")]


def mutate_user(
    actor_id, user_id, *, name=None, email=None, role=None, status=None, delete=False, db_path=None
):
    with closing(database.connect(db_path)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        target = authorize(
            conn, actor_id, target_id=user_id, create_role=role, role_change=role is not None
        )
        if role is not None and role not in {"USER", "ADMIN"}:
            raise ValueError("Invalid role")
        if status is not None and status not in {"ACTIVE", "DISABLED"}:
            raise ValueError("Invalid status")
        name, email = _profile(name, email)
        changes = {
            key: value
            for key, value in (("name", name), ("email", email), ("role", role), ("status", status))
            if value is not None
        }
        if delete:
            changes["status"] = "DELETED"
        if "status" in changes:
            changes["active"] = int(changes["status"] == "ACTIVE")
        if changes:
            assignments = ",".join(f"{key}=?" for key in changes)
            try:
                conn.execute(
                    f"UPDATE users SET {assignments},updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (*changes.values(), user_id),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("Email already exists") from exc
        if role is not None or status is not None or delete:
            conn.execute("DELETE FROM sessions WHERE username=?", (target["username"],))
        if status is not None or delete:
            database.bump_gallery_revision(conn)
        if changes:
            conn.execute(
                """INSERT INTO user_management_audit(actor_id,target_id,action,old_role,new_role)
                VALUES(?,?,?,?,?)""",
                (
                    actor_id,
                    user_id,
                    "delete" if delete else "update",
                    target["role"],
                    role or target["role"],
                ),
            )
        return public_user(conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone())
