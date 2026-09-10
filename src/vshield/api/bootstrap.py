"""Explicit offline first-super-admin promotion; never exposed over HTTP."""

from contextlib import closing

from vshield.api import database
from vshield.api.user_store import public_user


def bootstrap_super_admin(username, db_path=None):
    database.validate_username(username)
    with closing(database.connect(db_path)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("SELECT 1 FROM users WHERE role='SUPER_ADMIN' LIMIT 1").fetchone():
            raise PermissionError("A super-admin already exists; bootstrap is disabled")
        target = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if target is None:
            raise LookupError("Unknown account")
        if target["active"] != 1 or target["status"] != "ACTIVE" or not target["enrollment_id"]:
            raise ValueError("Bootstrap requires an active, face-enrolled account")
        conn.execute(
            "UPDATE users SET role='SUPER_ADMIN',updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (target["id"],),
        )
        conn.execute("DELETE FROM sessions WHERE username=?", (username,))
        conn.execute(
            """INSERT INTO user_management_audit(actor_id,target_id,action,old_role,new_role)
            VALUES(?,?,'bootstrap',?,'SUPER_ADMIN')""",
            (target["id"], target["id"], target["role"]),
        )
        return public_user(
            conn.execute("SELECT * FROM users WHERE id=?", (target["id"],)).fetchone()
        )
