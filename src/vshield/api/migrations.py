"""Additive, transactional migration of the existing SQLite account database."""


def ensure_user_constraints(conn):
    """Triggers add CHECK-equivalent guards without rebuilding legacy FK tables."""
    for operation in ("INSERT", "UPDATE"):
        conn.execute(f"""CREATE TRIGGER IF NOT EXISTS users_validate_{operation.lower()}
            BEFORE {operation} ON users
            WHEN NEW.role IS NULL OR NEW.role NOT IN ('SUPER_ADMIN','ADMIN','USER')
              OR NEW.status IS NULL OR NEW.status NOT IN ('ACTIVE','DISABLED','DELETED')
              OR NEW.active IS NULL OR NEW.active NOT IN (0,1)
              OR (NEW.status='ACTIVE' AND NEW.active<>1)
              OR (NEW.status<>'ACTIVE' AND NEW.active<>0)
            BEGIN SELECT RAISE(ABORT,'Invalid account role or status'); END""")


def migrate_access_logs(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY)")
    if conn.execute("SELECT 1 FROM schema_migrations WHERE version=2").fetchone():
        return
    conn.execute("""CREATE TABLE IF NOT EXISTS access_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER REFERENCES users(id),
        event_type TEXT NOT NULL,
        result TEXT NOT NULL,
        reason_code TEXT,
        recognition_distance REAL,
        spoof_score REAL,
        pad_status TEXT,
        pad_model_version TEXT,
        request_id TEXT,
        source TEXT,
        timestamp TEXT NOT NULL DEFAULT (datetime('now'))
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_access_logs_timestamp ON access_logs(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_access_logs_user_timestamp ON access_logs(user_id, timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_access_logs_event_timestamp ON access_logs(event_type, timestamp)")
    conn.execute("INSERT OR IGNORE INTO schema_migrations VALUES(2)")


def migrate_users(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY)")
    conn.execute("""CREATE TABLE IF NOT EXISTS identity_revision (
        id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL DEFAULT 0)""")
    conn.execute("INSERT OR IGNORE INTO identity_revision VALUES(1,0)")
    migrate_access_logs(conn)
    if conn.execute("SELECT 1 FROM schema_migrations WHERE version=1").fetchone():
        ensure_user_constraints(conn)
        return
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
    additions = {
        "name": "TEXT",
        "email": "TEXT",
        "password_hash": "TEXT",
        "status": "TEXT NOT NULL DEFAULT 'DISABLED'",
        "created_at": "TEXT",
        "updated_at": "TEXT",
        "created_by": "INTEGER REFERENCES users(id)",
    }
    for name, definition in additions.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE users ADD COLUMN {name} {definition}")
    conn.execute("""UPDATE users SET role=upper(role),
        status=CASE WHEN active=1 AND upper(role) IN ('USER','ADMIN','SUPER_ADMIN')
        THEN 'ACTIVE' ELSE 'DISABLED' END,
        active=CASE WHEN active=1 AND upper(role) IN ('USER','ADMIN','SUPER_ADMIN')
        THEN 1 ELSE 0 END,
        name=coalesce(name,username),created_at=coalesce(created_at,CURRENT_TIMESTAMP),
        updated_at=coalesce(updated_at,CURRENT_TIMESTAMP)""")
    # Unknown legacy roles remain denied; never infer SUPER_ADMIN from ADMIN.
    conn.execute(
        "UPDATE users SET role='USER' WHERE role IS NULL OR role NOT IN ('USER','ADMIN','SUPER_ADMIN')"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS users_email_unique ON users(lower(email)) WHERE email IS NOT NULL"
    )
    conn.execute("""CREATE TABLE IF NOT EXISTS user_management_audit (
        id INTEGER PRIMARY KEY, actor_id INTEGER NOT NULL REFERENCES users(id),
        target_id INTEGER NOT NULL REFERENCES users(id), action TEXT NOT NULL,
        old_role TEXT, new_role TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("DELETE FROM sessions")
    conn.execute("INSERT INTO schema_migrations VALUES(1)")
    ensure_user_constraints(conn)
