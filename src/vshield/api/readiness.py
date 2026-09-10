"""Read-only readiness diagnostics without paths, vectors, or implicit model downloads."""

import sqlite3
from contextlib import closing
from pathlib import Path

from vshield.api import database


def readiness(service):
    db_ready, revision = False, None
    try:
        uri = Path(database.get_db_path()).resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.execute("SELECT id FROM users LIMIT 1").fetchall()
            conn.execute("SELECT token_hash FROM sessions LIMIT 0").fetchall()
            revision = conn.execute("SELECT revision FROM identity_revision WHERE id=1").fetchone()[0]
            db_ready = True
    except Exception:
        pass
    pad_ready = getattr(getattr(service, "anti_spoof_model", None), "ready", False) is True
    facenet_ready = getattr(getattr(service, "face_embedder", None), "ready", False) is True
    index = getattr(service, "identity_index", None)
    index_ready = (getattr(index, "available", False) is True
                   and getattr(index, "_revision", None) == revision and db_ready)
    components = {"pad": pad_ready, "facenet": facenet_ready,
                  "identity_index": index_ready, "database": db_ready}
    return {"ready": all(components.values()), "components": components,
            "facenet_state": "LOADED" if facenet_ready else "NOT_LOADED"}
