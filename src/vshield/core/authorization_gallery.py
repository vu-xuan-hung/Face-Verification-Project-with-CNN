"""Load only explicit, consented account enrollments; never infer image identities."""

import hashlib
import json
from pathlib import Path

from vshield.api import database
from vshield.core.embedder import normalize_embedding
from vshield.core.identity_index_support import IdentityIndexError

EMBEDDING_CONTRACT = "facenet-512-l2-bgr-v1"


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError("Enrollment path escapes private gallery")
    return target


def load_authorization_gallery(root, db_path=None, *, include_inactive=False, by_user_id=False):
    root = Path(root).resolve()
    result = {}
    for account in database.list_accounts(db_path):
        if (not include_inactive and account["active"] != 1) or account[
            "role"
        ] not in database.ROLES:
            continue
        if account.get("status") == "DELETED":
            continue
        if not account["enrollment_id"]:
            continue
        try:
            username = database.validate_username(account["username"])
            folder = inside(root, username)
            manifest_path = inside(folder, "enrollment.json")
            if manifest_path.stat().st_size > 2_000_000:
                raise ValueError("Oversized enrollment manifest")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest["username"] != username
                or manifest["enrollment_id"] != account["enrollment_id"]
                or manifest["contract"] != EMBEDDING_CONTRACT
                or manifest["consent"] is not True
            ):
                raise ValueError("Enrollment association/consent/contract mismatch")
            templates = manifest["templates"]
            if not 1 <= len(templates) <= 20:
                raise ValueError("Enrollment needs 1-20 templates")
            vectors = []
            for template in templates:
                path = inside(folder, template["image"])
                if path.stat().st_size > 8 * 1024 * 1024:
                    raise ValueError("Oversized enrollment image")
                if file_digest(path) != template["sha256"]:
                    raise ValueError("Enrollment image checksum mismatch")
                vectors.append(normalize_embedding(template["embedding"]))
            if "user_id" in manifest and manifest["user_id"] != account["id"]:
                raise ValueError("Enrollment user ID mismatch")
            result[str(account["id"]) if by_user_id else username] = vectors
        except Exception as exc:
            # Do not silently remove a competing identity from the runner-up gate.
            raise IdentityIndexError(
                "Invalid active enrollment; repair gallery before startup"
            ) from exc
    return result
