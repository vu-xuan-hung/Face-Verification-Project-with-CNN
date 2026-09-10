"""Managed enrollment: server authorization, pretrained embeddings, atomic account publish."""

import json
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from vshield.api import database, user_store
from vshield.core.anti_spoof import build_pad_service
from vshield.core.authorization_gallery import (
    EMBEDDING_CONTRACT,
    inside,
    load_authorization_gallery,
)
from vshield.services.enrollment_images import (
    liveness_provenance,
    prepare_images,
    reject_duplicate_face,
)


class UserManagementService:
    def __init__(self, root, preprocessor, embedder, db_path=None, identity_index=None, pad_service=None):
        self.root = Path(root).resolve()
        self.preprocessor = preprocessor
        self.embedder = embedder
        self.db_path = db_path
        self.identity_index = identity_index
        self.pad_service = build_pad_service() if pad_service is None else pad_service

    def create(self, actor_id, profile, images, role):
        with closing(database.connect(self.db_path)) as conn:
            user_store.authorize(conn, actor_id, create_role=role)
        if profile.get("consent") is not True:
            raise ValueError("Explicit subject consent is required")
        username = database.validate_username(profile["username"])
        destination = inside(inside(self.root, "faces"), username)
        if destination.exists() or database.get_account(username, self.db_path):
            raise ValueError("Username is already registered")
        templates, contents = prepare_images(images, self.preprocessor, self.embedder, self.pad_service)
        enrollment_id = uuid.uuid4().hex
        draft = inside(self.root, f"drafts/{enrollment_id}")
        draft.mkdir(parents=True, exist_ok=False)
        for item, content in zip(templates, contents, strict=True):
            (draft / item["image"]).write_bytes(content)
        published = False
        try:
            with closing(database.connect(self.db_path)) as conn, conn:
                conn.execute("BEGIN IMMEDIATE")
                # Recheck permissions after potentially slow FaceNet inference.
                user_store.authorize(conn, actor_id, create_role=role)
                gallery = load_authorization_gallery(
                    self.root / "faces", self.db_path, include_inactive=True
                )
                reject_duplicate_face(templates, gallery)
                account = user_store.insert_user(
                    conn,
                    username=username,
                    name=profile["name"],
                    email=profile["email"],
                    role=role,
                    created_by=actor_id,
                    enrollment_id=enrollment_id,
                )
                manifest = {
                    "user_id": account["id"],
                    "username": username,
                    "enrollment_id": enrollment_id,
                    "contract": EMBEDDING_CONTRACT,
                    "consent": True,
                    "consented_at": datetime.now(timezone.utc).isoformat(),
                    "templates": templates,
                    **liveness_provenance(templates),
                }
                (draft / "enrollment.json").write_text(json.dumps(manifest), encoding="utf-8")
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise ValueError("Enrollment destination already exists")
                draft.rename(destination)
                published = True
        except Exception:
            if published:
                destination.rename(draft)
            raise
        result = user_store.public_user(account)
        if self.identity_index is not None:
            try:
                self.identity_index.refresh()
                result["vector_sync_status"] = "ready"
            except Exception:
                # Account commit succeeded. Do not pretend it failed and invite
                # duplicate retries; matching retries refresh and fails closed.
                result["vector_sync_status"] = "pending"
        return result
