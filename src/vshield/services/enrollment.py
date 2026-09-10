"""Trusted local enrollment with explicit consent and non-overwriting publication."""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from vshield.api import database
from vshield.api.roles import normalize_role
from vshield.core.authorization_gallery import (
    EMBEDDING_CONTRACT,
    file_digest,
    inside,
    load_authorization_gallery,
)
from vshield.core.embedder import FaceEmbedder, normalize_embedding
from vshield.core.face_preprocessor import FacePreprocessor
from vshield.services.enrollment_images import reject_duplicate_face


def enroll(
    root,
    username,
    images,
    *,
    role="user",
    consent=False,
    db_path=None,
    preprocessor=None,
    embedder=None,
):
    """Create a new enrollment. A failed draft is inert and retained for inspection."""
    database.validate_username(username)
    role = normalize_role(role)
    if not consent:
        raise ValueError("Explicit informed consent is required")
    if role not in database.ROLES:
        raise ValueError("Invalid role")
    if role == "SUPER_ADMIN":
        raise ValueError("Enroll as ADMIN, then use guarded super-admin bootstrap")
    if not 1 <= len(images) <= 20:
        raise ValueError("Supply 1-20 explicitly associated face photos")
    root = Path(root).resolve()
    faces_root = inside(root, "faces")
    destination = inside(faces_root, username)
    account = database.get_account(username, db_path)
    if destination.exists() or (account and account["enrollment_id"]):
        raise ValueError("Enrollment already exists; refusing to overwrite biometric data")
    preprocessor = preprocessor or FacePreprocessor()
    embedder = embedder or FaceEmbedder()
    templates = []
    encoded_images = []
    seen = set()
    for image_path in images:
        path = Path(image_path).resolve(strict=True)
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("Photo exceeds 8 MiB")
        with Image.open(path) as header:
            if header.width * header.height > 20_000_000:
                raise ValueError("Photo exceeds 20 megapixels")
        content = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot decode enrollment photo")
        crops = preprocessor.extract(image)
        vector = normalize_embedding(embedder.encode(crops.facenet))
        if templates and np.linalg.norm(vector - np.asarray(templates[0]["embedding"])) > 0.9:
            raise ValueError("Enrollment photos do not consistently match the same identity")
        # Store a canonical image without EXIF and retain the full validated frame.
        ok, encoded = cv2.imencode(".png", image)
        if not ok or encoded.nbytes > 8 * 1024 * 1024:
            raise ValueError("Cannot store photo within canonical size limit")
        digest = hashlib.sha256(encoded.tobytes()).hexdigest()
        if digest in seen:
            raise ValueError("Duplicate enrollment photo")
        seen.add(digest)
        encoded_images.append(encoded.tobytes())
        templates.append(
            {
                "image": f"{len(templates) + 1:02d}.png",
                "sha256": digest,
                "source_sha256": file_digest(path),
                "embedding": vector.tolist(),
            }
        )

    enrollment_id = uuid.uuid4().hex
    reject_duplicate_face(
        templates, load_authorization_gallery(faces_root, db_path, include_inactive=True)
    )
    draft = inside(root, f"drafts/{enrollment_id}")
    draft.mkdir(parents=True, exist_ok=False)
    for template, content in zip(templates, encoded_images, strict=True):
        (draft / template["image"]).write_bytes(content)
    manifest = {
        "username": username,
        "enrollment_id": enrollment_id,
        "contract": EMBEDDING_CONTRACT,
        "consent": True,
        "consented_at": datetime.now(timezone.utc).isoformat(),
        "templates": templates,
    }
    (draft / "enrollment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    faces_root.mkdir(parents=True, exist_ok=True)
    # Path.rename on Windows refuses an existing destination; never use replace here.
    if destination.exists():
        raise ValueError("Enrollment destination appeared during processing")
    draft.rename(destination)
    try:
        database.register_user(
            username,
            role,
            db_path,
            enrollment_id=enrollment_id,
            validate_enrollment=lambda: reject_duplicate_face(
                templates, load_authorization_gallery(faces_root, db_path, include_inactive=True)
            ),
        )
    except Exception:
        # Preserve photos for inspection, but unpublish a failed enrollment so a
        # subsequent retry can use the username. Both paths were checked above.
        destination.rename(draft)
        raise
    return {
        "username": username,
        "role": role,
        "templates": len(templates),
        "restart_required": True,
    }
