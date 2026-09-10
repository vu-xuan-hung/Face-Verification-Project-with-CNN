"""Bounded face template preparation shared by managed enrollment workflows."""

import hashlib

import cv2
import numpy as np

from vshield.core.embedder import normalize_embedding


def prepare_images(images, preprocessor, embedder):
    """Input is decoded BGR images, never caller-supplied embeddings or file paths."""
    if not 2 <= len(images) <= 10:
        raise ValueError("Capture 2-10 face photos")
    templates, contents, seen = [], [], set()
    for image in images:
        crops = preprocessor.extract(image)
        vector = normalize_embedding(embedder.encode(crops.facenet))
        ok, encoded = cv2.imencode(".png", image)
        if not ok or encoded.nbytes > 8 * 1024 * 1024:
            raise ValueError("Canonical photo exceeds 8 MiB")
        content = encoded.tobytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest in seen:
            raise ValueError("Duplicate enrollment photo; capture different views")
        if templates and np.linalg.norm(vector - np.asarray(templates[0]["embedding"])) > 0.9:
            raise ValueError("Enrollment photos do not consistently match the same identity")
        seen.add(digest)
        contents.append(content)
        templates.append(
            {
                "image": f"{len(templates) + 1:02d}.png",
                "sha256": digest,
                "embedding": vector.tolist(),
            }
        )
    return templates, contents


def reject_duplicate_face(templates, gallery):
    """Conservative duplicate gate using the existing unit-vector L2 threshold."""
    vectors = [np.asarray(item["embedding"], dtype=np.float32) for item in templates]
    for stored in gallery.values():
        if any(
            np.linalg.norm(query - reference) <= 0.9 for query in vectors for reference in stored
        ):
            # Do not expose the matching person's identity to the enrolling actor.
            raise ValueError("Face is already enrolled to an existing account")
