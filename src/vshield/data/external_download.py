"""Bounded, official-only source evidence acquisition, without archive extraction."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from urllib3.exceptions import HTTPError as StreamHTTPError

REPOSITORY = "https://raw.githubusercontent.com/ZhangYuanhan-AI/CelebA-Spoof/master/"
FOLDER = "https://drive.google.com/drive/folders/1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z"
FIRST_PART = "https://drive.usercontent.google.com/download?id=1gd0M4qpxzqrnOkCl7hv7vnCWV4JJTFwM&export=download"
ALLOWED_HOSTS = {"raw.githubusercontent.com", "drive.google.com", "drive.usercontent.google.com"}


def bounded_get(url: str, max_bytes: int, *, archive: bool = False,
                consume: Callable[[int], None] | None = None) -> tuple[bytes, dict]:
    """Enforce byte limits even for chunked responses and reject unexpected redirects."""
    if max_bytes <= 0:
        raise ValueError("Positive byte limit required")
    for _ in range(5):
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS or parsed.username:
            raise ValueError("Only explicit official HTTPS download hosts are allowed")
        response = requests.get(url, stream=True, allow_redirects=False, timeout=(10, 30),
                                headers={"Range": f"bytes=0-{max_bytes - 1}"} if archive else {})
        with response:
            if response.is_redirect:
                url = response.headers.get("Location", "")
                continue
            response.raise_for_status()
            size = int(response.headers.get("Content-Length", "0"))
            if size > max_bytes:
                raise ValueError(f"Remote response {size} bytes exceeds limit {max_bytes}")
            content = bytearray()
            while len(content) < max_bytes:
                amount = min(65536, max_bytes - len(content))
                # Reserve first: a broken read may consume bytes before raising.
                if consume:
                    consume(amount)
                chunk = response.raw.read(amount, decode_content=True)
                if consume:
                    consume(len(chunk) - amount)
                if not chunk:
                    break
                content.extend(chunk)
            if len(content) == max_bytes and (not size or response.headers.get("Content-Encoding")):
                # Unknown-length bodies may have more data: fail without an extra read.
                raise ValueError(f"Download exhausted {max_bytes} byte limit before confirmed EOF")
            return bytes(content), {"url": url, "status": response.status_code,
                                    "content_type": response.headers.get("Content-Type", ""),
                                    "content_range": response.headers.get("Content-Range", "")}
    raise ValueError("Too many redirects")


def acquire_source_evidence(output: Path, max_bytes: int = 200_000_000) -> dict:
    """Fetch author metadata documentation, listing and a bounded real archive probe.

    A Drive warning is retained as evidence, never mistaken for training data.
    No confirmation forms/EULAs are submitted and no unverified sample is released.
    """
    if not 4096 <= max_bytes <= 200_000_000:
        raise ValueError("Budget must be 4096..200000000 bytes")
    output.mkdir(parents=True, exist_ok=False)
    resources = [("source-readme.txt", REPOSITORY + "README.md", False),
                 ("source-label-client.txt", REPOSITORY + "intra_dataset_code/client.py", False),
                 ("official-drive-listing.html", FOLDER, False),
                 ("official-archive-probe.bin", FIRST_PART, True)]
    report = {"source": "celeba-spoof", "checked_at": datetime.now(timezone.utc).isoformat(),
              "budget_bytes": max_bytes, "downloaded_bytes": 0, "saved_bytes": 0, "image_samples": 0,
              "byte_accounting": "Payload bytes include failed resources; interrupted reads conservatively charge reserved bytes; excludes HTTP headers",
              "annotation_samples": 0, "released_samples": 0, "resources": [],
              "status": "blocked", "blockers": []}
    for name, url, archive in resources:
        try:
            # The initial archive request is intentionally 4KB, not a 1GB segment.
            cap = min(4096 if archive else 2_000_000, max_bytes - report["downloaded_bytes"])
            def consume(count):
                report["downloaded_bytes"] += count

            data, details = bounded_get(url, cap, archive=archive, consume=consume)
            if "text/html" in details["content_type"] and archive:
                name = "official-archive-warning.html"
                report["blockers"].append("Official archive endpoint returned HTML warning, not image data; confirmation not submitted")
            elif archive:
                report["blockers"].append("Only a bounded archive prefix acquired; multipart archive incomplete; not extracted")
            (output / name).write_bytes(data)
            report["saved_bytes"] += len(data)
            report["resources"].append({"file": name, "bytes": len(data),
                                         "sha256": hashlib.sha256(data).hexdigest(), **details})
            if name == "official-drive-listing.html":
                report["visible_archive_parts"] = len(set(re.findall(rb"CelebA_Spoof\.zip\.\d+", data)))
        except (requests.RequestException, StreamHTTPError, OSError, ValueError) as exc:
            report["blockers"].append(f"{name}: {exc}")
    report["blockers"].append("No standalone official labels/images obtained within bounded probe; supply legally obtained full archive and provenance sidecar")
    (output / "acquisition-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
