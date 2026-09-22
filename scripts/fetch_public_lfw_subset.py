"""Fetch 200 individually labeled LFW photos for an exploratory open-set protocol."""

import argparse
import csv
import hashlib
import io
import json
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO = "marcelohaps/lfw"
REVISION = "12a61458b56d0433d07269dc1d64368abf4f6b4d"
FIELDS = ["sample_id", "image_path", "subject_id", "split", "is_enrolled", "session_id", "notes"]


def _get(url: str, maximum: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "VShield-research-evaluation/1.0"})
    with urllib.request.urlopen(request, timeout=40) as response:
        if int(response.headers.get("Content-Length", "0")) > maximum:
            raise ValueError(f"Source exceeds bounded size: {url}")
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError(f"Source exceeds bounded size: {url}")
    return data


def _url(path: str) -> str:
    return f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{urllib.parse.quote(path, safe='/')}"


def _select(records):
    grouped = defaultdict(list)
    for row in records:
        identity = row["identity"]
        path = row["file_name"]
        if not path.startswith("images/") or ".." in Path(path).parts or not path.endswith(".jpg"):
            raise ValueError("Unexpected LFW metadata image path")
        grouped[identity].append(row)
    enrolled = sorted(identity for identity, rows in grouped.items() if len(rows) >= 5)[:20]
    unknown = sorted(identity for identity, rows in grouped.items() if len(rows) == 1)[:100]
    if len(enrolled) != 20 or len(unknown) != 100:
        raise ValueError("Source does not satisfy the frozen identity protocol")
    selected = []
    for identity in enrolled:
        photos = sorted(grouped[identity], key=lambda row: int(row["image_num"]))[:5]
        for split, photo in zip(("gallery", "gallery", "calibration", "test", "test"), photos, strict=True):
            selected.append((photo, split, "1"))
    for number, identity in enumerate(unknown):
        selected.append((grouped[identity][0], "calibration" if number < 20 else "test", "0"))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata = _get(_url("train/metadata.csv"), 4 * 1024 * 1024)
    records = list(csv.DictReader(io.StringIO(metadata.decode("utf-8-sig"))))
    selected = _select(records)
    manifest, provenance = [], []
    for number, (photo, split, enrolled) in enumerate(selected):
        source_path = f"train/{photo['file_name']}"
        target = (root / source_path).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Source image path escapes output directory")
        if not target.is_file():
            data = _get(_url(source_path), 2 * 1024 * 1024)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        identity = photo["identity"]
        manifest.append({
            "sample_id": f"lfw-{number:04d}", "image_path": source_path,
            "subject_id": identity, "split": split, "is_enrolled": enrolled,
            "session_id": "", "notes": "LFW web photograph; independent capture session unverified",
        })
        provenance.append({"source": source_path, "identity": identity, "split": split, "sha256": digest})
        if (number + 1) % 20 == 0:
            print(f"LFW: {number + 1}/{len(selected)} photographs", flush=True)
    with (root / "recognition.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(manifest)
    (root / "source_provenance.json").write_text(json.dumps({
        "repo": REPO, "revision": REVISION, "source": "LFW original unaligned imagefolder mirror",
        "source_split": "train in mirror; this is not model-training proof",
        "license": "other; verify original LFW terms before redistribution/publication",
        "images": len(manifest), "gallery": 40, "calibration": 40, "test": 120,
        "limitations": "Research-only LFW public-figure web photos, not consented VShield live captures. Subject session and FaceNet train overlap unverified. Never use these photos as PAD bona-fide or E2E live samples.",
        "photos": provenance,
    }, indent=2), encoding="utf-8")
    print("Recognition: 40 gallery + 40 calibration + 120 test = 200 photographs", flush=True)


if __name__ == "__main__":
    main()
