"""Acquire a bounded, attributed PAD candidate; never release or merge it."""

import argparse
import csv
import hashlib
import json
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from fetch_public_pad_subset import _middle_frame

REPO = "AxonData/face-anti-spoofing-dataset"
REVISION = "89d658fffa10ee93cc3010e53741d435fbadd3f6"
LICENSE = "cc-by-nc-4.0"
BUDGET = 200 * 1024 * 1024
FILE_CAP = 40 * 1024 * 1024


def fetch(url, limit):
    request = urllib.request.Request(url, headers={"User-Agent": "VShield-evaluation-acquisition/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if not response.url.startswith("https://"):
            raise ValueError("Non-HTTPS redirect")
        length = response.headers.get("Content-Length")
        if length and int(length) > limit:
            raise ValueError("Declared payload exceeds remaining budget")
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Payload exceeds remaining budget")
    return data


def select(names):
    result = []
    for name in sorted(names):
        if (name.startswith("Selfies/") or name.startswith("Replay_display_attacks/Real/")) and name.lower().endswith(".jpg"):
            result.append((name, "", "unknown", "source-still; bona-fide camera provenance unverified"))
    for prefix, attack in [("Replay_display_attacks/Screen/", "display_replay"),
                           ("Replay_mobile_attacks/", "mobile_replay"),
                           ("3D_paper_mask /", "paper_mask")]:
        candidates = sorted(n for n in names if n.startswith(prefix) and n.lower().endswith((".mp4", ".mov")))
        result.extend((name, "0", attack, "explicit source attack directory") for name in candidates[:2])
    return result


def main():
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    allowed = Path(__file__).resolve().parents[1] / "data" / "external"
    if not root.is_relative_to(allowed.resolve()):
        parser.error("Output must be below repository data/external")
    if root.exists():
        parser.error("Choose a new output directory; existing candidates are immutable")
    root.mkdir(parents=True)
    report = {"repo": REPO, "revision": REVISION, "license": LICENSE,
              "status": "candidate_not_released", "released_samples": 0,
              "download_budget_bytes": BUDGET, "saved_bytes": 0, "accounted_bytes": 0, "files": [], "failures": [],
              "limitations": ["Unknown subject/session/device IDs remain blank, not inferred from filenames.",
                              "Source stills are unlabeled pending bona-fide provenance review.",
                              "No train/validation/test split; no merging with existing evaluation data.",
                              "No fine-tuning or production model changes."]}
    rows = []
    try:
        api_url = f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}?blobs=true"
        report["accounted_bytes"] += 1_000_001
        api = fetch(api_url, 1_000_000)
        report["accounted_bytes"] -= 1_000_001 - len(api)
        metadata = json.loads(api)
        if metadata.get("sha") != REVISION or metadata.get("cardData", {}).get("license") != LICENSE:
            raise ValueError("Pinned revision or declared license differs")
        (root / "source-api.json").write_bytes(api)
        report["saved_bytes"] += len(api)
        report["accounted_bytes"] += 1_000_001
        card = fetch(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/README.md", 1_000_000)
        report["accounted_bytes"] -= 1_000_001 - len(card)
        (root / "source-card.md").write_bytes(card)
        report["saved_bytes"] += len(card)
        siblings = {entry["rfilename"]: entry for entry in metadata["siblings"]}
        for position, (name, label, attack, basis) in enumerate(select(siblings)):
            try:
                available = BUDGET - report["accounted_bytes"]
                cap = min(FILE_CAP, available)
                entry = siblings[name]
                expected_size = entry.get("size", entry.get("lfs", {}).get("size"))
                if cap <= 0 or (expected_size is not None and expected_size > cap):
                    raise ValueError("Source size exceeds per-file or remaining budget")
                if expected_size is not None:
                    cap = expected_size
                url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{urllib.parse.quote(name, safe='/')}"
                # Reserve the worst-case read, including the overflow sentinel.
                if cap + 1 > available:
                    raise ValueError("No room for bounded-read overflow sentinel")
                report["accounted_bytes"] += cap + 1
                payload = fetch(url, cap)
                report["accounted_bytes"] -= cap + 1 - len(payload)
                digest = hashlib.sha256(payload).hexdigest()
                expected_sha = entry.get("lfs", {}).get("sha256")
                if expected_sha and digest != expected_sha:
                    raise ValueError("Source LFS SHA-256 mismatch")
                if expected_size is not None and len(payload) != expected_size:
                    raise ValueError("Source size mismatch")
                suffix = Path(name).suffix.lower()
                source = root / "originals" / f"{position:03d}{suffix}"
                source.parent.mkdir(exist_ok=True)
                source.write_bytes(payload)
                report["saved_bytes"] += len(payload)
                sample = source
                frame_position = None
                if suffix in {".mp4", ".mov"}:
                    frame_position, frame = _middle_frame(source)
                    sample = root / "frames" / f"{position:03d}.png"
                    sample.parent.mkdir(exist_ok=True)
                    if not cv2.imwrite(str(sample), frame):
                        raise ValueError("Cannot save middle frame")
                elif cv2.imread(str(source)) is None:
                    raise ValueError("Source still cannot be decoded")
                sample_sha = hashlib.sha256(sample.read_bytes()).hexdigest()
                rows.append({"sample_id": f"axon-{position:03d}", "relative_path": sample.relative_to(root).as_posix(),
                             "label": label, "attack_type": attack, "subject_id": "", "session_id": "",
                             "clip_id": name if frame_position is not None else "", "device_id": "",
                             "source_path": name, "sha256": sample_sha, "label_basis": basis,
                             "status": "quarantine", "split": "",
                             "reason_code": "MISSING_CAPTURE_PROVENANCE" + (";UNVERIFIED_BONA_FIDE" if not label else "")})
                report["files"].append({"source_path": name, "source_url": url, "source_sha256": digest,
                                        "bytes": len(payload), "sample_sha256": sample_sha,
                                        "frame_position": frame_position, "lfs_hash_verified": bool(expected_sha)})
                print(f"saved {name}: {len(payload)} bytes", flush=True)
            except Exception as exc:
                report["failures"].append({"source_path": name, "error": str(exc)})
                print(f"skipped {name}: {exc}", flush=True)
        if rows:
            with (root / "candidate-manifest.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        report["candidate_samples"] = len(rows)
        report["candidate_labels"] = dict(Counter(row["label"] or "UNVERIFIED" for row in rows))
    finally:
        (root / "acquisition-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in ["status", "candidate_samples", "candidate_labels", "saved_bytes", "released_samples"]}))
    if not rows:
        raise SystemExit("No candidate samples acquired")


if __name__ == "__main__":
    main()
