"""Build date-disjoint local admin manifests from user-labeled face crops.

One representative crop per encoded capture second limits adjacent-frame
correlation. The dates and subject identity are provided explicitly by CLI.
These 128x128 crops are NOT a valid native webcam PAD collection protocol.
"""

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def _capture_second(image: Path) -> int:
    if len(image.stem) < 10 or not image.stem[:10].isdigit():
        raise ValueError(f"Cannot infer capture second from file name: {image.name}")
    return int(image.stem[:10])


def select_per_second(images: list[Path], stride_seconds: int, used_hashes: set[str]) -> list[Path]:
    if stride_seconds < 1:
        raise ValueError("--stride-seconds must be positive")
    groups: dict[int, list[Path]] = defaultdict(list)
    for image in images:
        groups[_capture_second(image)].append(image)
    selected = []
    first = min(groups) if groups else 0
    for second in sorted(groups):
        if (second - first) % stride_seconds:
            continue
        group = sorted(groups[second])
        middle = len(group) // 2
        candidates = sorted(range(len(group)), key=lambda position: (abs(position - middle), position))
        for position in candidates:
            image = group[position]
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest not in used_hashes:
                used_hashes.add(digest)
                selected.append(image)
                break
    return selected


def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--gallery-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--calibration-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--test-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--stride-seconds", type=int, default=1)
    parser.add_argument("--spoof-manifest", type=Path, help="Optional non-targeted public spoof E2E probes")
    args = parser.parse_args()
    dates = (args.gallery_date, args.calibration_date, args.test_date)
    if len(set(dates)) != 3:
        raise ValueError("Gallery, calibration and test must use distinct capture dates")
    for date in dates:
        datetime.strptime(date, "%Y-%m-%d")
    if not args.subject_id.strip():
        raise ValueError("--subject-id is required")
    source = args.source.resolve()
    output = args.output.resolve()
    photos = sorted(source.glob("*.jpg"))
    if not photos:
        raise ValueError(f"No JPG photos in {source}")
    by_date: dict[str, list[Path]] = defaultdict(list)
    hashes = Counter()
    for photo in photos:
        by_date[datetime.fromtimestamp(_capture_second(photo)).strftime("%Y-%m-%d")].append(photo)
        hashes[hashlib.sha256(photo.read_bytes()).hexdigest()] += 1
    if any(not by_date[date] for date in dates):
        raise ValueError(f"Requested capture date has no images: {dates}")
    output.mkdir(parents=True, exist_ok=True)
    selected: dict[str, list[Path]] = {}
    used_hashes: set[str] = set()
    for split, date in zip(("gallery", "calibration", "test"), dates, strict=True):
        selected[split] = select_per_second(by_date[date], args.stride_seconds, used_hashes)
        if not selected[split]:
            raise ValueError(f"No independent exact-file samples for {split}")
    recognition = []
    for split, date in zip(("gallery", "calibration", "test"), dates, strict=True):
        for index, photo in enumerate(selected[split], start=1):
            recognition.append({
                "sample_id": f"admin-{split}-{index}",
                "image_path": os.path.relpath(photo, output),
                "subject_id": args.subject_id,
                "split": split,
                "is_enrolled": "1",
                "session_id": date,
                "notes": "One crop per encoded capture second; same-day frames remain correlated",
            })
    _write(output / "recognition.csv", list(recognition[0]), recognition)
    e2e = [{
        "sample_id": f"admin-real-{index}",
        "image_path": os.path.relpath(photo, output),
        "subject_id": args.subject_id,
        "presentation": "REAL",
        "expected_identity_state": "KNOWN",
        "expected_access": "ALLOW",
        "attack_type": "none",
        "notes": "User-labeled real face CROP, not native full webcam frame; E2E exploratory only",
    } for index, photo in enumerate(selected["test"], start=1)]
    _write(output / "e2e_real_known.csv", list(e2e[0]), e2e)
    if args.spoof_manifest:
        spoof_manifest = args.spoof_manifest.resolve()
        with spoof_manifest.open("r", encoding="utf-8-sig", newline="") as stream:
            spoof_rows = list(csv.DictReader(stream))
        if any(row.get("presentation", "").upper() != "SPOOF" for row in spoof_rows):
            raise ValueError("Optional E2E spoof manifest must contain only SPOOF probes")
        combined = [*e2e]
        for row in spoof_rows:
            converted = {field: row.get(field, "") for field in e2e[0]}
            image = Path(row["image_path"])
            resolved = image if image.is_absolute() else spoof_manifest.parent / image
            converted["image_path"] = os.path.relpath(resolved.resolve(), output)
            converted["notes"] = "Non-targeted public spoof; upstream video identity is unknown, not verified admin"
            combined.append(converted)
        _write(output / "e2e_combined.csv", list(e2e[0]), combined)
    (output / "source_protocol.json").write_text(json.dumps({
        "source": str(source), "subject_id": args.subject_id,
        "user_label": "All photos in source/real were described by the user as admin",
        "roles_by_date": dict(zip(("gallery", "calibration", "test"), dates, strict=True)),
        "source_photos": len(photos), "source_counts_by_date": {day: len(group) for day, group in by_date.items()},
        "exact_duplicate_hash_groups": sum(count > 1 for count in hashes.values()),
        "images_in_duplicate_hash_groups": sum(count for count in hashes.values() if count > 1),
        "stride_seconds": args.stride_seconds,
        "selected_counts": {split: len(group) for split, group in selected.items()},
        "nontargeted_spoof_manifest": str(args.spoof_manifest.resolve()) if args.spoof_manifest else None,
        "selected_exact_hash_leakage": False,
        "limitations": "All source images are pre-cropped 128x128. Encoded timestamps and user label do not prove distinct video sessions or model-training independence. Unknown-person and targeted-spoof labels remain unavailable.",
    }, indent=2), encoding="utf-8")
    print({split: len(group) for split, group in selected.items()})


if __name__ == "__main__":
    main()
