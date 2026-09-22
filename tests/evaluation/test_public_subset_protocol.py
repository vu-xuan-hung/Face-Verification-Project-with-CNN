from collections import Counter

import cv2
import numpy as np
from scripts.build_public_spoof_e2e_manifest import _relative_image_path
from scripts.fetch_public_lfw_subset import _select
from scripts.fetch_public_pad_subset import _middle_frame

from vshield.evaluation.calibration import calibrate_recognition_predictions


def test_middle_video_frame_is_selected_once(tmp_path):
    video = tmp_path / "short.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 32))
    assert writer.isOpened()
    for index in range(11):
        writer.write(np.full((32, 32, 3), index * 20, dtype=np.uint8))
    writer.release()
    position, frame = _middle_frame(video)
    assert position == 5
    assert frame.shape == (32, 32, 3)
    assert 90 <= frame.mean() <= 110


def test_spoof_manifest_rebases_image_paths_to_output_directory(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "nested" / "output"
    source_dir.mkdir()
    assert _relative_image_path("frames/one.png", source_dir, output_dir) == (
        "../../source/frames/one.png"
    )


def test_lfw_selection_keeps_roles_and_identities_disjoint():
    records = []
    for identity in range(20):
        for number in range(1, 6):
            records.append({
                "identity": f"known_{identity:02d}", "image_num": str(number),
                "file_name": f"images/000/known_{identity:02d}_{number:04d}.jpg",
            })
    for identity in range(100):
        records.append({
            "identity": f"unknown_{identity:03d}", "image_num": "1",
            "file_name": f"images/001/unknown_{identity:03d}_0001.jpg",
        })
    selected = _select(records)
    assert len(selected) == 200
    assert Counter(split for _, split, _ in selected) == {
        "gallery": 40, "calibration": 40, "test": 120
    }
    gallery = {photo["identity"] for photo, split, _ in selected if split == "gallery"}
    assert all(photo["identity"] not in gallery for photo, _, enrolled in selected if enrolled == "0")


def test_recognition_calibration_counts_failed_probes(tmp_path):
    import csv

    source = tmp_path / "calibration.csv"
    fields = ["sample_id", "is_enrolled", "subject_id", "best_identity", "best_score", "margin", "error"]
    samples = [
        ["known-ok", "1", "a", "a", "0.5", "0.2", ""],
        ["known-error", "1", "a", "a", "0.5", "0.2", "model failed"],
        ["unknown-ok", "0", "b", "a", "0.8", "0.2", ""],
        ["unknown-error", "0", "c", "", "", "", "model failed"],
    ]
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(samples)
    result = calibrate_recognition_predictions(
        source, tmp_path / "out", current_distance=0.9, current_ambiguity=0.05,
        distance_grid=[0.6], ambiguity_grid=[0.0], objective="min-open-set-error",
    )
    metrics = result["calibration_metrics"]
    assert metrics["n_probes"] == 4
    assert metrics["inference_errors"] == 2
    assert metrics["known_identification_accuracy"] == 0.5
    assert metrics["unknown_rejection_rate"] == 0.5
