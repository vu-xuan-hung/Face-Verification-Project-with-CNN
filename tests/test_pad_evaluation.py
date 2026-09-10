"""Tests for anti-spoof evaluation script."""

import cv2
import numpy as np
from scripts.evaluate_pad_directory import evaluate_dataset


def test_evaluate_pad_directory_no_dataset(tmp_path, capsys):
    empty_dir = tmp_path / "empty_eval"
    empty_dir.mkdir()
    code = evaluate_dataset(empty_dir)
    assert code == 0
    captured = capsys.readouterr().out
    assert "NO EVALUATION DATASET PROVIDED" in captured


def test_evaluate_pad_directory_with_samples(tmp_path, capsys, monkeypatch):
    eval_dir = tmp_path / "eval_data"
    real_dir = eval_dir / "real"
    fake_dir = eval_dir / "fake_print"
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)

    # Create dummy images
    for i in range(2):
        img = np.full((64, 64, 3), 120, dtype=np.uint8)
        cv2.imwrite(str(real_dir / f"real_{i}.png"), img)
    for i in range(2):
        img = np.full((64, 64, 3), 40, dtype=np.uint8)
        cv2.imwrite(str(fake_dir / f"fake_{i}.png"), img)

    # Mock check_pad to return deterministic scores
    from vshield.core.anti_spoof import PadResult, PadStatus

    def mock_check_pad(service, image, bbox):
        # Brightness mock: brighter = higher liveness score
        score = float(np.mean(image) / 255.0)
        status = PadStatus.REAL if score >= 0.8 else PadStatus.FAKE
        return PadResult(
            status=status,
            score=score,
            class_index=1 if status == PadStatus.REAL else 0,
            model_version="mock-v2",
            threshold=0.8,
            reason="PAD_REAL" if status == PadStatus.REAL else "PAD_SPOOF",
        )

    import scripts.evaluate_pad_directory as epd
    monkeypatch.setattr(epd, "check_pad", mock_check_pad)

    code = evaluate_dataset(eval_dir, thresholds=[0.50, 0.80])
    assert code == 0
    captured = capsys.readouterr().out
    assert "MINIFASNET V2 EMPIRICAL PAD EVALUATION REPORT" in captured
    assert "Total Evaluated: 4" in captured
    assert "0.80" in captured
