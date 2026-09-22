import csv

import numpy as np
from scripts.evaluate_all import _incomplete_run_error

from vshield.core.identity_index import IdentityIndex
from vshield.evaluation.calibration import calibrate_pad_predictions
from vshield.evaluation.e2e_metrics import evaluate_e2e
from vshield.evaluation.pad_metrics import error_rates
from vshield.evaluation.recognition_metrics import (
    apply_open_set_thresholds,
    open_set_metrics,
)


def unit(index=0):
    vector = np.zeros(512, dtype=np.float32)
    vector[index] = 1
    return vector


def test_pad_apcer_bpcer_acer_and_inclusive_boundary():
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.8, 0.2, 0.8, 0.7])
    rates = error_rates(labels, scores, 0.8)
    assert rates == {"apcer": 0.5, "bpcer": 0.5, "acer": 0.5}


def test_recognition_open_set_outcomes_include_rejections_in_denominator():
    rows = [
        {"is_enrolled": True, "subject_id": "a", "decision": "KNOWN", "identity": "a", "best_identity": "a"},
        {"is_enrolled": True, "subject_id": "b", "decision": "UNKNOWN", "identity": None, "best_identity": "b"},
        {"is_enrolled": True, "subject_id": "c", "decision": "KNOWN", "identity": "a", "best_identity": "a"},
        {"is_enrolled": True, "subject_id": "d", "decision": "AMBIGUOUS", "identity": None, "best_identity": "d"},
        {"is_enrolled": False, "subject_id": "u1", "decision": "UNKNOWN", "identity": None, "best_identity": "a"},
        {"is_enrolled": False, "subject_id": "u2", "decision": "KNOWN", "identity": "a", "best_identity": "a"},
    ]
    metrics = open_set_metrics(rows)
    assert metrics["known_identification_accuracy"] == 0.25
    assert metrics["known_rejection_rate"] == 0.5
    assert metrics["wrong_identity_rate"] == 0.25
    assert metrics["unknown_rejection_rate"] == 0.5
    assert metrics["unknown_false_acceptance_rate"] == 0.5


def test_recognition_rank_one_treats_inference_error_as_failure():
    rows = [
        {
            "is_enrolled": True,
            "subject_id": "a",
            "decision": "ERROR",
            "identity": None,
            "best_identity": "a",
            "error": "model failed",
        }
    ]
    assert open_set_metrics(rows)["rank_1_identification_rate"] == 0.0


def test_aggregate_evaluation_reports_component_failures():
    message = _incomplete_run_error(
        {"pad": "RuntimeError: failed"},
        {"recognition": {"available": False, "explanation": "missing calibration"}},
    )
    assert message == (
        "Evaluation incomplete: pad: RuntimeError: failed; "
        "recognition calibration: missing calibration"
    )


def test_production_distance_threshold_is_inclusive():
    index = IdentityIndex({"a": [unit()]}, prefer_faiss=False)
    assert index._decide_details([("a", 0.9)]).decision == "MATCH"
    assert index._decide_details([("a", 0.9001)]).decision == "UNKNOWN"
    assert index._decide_details([("a", 0.8999)]).decision == "MATCH"


def test_ambiguity_boundary_and_close_winner():
    index = IdentityIndex({"a": [unit()], "b": [unit(1)]}, prefer_faiss=False)
    assert index._decide_details([("a", 0.0), ("b", 0.05)]).decision == "MATCH"
    assert index._decide_details([("a", 0.0), ("b", 0.049)]).decision == "AMBIGUOUS"
    assert index._decide_details([("a", 0.0), ("b", 0.2)]).decision == "MATCH"


def test_threshold_sweep_uses_same_inclusive_rules():
    rows = [{"best_score": 0.9, "margin": 0.05, "best_identity": "a", "is_enrolled": True, "subject_id": "a"}]
    assert apply_open_set_thresholds(rows, 0.9, 0.05)[0]["decision"] == "KNOWN"
    assert apply_open_set_thresholds(rows, 0.8999, 0.05)[0]["decision"] == "UNKNOWN"
    assert apply_open_set_thresholds(rows, 0.9, 0.0501)[0]["decision"] == "AMBIGUOUS"


def test_end_to_end_security_metrics():
    rows = [
        {"presentation": "REAL", "expected_identity_state": "KNOWN", "subject_id": "a", "final_access": "ALLOW", "predicted_identity": "a"},
        {"presentation": "REAL", "expected_identity_state": "UNKNOWN", "subject_id": "u", "final_access": "DENY", "predicted_identity": None},
        {"presentation": "SPOOF", "expected_identity_state": "KNOWN", "subject_id": "a", "final_access": "DENY", "predicted_identity": None},
    ]
    metrics = evaluate_e2e(rows)
    assert metrics["legitimate_access_success_rate"] == 1
    assert metrics["unknown_blocking_rate"] == 1
    assert metrics["spoof_blocking_rate"] == 1
    assert metrics["spoof_attack_success_rate"] == 0


def test_pad_calibration_keeps_zero_score(tmp_path):
    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["ground_truth", "pad_score", "error"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {"ground_truth": "SPOOF", "pad_score": "0.0", "error": ""},
                {"ground_truth": "REAL", "pad_score": "0.9", "error": ""},
            ]
        )
    result = calibrate_pad_predictions(
        predictions, tmp_path / "output", current_threshold=0.8
    )
    assert result["calibration_metrics"]["sample_count"] == 2
