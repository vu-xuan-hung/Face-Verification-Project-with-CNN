"""End-to-end biometric access-control metrics and failure attribution."""

from __future__ import annotations

from collections.abc import Sequence


def evaluate_e2e(rows: Sequence[dict]) -> dict[str, object]:
    legitimate = [
        row
        for row in rows
        if row["presentation"] == "REAL" and row["expected_identity_state"] == "KNOWN"
    ]
    unknown = [
        row
        for row in rows
        if row["presentation"] == "REAL" and row["expected_identity_state"] == "UNKNOWN"
    ]
    spoofs = [row for row in rows if row["presentation"] == "SPOOF"]
    legitimate_allowed = [
        row
        for row in legitimate
        if row["final_access"] == "ALLOW" and row.get("predicted_identity") == row["subject_id"]
    ]
    wrong = [
        row
        for row in legitimate
        if row["final_access"] == "ALLOW" and row.get("predicted_identity") != row["subject_id"]
    ]
    return {
        "n_samples": len(rows),
        "n_legitimate": len(legitimate),
        "n_real_unknown": len(unknown),
        "n_spoof": len(spoofs),
        "legitimate_access_success_rate": _rate(len(legitimate_allowed), len(legitimate)),
        "legitimate_denial_rate": _rate(
            sum(row["final_access"] == "DENY" for row in legitimate), len(legitimate)
        ),
        "wrong_identity_acceptance_rate": _rate(len(wrong), len(legitimate)),
        "unknown_blocking_rate": _rate(
            sum(row["final_access"] == "DENY" for row in unknown), len(unknown)
        ),
        "unknown_access_false_acceptance_rate": _rate(
            sum(row["final_access"] == "ALLOW" for row in unknown), len(unknown)
        ),
        "spoof_blocking_rate": _rate(
            sum(row["final_access"] == "DENY" for row in spoofs), len(spoofs)
        ),
        "spoof_attack_success_rate": _rate(
            sum(row["final_access"] == "ALLOW" for row in spoofs), len(spoofs)
        ),
        "inference_errors": sum(bool(row.get("error")) for row in rows),
    }


def failure_reason(row: dict) -> str | None:
    if row.get("error"):
        return row.get("failure_code") or "MODEL_ERROR"
    presentation = row["presentation"]
    expected_state = row["expected_identity_state"]
    if presentation == "SPOOF":
        if row.get("final_access") == "ALLOW":
            return "SPOOF_FALSE_ACCEPT"
        return None
    if row.get("pad_decision") != "REAL":
        return "PAD_FALSE_REJECT"
    if expected_state == "UNKNOWN" and row.get("final_access") == "ALLOW":
        return "UNKNOWN_FALSE_ACCEPT"
    if expected_state == "KNOWN":
        if row.get("recognition_decision") == "UNKNOWN":
            return "RECOGNITION_UNKNOWN"
        if row.get("recognition_decision") == "AMBIGUOUS":
            return "RECOGNITION_AMBIGUOUS"
        if row.get("final_access") == "ALLOW" and row.get("predicted_identity") != row["subject_id"]:
            return "WRONG_IDENTITY"
        if row.get("final_access") == "DENY":
            return row.get("failure_code") or "ACCESS_DENIED"
    return None


def spoof_failure_stage(row: dict) -> str | None:
    if row["presentation"] != "SPOOF":
        return None
    if row.get("pad_score") is None:
        return "BLOCKED_BEFORE_PAD"
    if row.get("pad_decision") != "REAL":
        return "BLOCKED_BY_PAD"
    if row.get("final_access") == "DENY":
        return "BLOCKED_BY_RECOGNITION"
    return "ACCEPTED"


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
