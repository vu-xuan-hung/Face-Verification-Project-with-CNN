# Collecting VShield evaluation data

This directory contains manifest templates only. It does not contain a valid evaluation dataset and must not be used to claim biometric performance.

Image paths are resolved relative to the CSV file. Keep source images outside Git when they contain biometric data, and apply appropriate consent, retention, access-control, and deletion policies.

## Scientific split rules

Keep four roles logically separate:

1. Training/model data: used to fit model weights.
2. Gallery/enrollment data: used only to create enrolled identity templates.
3. Calibration/validation data: used to choose PAD, recognition, or ambiguity thresholds.
4. Final test data: evaluated once after thresholds and protocol are frozen.

Never copy the same photograph into two roles. An exact copy with a different filename is still leakage. Neighboring frames from one video can also make results overly optimistic, so use different capture sessions for gallery, calibration, and final test whenever possible. Unknown subjects must not appear in the gallery under any identity.

## PAD manifests

Use `pad_calibration.csv` to select a PAD threshold and `pad_test.csv` only for the final locked evaluation.

Required columns:

- `sample_id`: unique stable identifier.
- `image_path`: path relative to the manifest.
- `label`: `REAL` or `SPOOF`.

Optional columns in the template are `attack_type`, `subject_id`, `session_id`, `device`, and `notes`. Use `attack_type=none` for bona-fide samples and descriptive values such as `print`, `screen`, `replay`, or `other` for attacks. Do not claim an attack category that was not deliberately performed and human-labelled.

Collect bona-fide live captures and, where safe and authorized, printed-photo, screen/display, and replay attacks. Aim for multiple people, capture sessions, lighting conditions, poses, distances, and target devices. These are collection recommendations; the framework never assumes the current data has that coverage.

## Recognition manifest

`recognition_test.csv` contains all recognition roles but each row has exactly one split:

- `gallery`: enrollment templates only.
- `calibration`: independent probes used for threshold selection.
- `test`: untouched final probes.

Required fields are `sample_id`, `image_path`, `subject_id`, `split`, and `is_enrolled`. Set `is_enrolled=1` for a probe whose subject has gallery images and `is_enrolled=0` for an unknown subject. Gallery rows must use `is_enrolled=1`.

Use multiple gallery captures per enrolled identity where possible. Calibration and test images must be different captures, preferably from different sessions. Each known probe's `subject_id` must exactly match its gallery identity. Each unknown probe's subject must be absent from every gallery row.

The evaluator compares every probe with gallery identities using production VShield's minimum-template normalized-L2 strategy. It does not use calibration or test images to build the gallery.

## End-to-end manifest

`e2e_test.csv` fields are:

- `presentation`: `REAL` or `SPOOF`.
- `expected_identity_state`: `KNOWN` or `UNKNOWN`.
- `expected_access`: `ALLOW` or `DENY`.
- `subject_id`: enrolled identity for known attempts, or stable unknown/attack subject label.
- `attack_type`: optional attack category.

The typical expectations are:

- `REAL + KNOWN -> ALLOW` as the correct identity.
- `REAL + UNKNOWN -> DENY`.
- `SPOOF -> DENY`, with recognition not executed when PAD blocks it.

The end-to-end command also requires the recognition manifest so it can build an independent gallery. It aborts by default if an exact gallery image hash appears in the end-to-end test manifest.

## Validation and leakage behavior

Every evaluator checks missing and corrupted files, duplicate paths, duplicate sample IDs, exact SHA-256 duplicates, invalid labels, and relevant identity/split rules. Exact gallery-to-test or calibration-to-test hash leakage aborts evaluation unless `--allow-leakage` is explicitly supplied. That flag is for investigating known-bad data; results from leaked data should not be presented as final performance.

## Commands

```powershell
uv run python scripts/evaluate_pad.py --manifest evaluation_data/pad_test.csv --output outputs/evaluation/pad --save-plots

uv run python scripts/calibrate_pad.py --manifest evaluation_data/pad_calibration.csv --output outputs/evaluation/pad-calibration --objective min-acer

uv run python scripts/evaluate_recognition.py --manifest evaluation_data/recognition_test.csv --output outputs/evaluation/recognition --save-plots

uv run python scripts/calibrate_recognition.py --manifest evaluation_data/recognition_test.csv --output outputs/evaluation/recognition-calibration --objective min-open-set-error

uv run python scripts/evaluate_e2e.py --manifest evaluation_data/e2e_test.csv --gallery-manifest evaluation_data/recognition_test.csv --output outputs/evaluation/e2e --save-plots

uv run python scripts/evaluate_all.py --config configs/evaluation.yaml
```

Calibration commands only recommend values. They never edit `configs/ai-models.yaml` or production recognition constants. Validate any recommendation on the untouched final test set.
