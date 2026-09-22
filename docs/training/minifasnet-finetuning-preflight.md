# MiniFASNetV2 fine-tuning preflight

Date: 2026-09-17 (Asia/Bangkok).
Status: DATA_GATE_BLOCKED; no fine-tuning has been executed.

## Authorization and scope

The user stated that people in the supplied images consented to their use. This statement is recorded as user-provided authorization for inspecting the local collection; it does not supply missing capture metadata or establish third-party dataset rights. External public previews, authorization/enrollment galleries, and prior evaluation probes are not selected for training.

## Current inventory

Image counts (JPG/JPEG/PNG/BMP/WebP, including git-ignored files):

| Directory | Images | Notes |
| --- | ---: | --- |
| data/DataCollect | 2,892 | 1,389 under real; 1,488 under fake; 15 at the root, not assigned a folder label |
| data/Real | 696 | Directory name alone does not establish capture provenance |
| data/All | 2,416 | Not assumed independent of DataCollect or SplitData |
| data/SplitData | 2,415 | Existing paired labels and legacy train/val/test membership |
| data/faces | 54 | Not selected as PAD training data |
| data/authorization | 1 | Enrollment gallery, excluded |
| data/external | 270 | Separate third-party or prior evaluation data; excluded |

These are file counts, not unique independent samples.

## Fresh SplitData audit

Command from repository root:

```powershell
.\.venv\Scripts\python.exe scripts/build-dataset-manifest.py --input data/SplitData --output outputs/training-preflight-20260917/SplitData-manifest.csv
```

- Images: 2,415; unique SHA-256 values: 1,608.
- Labels: 1,153 real (`1`), 1,262 fake (`0`).
- Existing membership: 1,691 train; 483 validation; 241 test.
- Exact duplicate groups crossing existing splits: 264, involving 746 rows.
- Exact-hash conflicting-label groups: 0. This does not establish that all semantic labels are correct.
- All 2,415 rows are quarantined because subject_id, session_id, clip_id, and device_id are absent.
- Legacy splits cannot be treated as an independent final evaluation. Deduplication alone does not resolve neighboring-frame, session, subject, or device overlap.

Raw fresh manifest: `outputs/training-preflight-20260917/SplitData-manifest.csv`. Existing v1/v2 evidence and source images were not overwritten.

## Next steps, not yet executed

1. Obtain authentic capture-to-subject/session/clip/device mapping, using pseudonyms rather than personal names; distinguish bona-fide captures from print/screen/replay presentations.
2. Build a new version from the original collection, retaining source hashes and previous test membership as history. Never describe previously examined probes as a new untouched test set.
3. Quarantine unlabeled/ambiguous samples; group exact and near duplicates; audit label-independent shortcuts and collection coverage before release.
4. Produce a separate PyTorch MiniFASNetV2 fine-tuning candidate. Existing `src/vshield/training/train.py` trains a legacy Keras CNN, not MiniFASNetV2, and must not be used to claim MiniFASNet training.
5. Record seed, source checkpoint hash, data manifest, split protocol, environment, optimization settings, per-epoch history, selected candidate checkpoint, and baseline-versus-candidate results in an executed notebook.
6. Preserve the deployed ONNX artifact and production thresholds until an independently reviewed promotion is authorized.

## Unresolved questions

- Which images belong to each actual person, capture session, source clip, and device?
- Which spoof presentations were deliberately performed, and what were their attack instruments?
- Are original full-camera frames or source videos available for matching the deployed PAD context-crop contract?
- Is there a new, unexamined collection suitable for final independent evaluation?
