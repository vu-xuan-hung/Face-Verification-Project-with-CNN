# External PAD datasets

External PAD images are not authorized login identities. Store research imports under private `data/external/`, never `data/authorization/`. The source catalog `configs/external-datasets.yaml` documents fixed adapter defaults; CLI overrides are explicit, not loaded silently from YAML.

## Sources and access

[CelebA-Spoof authors](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) restrict data to non-commercial research and prohibit sharing copies outside the permitted organization. Confirm compliant use before obtaining images. Scripts do not accept agreements.

The [official evaluation client](https://github.com/ZhangYuanhan-AI/CelebA-Spoof/blob/master/intra_dataset_code/client.py) uses attribute index 43: **0 live, 1 spoof**. The adapter inverts to project **0 fake, 1 real**. Attribute 40 supplies attack type; conflicting labels are rejected.

[OULU-NPU access](https://sites.google.com/site/oulunpudatabase/download) requires institutional authorization. This is a source lead, not an implemented adapter; no EULA is submitted.

## Bounded acquisition

```powershell
.\.venv\Scripts\python.exe scripts/download-external-dataset.py --output data/external/celeba-spoof/acquisition-NEW --max-bytes 200000000
```

Fetches official documentation, archive listing and a 4KB first-part probe. No Drive confirmation, full multipart download, archive extraction, arbitrary mirrors or overwrite. Payload cap is cumulative, including failed resources; HTTP headers are excluded. Interrupted reads conservatively retain the reserved budget, so accounted bytes may exceed received bytes after a connection failure. `saved_bytes` counts successful local files only. Blocked acquisition exits nonzero and records evidence in `acquisition-report.json`; HTML warnings are not image samples. Always choose a fresh directory.

Actual attempt, 2026-09-09: **799,972 bytes** of documentation/listing/warning, **0 image samples, 0 annotation samples, 0 released samples**. Listing showed 50 archive parts; visible part sizes were approximately 1,000MB each. The 4KB probe returned a 2,439-byte virus-scan-warning HTML page. Evidence: `data/external/celeba-spoof/acquisition-20260909/acquisition-report.json`. No sample download success is claimed.

## Import legally obtained source data

```powershell
.\.venv\Scripts\python.exe scripts/import-external-dataset.py --source-root D:/datasets/CelebA_Spoof --train-annotations D:/datasets/CelebA_Spoof/metas/intra_test/train_label.json --test-annotations D:/datasets/CelebA_Spoof/metas/intra_test/test_label.json --output data/external/celeba-spoof/candidate-NEW --limit 200
```

Expected layout: `Data/train/<numeric-subject>/live/<image>` or `Data/train/<numeric-subject>/spoof/<image>`, likewise `test` or `val` only when supplied by the official source. JSON maps paths to exactly 44 attributes. Annotation flags declare official membership; conflicting paths fail. Third-party layouts are not guessed.

Creates a new candidate directory, never modifies existing v2. Original images and annotation JSON are copied with hashes. Normalized lossless PNG copies use OpenCV BGR 128x128 INTER_AREA; float32 `/255` occurs at load. No guessed face crop. Images are bounded to 32MB compressed and 25 million pixels before OpenCV decoding. Hash ledger records source labels, official split, originals and transformed copies. A partial import after an error remains inspectable but is incomplete without final `import-report.json`.

Optional `--metadata capture-provenance.csv`, schema only:

```csv
source_path,subject_id,session_id,clip_id,device_id,attack_instrument_id,capture_time,provenance_source
```

Populate only from authentic capture records. Optional subject must match `celeba-spoof:<official numeric subject>`. Capture IDs must be globally scoped stable IDs, not row numbers. Required nonempty `provenance_source` points to real supporting evidence. Missing capture IDs are never invented. Official numeric identity does not identify a real-world person by name.

## Candidate protocols and release

- `official-membership.csv`: unchanged source membership; never strict-v2 release.
- `proposed-protocols/{train,val,test}-v2.csv`: separate custom grouped proposal targeting 70/15/15; reuses existing SHA256/dHash checks and subject/session/clip/device/attack-instrument grouping.
- `quarantine.csv`: missing provenance, duplicate conflicts/exclusions.
- `manifest.csv`: proposed rows remain candidate with `RELEASE_AUDIT_REQUIRED`.
- `release-audit.json`: strict audit of proposed grouping. Missing shortcut report blocks release; evaluated proposal count is separate from zero released samples.

Ratios are approximate; indivisible groups can leave splits empty. Source test rows remain unchanged in official membership but may move in the separately named custom proposal; custom results are not official benchmark results. Real metadata, class/attack coverage, manifest-bound shortcut probes, independent release audit and explicit promotion are required before training. No registry is generated automatically. Existing quadratic deduplication bounds imports to 10,000 rows, default 200.

Existing `scripts/shortcut-baselines.py`, `scripts/audit-dataset-release.py` and `scripts/split-dataset-grouped.py` remain authoritative for a separately approved release. Keep all outputs inside a fresh version. Never alter existing v2 evidence or mark candidates released merely to pass a gate. Synthetic unit-test images are not downloaded research data.

## Unresolved inputs

- Legally obtained complete images and official annotation JSON, or an accessible small official subset.
- Authentic session/clip/device records required by strict-v2 policy; published attributes do not establish these capture fields.
- Institutional OULU-NPU access if selected later.
