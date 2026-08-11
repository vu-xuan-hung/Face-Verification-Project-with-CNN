# Phase 05 — Enforce data and model governance in CI

## Context

- [Plan](./plan.md)
- Outputs from Phases 01–04.

## Overview

**Priority:** P1  
**Effort:** 2.5 days  
**Goal:** Không để leakage/shortcut quay lại khi thêm dữ liệu hoặc retrain.

## CI stages

```text
Schema/decode
  -> hash + duplicate components
  -> group-overlap checks
  -> shortcut probes
  -> coverage/slice checks
  -> train/val calibration
  -> locked test evaluation
  -> signed promotion manifest
```

PR CI dùng fixture nhỏ; dataset-release job chạy full audit. Model promotion chỉ nhận immutable dataset version đã pass.

## Proposed files

- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\scripts\audit-dataset-release.py`
- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\configs\dataset-release-gates.yaml`
- Modify CI workflow hiện có để chạy audit/test.
- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\artifacts\registry\dataset-v2-manifest.json`
- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\artifacts\registry\model-v2-manifest.json`

## Promotion manifest

Phải chứa:

- Dataset/protocol/model/preprocess SHA-256.
- Code commit, dependency lock, seed, hardware.
- Label map and threshold comparator.
- Validation-selected threshold and objective.
- Overall/slice metrics with CI.
- Leakage/shortcut gate results.
- Owner, reviewer, timestamp, supported/unsupported domains.

## Regression tests

- Add one duplicate across train/test → CI fails.
- Add near-duplicate to another split → CI fails or manual-review block.
- Change subject/session group → overlap check fails.
- Create class/brightness confound → shortcut gate fails.
- Mutate locked test file → checksum fails.
- Try to tune threshold with test scores → evaluation contract fails.
- Missing provenance or attack slice → release fails.

## Operational monitoring

- Production drift: input resolution, brightness, blur, device, PAD score distribution.
- Track challenge outcome/confirmed attack where legally available.
- Không tự động retrain từ production captures chưa consent/label review.
- Trigger re-audit khi camera pipeline, detector, codec, model hoặc preprocessing thay đổi.

## Todo

- [x] Encode release gates as config
- [x] Add small deterministic CI fixtures
- [ ] Add full dataset-release job with access to external dataset storage
- [x] Add checksum-bound model/dataset manifests
- [ ] Configure cryptographic signing with a deployment-owned key
- [ ] Add production drift dashboard and alert thresholds
- [ ] Define operational owner and rollback approval process

## Success criteria

- Các leakage/shortcut fixtures đều bị block.
- Model không thể promote nếu thiếu manifest hoặc failed gate.
- Có thể tái lập metric từ artifact immutable.
- Thay preprocessing/camera contract buộc tạo evaluation version mới.

## Risks

- Full perceptual scan tốn thời gian: cache hashes/features theo content SHA; incremental candidate search.
- Gate bị bypass thủ công: yêu cầu reviewer/sign-off và lưu exception có expiry.
- Drift alert không có ground truth: dùng làm tín hiệu điều tra, không suy ra accuracy production.

## Rollback

Giữ model/dataset generation trước dưới dạng immutable. Rollback theo manifest; không ghi đè artifact và không ghép score từ protocol khác nhau.
