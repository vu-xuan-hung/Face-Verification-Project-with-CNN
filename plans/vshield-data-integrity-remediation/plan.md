---
title: "V-Shield Data Integrity Remediation"
description: "Loại bỏ leakage, phá shortcut dữ liệu và xây lại protocol đánh giá anti-spoof có giá trị."
status: in-progress
priority: P1
effort: "15 engineer-days + 1-3 weeks data collection"
branch: fixgit
tags: [critical, data, anti-spoofing, evaluation, ml-governance]
created: 2026-07-30
---

# V-Shield Data Integrity Remediation

## Overview

Dataset/test v1 bị vô hiệu cho đánh giá model: 264 nhóm exact duplicate xuyên split (746 ảnh) và metadata-only classifier đạt 99,59% accuracy/AUC 1,0. Kế hoạch tạo dataset v2 immutable, group-disjoint, không shortcut rõ ràng; sau đó mới retrain và test.

## Non-negotiable decisions

- Không dùng test v1 để chọn model, threshold hoặc công bố accuracy.
- Không chỉ dedupe rồi random split lại: leakage và confound là hai gate độc lập.
- Split theo `subject/session/clip/device/attack`, không theo file/frame.
- Threshold chọn duy nhất trên validation; test v2 khóa trước khi train.
- Nếu thiếu provenance để xác định group, mẫu phải quarantine hoặc thu lại; không đoán.

## Phases

| # | Phase | Status | Effort | Gate |
|---|---|---|---:|---|
| 1 | [Containment](./phase-01-containment.md) | Complete | 0.5d | Dataset/model v1 read-only, metric marked invalid |
| 2 | [Manifest, dedupe, grouped split](./phase-02-manifest-dedupe-split.md) | Blocked on provenance review | 3d | Zero cross-split content/group overlap |
| 3 | [Remove shortcut confounds](./phase-03-remove-shortcut-confounds.md) | Blocked on data collection | 5d engineering + collection | Nuisance baselines near chance |
| 4 | [Retrain and locked evaluation](./phase-04-retrain-evaluate.md) | Tooling complete; model run blocked | 4d | PAD metrics + confidence intervals on test v2 |
| 5 | [CI and data governance](./phase-05-ci-data-governance.md) | Partially complete | 2.5d | Bad dataset release blocked automatically |

## Dependency flow

```text
Contain v1 -> Build manifest -> Cluster/dedupe -> Grouped split
                                            -> Shortcut gate
                                            -> Recollect if failed
                                            -> Freeze test v2
                                            -> Train/calibrate on train+val
                                            -> One locked test evaluation
                                            -> CI promotion gate
```

## Global release gates

- Exact SHA-256 duplicate groups crossing splits: `0`.
- Confirmed perceptual duplicate components crossing splits: `0`.
- Subject/session/clip/device group overlap: `0`.
- Manifest/provenance coverage: `100%` for released samples.
- Metadata-only AUC point estimate `<= 0.60`, upper bootstrap 95% CI `<= 0.65`.
- Metadata-only accuracy `<= majority baseline + 10 percentage points`.
- Every reported attack type appears in val/test; metrics reported by attack/device/session.
- Test manifest, files and protocol have checksums and are immutable.

## Critical risk

Nếu real và fake hiện được thu từ identity, camera hoặc bối cảnh tách biệt, dedupe/re-split không thể cứu dataset. Phase 3 phải thu lại dữ liệu theo ma trận đối chứng; hậu xử lý brightness/size không phải phương án sửa hợp lệ.

## Execution status — 2026-07-30

- Historical inventory: 2,415 samples; all quarantined due missing provenance.
- Exact leakage reproduced: 264 cross-split groups / 746 references.
- Conservative OpenCV dHash scan: 5,998 cross-split candidates pending review.
- Metadata, low-resolution, color-histogram, background-only and center-only probes: ROC-AUC 1.0.
- Release v2: blocked; train/val/test released counts are 0/0/0.
- Seeded train-only appearance augmentation is implemented and tested; it does not replace balanced data collection.
- Retraining and locked test evaluation intentionally not run.

## Source evidence

- [AI/ML audit](../vshield-ai-ml-audit-2026-07-30.md)
- `scripts/split_data.py:24-43`: shuffle/split theo basename, không có group.
- `scripts/split_data.py:34-36`: floor từng tỷ lệ làm rơi một mẫu.
- `src/vshield/training/train.py:24-27`: chỉ evaluate validation.
