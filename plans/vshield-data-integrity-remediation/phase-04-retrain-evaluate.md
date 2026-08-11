# Phase 04 — Retrain, calibrate, and evaluate once

## Context

- [Plan](./plan.md)
- `C:\Users\LOQ\Desktop\ml-ai-dl\project\src\vshield\training\train.py:15-27`
- `C:\Users\LOQ\Desktop\ml-ai-dl\project\src\vshield\models\cnn.py:21`

## Overview

**Priority:** P1  
**Effort:** 4 days after data v2 is ready  
**Goal:** Tạo kết quả anti-spoof có thể audit, không dùng test để chọn model/threshold.

## Training protocol

1. Loader nhận protocol CSV và fail nếu path rỗng, hash sai, group overlap hoặc label ngoài schema.
2. Pin seed, framework/dependency versions, preprocessing version và hardware.
3. Augmentation chỉ áp dụng train, đối xứng giữa labels; ablate `RandomRotation(0.2)` vì tương đương khoảng ±72°.
4. Model selection/early stopping chỉ trên validation.
5. Lưu per-sample score trước threshold, model hash, dataset hash, code commit.
6. Chọn operating threshold trên validation theo product cost:
   - Security-first: giới hạn APCER rồi tối thiểu BPCER.
   - Usability-first: giới hạn BPCER nhưng phải có APCER ceiling.
7. Freeze threshold, sau đó chạy test đúng một lần.

## Evaluation outputs

- ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix.
- APCER theo từng attack type, BPCER, ACER.
- Nếu tích hợp authentication: FAR/FRR end-to-end báo riêng, không đánh đồng với APCER/BPCER.
- Slice theo device, session, lighting, subject cohort, attack instrument.
- 95% CI bootstrap theo subject/session/clip.
- Latency p50/p95/p99, throughput, peak RSS/VRAM trên hardware mục tiêu.
- External/camera holdout riêng nếu có; không trộn với test nội bộ.

## Proposed files

- Modify `C:\Users\LOQ\Desktop\ml-ai-dl\project\src\vshield\training\train.py`
- Modify `C:\Users\LOQ\Desktop\ml-ai-dl\project\src\vshield\data\loader.py`
- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\scripts\evaluate-pad.py`
- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\scripts\benchmark-pad.py`
- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\artifacts\evaluation\v2\model-card.md`
- Create tests under `C:\Users\LOQ\Desktop\ml-ai-dl\project\tests\data\` and `tests\evaluation\`.

## Test cases

- Empty/wrong config path exits non-zero.
- Test rows cannot enter fit/early stopping/threshold search.
- Threshold exactly at boundary follows documented comparator.
- Label 0=fake, 1=real verified by golden scores/confusion matrix.
- Shuffled row order does not change metrics.
- Missing attack slice causes report failure, not silent omission.
- Bootstrap unit is group, not frame.
- Model/data/preprocess hash appears in every report.

## Todo

- [x] Version preprocessing and deterministic train config
- [x] Add seeded, label-symmetric, train-only appearance augmentation
- [x] Add fail-fast manifest loader
- [ ] Retrain candidate models on train v2
- [ ] Select model and threshold on val v2
- [ ] Run locked test v2 once
- [ ] Produce PAD, slice, latency and resource reports
- [ ] Independent audit/sign-off before promotion

## Success criteria

- No test access before model+threshold freeze.
- Metrics reproducible from score dump and protocol checksum.
- APCER/BPCER reported overall and by supported attack/device.
- Confidence intervals use independent groups.
- Production threshold traceable to validation objective.

## Risks

- Metric giảm mạnh so với v1 là expected; không “sửa” bằng mở test.
- Test v2 quá nhỏ cho CI hữu ích: thu thêm groups, không thêm correlated frames.
- Accuracy cao nhưng một attack slice thất bại: không promote cho attack scope đó.

## Next

Phase 05 tự động hóa các gates trước mọi dataset/model release.
