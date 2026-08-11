# Phase 03 — Remove shortcut and acquisition confounds

## Context

- [Plan](./plan.md)
- Evidence: metadata-only `[H,W,bytes,brightness,std,blur]` đạt 99,59% accuracy, AUC 1,0.

## Overview

**Priority:** P0  
**Effort:** 5 engineering days + 1–3 weeks collection elapsed  
**Goal:** Buộc label real/fake không còn được suy ra đáng tin cậy từ identity, thiết bị, session hoặc thống kê ảnh thô.

## Root-cause decision

Không sửa bằng cách resize/normalize brightness hậu kỳ đơn thuần. Brightness, moiré, blur và texture có thể là tín hiệu PAD thật; cưỡng ép phân phối sau thu thập có thể xóa tín hiệu hợp lệ và tạo artifact mới. Cách sửa chính là thiết kế thu thập đối chứng.

## Balanced collection matrix

Với mỗi `subject_id`, thu cả bona fide và attacks dưới các nuisance factor giao nhau:

- Ít nhất 2–3 session khác ngày.
- Ít nhất 2 device/camera khi scope production có nhiều device.
- Nhiều lighting/background/distance/pose, áp dụng cho cả hai label.
- Fake dùng ảnh/video của **cùng target identity**, không dùng người khác làm fake.
- Attack types ghi rõ: print matte/glossy, phone/tablet replay, video replay; mask/deepfake chỉ khi nằm trong product scope.
- Cùng face detector, crop, codec, resolution policy và storage pipeline cho hai label.
- Train augmentations áp dụng đối xứng sau khi split; không tạo augment trước split.

Không đặt quota tuyệt đối nếu chưa biết nguồn lực. Trước release, coverage matrix phải cho thấy mỗi label có mặt trong từng domain quan trọng; ô trống phải được ghi là unsupported domain.

## Proposed files

- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\scripts\shortcut-baselines.py`
- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\scripts\dataset-slice-report.py`
- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\configs\shortcut-gates.yaml`
- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\data\manifests\collection-matrix-v2.csv`
- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\artifacts\evaluation\v2\data-card.md`

## Shortcut probes

Chạy trên train→test theo protocol v2, không tune trên test:

1. Metadata-only logistic/tree: shape, bytes, brightness, contrast, blur, codec.
2. Brightness-only và dimensions-only.
3. Subject/device/session ID classifier.
4. Background-only image: mask toàn face.
5. Face-only central crop: loại background.
6. Very-low-resolution/color-histogram baseline.
7. Distribution tests theo label/split: KS/JS/PSI + plots; không dùng p-value đơn lẻ làm gate.

## Gate policy

- Metadata-only AUC point `<=0.60`; upper subject/session-bootstrap 95% CI `<=0.65`.
- Accuracy `<= majority baseline +10pp`.
- Nếu probe fail: block dataset release, xác định feature/domain gây shortcut, recollect/rebalance; không “chấp nhận vì CNN vẫn tốt”.
- Threshold trên là governance policy, không phải chứng minh toán học rằng không còn mọi shortcut.

## Tests

- Synthetic brightness-label correlation phải làm gate fail.
- Subject-disjoint nhưng device-confounded dataset phải fail device probe.
- Face-masked baseline không được vượt gate.
- Feature extraction deterministic và không đọc pixel vùng đã mask.
- Bootstrap resample theo subject/session, không theo frame.

## Todo

- [x] Lập collection/coverage matrix
- [ ] Thu bổ sung bona fide và attacks đối chứng
- [ ] Đồng nhất capture/preprocessing contract giữa labels
- [x] Chạy nuisance and masked-image probes
- [x] Điều tra mọi failed gate
- [ ] Phê duyệt data card v2

## Success criteria

- Nuisance baseline đạt gate.
- Không identity/device/session nào ánh xạ một-một sang label.
- Mỗi supported attack/device có đủ val/test group độc lập.
- Data card mô tả rõ unsupported domains và residual risks.

## Risks

- Nếu nguồn v1 dùng identity khác nhau cho real/fake, phần lớn dữ liệu phải bỏ hoặc thu lại.
- “Cân bằng histogram” có thể che confound nhưng không tạo dữ liệu production thật.
- Probe gần chance không chứng minh model học đúng; Phase 04 vẫn cần attack/domain evaluation.

## Next

Chỉ freeze dataset v2 khi cả leakage gate và shortcut gate pass.
