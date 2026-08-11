# Phase 01 — Contain contaminated dataset and claims

## Context

- [Plan](./plan.md)
- [Audit evidence](../vshield-ai-ml-audit-2026-07-30.md)
- Current splitter: `C:\Users\LOQ\Desktop\ml-ai-dl\project\scripts\split_data.py:24-43`

## Overview

**Priority:** P0  
**Effort:** 0.5 day  
**Goal:** Ngăn test v1 tiếp tục ảnh hưởng quyết định model trong khi giữ đủ bằng chứng để tái lập audit.

## Requirements

- Đổi trạng thái dataset/split/model v1 thành `invalid_for_model_evaluation`.
- Không xóa dữ liệu. Chuyển sang read-only archive có checksum.
- Ngừng mọi claim accuracy/AUC từ test v1; không dùng test v1 cho early stopping, threshold hoặc model selection.
- Ghi rõ 99,59%/AUC 1,0 là kết quả shortcut detector, không phải anti-spoof model.

## Files to create or modify during implementation

- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\data\manifests\dataset-v1-invalid.json`
- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\artifacts\evaluation\v1\INVALID.md`
- Modify tài liệu/README nào đang công bố metric v1, nếu có.
- Không overwrite `artifacts/models/face_verify_v1.keras`; model mới phải có version mới.

## Implementation steps

1. Chụp inventory: relative path, size, SHA-256 cho toàn bộ source data, split và artifact.
2. Lưu reason codes:
   - `EXACT_DUPLICATE_CROSS_SPLIT`
   - `PERCEPTUAL_DUPLICATE_CROSS_SPLIT`
   - `METADATA_SHORTCUT`
   - `NON_GROUPED_SPLIT`
3. Gắn manifest v1 với audit timestamp, code commit, config và audit report.
4. Thêm banner fail-fast vào workflow evaluation nếu input protocol là v1.
5. Giữ v1 chỉ cho regression/debug, không trộn vào v2.

## Todo

- [x] Lập checksum inventory v1
- [x] Gắn trạng thái invalid và reason codes
- [x] Chặn metric/model promotion dùng test v1
- [x] Rà soát và thu hồi claim accuracy hiện tại

## Success criteria

- Không job train/eval chính thức nào tham chiếu test v1.
- Dữ liệu v1 vẫn tái lập được audit bằng checksum.
- Dashboard/report phân biệt rõ `invalid historical result` với benchmark hợp lệ.

## Risks and mitigations

- **Mất traceability do xóa/sửa tại chỗ:** archive immutable, không destructive cleanup.
- **Nhầm v1 thành baseline:** namespace/artifact version riêng; pipeline promotion reject v1.
- **So sánh “model mới tốt hơn v1”:** cấm so sánh trực tiếp vì protocol thay đổi.

## Security/data protection

- Manifest dùng opaque `subject_id`; không ghi tên thật.
- Hạn chế quyền đọc ảnh mặt gốc; log chỉ lưu hash/ID, không lưu ảnh.

## Next

Phase 02 chỉ đọc archive v1 và tạo dataset v2 qua manifest; không copy ngẫu nhiên theo filename.
