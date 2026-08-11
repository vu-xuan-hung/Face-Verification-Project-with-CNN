# Phase 02 — Build manifest, deduplicate, and split by groups

## Context

- [Plan](./plan.md)
- `C:\Users\LOQ\Desktop\ml-ai-dl\project\scripts\split_data.py:24-64`
- `C:\Users\LOQ\Desktop\ml-ai-dl\project\configs\data.yaml:1-4`

## Overview

**Priority:** P0  
**Effort:** 3 days  
**Goal:** Tạo protocol v2 với zero content/group leakage và provenance đầy đủ.

## Data contract

Mỗi row cần: `sample_id`, `relative_path`, `label`, `subject_id`, `session_id`, `clip_id`, `device_id`, `attack_type`, `capture_time`, `sha256`, `dhash`, `phash`, `height`, `width`, `bytes`, `brightness`, `contrast`, `blur`, `duplicate_component_id`, `split`, `status`, `reason_code`.

Group key ưu tiên:

```text
subject_id + session_id + clip_id + device_id + attack_instrument_id
```

Thiếu field định danh group không được tự động gán split; đưa vào quarantine để bổ sung metadata.

## Proposed files

- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\scripts\build-dataset-manifest.py`
- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\scripts\deduplicate-dataset.py`
- Replace logic in `C:\Users\LOQ\Desktop\ml-ai-dl\project\scripts\split_data.py` with manifest-driven grouped split, hoặc retire script bằng wrapper rõ ràng.
- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\configs\dataset-schema.yaml`
- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\data\manifests\dataset-v2.csv`
- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\data\manifests\quarantine-v2.csv`
- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\data\protocols\{train,val,test}-v2.csv`

## Duplicate policy

1. SHA-256 giống nhau: tạo một connected component.
2. Cùng label/group: giữ một canonical sample; các bản còn lại `duplicate_excluded`.
3. Khác label: quarantine cả component và manual review; không “majority vote”.
4. Near-duplicate:
   - Candidate bằng dHash/pHash threshold.
   - Confirm bằng SSIM hoặc LPIPS/embedding, crop tương đồng, temporal adjacency.
   - Union-find để bắt transitive chain `A≈B≈C`.
   - Toàn component vào đúng một split, kể cả candidate chưa review xong.
5. Không chỉ xóa 1.567 cặp độc lập; pairwise deletion có thể để lại leakage bắc cầu.

## Grouped split

1. Chọn một protocol chính trước khi nhìn model:
   - Recommended: test held-out theo subject **và** session/device khi đủ dữ liệu.
   - Fallback: group-disjoint theo capture clip/session; ghi confidence thấp hơn.
2. Stratify theo label và `attack_type` ở cấp group, không phá group để cân bằng số lượng.
3. Không dùng basename/time-frame làm sample độc lập.
4. Phân bổ toàn bộ row; không floor rồi làm rơi mẫu như splitter hiện tại.
5. Freeze test manifest + file hashes trước training.

## Tests

- Exact duplicate xuyên split làm job fail.
- Duplicate khác label bị quarantine.
- Near chain A–B, B–C tạo một component dù A–C ngoài threshold.
- Cùng subject/session/clip/device không vượt split.
- Split deterministic với cùng manifest/config.
- Mọi sample thuộc đúng một trong train/val/test/quarantine.
- File thiếu/không decode/label không hợp lệ làm audit fail.

## Todo

- [x] Định nghĩa schema và reason codes
- [x] Sinh manifest và provenance coverage report
- [x] Cluster exact/perceptual duplicates
- [ ] Manual-review cross-label/borderline clusters
- [ ] Tạo grouped split v2
- [ ] Khóa test v2 bằng checksum
- [x] Viết test leakage/split determinism

## Success criteria

- Exact SHA-256 groups crossing splits: `0`.
- Confirmed near-duplicate components crossing splits: `0`.
- Subject/session/clip/device overlap: `0`.
- Released rows có provenance bắt buộc: `100%`.
- Không sample bị mất âm thầm; counts reconcile chính xác.

## Risks

- dHash ≤4 có false positive/negative: chỉ dùng candidate generation; confirm và review.
- Metadata cũ không có subject/session: không suy diễn identity từ khuôn mặt tự động; recollect hoặc quarantine.
- Grouped split làm tập hiệu dụng nhỏ: báo CI rộng thay vì tái dùng frame tương quan.

## Next

Chạy toàn bộ shortcut probes ở Phase 03. Zero duplicate chưa đủ điều kiện train.
