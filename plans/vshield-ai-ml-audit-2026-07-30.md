# V-Shield AI/ML Audit

**Ngày audit:** 2026-07-30 (Asia/Bangkok)
**Phạm vi:** `src/vshield`, `configs`, `scripts`, `data`, `artifacts`, `tests`, `frontend`, Docker/build config
**Vai trò:** Senior AI Engineer / Computer Vision Engineer / ML Auditor
**Nguyên tắc:** source-first, evidence-first; không sửa source sản phẩm trong audit

## Executive summary

V-Shield **chưa ở trạng thái có thể xác thực khuôn mặt hoặc đánh giá model một cách hợp lệ**.

1. `artifacts/models/face_verify_v1.keras` không tồn tại. Runtime thực tế trả HTTP 503 cho mọi ảnh hợp lệ trước cả face detection.
2. `data/faces` có 54 ảnh nằm trực tiếp ở root, không có `data/faces/<username>/`; identity index vì vậy rỗng. Ngay cả khi bổ sung anti-spoof model, không user nào đăng nhập được.
3. Tập anti-spoof bị leakage nghiêm trọng: 264 nhóm duplicate tuyệt đối xuyên split, chứa 746 ảnh; thêm 1.567 cặp near-duplicate dHash ≤ 4 bit.
4. Test split bị shortcut/confound: classifier logistic chỉ dùng metadata ảnh, không dùng pixel, đạt **99,59% accuracy / AUC 1,0**. Median brightness của fake là 160,08, real là 78,74; kích thước và dung lượng cũng khác mạnh. Accuracy cao trên test hiện tại không chứng minh anti-spoof.
5. Script train chính thức resolve sai đường dẫn và trả toàn bộ tensor rỗng. Code train chỉ evaluate validation, không evaluate test.
6. Trên chính 122 ảnh `real` của test, Haar pipeline chỉ chấp nhận đúng một mặt ở 91 ảnh. Face gate đã tạo mức reject tối thiểu **31/122 = 25,41%** trước anti-spoof/FaceNet.
7. L2 threshold 0,9 trên unit embedding tương đương cosine similarity 0,595, rất permissive về mặt hình học; chưa có genuine/impostor pairs, ROC, FAR/FRR/EER để kết luận phù hợp.

Kết luận chất lượng:

- **Accuracy / FAR / FRR hiện tại: chưa xác định được.**
- **Anti-spoof accuracy hiện tại: chưa xác định được; test hiện có không hợp lệ để tuyên bố.**
- **Production readiness: không đạt.**
- **Điểm tích cực đã chứng minh:** pipeline fail-closed khi model anti-spoof lỗi; FaceNet chỉ chạy sau khi anti-spoof trả `real`; embedding được L2-normalize; multi-face bị reject; FAISS squared L2 được lấy căn đúng; 50/50 unit/integration test hiện có pass.

## Phương pháp và bằng chứng đã chạy

- Đọc toàn bộ source/config/test/frontend liên quan.
- Kiểm kê artifact, enrollment và 2.416 cặp image-label gốc.
- Decode toàn bộ 2.415 ảnh trong train/val/test; kiểm tra label, shape, dtype, EXIF.
- SHA-256 duplicate audit và dHash near-duplicate audit xuyên split.
- Baseline logistic chỉ dùng metadata ảnh.
- Chạy Haar detector với đúng tham số production trên toàn bộ test và enrollment.
- Chạy API bằng `TestClient`, gồm invalid file, validation error, model thiếu, decompression bomb và `MemoryError`.
- Chạy model cache FaceNet thật trong môi trường hiện tại để xác minh shape, normalization và latency.
- Chạy full test suite: **50 passed, 0 failed, 3 warnings, 14,33 s**.
- Chạy frontend production build: pass, 1.501 modules, 791 ms.
- Môi trường thực tế: TensorFlow 2.18.1, Keras 3.15.1, keras-facenet 0.3.2, OpenCV 4.11.0, FAISS CPU 1.14.3, NumPy 1.26.4; TensorFlow chỉ thấy CPU và binary không build CUDA.

## A. Sơ đồ pipeline hiện tại

```mermaid
flowchart TD
    C[React camera 640x480 hoặc upload <=1024 px<br/>JPEG quality 0.9 data URI]
    P[Pydantic ImagePayload<br/>string <= 11,200,000 chars]
    D[Base64 decode <=8 MiB<br/>PIL header size <=20 MP<br/>cv2.imdecode IMREAD_COLOR]
    M{Anti-spoof model<br/>đã load?}
    F[Haar frontal face detect<br/>equalized grayscale]
    N{Exactly one face?}
    X[Expanded BGR crop<br/>clip vào biên]
    A[Anti-spoof crop<br/>128x128 BGR uint8<br/>float32 /255 -> NHWC batch]
    S{scalar score > 0.5?}
    L[Haar eye alignment<br/>FaceNet crop]
    E[BGR->RGB<br/>keras-facenet resize 160x160<br/>(x-127.5)/127.5]
    V[FaceNet 20180402-114759<br/>512-D -> L2 normalize]
    I[FAISS IndexFlatL2 hoặc NumPy<br/>sqrt squared distance]
    T{best L2 <=0.9<br/>và margin >=0.05?}
    DB[SQLite role lookup + login log]
    R[200 username/role]

    C --> P --> D --> M
    M -- Không --> U1[503 unavailable]
    M -- Có --> F --> N
    N -- Không --> U2[422 invalid_face]
    N -- Có --> X --> A --> S
    S -- Lỗi --> U3[503 unavailable]
    S -- Không --> U4[403 spoof]
    S -- Có --> L --> E --> V --> I --> T
    I -- Index rỗng/lỗi --> U5[503 unavailable]
    T -- Không --> U6[403 unknown]
    T -- Có --> DB
    DB -- Lỗi --> U7[503 database unavailable]
    DB -- OK --> R
```

Điểm khác sơ đồ kỳ vọng: `AuthenticationService.authenticate()` kiểm tra model anti-spoof tồn tại **trước face detection** (`services/authentication.py:64-72`). Trong trạng thái repo hiện tại, pipeline dừng ở đó.

### Trace chi tiết theo bước

| Bước | File / hàm | Input → output | Màu / dtype / shape | Dừng và exception | Tác động |
|---|---|---|---|---|---|
| Client camera | `frontend/src/pages/Login.jsx:34-43` | video frame → JPEG data URI | Canvas RGB nội bộ; thường 640×480; JPEG quality 0,9 | Camera lỗi chỉ hiển thị upload fallback | JPEG và camera domain ảnh hưởng texture PAD |
| Client upload | `Login.jsx:82-103` | file → canvas max side 1024 → JPEG | Giữ aspect; bỏ alpha; browser thường áp orientation khi render | `img.onerror` không xử lý | Upload tĩnh làm replay attack dễ thử |
| Schema | `api/schemas.py:17-27` | JSON string | max 11.200.000 chars | FastAPI 422 `{"detail":[...]}` | Frontend đọc sai error schema |
| Decode | `api/app.py:33-64` | data URI → image | `(H,W,3)`, `uint8`, BGR | custom 400 cho format/base64/PIL thường; bomb/MemoryError có thể 500 | Chặn 8 MiB/20 MP nhưng vẫn có DoS window |
| Model availability | `services/authentication.py:64-69` | service state | — | model `None` → 503 | Hiện tại mọi request dừng tại đây |
| Face detect | `core/face_preprocessor.py:51-71` | BGR image → boxes | gray `(H,W)` uint8; boxes `(N,4)` | 0 hoặc >1 → 422 | Haar yếu với pose/light/small face; false reject |
| Crop | `face_preprocessor.py:73-90` | box → expanded crop | BGR `uint8`, `(Hc,Wc,3)` | crop rỗng → 422 | Clip an toàn; crop thay đổi khi sát biên |
| Anti crop | `face_preprocessor.py:77`; `authentication.py:89` | crop → batch | `(1,128,128,3)` float32 BGR, `x/255` ∈ [0,1] | resize/OpenCV lỗi → 503/422 tùy nơi | Khớp loader train về BGR/[0,1], nhưng detector/crop khác |
| Anti inference | `core/anti_spoof.py:45-67` | batch → scalar | output phải đúng 1 finite numeric score trong [0,1] | lỗi/score sai → 503 fail-closed; score ≤0,5 → spoof | Mapping 0=fake, 1=real đúng theo config/code |
| Eye alignment | `face_preprocessor.py:92-129` | crop → rotated crop | BGR uint8, shape giữ nguyên | không đúng 2 eye → dùng crop không align | Hai detection lớn nhất không được kiểm tra hình học |
| FaceNet wrapper | `core/embedder.py:61-75`; dependency `keras_facenet` | BGR crop → embedding | RGB; dependency resize `(160,160,3)`; `(x-127.5)/127.5`; output `(1,512)` | load/inference lỗi → 503 | Enrollment/verification cùng code; Haar crop khác MTCNN reference |
| Embedding normalize | `embedder.py:20-38` | vector → unit vector | `(512,)`, float32 contiguous, norm 1 | zero/NaN/wrong dim → 503 | Cho phép quan hệ L2–cosine chính xác |
| Identity search | `core/identity_index.py:121-193` | query + `(M,512)` index | FAISS trả squared L2; code `sqrt`; NumPy trả L2 | index rỗng/lỗi → 503 | nearest template 1:N; outlier enrollment tăng FAR |
| Decision | `identity_index.py:185-193` | ranked identities | best ≤0,9; runner-up gap ≥0,05 | fail → unknown 403 | Threshold chưa calibrate; chỉ 1 identity thì không có margin gate |
| DB/response | `api/app.py:114-135` | result → JSON | success `{success,username,role}` | DB lỗi → 503 | Auth đúng vẫn có thể bị từ chối do logging lỗi |

## B. Bảng model / AI component

| Model | Chức năng | Nguồn / version | Input thực tế | Preprocessing | Output | Threshold | Vấn đề |
|---|---|---|---|---|---|---|---|
| OpenCV Haar `haarcascade_frontalface_default.xml` | Face detection | OpenCV bundled; env 4.11.0 | BGR frame bất kỳ | gray + equalizeHist | boxes `(N,4)` | `scaleFactor=1.05`, `minNeighbors=4`, `minSize=30` | Frontal-only, không confidence score, test real reject 25,41% |
| OpenCV Haar `haarcascade_eye.xml` | Roll alignment | OpenCV bundled | BGR face crop | gray, upper 65% | eye boxes | `scaleFactor=1.1`, `minNeighbors=4`, `minSize=8` | Chọn hai eye lớn nhất; không landmark canonical |
| `face_verify_v1.keras` | Anti-spoof binary | Custom; **artifact thiếu, version/weights/hash không có** | `(1,128,128,3)` | BGR, float32 `/255` | một sigmoid score | real khi `score > 0.5` | Không thể inspect/eval; source CNN và notebook không cùng architecture |
| `CNNModel` hiện tại | Code để train anti-spoof mới | `models/cnn.py`; custom Sequential | `(N,128,128,3)` | BGR/[0,1], NHWC | `(N,1)` sigmoid | 0,5 | 4.289.217 params; khác notebook/README artifact ~515 MB |
| Notebook anti-spoof | Nguồn artifact có khả năng trước đây | `notebooks/model.ipynb`; custom MBConv/Fused-MBConv | `(N,128,128,3)` | BGR/[0,1] | `(N,1)` sigmoid | 0,5 | Không model card/checksum; không chứng minh artifact từng dùng architecture này |
| FaceNet | Face embedding | `keras-facenet` 0.3.2, implicit key `20180402-114759`, weights từ David Sandberg wrapper | cropped RGB | resize 160², `(x-127.5)/127.5` | 512-D | project L2 0,9 | Version key implicit; dependency reference dùng MTCNN/rough alignment, project dùng Haar |
| FAISS `IndexFlatL2` | Exact vector search | faiss-cpu 1.14.3 | `(1,512)` vs `(M,512)` unit float32 | none | squared distances + IDs | sqrt rồi L2 gate | CPU only; exact search; snapshot immutable |
| `blaze_face_short_range.tflite` | Không có vai trò | file root, untracked | — | — | — | — | Không có reference trong source; **không phải detector runtime** |

Keras/TensorFlow không dùng `eval()`/`torch.no_grad()` vì đó là API PyTorch. `model.predict()` chạy inference mode, nên Dropout/augmentation tắt và BatchNorm dùng moving statistics. Tuy vậy anti model được `load_model()` với compile mặc định, không có signature validation và không pin device.

## C. Bảng findings

| Severity | Thành phần | Vấn đề | Ảnh hưởng | Bằng chứng | Ưu tiên |
|---|---|---|---|---|---|
| Critical | Runtime model | Anti-spoof artifact thiếu | Mọi ảnh hợp lệ → 503 | artifact count=0; API chạy thật trả 503 | P0 |
| Critical | Enrollment | 54 ảnh sai cấu trúc, 0 username dir | Identity index rỗng; không ai login | filesystem + `verifier.py:40-63` | P0 |
| Critical | PAD data | Duplicate/near-duplicate xuyên split | Test leakage, metric lạc quan giả | 264 groups/746 images; 1.567 near pairs | P0 |
| Critical | PAD data | Identity/device/brightness/size confound | Model học shortcut thay vì spoof | metadata-only 99,59% accuracy, AUC 1,0 | P0 |
| High | Training | Config resolve sai; không eval test | Train chính thức không chạy; không có test metric | loader trả toàn tensor `(0,)` | P0 |
| High | Model governance | Cùng tên artifact cho hai architecture | Không tái lập/không biết model production | `cnn.py`, notebook, README mismatch | P0 |
| High | Face gate | Haar false reject và false multi-face | Tăng FRR, giảm camera robustness | real test: 91/122 exactly-one | P1 |
| High | PAD security | Single-frame passive CNN + upload tĩnh | Replay/print/screen chưa thấy có thể pass | pipeline/source; không challenge/temporal signal | P1 |
| High | Recognition | 1:N nearest-template + L2 0,9 chưa calibrate | FAR tăng theo số user/template | `identity_index.py:14,174-193` | P1 |
| High | Input/API | Bomb/MemoryError → 500; decode ảnh lớn trước CV | DoS, RAM spike, API instability | manual test 500 | P1 |
| High | API security | `/logs*` không auth, không rate limit; role phía client | Rò log/brute-force/production insecure | `app.py:91-155`, frontend localStorage | P1 |
| Medium | Performance | CPU-only, cold FaceNet 27 s, locks serialize stage | Cold start/latency/throughput thấp | benchmark local | P1 |
| Medium | Lifecycle | Index chỉ rebuild startup, model lỗi không retry/health | Stale enrollment, service “up” nhưng unusable | `authentication.py:133-159` | P1 |
| Medium | Preprocessing | Haar vs CvZone/MTCNN; eye alignment không canonical; EXIF bỏ qua | Embedding/PAD domain shift, FRR | source trace | P2 |
| Medium | Frontend/API | FastAPI validation `detail`, UI đọc `message` | Hiển thị `Thất bại: undefined` | manual 422 + `Login.jsx:67-69` | P2 |
| Medium | Dependency/deploy | Docker không copy lock; broad model deps; global Keras monkeypatch | Build không tái lập, compatibility risk | Dockerfile/pyproject/anti_spoof.py | P2 |

## Findings chi tiết

### F-01 — [Severity: Critical] Runtime hiện tại không có anti-spoof model

**Phân loại:** Đã chứng minh bằng filesystem và test runtime.

**Vị trí:**
- `src/vshield/services/authentication.py:64-69, 138-147`
- `src/vshield/core/anti_spoof.py:36-43`
- `artifacts/models/` — 0 file

**Hiện trạng:** `load_model()` fail, trả `None`; app vẫn startup. `authenticate()` trả unavailable trước face detection. Valid PNG qua API trả 503 `Anti-spoofing service unavailable`.

**Nguyên nhân:** Artifact gitignored và không được provision; không readiness gate/checksum.

**Ảnh hưởng đến AI:** Không có anti-spoof, embedding hay threshold decision nào chạy.

**Ảnh hưởng hệ thống:** 100% authentication request hợp lệ thất bại; process vẫn trông như healthy.

**Cách tái hiện:** Gọi `build_default_authentication_service(Path.cwd())`; model `None`, index size 0; POST valid data URI → 503.

**Đề xuất sửa:** Provision immutable artifact theo version + SHA-256; validate input/output signature khi startup; readiness fail nếu model/index không usable.

**Test cần bổ sung:** Startup contract test với model thật; checksum mismatch; wrong input/output shape; readiness status.

### F-02 — [Severity: Critical] Enrollment sai cấu trúc làm index rỗng

**Phân loại:** Đã chứng minh.

**Vị trí:**
- `src/vshield/core/verifier.py:27-63`
- `src/vshield/services/authentication.py:149-153`
- `data/faces/`

**Hiện trạng:** 54 file root-level, 0 user directory. Loader cảnh báo rồi bỏ qua root images. 11/54 ảnh Haar không thấy mặt; 17/54 bị phát hiện >1 mặt; có diagram như `distance_matrix.png`.

**Nguyên nhân:** Dữ liệu không theo `faces/<username>/*`; không có enrollment API/validator.

**Ảnh hưởng đến AI:** 0 embedding, index rỗng; ảnh diagram/low-quality sẽ bị skip nếu chuyển bừa vào folder.

**Ảnh hưởng hệ thống:** Khi anti model có mặt, ảnh `real` vẫn chạy FaceNet rồi search ném `IdentityIndexUnavailableError` → 503; không user thật login được.

**Cách tái hiện:** `load_database("data/faces", stub_encoder)` trả `{}`; test hiện có cũng xác nhận root file bị ignore.

**Đề xuất sửa:** Tạo manifest enrollment rõ username/role/template ID; validate exactly-one face + quality + duplicate; build index atomic; không startup-ready nếu index rỗng.

**Test cần bổ sung:** Wrong-folder integration test trên cấu trúc dữ liệu thật; mixed valid/invalid; duplicate identity giữa folders.

### F-03 — [Severity: Critical] Train/val/test leakage do split từng frame

**Phân loại:** Đã chứng minh bằng SHA-256 và source split.

**Vị trí:**
- `scripts/split_data.py:6-43, 53-64`
- `data/SplitData/*`

**Hiện trạng:** 264 nhóm duplicate tuyệt đối xuyên split, tổng 746 image references; 1.567 cross-split dHash pairs ≤4 bit. Split random từng timestamp frame từ cùng video/session.

**Nguyên nhân:** Không group theo subject/session/device/attack clip; consecutive frames gần như giống nhau được shuffle độc lập.

**Ảnh hưởng đến AI:** Validation/test thấy lại train content; accuracy/precision/recall/F1 bị inflate; threshold 0,5 có thể overfit session.

**Ảnh hưởng hệ thống:** Tạo niềm tin sai khi production gặp camera/device/attack mới.

**Cách tái hiện:** SHA-256 từng ảnh và group hash theo split; ví dụ cùng content xuất hiện train/val/test quanh timestamp `177434219...`.

**Đề xuất sửa:** Deduplicate trước split; GroupShuffleSplit theo `subject_id + capture_session + device + attack_clip`; khóa test trước train.

**Test cần bổ sung:** CI assert zero exact/perceptual duplicate xuyên split và zero group-ID overlap.

### F-04 — [Severity: Critical] Dataset có shortcut identity/device/brightness, không đo liveness

**Phân loại:** Đã chứng minh định lượng; identity confound được quan sát trên sample, cần metadata subject để định lượng đầy đủ.

**Vị trí:**
- `src/vshield/data/collector.py:79-187`
- `scripts/collect_data.py:24-34, 180-194`
- `data/All`, `data/SplitData`

**Hiện trạng:** Fake median brightness 160,08 vs real 78,74; fake median shape 141×126 vs real 206×183. Logistic chỉ dùng height/width/file-size/brightness/std/blur đạt test accuracy 99,59%, AUC 1,0. Sample fake và real là các danh tính khác nhau.

**Nguyên nhân:** Thu thập theo các chuỗi riêng, label manual; không cân bằng identity/background/device/light giữa bona fide và attack. Blink/motion objects trong collector mới không được dùng để label/filter.

**Ảnh hưởng đến AI:** CNN có thể nhận diện người, độ sáng, crop size hoặc camera thay vì dấu hiệu presentation attack; ảnh giả đúng identity/ánh sáng mới có FAR cao.

**Ảnh hưởng hệ thống:** Anti-spoof dễ vỡ cross-camera/cross-person; metric test không transferable.

**Cách tái hiện:** Fit logistic metadata trên train, evaluate frozen trên test: confusion `[[118,1],[0,122]]`.

**Đề xuất sửa:** Mỗi subject phải có cả bona fide và nhiều PAI; cùng device/background/light; thêm cross-device/cross-subject holdout và public PAD datasets.

**Test cần bổ sung:** Metadata-only baseline phải gần chance; identity-prediction probe; cross-domain PAD benchmark.

### F-05 — [Severity: High] Training entrypoint không load data và không evaluate test

**Phân loại:** Đã chứng minh.

**Vị trí:**
- `configs/data.yaml:1-4`
- `src/vshield/data/loader.py:47-65, 98-109`
- `src/vshield/training/train.py:11-29`
- `scripts/split_data.py:34-43, 69-79`

**Hiện trạng:** Loader neo `./data/SplitData` theo thư mục `configs`, thành `configs/data/SplitData`. Kết quả chạy thật: mọi X/y shape `(0,)`. Train chỉ evaluate `X_val`; `X_test` không dùng. Split 2.416 sample thành 2.415 và drop stem `17743470786603053`.

**Nguyên nhân:** Semantics path không thống nhất; floor ratios không cấp remainder; data YAML trong split dùng leading `/`; file YAML mở append; `set()` làm reproducibility phụ thuộc hash seed.

**Ảnh hưởng đến AI:** Không thể retrain/reproduce; test set tồn tại nhưng không có model metric.

**Ảnh hưởng hệ thống:** `make train` in “Không tìm thấy dữ liệu training”; model cũ/thiếu không được thay.

**Cách tái hiện:** Gọi `load_data_from_config("configs/data.yaml")`.

**Đề xuất sửa:** Schema config với root resolve rõ theo project root; validate split non-empty/class IDs; deterministic grouped split; evaluate test một lần sau freeze.

**Test cần bổ sung:** Config path integration test; ratio/remainder; repeatability; train must fail non-zero on empty split.

### F-06 — [Severity: High] Không có model provenance; code và artifact name bị architecture drift

**Phân loại:** Đã chứng minh ở source; architecture artifact thực tế không xác minh được vì file thiếu.

**Vị trí:**
- `src/vshield/models/cnn.py:6-72`
- `notebooks/model.ipynb` code cells quanh lines 18-203
- `README.md:128-150`

**Hiện trạng:** `CNNModel` hiện tại là 3 Conv blocks + Flatten, 4.289.217 params. Notebook dùng hàng chục MBConv/Fused-MBConv blocks. Cả hai save cùng tên `face_verify_v1.keras`; README mô tả khoảng 515 MB.

**Nguyên nhân:** Không model registry/model card/git SHA/data hash/preprocess signature.

**Ảnh hưởng đến AI:** Không biết weights được train bằng architecture/data nào; không so sánh train vs inference đáng tin cậy.

**Ảnh hưởng hệ thống:** Một lần `make train` có thể âm thầm thay model lớn bằng model khác cùng version filename.

**Cách tái hiện:** Build source model và `count_params()`; so notebook.

**Đề xuất sửa:** Semantic version immutable (`pad-cnn-v2.1.0`), manifest JSON chứa code SHA, dataset hash, labels, input, normalization, metrics, threshold, framework versions, artifact SHA.

**Test cần bổ sung:** Artifact contract + golden tensor output + manifest/hash verification.

### F-07 — [Severity: High] Face detector/alignment là nguồn FRR lớn

**Phân loại:** Đã chứng minh trên project test; generalization camera thật cần thêm dữ liệu.

**Vị trí:**
- `src/vshield/core/face_preprocessor.py:37-129`

**Hiện trạng:** Haar frontal face; exactly-one strict. Trong 122 real test: 91 one-face, 21 no-face, 9 two-face, 1 three-face. Pre-model reject 25,41%. Eye alignment dùng hai detections lớn nhất hoặc bỏ alignment.

**Nguyên nhân:** Detector cũ, no confidence/landmarks; test crops cũng có false multi-face.

**Ảnh hưởng đến AI:** FRR cao; crop/rotation không ổn định làm embedding cùng người lệch; anti-spoof nhận crop khác train.

**Ảnh hưởng hệ thống:** Người dùng thấy 422 dù ảnh có mặt; detector lock giới hạn throughput.

**Cách tái hiện:** Chạy đúng `detectMultiScale()` production trên test.

**Đề xuất sửa:** Benchmark RetinaFace/SCRFD/MediaPipe detector có 5-point landmarks; align canonical; quality gate pose/size/occlusion; vẫn reject multi-face.

**Test cần bổ sung:** Detector recall theo yaw/pitch/light/size; alignment consistency; boundary crop.

### F-08 — [Severity: High] Anti-spoof chỉ là passive single-frame classifier

**Phân loại:** Rủi ro kiến trúc đã chứng minh bằng source; attack success rate cần red-team data.

**Vị trí:**
- `services/authentication.py:89-103`
- `core/anti_spoof.py:45-67`
- `frontend/src/pages/Login.jsx:82-103`

**Hiện trạng:** Một JPEG/upload tĩnh → một sigmoid score. Không temporal consistency, challenge-response, sensor attestation, depth/IR hoặc injection detection.

**Nguyên nhân:** PAD architecture tối giản và data hạn chế.

**Ảnh hưởng đến AI:** Print/screen/replay/domain mới có thể qua; fail-closed chỉ xử lý model error, không xử lý model confidently wrong.

**Ảnh hưởng hệ thống:** Endpoint công khai cho phép thử attack không giới hạn.

**Cách tái hiện:** Source trace; model thiếu nên chưa thể đo APCER/IAPAR thực tế.

**Đề xuất sửa:** Trước mắt passive PAD được calibrate cross-domain; production high-risk cần short video/challenge + anti-injection/device binding và rate limit.

**Test cần bổ sung:** Print glossy/matte, phone/tablet nhiều brightness, replay video, mask, deepfake, camera virtual/injected frame.

### F-09 — [Severity: High] Recognition là open-set 1:N với threshold chưa calibrate

**Phân loại:** Logic đã chứng minh; mức FAR/FRR cần dữ liệu.

**Vị trí:**
- `core/identity_index.py:14-15, 67-193`
- `core/embedder.py:20-38`

**Hiện trạng:** Unit-normalized 512-D, nearest individual template, accept best L2 ≤0,9 và margin ≥0,05. Nếu chỉ một identity, margin không áp dụng.

**Nguyên nhân:** Threshold hard-code, không genuine/impostor calibration; endpoint không có claimed username nên là identification 1:N chứ không phải verification 1:1.

**Ảnh hưởng đến AI:** FAR tăng theo số identity/template; template outlier có thể hút impostor; threshold 0,9 tương đương cosine 0,595 và góc 53,49°.

**Ảnh hưởng hệ thống:** Scale enrollment thay đổi security mà config không đổi.

**Cách tái hiện:** `cos = 1 - 0.9²/2 = 0.595`. Code lấy căn FAISS đúng, không nhầm squared distance.

**Đề xuất sửa:** Nếu login có username, chuyển 1:1 claimed identity; quality-weighted templates/centroid; calibrate threshold/margin trên validation camera thật và freeze cho test.

**Test cần bổ sung:** Genuine/impostor pairs; population scaling; template outlier; exact boundary/float32; FAISS/NumPy parity.

### F-10 — [Severity: High] Image bomb và memory error vẫn thành HTTP 500

**Phân loại:** Đã chứng minh bằng test.

**Vị trí:**
- `api/app.py:33-64, 100-112`
- `api/schemas.py:20-26`

**Hiện trạng:** Có 8 MiB/20 MP guards và content decode thật, nhưng chỉ catch `UnidentifiedImageError`, `OSError`, `ValueError`. Mock `PIL.Image.DecompressionBombError` và `MemoryError` đều trả 500 text/plain.

**Nguyên nhân:** Exception taxonomy thiếu; ảnh được decode full-resolution trước downscale.

**Ảnh hưởng đến AI:** Không trực tiếp thay accuracy; concurrent OOM có thể làm model/process lỗi.

**Ảnh hưởng hệ thống:** DoS/RAM spike; response schema bất nhất; 20 MP BGR riêng đã khoảng 60 MB/request chưa tính copies.

**Cách tái hiện:** Patch `Image.open` raise bomb/MemoryError, POST `/predict` với `raise_server_exceptions=False`.

**Đề xuất sửa:** Catch bomb rõ ràng; map invalid input 400/413/422; hard pixel/dimension limit trước allocation khi decoder hỗ trợ; request/rate/concurrency limits; worker memory limits.

**Test cần bổ sung:** Real crafted decompression bomb, truncated image, huge dimensions, parallel 20 MP requests, cv2.error.

### F-11 — [Severity: Medium] CPU-only + cold start và locks hạn chế throughput

**Phân loại:** Đã benchmark local; production hardware khác cần benchmark lại.

**Vị trí:**
- `core/embedder.py:44-75`
- `services/authentication.py:62, 89-92`
- `core/face_preprocessor.py:42, 62, 96`

**Hiện trạng:** TF chỉ thấy CPU; OpenCV/FAISS CPU. FaceNet cached weights cold encode 27,015 s; warm 223,75–279,22 ms trên sample. Haar first call ~1,75 s, warm ~6,6–18 ms. Hai request mock concurrent hoàn tất 0,606 s với mỗi anti/embedding stage 0,2 s, xác nhận stage locks serialize inference.

**Nguyên nhân:** Heavy TensorFlow initialization; one shared lock/model; no warmup/batching/metrics.

**Ảnh hưởng đến AI:** Không đổi metric offline, nhưng timeout/load có thể tăng system-level FRR.

**Ảnh hưởng hệ thống:** Low throughput; nhiều workers nhân RAM; first request rất chậm.

**Cách tái hiện:** Benchmark local đã ghi ở phương pháp.

**Đề xuất sửa:** Startup warmup; per-stage timing; bounded concurrency; benchmark ONNX/TFLite; model pool chỉ sau khi xác minh thread safety và RAM.

**Test cần bổ sung:** cold/warm p50/p95/p99, concurrency 1/2/4/8, RSS/VRAM, soak test.

### F-12 — [Severity: Medium] Lifecycle model/index không production-safe

**Phân loại:** Đã chứng minh bằng source.

**Vị trí:**
- `api/app.py:81-89`
- `services/authentication.py:133-159`
- `core/embedder.py:49-59`

**Hiện trạng:** Snapshot enrollment chỉ dựng startup. Ảnh thay đổi không invalidate. Anti model load fail không retry. FaceNet load fail giữ `_model=None`, nên request sau có thể retry load lặp lại. Enrollment chỉ được build nếu anti model load thành công.

**Nguyên nhân:** Không registry/reload/readiness/generation ID.

**Ảnh hưởng đến AI:** Embedding stale cho tới restart; model và index có thể lệch generation.

**Ảnh hưởng hệ thống:** Service “up” nhưng 503; restart tốn cold start; first-request retry storm tiềm năng.

**Cách tái hiện:** Test hiện có xác nhận anti model thiếu thì enrollment encoder không được gọi.

**Đề xuất sửa:** Explicit startup state machine; atomic index generations; admin rebuild job; readiness includes model/index generation; no implicit network download lúc request.

**Test cần bổ sung:** Hot rebuild atomicity, concurrent reads during swap, failed generation rollback.

### F-13 — [Severity: High] API auth/security và frontend error contract chưa đúng

**Phân loại:** Đã chứng minh bằng source/test.

**Vị trí:**
- `api/app.py:91-98, 137-155`
- `api/schemas.py:35-56`
- `frontend/src/pages/Login.jsx:55-69`
- `frontend/src/pages/AdminDashboard.jsx:11-31`

**Hiện trạng:** CORS `*`; `/predict`, `/logs`, `/logs/export` không auth/rate limit. Admin chỉ kiểm tra `localStorage`. Pydantic error dùng `detail`; UI đọc `message` → `undefined`. Response models khai báo nhưng route không dùng.

**Nguyên nhân:** Client-side authorization và thiếu unified error schema.

**Ảnh hưởng đến AI:** Brute-force làm tăng số impostor attempts, biến FAR nhỏ thành compromise probability lớn.

**Ảnh hưởng hệ thống:** Bất kỳ ai gọi logs/export; localStorage role giả được; production frontend hard-code `localhost:8000`.

**Cách tái hiện:** POST `{}` → 422 `detail`; source UI hiển thị `Thất bại: undefined`.

**Đề xuất sửa:** Server-side session/JWT/RBAC, rate limit/lockout/audit; API base URL env; exception handler trả schema thống nhất.

**Test cần bổ sung:** AuthZ logs, brute-force/rate limit, validation contract, remote deployment URL.

### F-14 — [Severity: Medium] Preprocessing train/inference chỉ khớp một phần

**Phân loại:** Source mismatch đã chứng minh; mức accuracy impact cần experiment.

**Vị trí:**
- `data/loader.py:70-94`
- `data/collector.py:118-187`
- `face_preprocessor.py:60-79`
- `embedder.py:61-75`

**Hiện trạng:**
- Anti train và inference đều BGR, resize 128, float32/[0,1] — phần này đúng.
- Train crop do CvZone detector hoặc script cũ; inference crop do Haar.
- Enrollment và verification cùng FacePreprocessor/FaceEmbedder — nhất quán.
- FaceNet nhận RGB, dependency tự resize 160 và normalize [-1,1] — đúng theo installed model metadata.
- OpenCV decode bỏ EXIF orientation; Haar eye alignment không tương đương MTCNN/5-point alignment.
- Grayscale/RGBA được `IMREAD_COLOR` ép thành 3-channel BGR — không crash, nhưng alpha bị bỏ.

**Nguyên nhân:** Nhiều detector và toolchain không có shared preprocess contract.

**Ảnh hưởng đến AI:** Crop/domain shift làm PAD score và embedding dao động; cùng người có thể tăng L2.

**Ảnh hưởng hệ thống:** Camera/browser/device khác cho kết quả không ổn định.

**Cách tái hiện:** Spy tensor tại train loader, enrollment và request; so hash/stat/shape.

**Đề xuất sửa:** Một versioned preprocessing module/manifest; canonical landmarks; EXIF transpose ở upload; golden images/tensors.

**Test cần bổ sung:** Enrollment/verification tensor equivalence; BGR/RGB color-patch test; EXIF rotation; grayscale/RGBA.

### F-15 — [Severity: Medium] Build/dependency không tái lập và monkeypatch Keras toàn cục

**Phân loại:** Source risk đã chứng minh.

**Vị trí:**
- `Dockerfile.backend:16-24`
- `pyproject.toml:31-55`
- `core/anti_spoof.py:9-33`

**Hiện trạng:** Docker copy `pyproject.toml` nhưng không copy `uv.lock`, cài broad ranges. Cài cả `opencv-python` và `opencv-contrib-python`. Import anti-spoof monkeypatch `from_config` của nhiều Keras layer class toàn process để bỏ `quantization_config`.

**Nguyên nhân:** Artifact/framework incompatibility được xử lý bằng global patch thay vì pin/migrate model.

**Ảnh hưởng đến AI:** Model deserialize có thể thay semantics; FaceNet load sau patch chịu global side effects chưa test.

**Ảnh hưởng hệ thống:** Build theo ngày có dependency khác; image lớn; compatibility/cold-start rủi ro.

**Cách tái hiện:** So Dockerfile với lock; inspect patched classes.

**Đề xuất sửa:** Copy/use lock; split runtime/training extras; pin tested matrix; convert artifact một lần bằng tool riêng, không monkeypatch global runtime.

**Test cần bổ sung:** Clean Docker build, offline startup, model load golden output, dependency matrix.

## Threshold inventory

| Threshold/config | Vị trí | Giá trị | Metric / quyết định | Cơ sở hiện có | Rủi ro |
|---|---|---:|---|---|---|
| Anti-spoof | `core/anti_spoof.py:67` | 0,5 | sigmoid; `>0.5` real | Hard-code | Chưa ROC/APCER/BPCER; 0,5 đúng mapping nhưng chưa chắc operating point |
| Face L2 | `core/identity_index.py:14,186` | 0,9 | unit-vector Euclidean; accept `<=` | Hard-code | cosine 0,595, chưa FAR/FRR |
| Identity margin | `identity_index.py:15,189-191` | 0,05 | runner-up L2 minus best | Hard-code | Không áp dụng khi chỉ một identity |
| Face cascade | `face_preprocessor.py:65-67` | 1,05 / 4 / 30 px | scale/minNeighbors/minSize | Hard-code | Không probability confidence; false reject/multi-face |
| Eye cascade | `face_preprocessor.py:99-101` | 1,1 / 4 / 8 px | scale/minNeighbors/minSize | Hard-code | False eye → rotation sai |
| Collector face confidence | `data/collector.py:38,97,139` | 0,80 | CvZone detection score | Hard-code | Không phải inference detector |
| Collector blur | `data/collector.py:36,162` | >35 | Laplacian variance | Hard-code | Resolution-dependent; không dùng inference/enrollment |
| Blink | `data/collector.py:37` | <0,20 | EAR | Khai báo nhưng collector mới không dùng | Tài liệu gây hiểu nhầm |
| MOG2 var | `data/collector.py:99-100` | 16 | background subtraction | Object tạo nhưng không dùng | Tốn resource collector, không tạo liveness label |
| Payload | `api/app.py:29` | 8 MiB decoded bytes | reject input | Hard safety limit | Compressed bomb vẫn có window |
| Pixel | `api/app.py:30` | 20 MP | reject input | Hard safety limit | ~60 MB BGR/request, chưa tính copies |
| Schema chars | `api/schemas.py:22` | 11.200.000 | Pydantic length | xấp xỉ base64 8 MiB | Lỗi schema khác custom API |
| Frontend upload | `Login.jsx:90` | max side 1024 | resize client | UI-only | API caller bypass được |

Quan hệ chính xác với unit embedding:

```text
L2² = ||u-v||² = 2 - 2*cosine_similarity
cosine_similarity = 1 - L2²/2

L2 = 0,9  =>  cosine = 1 - 0,81/2 = 0,595
```

Không được coi 0,9 hoặc 0,5 là “chuẩn” nếu chưa calibrate bằng dữ liệu deployment.

## Enrollment audit

### Luồng code

```text
Không có enrollment API
→ admin đặt file thủ công tại data/faces/<username>/*
→ startup chỉ khi anti model load được
→ cv2.imread BGR
→ Haar exactly-one face
→ expanded crop + optional Haar-eye rotation
→ BGR→RGB
→ keras-facenet resize 160 + [-1,1]
→ 512-D + L2 normalize
→ giữ trong RAM dict/list
→ flatten thành immutable FAISS/NumPy snapshot
```

### Dữ liệu thực tế

- Root files: 54; user directories: 0.
- Extension: 43 JPG, 11 PNG; 0 unreadable; 0 exact duplicate.
- Haar: 26 exactly-one, 11 no-face, 17 multi-face.
- Nhiều diagram/non-enrollment image; nhiều ảnh tối/mờ nghiêm trọng.
- Không quality gate blur/brightness/pose/occlusion.
- Không cache embedding trên disk, nên không có “embedding cũ” qua restart; nhưng snapshot trong RAM stale sau khi ảnh đổi cho tới restart.
- Không rebuild/invalidate/index version.
- Không anti-spoof hoặc trusted enrollment ceremony.

## Data/training audit

### Split hiện có

| Split | Ảnh | Fake (0) | Real (1) | Missing label | Invalid class | Unreadable |
|---|---:|---:|---:|---:|---:|---:|
| Train | 1.691 | 874 | 817 | 0 | 0 | 0 |
| Val | 483 | 269 | 214 | 0 | 0 | 0 |
| Test | 241 | 119 | 122 | 0 | 0 | 0 |

Điểm đúng: label mapping 0=fake, 1=real nhất quán giữa config, collector, BCE sigmoid và inference.

Điểm sai/rủi ro:

- `data/Real` vẫn chứa 192 label 0 và 504 label 1; folder name không đáng tin, may là loader đọc label file.
- Loader gán missing label thành fake (`loader.py:86-88`) thay vì fail; current split chưa có missing label.
- Augmentation `RandomRotation(0.2)` tương đương khoảng ±72°, quá mạnh cho face PAD và chưa ablation.
- Không seed TensorFlow/NumPy; không deterministic training.
- Chỉ metric accuracy; không AUC/precision/recall/F1/APCER/BPCER.
- Không class/device/subject/session balancing.
- Không model checkpoint manifest; chỉ EarlyStopping restore best val weights.

## Eval trên tập test — kết quả có thể và không thể báo cáo

### Không thể báo cáo model anti-spoof accuracy

Lý do:

1. `face_verify_v1.keras` không tồn tại.
2. `train.py` không load được data do path bug.
3. `train.py` không gọi evaluate trên `X_test`.
4. Test bị duplicate leakage và metadata shortcut, nên dù có artifact, metric trực tiếp cũng không đủ giá trị production.

Do đó báo một con số “anti-spoof test accuracy” lúc này sẽ là bịa hoặc đánh giá nhầm model.

### Kết quả audit trên test đã chạy

| Kiểm tra | Kết quả | Diễn giải |
|---|---:|---|
| Decode/label integrity | 241/241 readable, label hợp lệ | File-level integrity ổn |
| Exactly-one Haar face | 185/241 = 76,76% | 56 ảnh bị chặn trước PAD |
| Exactly-one trên real | 91/122 = 74,59% | Gate-only real reject 25,41% |
| Exactly-one trên fake | 94/119 = 78,99% | 25 fake không tới PAD |
| Metadata-only baseline | Accuracy 99,59%; AUC 1,0 | Test đo shortcut rất mạnh |
| Exact cross-split leakage | 264 groups / 746 references | Test không độc lập |
| dHash ≤4 cross-split | 1.567 pairs | Near-duplicate leakage |

### Face recognition eval

Không có identity-labelled probe/enrollment protocol hoặc genuine/impostor pair list. Vì vậy chưa thể tính FAR, FRR, EER cho FaceNet. `data/faces` cũng không theo username structure.

## API/frontend response audit

| Case | Backend hiện tại | HTTP | Frontend |
|---|---|---:|---|
| Success | `{success:true, username, role}` | 200 | Đúng |
| Custom invalid image | `{success:false,message}` | 400 | Đúng |
| No/multi face | `{success:false,message}` | 422 | Đúng nếu route chạy |
| Spoof | `{success:false,message}` | 403 | Đúng |
| Unknown | `{success:false,message}` | 403 | Đúng |
| Empty index/model error | `{success:false,message}` | 503 | Đúng text, không phân loại retry |
| Pydantic validation | `{"detail":[...]}` | 422 | Hiển thị `undefined` |
| Decompression bomb/MemoryError | text `Internal Server Error` | 500 | JSON parse throw → generic connection error |
| `/logs` | raw list | 200 | Khớp frontend hiện tại, khác `LogsResponse` class không dùng |

## Hiệu năng và resource

Thứ tự bottleneck dự kiến:

1. FaceNet CPU inference, lock serialize; warm local khoảng 224–279 ms/sample.
2. Anti-spoof artifact theo README khoảng 515 MB, nhưng file thiếu nên không benchmark được; architecture notebook rất lớn.
3. Cold model initialization: FaceNet local 27 s; Haar first call ~1,75 s.
4. Decode full-resolution ảnh và các copies; 20 MP gây RAM spike.
5. TensorFlow locks và one-request batch; không batching.
6. Enrollment recompute tuần tự mọi ảnh lúc startup.
7. SQLite logging sau auth có thể serialize/lock và biến auth thành 503.

Endpoint `/predict` là synchronous `def`, nên FastAPI chạy trong thread pool thay vì block event loop. Đây là lựa chọn đúng hơn `async def` cho code blocking, nhưng thread pool không giải quyết GPU/TF locks hoặc RAM concurrency.

## D. Top 5 yếu tố ảnh hưởng lớn nhất tới độ chính xác

1. Dataset PAD bị leakage + identity/device/brightness shortcut.
2. Haar face detector/alignment: gate-only reject 25,41% real test.
3. Không có calibration threshold cho anti-spoof, L2 và margin.
4. Single-frame PAD và domain shift collector CvZone → inference Haar/camera thật.
5. Enrollment không hợp lệ/không quality-controlled; nearest outlier template 1:N.

## E. Top 5 yếu tố ảnh hưởng lớn nhất tới production

1. Thiếu anti model và index rỗng: toàn hệ thống auth không dùng được.
2. Không artifact/model/data provenance; train entrypoint hỏng.
3. API/logs không auth/rate limit; frontend role chỉ localStorage.
4. CPU-only cold start/serialized inference/RAM spike ảnh lớn.
5. Không readiness/metrics/rebuild atomic; exception input gây 500.

## F. Kế hoạch sửa

### Sửa ngay trong 1 ngày

1. Provision đúng model artifact kèm SHA/manifest; readiness fail nếu thiếu.
2. Sắp xếp `data/faces/<username>/`, loại diagram/no-face/multi-face; build index và xác minh size.
3. Sửa path config; train phải exit non-zero nếu split rỗng; thêm test evaluate invocation.
4. Unified error schema; catch decompression bomb; frontend xử lý cả `detail`.
5. Thêm server-side auth cho logs và rate limit `/predict`.
6. Ghi log structured cho model/index state, không chỉ `print`.

### Sửa trong 1 tuần

1. Viết dataset audit/grouped split, dedupe và rebuild clean train/val/test.
2. Thay detector + 5-point alignment sau benchmark; thêm image quality gate.
3. Model registry/manifest/checksum + golden tensor tests.
4. Script evaluate PAD và recognition; threshold calibrate trên validation.
5. Warmup, per-stage latency metrics, readiness/liveness endpoints.
6. Atomic enrollment index rebuild; API/admin job có auth.
7. Pin lock trong Docker; tách runtime/training dependencies.

### Cần thu thập dữ liệu/thực nghiệm

1. Genuine/impostor camera pairs khác ngày/device/session.
2. Bona fide và fake của cùng subject, cùng background/device/light.
3. Print, phone/tablet, replay video, mask/deepfake; nhiều camera/brightness.
4. Pose, blur, glasses, occlusion, skin tone/sex/age coverage.
5. Threshold sweep với confidence interval; cross-device/cross-subject test.

### Cần thay đổi kiến trúc

1. Claimed-identity 1:1 verification nếu UX cho phép, thay vì open-set 1:N login.
2. Temporal/challenge PAD và anti-injection/device attestation cho mức rủi ro cao.
3. Versioned preprocessing contract dùng chung train/enrollment/inference.
4. Model serving state machine + atomic generations + bounded queue/backpressure.

## G. Kế hoạch đánh giá model

### Cấu trúc dữ liệu đề xuất

```text
evaluation/
  manifest.parquet
  pad/
    <subject_id>/<session_id>/<device_id>/<bona_fide_or_attack>/<attack_type>/*
  recognition/
    enrollment/<subject_id>/<session_A>/*
    genuine_probe/<subject_id>/<session_B_or_C>/*
    impostor_probe/<other_subject_id>/<session>/*
  protocols/
    pad_train.csv
    pad_val.csv
    pad_test.csv
    genuine_pairs_val.csv
    impostor_pairs_val.csv
    genuine_pairs_test.csv
    impostor_pairs_test.csv
```

Ràng buộc:

- Group-disjoint theo subject/session/device/attack clip.
- Zero exact/perceptual duplicate xuyên split.
- Threshold chọn trên val, khóa trước test.
- Test camera thật độc lập; không dùng test để early stop/chọn model.
- Lưu demographic/device/lighting metadata để slice metrics, không dùng làm shortcut label.

### Script đề xuất

| Script | Chức năng |
|---|---|
| `scripts/audit_dataset.py` | schema, label, duplicate, near-duplicate, group overlap, quality, class/domain distribution |
| `scripts/evaluate_pad.py` | score dump, ROC/PR, threshold sweep, precision/recall/F1, APCER/BPCER/ACER/IAPAR |
| `scripts/evaluate_recognition.py` | genuine/impostor distance, ROC, FAR/FRR/EER, threshold and margin sweep |
| `scripts/evaluate_open_set.py` | FPIR/FNIR theo gallery size cho 1:N |
| `scripts/benchmark_pipeline.py` | stage latency cold/warm p50/p95/p99, throughput, RSS/VRAM, concurrency |

### Công thức/report bắt buộc

```text
FAR = false accepts / impostor attempts
FRR = false rejects / genuine attempts
EER = operating point nơi FAR xấp xỉ FRR
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * precision * recall / (precision + recall)
```

Ngoài ra với PAD nên báo APCER/BPCER/ACER và attack-type slices; với 1:N nên báo FPIR/FNIR theo gallery size. Mỗi metric cần bootstrap 95% CI và confusion matrix tại threshold đã freeze.

### Benchmark resource

- Hardware + framework versions + model SHA.
- Cold start, first request, warm steady-state.
- Concurrency 1/2/4/8; payload 640×480, 1024 px và giới hạn server.
- Per-stage decode/detect/PAD/align/embed/search/DB.
- Process RSS peak, steady RSS, GPU VRAM nếu có.
- Soak test ≥30 phút; error/timeout/OOM/SQLite lock rate.

## Test matrix yêu cầu

| # | Mục tiêu / input | Expected | Kết quả hiện tại đã chạy | Test file đề xuất / mock |
|---:|---|---|---|---|
| 1 | Ảnh hợp lệ, mặt thật, enrolled | 200 đúng user | Actual: 503 vì model thiếu; mock path 200 pass | `tests/e2e/test_real_auth.py`; model thật + test DB |
| 2 | Không có mặt | 422, không PAD/FaceNet | Unit/mock API pass; current default model thiếu trả 503 trước detect | `tests/api/test_face_count.py`; detector stub |
| 3 | Nhiều mặt | 422, không PAD/FaceNet | Unit preprocessor pass; chưa full API case | cùng file; 2-box detector |
| 4 | Print/screen/replay | 403 | Chỉ mock score=0,5 pass; chưa model thật | `tests/model/test_pad_attacks.py`; không mock model |
| 5 | Non-image bytes có MIME image | 400 JSON | Manual test: 400 pass | `tests/api/test_image_decode.py`; bytes thật |
| 6 | Extreme image / bomb | 413/422, không 500 | Size guards pass; bomb + MemoryError manual → 500 | cùng file; crafted image + fault injection |
| 7 | Empty index | 503 explicit readiness/unavailable | Unit raises; runtime index rỗng nhưng bị missing model mask | `tests/integration/test_empty_index.py` |
| 8 | Model không load | Startup not-ready/503 fail-closed | Existing pass | `test_anti_spoof_fail_closed.py`; load_model mock |
| 9 | Enrollment/verify preprocess khác | Golden tensors phải bằng contract | Chưa có; source enrollment/verify cùng code, train detector khác | `tests/model/test_preprocess_contract.py`; spy model |
| 10 | Threshold sát biên | Quy tắc `==`, ±epsilon rõ | Anti 0,5 tested; L2 0,9 chưa có; float32 constructed 0,90000004 bị reject | `tests/model/test_threshold_boundaries.py` |
| 11 | Hai request concurrent | Không crash; bounded latency/memory | Manual mock: 2 authenticated, 0,606 s; locks serialize stages | `tests/perf/test_concurrency.py`; slow thread-safe fake |
| 12 | FastAPI validation error | UI hiển thị message đúng | Manual backend `detail`; source UI sẽ `undefined` | `frontend/src/pages/Login.test.jsx`; mocked fetch 422 |
| 13 | Ảnh enrollment sai folder | Readiness fail/index empty rõ | Existing unit pass; actual 54 root files → index empty | `tests/integration/test_enrollment_layout.py` |

## Những điều chưa thể kết luận

- Anti-spoof test accuracy/precision/recall/F1 vì artifact thiếu.
- FAR/FRR/EER recognition vì không có genuine/impostor protocol.
- Threshold 0,5/0,9 tốt hay xấu trên camera deployment.
- APCER theo từng print/screen/replay/mask attack.
- Production p95/p99, RAM và throughput với anti model thật.
- Bias theo demographic vì dataset không có metadata/protocol.

## Nguồn sơ cấp đối chiếu

- FaceNet paper: https://arxiv.org/abs/1503.03832
- keras-facenet wrapper/source: https://github.com/faustomorales/keras-facenet
- NIST FATE passive software PAD evaluation: https://www.nist.gov/publications/face-analysis-technology-evaluation-fate-part-10-performance-passive-software-based
- NIST SP 800-63B biometric/PAD guidance: https://pages.nist.gov/800-63-4/sp800-63b/authenticators/
