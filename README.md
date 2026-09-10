# 👁️ V-Shield — Face Authentication System

<div align="center">

**Hệ thống đăng nhập bằng khuôn mặt tích hợp chống giả mạo (Anti-Spoofing)**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![uv](https://img.shields.io/badge/package_manager-uv-DE5FE9)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

**Trạng thái 2026-09-10:** Đã có UI/API quản lý `SUPER_ADMIN`/`ADMIN`/`USER`, enrollment nhiều ảnh và phân quyền backend; FaceNet tìm `user_id`, SQLite quyết định role/status, không train lại khi thêm người. Chưa enroll người thật. Chưa tải được ảnh dataset mới (**0 mẫu mới/được phát hành**), chưa có PAD model nên đăng nhập thực tế trả `503`. Không tự gán 54 ảnh rời thành tài khoản. Xem [hướng dẫn phân quyền](docs/face-authorization.md) và [nhập/chia dataset ngoài](docs/external-datasets.md).

## Dataset v2 integrity workflow

The historical split is marked `invalid_for_model_evaluation`. Do not use it
for model selection, threshold calibration, or accuracy claims.

Build a v2 candidate only after completing
`data/manifests/collection-matrix-v2.csv` with real subject/session/clip/device
provenance:

```bash
uv run python scripts/build-dataset-manifest.py \
  --input data/All \
  --metadata data/manifests/capture-metadata-v2.csv \
  --output data/manifests/dataset-v2.csv

uv run python scripts/deduplicate-dataset.py \
  --manifest data/manifests/dataset-v2.csv \
  --output data/manifests/dataset-v2-deduplicated.csv \
  --candidates data/manifests/near-duplicate-candidates-v2.csv

uv run python scripts/split-dataset-grouped.py \
  --manifest data/manifests/dataset-v2-deduplicated.csv \
  --output-dir data/protocols \
  --quarantine data/manifests/quarantine-v2.csv \
  --output-manifest data/manifests/dataset-v2-split.csv

uv run python scripts/shortcut-baselines.py \
  --manifest data/manifests/dataset-v2-split.csv \
  --dataset-root data/All \
  --output artifacts/evaluation/v2/shortcut-report.json
```

Training remains blocked until `configs/data-v2.yaml` references a released
protocol and the release audit passes. Threshold selection is performed from
validation scores by `scripts/evaluate-pad.py`; test scores are never used by
the training entry point.

---

## 📖 Giới thiệu

V-Shield là hệ thống xác thực người dùng sử dụng Deep Learning — thay thế mật khẩu bằng khuôn mặt. Người dùng chỉ cần nhìn vào camera, hệ thống tự động:

1. **Phát hiện giả mạo** — phân biệt khuôn mặt thật (3D) với ảnh in / màn hình điện thoại (CNN Anti-Spoofing)
2. **Nhận diện danh tính** — so khớp khuôn mặt với cơ sở dữ liệu (FaceNet One-Shot Learning)
3. **Phân quyền backend** — tra role SUPER_ADMIN/ADMIN/USER và trạng thái hiện tại từ SQLite, không suy quyền từ model

Kiến trúc **Client — Server**: Frontend React gửi ảnh base64 lên Backend FastAPI để xử lý.

---

## ✨ Điểm nổi bật

| Tính năng | Mô tả |
|---|---|
| 🛡️ **Anti-Spoofing** | CNN tự xây dựng phân loại Real/Fake, chặn replay attack bằng ảnh giấy |
| ⚡ **One-Shot Learning** | FaceNet 512-D embeddings — thêm người mới không cần retrain model |
| 🔐 **RBAC** | ADMIN quản lý USER; chỉ SUPER_ADMIN tạo/quản lý ADMIN và đổi role; backend kiểm tra độc lập UI |
| 📋 **Audit Logging** | Lịch sử đăng nhập lưu SQLite, có API query & xuất CSV |
| 📦 **Modern Tooling** | `uv` + `pyproject.toml` + `src-layout` — chuẩn Python 2026 |

---

## 🏗️ Kiến trúc hệ thống

```
[ React Frontend ]
       │  base64 image (POST /predict)
       ▼
[ FastAPI Backend ]
       │
       └─► Detect exactly one face ──► Align & crop
                    │
                    ├─► Anti-Spoofing CNN ──► FAKE? → Reject
                    │         (face_verify_v1.keras)
                    │
                    └─► FaceNet ──► L2 normalize ──► ChromaDB local search
                                                    └─► FAISS/NumPy fallback
                                                    └─► Identity match?
                                                      │
                                      ┌───────────────┴───────────────┐
                                   Known                           Unknown
                                      │                               │
                             Log to SQLite                        Reject
                             Check active account + issue session
```

---

## 📂 Cấu trúc thư mục

```
project/
├── src/
│   └── vshield/                 # Python package chính (src-layout)
│       ├── api/
│       │   ├── app.py           # FastAPI server
│       │   └── database.py      # SQLite helper
│       ├── core/
│       │   ├── anti_spoof.py    # Load & chạy CNN Anti-Spoofing
│       │   ├── embedder.py      # FaceNet embedding
│       │   └── verifier.py      # Tìm kiếm L2 + nhận diện
│       ├── data/
│       │   ├── loader.py        # Đọc dataset từ YAML config
│       │   └── collector.py     # Thu thập ảnh từ webcam
│       ├── models/
│       │   └── cnn.py           # Kiến trúc mô hình CNN Anti-Spoofing
│       └── training/
│           └── train.py         # Script huấn luyện
├── frontend/                    # React + Vite UI
├── artifacts/
│   └── models/                  # Chứa face_verify_v1.keras (gitignored)
├── data/
│   ├── authorization/           # faces/<username>/ + Chroma phân quyền riêng
│   ├── external/                # Dataset nghiên cứu nhập riêng, không cấp quyền
│   ├── faces/                   # Ảnh legacy, không tự dùng để đăng nhập
│   ├── SplitData/               # Dataset train/val/test
│   └── DataCollect/             # Ảnh raw thu thập từ webcam
├── configs/
│   └── data.yaml                # Cấu hình đường dẫn dataset
├── scripts/
│   ├── collect_data.py          # Thu thập data Real/Fake
│   └── split_data.py            # Chia dataset
├── notebooks/
│   └── model.ipynb              # Notebook khám phá mô hình
├── pyproject.toml               # Dependencies (thay requirements.txt)
├── Makefile                     # Lệnh tắt tiện lợi
└── README.md
```

---

## 🚀 Workflow — Hướng dẫn chạy

### Yêu cầu
- Python **3.10+**
- Node.js **16+**
- Webcam

### Bước 1 — Cài đặt môi trường

```bash
# Cài uv (lần đầu, nếu chưa có)
pip install uv

# Cài toàn bộ thư viện Python
make install

# Hoặc không dùng make:
uv sync --all-extras
```

### Bước 2 — Chuẩn bị model

> **Lưu ý:** File `face_verify_v1.keras` (~515MB) bị gitignore, không có trong repo.

**Option A — Dùng model đã train sẵn:**
```bash
# Đặt file face_verify_v1.keras vào:
artifacts/models/face_verify_v1.keras
```

**Option B — Tự train lại:**
```bash
# 1. Thu thập ảnh khuôn mặt Real (class_id=1)
uv run python scripts/collect_data.py --class-id 1 --output data/DataCollect/real

# 2. Thu thập ảnh giả mạo Fake (class_id=0)
uv run python scripts/collect_data.py --class-id 0 --output data/DataCollect/fake

# 3. Chia dataset (train/val/test)
uv run python scripts/split_data.py --help

# 4. Train model
# Training is blocked until the Dataset v2 integrity workflow below passes.
```

### Bước 3 — Đăng ký khuôn mặt người dùng

Dừng backend. Người quản trị chạy CLI với ảnh đã được chủ thể đồng ý đăng ký; thay đường dẫn ví dụ bằng ảnh đúng người:

```powershell
.\.venv\Scripts\python.exe scripts/manage-identities.py enroll --username owner --role ADMIN --images "C:\path\to\consented-owner-photo.jpg" --consent
.\.venv\Scripts\python.exe scripts/manage-identities.py bootstrap-super-admin --username owner
.\.venv\Scripts\python.exe scripts/manage-identities.py sync
.\.venv\Scripts\python.exe scripts/manage-identities.py list
```

Khởi động lại backend. Bootstrap chỉ định SUPER_ADMIN đầu tiên một lần, không có username tự nhận quyền. Sau đăng nhập, dùng UI quản lý để enroll USER (ADMIN/SUPER_ADMIN) hoặc ADMIN (chỉ SUPER_ADMIN), capture/upload 2–10 ảnh có consent; thay đổi HTTP không cần restart hoặc train lại. Ảnh/manifest ở `data/authorization/faces/<username>/`; Chroma giữ embedding/user_id, SQLite quyết định quyền. Không dùng dataset công khai làm tài khoản đăng nhập. Xem [runbook](docs/face-authorization.md) về migration, API, test và retention; xóa tài khoản là soft delete, không xóa ảnh sinh trắc học.

### Bước 4 — Chạy ứng dụng

**Terminal 1 — Backend:**
```bash
make dev
# Server chạy tại http://localhost:8000
# Docs API tại  http://localhost:8000/docs
```

**Terminal 2 — Frontend:**
```bash
make frontend
# Hoặc: cd frontend && npm install && npm run dev
# Mở http://localhost:5173
```

### Tổng hợp lệnh Makefile

```bash
make install     # Cài thư viện với uv
make dev         # Chạy Backend FastAPI
make frontend    # Chạy Frontend React
make docker      # Chạy toàn bộ hệ thống bằng Docker
make train       # Train lại mô hình CNN
make collect     # Mở webcam thu thập data
make format      # Format code bằng ruff
make clean       # Xóa __pycache__, .pytest_cache
make help        # Xem toàn bộ lệnh
```

---

## 🔧 Những điểm đã cải tiến (so với v1)

| Hạng mục | Trước (v1) | Sau (v2 — hiện tại) |
|---|---|---|
| **Cấu trúc thư mục** | File rải rác, không có package | `src-layout` chuẩn PEP 517 |
| **Quản lý thư viện** | `pip` + `requirements.txt` | `uv` + `pyproject.toml` (nhanh hơn ~10x) |
| **Đăng ký user** | Hardcode tên trong source code | UI/API enrollment có consent, CLI bootstrap owner, SQLite cấp quyền, Chroma lưu embedding/user_id |
| **Cấu hình** | Magic path rải rắc khắp code | Tập trung trong `configs/data.yaml` |
| **Database module** | Import trực tiếp `database.py` cùng thư mục | Module riêng trong package `vshield.api` |
| **Chạy project** | Copy từng lệnh dài | `make dev`, `make train`,... |

---

## ⚠️ Nhược điểm & Hướng cải tiến

### 1. Hiệu suất tìm kiếm khuôn mặt `Đã cải tiến`
- **Giới hạn:** Truy vấn toàn bộ template để kiểm tra khoảng cách với danh tính thứ hai; chưa có benchmark khả năng mở rộng thực tế.
- **Cải tiến đã làm:** Dùng **ChromaDB PersistentClient** tại `data/authorization/chroma/` làm vector store local ưu tiên. FAISS/NumPy giữ vai trò fallback từ gallery đã kiểm tra.
  > Chroma lưu FaceNet embedding 512-D và metadata identity tối thiểu; ảnh khuôn mặt không được lưu trong vector store.
  > Backend kiểm tra gallery revision trước truy vấn và đổi snapshot hợp lệ sau thay đổi HTTP. Chỉ quy trình CLI `sync` vẫn yêu cầu dừng backend rồi restart. Collection revision cũ còn lưu trên đĩa; `.gitignore` không thay thế mã hóa, ACL hoặc chính sách retention/xóa dữ liệu.

### 2. Dữ liệu train còn hạn chế
- **Vấn đề:** Split v1 không hợp lệ cho đánh giá do trùng/rò rỉ; v2 chưa có mẫu phát hành. Chưa được tuyên bố độ chính xác.
- **Đã bổ sung:** Adapter CelebA-Spoof chuẩn hóa nhãn và BGR 128×128, đề xuất chia nhóm 70/15/15, quarantine khi thiếu provenance. Cần ảnh/annotation nguồn hợp lệ; [lần tải hiện tại chưa lấy được mẫu](docs/external-datasets.md).

### 3. Bảo mật API
- **Đã làm:** `/predict` chỉ cấp bearer token opaque sau xác thực và kiểm tra ACTIVE; SQLite lưu hash token, hết hạn sau 1 giờ. `/logs` và `/logs/export` cho ADMIN/SUPER_ADMIN. API quản lý kiểm tra actor/target, chỉ SUPER_ADMIN được tạo ADMIN/đổi role; không có HTTP tạo/sửa SUPER_ADMIN. `/auth/me` đọc quyền hiện tại; đổi role/disable/delete thu hồi phiên. Frontend dùng `sessionStorage`, không quyết định quyền.
- **Còn thiếu:** HTTPS khi triển khai, rate limiting, mã hóa/ACL dữ liệu sinh trắc học và quy trình retention/xóa đầy đủ; xem [runbook bảo mật](docs/face-authorization.md).

## 🧠 Công nghệ sử dụng

| Thành phần | Công nghệ | Lý do chọn |
|---|---|---|
| Backend | FastAPI | Async, tự sinh docs, nhanh hơn Flask |
| ML Framework | TensorFlow / Keras | Dễ build CNN, nhiều tài liệu |
| Face Embedding | keras-facenet | Pretrained, không cần retrain khi thêm user |
| Vector Database | ChromaDB local | Lưu/query embedding bền vững qua restart |
| Computer Vision | OpenCV | Xử lý ảnh nhanh, dùng rộng rãi |
| Hand Tracking | MediaPipe | Blink detection (EAR) cho data collector |
| Frontend | React + Vite | HMR nhanh, code tổ chức tốt |
| Database | SQLite | Nhẹ, zero-config, phù hợp scale nhỏ |
| Package Manager | uv | Thay pip, cài nhanh hơn ~10x |

---

<div align="center">
<i>Face Verification Project — Deep Learning Application · VKU 2026</i>
</div>
