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

---

## 📖 Giới thiệu

V-Shield là hệ thống xác thực người dùng sử dụng Deep Learning — thay thế mật khẩu bằng khuôn mặt. Người dùng chỉ cần nhìn vào camera, hệ thống tự động:

1. **Phát hiện giả mạo** — phân biệt khuôn mặt thật (3D) với ảnh in / màn hình điện thoại (CNN Anti-Spoofing)
2. **Nhận diện danh tính** — so khớp khuôn mặt với cơ sở dữ liệu (FaceNet One-Shot Learning)
3. **Phân quyền tự động** — Admin hoặc User tùy theo danh tính

Kiến trúc **Client — Server**: Frontend React gửi ảnh base64 lên Backend FastAPI để xử lý.

---

## ✨ Điểm nổi bật

| Tính năng | Mô tả |
|---|---|
| 🛡️ **Anti-Spoofing** | CNN tự xây dựng phân loại Real/Fake, chặn replay attack bằng ảnh giấy |
| ⚡ **One-Shot Learning** | FaceNet 512-D embeddings — thêm người mới không cần retrain model |
| 🔐 **RBAC** | Tự phân quyền Admin/User ngay sau khi nhận diện thành công |
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
                    └─► FaceNet ──► L2 normalize ──► FAISS/NumPy search
                                                    └─► Identity match?
                                                      │
                                      ┌───────────────┴───────────────┐
                                   Known                           Unknown
                                      │                               │
                             Log to SQLite                        Reject
                             Return role
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
│   ├── faces/                   # Ảnh khuôn mặt đăng ký của từng user
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
python scripts/collect_data.py --class-id 1 --output data/DataCollect/real

# 2. Thu thập ảnh giả mạo Fake (class_id=0)
python scripts/collect_data.py --class-id 0 --output data/DataCollect/fake

# 3. Chia dataset (train/val/test)
python scripts/split_data.py

# 4. Train model
make train
```

### Bước 3 — Đăng ký khuôn mặt người dùng

Tạo thư mục con trong `data/faces/` với tên username, đặt ảnh vào trong:

```
data/faces/
├── hung/        ← ảnh 1.jpg, 2.jpg, ... (tự động nhận role "admin")
├── alice/       ← ảnh 1.jpg, 2.jpg, ...
└── bob/         ← ảnh 1.jpg, 2.jpg, ...
```

Các ảnh đặt trực tiếp ở `data/faces/` không có username nên sẽ bị bỏ qua. Index danh tính
được dựng thành snapshot khi backend khởi động; hãy khởi động lại backend sau khi thay đổi
ảnh đăng ký.

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
| **Đăng ký user** | Hardcode tên trong source code | Tự động đọc từ thư mục `data/faces/` |
| **Cấu hình** | Magic path rải rắc khắp code | Tập trung trong `configs/data.yaml` |
| **Database module** | Import trực tiếp `database.py` cùng thư mục | Module riêng trong package `vshield.api` |
| **Chạy project** | Copy từng lệnh dài | `make dev`, `make train`,... |

---

## ⚠️ Nhược điểm & Hướng cải tiến

### 1. Hiệu suất tìm kiếm khuôn mặt `Đã cải tiến`
- **Vấn đề:** Đang dùng vòng lặp `for` tính L2 distance từng cặp embedding. Khi có hàng nghìn user, tốc độ rất chậm.
- **Cải tiến đã làm:** Tích hợp **FAISS** (được cài cùng project). Nếu FAISS không khả dụng hoặc search lỗi, code dùng exact NumPy fallback với cùng metric L2.
  > `IndexFlatL2` là exact search `O(N·d)`, nhưng được tối ưu native/SIMD. Với dữ liệu lớn hơn, có thể chuyển sang HNSW hoặc IVF sau khi benchmark.

### 2. Dữ liệu train còn hạn chế
- **Vấn đề:** Dataset Anti-Spoofing tự thu thập nhỏ, ít đa dạng → model dễ nhầm khi ánh sáng yếu hoặc góc nghiêng.
- **Hướng cải tiến:** Tăng cường Augmentation mạnh hơn (đổi sáng, contrast, blur ngẫu nhiên). Bổ sung dataset công khai như **CelebA-Spoof** hoặc **LCC-FASD**.

### 3. Bảo mật API
- **Vấn đề:** Endpoint `/predict` và `/logs/export` không có xác thực. Ai biết URL đều gọi được.
- **Hướng cải tiến:** Tích hợp **JWT Token** — server trả về token sau khi nhận diện thành công, các request tới `/logs/*` phải đính kèm `Authorization: Bearer <token>`.



## 🧠 Công nghệ sử dụng

| Thành phần | Công nghệ | Lý do chọn |
|---|---|---|
| Backend | FastAPI | Async, tự sinh docs, nhanh hơn Flask |
| ML Framework | TensorFlow / Keras | Dễ build CNN, nhiều tài liệu |
| Face Embedding | keras-facenet | Pretrained, không cần retrain khi thêm user |
| Computer Vision | OpenCV | Xử lý ảnh nhanh, dùng rộng rãi |
| Hand Tracking | MediaPipe | Blink detection (EAR) cho data collector |
| Frontend | React + Vite | HMR nhanh, code tổ chức tốt |
| Database | SQLite | Nhẹ, zero-config, phù hợp scale nhỏ |
| Package Manager | uv | Thay pip, cài nhanh hơn ~10x |

---

<div align="center">
<i>Face Verification Project — Deep Learning Application · VKU 2026</i>
</div>
