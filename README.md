# VShield — AI-Powered Biometric Access Control

VShield là hệ thống kiểm soát truy cập bằng khuôn mặt dành cho đồ án nền tảng đại học. Hệ thống kết hợp nhận diện danh tính, chống giả mạo thụ động, phân quyền theo vai trò, quản lý người dùng và nhật ký truy cập trong một kiến trúc client–server dễ chạy và dễ trình diễn.

> VShield nhận diện **người dùng là ai** trước, sau đó lấy `role` và `status` từ database để quyết định quyền. Model AI không phân loại trực tiếp khuôn mặt thành `ADMIN` hay `USER`.

## Tính năng chính

- Đăng nhập bằng frame capture trực tiếp từ webcam; không có upload ảnh tĩnh trên giao diện login.
- Face detection và kiểm tra chính xác một khuôn mặt.
- Passive anti-spoofing bằng MiniFASNetV2 pretrained, chạy với ONNX Runtime CPU.
- Face recognition bằng `keras-facenet`, embedding 512 chiều và L2 normalization.
- Similarity search bằng ChromaDB persistent; có fallback FAISS/NumPy.
- Từ chối khuôn mặt lạ, kết quả mơ hồ và gallery rỗng.
- Phân quyền backend với `USER`, `ADMIN`, `SUPER_ADMIN`.
- Enrollment 2–10 mẫu ảnh, consent bắt buộc, kiểm tra liveness cho từng mẫu.
- Kiểm tra khuôn mặt trùng trước khi tạo identity mới.
- Disable account, đổi role và thu hồi session.
- Access logs, dashboard thống kê, lọc và xuất CSV.
- Giao diện React/Vite responsive theo phong cách enterprise security.
- Cơ chế fail-closed: PAD/model lỗi hoặc không sẵn sàng thì từ chối truy cập.

## Kiến trúc

```text
Browser camera capture
              │
              ▼
        FastAPI /predict
              │
              ▼
        Face detection
              │ exactly one face
              ▼
 MiniFASNetV2 passive PAD (ONNX CPU)
       │                     │
   FAKE/ERROR             REAL only
       │                     │
       ▼                     ▼
  Deny + access log     FaceNet embedding
                             │
                             ▼
                   ChromaDB / FAISS search
                             │
                   user_id or UNKNOWN
                             │
                             ▼
                   SQLite role + status
                             │
                             ▼
                 Session + backend RBAC
```

FaceNet và MiniFASNet sử dụng hai preprocessing path riêng. PAD nhận crop có context từ ảnh gốc và bounding box; FaceNet giữ crop/alignment dành cho recognition.

## Công nghệ

| Thành phần | Công nghệ |
|---|---|
| Backend | Python 3.10+, FastAPI, Uvicorn |
| Frontend | React 18, React Router, Vite, plain CSS |
| Database | SQLite, SQL trực tiếp |
| Face recognition | pretrained `keras-facenet` FaceNet |
| Anti-spoofing | pretrained MiniFASNetV2, ONNX Runtime CPU |
| Vector search | ChromaDB persistent, FAISS/NumPy fallback |
| Image processing | OpenCV, Pillow, MediaPipe/Haar pipeline hiện tại |
| Tests | pytest, Node test runner |

## Phân quyền

| Chức năng | USER | ADMIN | SUPER_ADMIN |
|---|:---:|:---:|:---:|
| Đăng nhập bằng khuôn mặt | Có | Có | Có |
| Xem profile và access history của bản thân | Có | Có | Có |
| Xem dashboard và toàn bộ access logs | Không | Có | Có |
| Tạo/sửa/disable USER | Không | Có | Có |
| Tạo/sửa/disable ADMIN | Không | Không | Có |
| Đổi role USER ↔ ADMIN | Không | Không | Có |
| Tạo SUPER_ADMIN qua HTTP | Không | Không | Không |

Frontend chỉ ẩn các action không phù hợp để cải thiện UX. Mọi thao tác nhạy cảm vẫn được backend kiểm tra lại bằng authenticated session và dữ liệu role/status trong SQLite.

## Cấu trúc thư mục

```text
project/
├── configs/                   # AI model contract, dataset/training config
├── docs/                      # Tài liệu model và dataset
├── frontend/                  # React + Vite UI
│   ├── src/components/        # App shell, badges, cards, logs, dialogs
│   ├── src/pages/             # Login, admin và user dashboard
│   └── tests/                 # Frontend contract/permission tests
├── scripts/                   # Model setup, identity management, evaluation
├── src/vshield/
│   ├── api/                   # FastAPI, sessions, SQLite, migrations, RBAC routes
│   ├── core/                  # Face preprocessing, PAD, embeddings, vector index
│   ├── data/                  # Dataset acquisition/import helpers
│   └── services/              # Authentication và enrollment workflows
├── tests/                     # Backend/unit/integration/security tests
├── pyproject.toml
└── uv.lock
```

## Yêu cầu môi trường

- Python 3.10 hoặc 3.11 được khuyến nghị.
- Node.js 18+ và npm.
- Webcam cho luồng đăng nhập/capture trực tiếp.
- Git và kết nối Internet cho lần cài dependency/model đầu tiên.
- Windows PowerShell, Linux hoặc macOS shell.

Model ONNX, database, vector gallery và dữ liệu khuôn mặt không được commit. Sau khi clone cần tạo model artifact và enrollment riêng trên máy chạy.

## Cài đặt nhanh bằng `uv`

### 1. Clone repository

```bash
git clone https://github.com/vu-xuan-hung/Face-Verification-Project-with-CNN.git
cd Face-Verification-Project-with-CNN
git checkout add
```

### 2. Cài backend

Windows PowerShell:

```powershell
uv sync --extra dev
```

Linux/macOS:

```bash
uv sync --extra dev
```

Nếu không dùng `uv`, có thể tạo virtual environment rồi chạy:

```bash
python -m pip install -e ".[dev]"
```

### 3. Cài frontend

```bash
cd frontend
npm install
cd ..
```

### 4. Tạo MiniFASNetV2 ONNX artifact

Runtime production chỉ cần `onnxruntime`. Script setup cần thêm CPU PyTorch và `onnx` để tải checkpoint chính thức đã pin hash, export và kiểm tra parity.

Windows PowerShell:

```powershell
uv pip install torch --index https://download.pytorch.org/whl/cpu
uv pip install onnx
uv run --no-sync python scripts/setup-minifasnet.py
uv run --no-sync python scripts/validate-minifasnet.py
```

Kết quả cần có:

```text
artifacts/models/minifasnet-v2.onnx
artifacts/models/minifasnet-v2-parity.json
```

Runtime đối chiếu SHA-256 trong `configs/ai-models.yaml`. Thiếu file, sai checksum, sai input/output contract hoặc ONNX load lỗi đều khiến hệ thống fail closed.

Thông tin provenance và preprocessing chính xác nằm trong [docs/minifasnet-model.md](docs/minifasnet-model.md).

## Khởi tạo SUPER_ADMIN đầu tiên

Hệ thống không hỗ trợ password login trong phiên bản hiện tại. Đăng nhập dùng face authentication và opaque bearer session; database không nhận role từ frontend.

Chuẩn bị ít nhất hai ảnh khác nhau của cùng một người, có sự đồng thuận sinh trắc học. Model PAD phải sẵn sàng trước khi enroll.

Windows PowerShell:

```powershell
uv run python scripts/manage-identities.py enroll `
  --username admin_root `
  --role ADMIN `
  --images "D:\faces\admin-1.jpg" "D:\faces\admin-2.jpg" `
  --consent

uv run python scripts/manage-identities.py bootstrap-super-admin --username admin_root
uv run python scripts/manage-identities.py sync
```

Linux/macOS:

```bash
uv run python scripts/manage-identities.py enroll \
  --username admin_root \
  --role ADMIN \
  --images ./faces/admin-1.jpg ./faces/admin-2.jpg \
  --consent

uv run python scripts/manage-identities.py bootstrap-super-admin --username admin_root
uv run python scripts/manage-identities.py sync
```

Các lệnh quản trị offline khác:

```bash
uv run python scripts/manage-identities.py list
uv run python scripts/manage-identities.py disable --username username
uv run python scripts/manage-identities.py role --username username --role ADMIN
uv run python scripts/manage-identities.py sync
```

`bootstrap-super-admin` chỉ dành cho owner đầu tiên. API không cho ADMIN tự nâng quyền hoặc tạo ADMIN mới.

## Chạy ứng dụng

Mở hai terminal tại thư mục gốc.

Terminal 1 — backend:

```bash
uv run uvicorn vshield.api.app:app --host 127.0.0.1 --port 8000 --reload
```

Terminal 2 — frontend:

```bash
cd frontend
npm run dev
```

Truy cập:

- Giao diện: <http://localhost:5173>
- Swagger API: <http://localhost:8000/docs>
- Readiness: <http://localhost:8000/health/ready>

Nếu frontend chạy trên origin khác, cấu hình CORS trước khi mở backend:

Windows PowerShell:

```powershell
$env:VSHIELD_CORS_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"
```

Production frontend có thể trỏ sang backend khác bằng biến build-time:

```bash
VITE_API_URL=https://api.example.com npm run build
```

PowerShell:

```powershell
$env:VITE_API_URL="https://api.example.com"
npm run build
```

## Luồng sử dụng

### Đăng nhập

1. Mở trang login và cho phép camera.
2. Đặt duy nhất một khuôn mặt trong khung hướng dẫn.
3. Chọn **Capture & Login**.
4. Backend kiểm tra face count → liveness → identity → account status.
5. Thành công sẽ tạo session và chuyển tới dashboard phù hợp với role.

Các trạng thái được UI phân biệt rõ: `NO_FACE`, `MULTIPLE_FACES`, `SPOOF`, `UNKNOWN`, `AMBIGUOUS`, `USER_DISABLED`, `PAD_UNAVAILABLE`, `MODEL_ERROR`, `ACCESS_GRANTED`.

### Enroll USER hoặc ADMIN

1. Đăng nhập bằng ADMIN/SUPER_ADMIN.
2. Mở **Enroll identity**.
3. Điền username, tên và email.
4. Xác nhận biometric consent.
5. Capture/upload từ 2 đến 10 mẫu khác nhau.
6. Submit để backend kiểm tra từng mẫu độc lập.
7. Chỉ mẫu `REAL` mới được tạo FaceNet embedding và duplicate check.

ADMIN chỉ tạo được USER. Chỉ SUPER_ADMIN thấy và sử dụng tùy chọn tạo ADMIN.

### Quản lý người dùng

- Edit name/email.
- Disable/enable account.
- SUPER_ADMIN đổi role USER ↔ ADMIN.
- Soft-delete account theo phạm vi quyền.

Disable account và delete biometric data là hai hành động khác nhau. Backend hiện chưa cung cấp endpoint xóa riêng biometric templates; UI không giả lập chức năng này.

## API chính

| Method | Endpoint | Quyền | Mục đích |
|---|---|---|---|
| `POST` | `/predict` | Public input | Face authentication, trả session nếu thành công |
| `GET` | `/health/ready` | Public | Trạng thái PAD, FaceNet, identity index và database |
| `GET` | `/auth/me` | Authenticated | Lấy identity từ session |
| `POST` | `/auth/logout` | Authenticated | Thu hồi session hiện tại |
| `GET` | `/users` | ADMIN+ | Danh sách account trong management scope |
| `POST` | `/users` | ADMIN+ | Enroll USER |
| `POST` | `/admins` | SUPER_ADMIN | Enroll ADMIN |
| `PATCH` | `/users/{id}` | ADMIN+ theo scope | Sửa profile |
| `PATCH` | `/users/{id}/status` | ADMIN+ theo scope | Enable/disable |
| `PATCH` | `/users/{id}/role` | SUPER_ADMIN | Đổi USER/ADMIN |
| `DELETE` | `/users/{id}` | ADMIN+ theo scope | Soft-delete account |
| `GET` | `/access-logs/me` | Authenticated | Access history của chính mình |
| `GET` | `/access-logs` | ADMIN+ | Lọc/paginate access events |
| `GET` | `/access-logs/export` | ADMIN+ | Xuất CSV |
| `GET` | `/dashboard/stats` | ADMIN+ | Thống kê dashboard |

Không gửi embedding vector, filesystem path, stack trace hoặc secret config trong API response.

## Database và dữ liệu runtime

SQLite được tạo tự động tại `login_logs.db`. Các bảng chính:

- `users`: identity metadata, role, status và enrollment reference.
- `sessions`: hash opaque session token và thời hạn.
- `access_logs`: kết quả PAD/recognition/authorization.
- `identity_revision`: revision để refresh vector index.
- `user_management_audit`: audit thay đổi account.

Dữ liệu khuôn mặt và vector nằm trong `data/authorization/`; ChromaDB persistent nằm trong vùng data runtime. Các path này cùng `*.db` và model binaries đã được `.gitignore` để tránh đẩy dữ liệu nhạy cảm lên Git.

Không tự đánh dấu identity cũ là đã vượt liveness. Enrollment mới lưu provenance PAD; legacy identities giữ trạng thái `legacy_unverified` nếu metadata cho phép.

## Model contract MiniFASNetV2

Contract runtime nằm tại `configs/ai-models.yaml`:

- Input: BGR, float32, NCHW `[1, 3, 80, 80]`.
- Giá trị pixel: `0–255`, không `/255`, không mean/std normalization.
- Crop scale: `2.7` quanh bounding box.
- Output: raw logits `[1, 3]`.
- Runtime áp dụng stable softmax đúng một lần.
- Real class index: `1`.
- Threshold hiện tại: `0.8`, **provisional**, chưa được tuyên bố tối ưu.
- ONNX checksum: `04e8890346498741e655adfcb67b48ee2620fc2d251321a8f0ad3262bb842067`.

Artifact được export từ checkpoint chính thức tại commit `b6d5f04ad78778917853b25c778acef6d5626d15`. Chi tiết checksum và parity xem trong tài liệu model.

## Dataset nghiên cứu

Dataset PAD nghiên cứu không được dùng làm identity đăng nhập và không được đặt trong `data/authorization/`.

Các nguồn đã khảo sát:

- [CelebA-Spoof](https://github.com/ZhangYuanhan-AI/CelebA-Spoof): research/non-commercial terms; cần tự xác nhận quyền truy cập.
- [OULU-NPU](https://sites.google.com/site/oulunpudatabase/): yêu cầu institutional EULA; adapter chưa được triển khai.

Catalog, mapping label, download budget và import policy nằm tại:

- `configs/external-datasets.yaml`
- [docs/external-datasets.md](docs/external-datasets.md)

Không commit hoặc phân phối lại ảnh dataset có hạn chế giấy phép.

## Kiểm thử

Backend:

```bash
uv run pytest -q
```

Chỉ các test PAD/RBAC chính:

```bash
uv run pytest -q tests/test_minifasnet_onnx.py tests/test_anti_spoof_fail_closed.py tests/test_authentication_architecture.py tests/test_access_logs.py tests/test_rbac_api.py tests/test_rbac_identity.py
```

Frontend:

```bash
cd frontend
node tests/auth-api.test.js
node tests/role-permissions.test.js
npm run build
```

Integration test ONNX thật chỉ chạy khi artifact đã được setup. Không được hiểu unit test mock là bằng chứng độ chính xác camera thực tế.

## Đánh giá PAD trên camera mục tiêu

Chuẩn bị dữ liệu riêng, có consent:

```text
evaluation/
├── real/
├── fake_print/
└── fake_screen/
```

Chạy:

```bash
uv run python scripts/evaluate_pad_directory.py --eval-dir evaluation
```

Script báo APCER, BPCER, ACER và accuracy cho nhiều threshold. Nếu không có dataset, script báo `NO EVALUATION DATASET PROVIDED` và không tạo số liệu giả.

## Troubleshooting

### `/health/ready` trả 503

Kiểm tra từng component trong response:

- `pad=false`: chạy lại setup/validate MiniFASNet và kiểm tra checksum config.
- `facenet=false`: kiểm tra TensorFlow/keras-facenet và lần tải weight đầu tiên.
- `identity_index=false`: enroll identity, chạy `manage-identities.py sync`, rồi restart backend.
- `database=false`: kiểm tra quyền ghi thư mục project và SQLite migration.

### Frontend không gọi được backend

- Backend phải chạy tại `http://localhost:8000`, hoặc đặt `VITE_API_URL` trước khi build.
- Origin frontend phải có trong `VSHIELD_CORS_ORIGINS`.
- Kiểm tra Network tab; HTTP 401 sẽ xóa session, HTTP 403 không tự đăng xuất.

### Camera không mở

- Cho phép camera trong browser.
- Dùng `localhost` hoặc HTTPS; browser thường chặn camera trên HTTP remote host.
- Đóng ứng dụng khác đang giữ webcam.
- Đăng nhập chỉ cho phép capture trực tiếp từ camera; giao diện không hỗ trợ upload ảnh tĩnh.

### Enrollment bị từ chối

- Mỗi ảnh phải có đúng một khuôn mặt.
- Tất cả mẫu phải vượt PAD; một mẫu spoof/uncertain sẽ bị loại theo policy fail-closed.
- Các ảnh phải khác nhau và cùng một người.
- Identity hoặc email không được trùng.
- ADMIN không được tạo ADMIN.

## Kịch bản demo đề xuất

1. Mở `/health/ready` để xác nhận toàn bộ component sẵn sàng.
2. Đăng nhập SUPER_ADMIN bằng webcam.
3. Trình bày dashboard, thống kê và access chart.
4. Enroll USER mới qua 5 bước: details → consent → samples → review → submit.
5. Đăng nhập USER mới và xem personal access history.
6. Thử một người chưa enroll để tạo `UNKNOWN_FACE`.
7. Thử ảnh in/màn hình để tạo `SPOOF_ATTEMPT` và chứng minh FaceNet bị bỏ qua.
8. Disable USER rồi thử lại để tạo `USER_DISABLED`.
9. Quay về dashboard, lọc log và export CSV.

## Giới hạn và lưu ý bảo mật

- MiniFASNetV2 là passive 2D RGB PAD; không đảm bảo chống mọi replay, deepfake hoặc mặt nạ 3D.
- Threshold `0.8` chưa được calibration trên webcam triển khai thực tế.
- Single-frame HTTP không có temporal voting hoặc challenge-response.
- Browser frame không có hardware camera attestation.
- Haar/face detector hiện tại được giữ để tránh thay đổi scope; domain khác reference pipeline có thể ảnh hưởng PAD.
- Opaque session được lưu phía client trong `sessionStorage`; hệ thống hiện không có password/OAuth/MFA.
- Đây là đồ án nghiên cứu/giáo dục, không tuyên bố đạt chuẩn triển khai ngân hàng/chính phủ.
- Cần đánh giá license checkpoint và dataset trước mục đích thương mại.
- Không commit `login_logs.db`, model binary, face images, embeddings, session data hoặc dataset restricted.

## License

Source code authored for VShield is released under the MIT License. See [LICENSE](LICENSE) for details.

Third-party libraries, pretrained models, checkpoints, datasets, and other external artifacts remain subject to their respective licenses and terms of use. The VShield MIT License does not grant redistribution rights for those assets.

This license statement does not assert that MiniFASNet, FaceNet, CelebA-Spoof, OULU-NPU, user biometric data, or any other third-party artifact is licensed under MIT. Verify the applicable upstream terms before use or redistribution.

---

VShield 2026 — Face identity → database role → server-side authorization.
