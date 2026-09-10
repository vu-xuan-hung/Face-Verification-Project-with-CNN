# Quản lý người dùng, khuôn mặt và RBAC

Cập nhật: 2026-09-10. Giữ FastAPI, React/Vite, SQLite và FaceNet/Chroma hiện tại. Thêm người hoặc đổi quyền không train lại model.

## Kiến trúc trước và sau

Trước: FaceNet đã tạo embedding → tìm username → SQLite với hai role admin/user. Enrollment chủ yếu bằng CLI và đồng bộ offline. Model không phân loại trực tiếp ADMIN/USER.

Sau:

```text
Camera/ảnh → detect đúng một mặt → PAD → FaceNet 512D, chuẩn hóa L2
→ similarity search → user_id → SQLite users(role, status)
→ phiên bearer opaque → backend RBAC + kiểm tra tài khoản đích
```

Giữ nhiều template theo ảnh; không phải tạo class/model mới hay lấy trung bình embedding. Model quyết định danh tính, SQLite quyết định quyền tại mỗi request.

## Quyền và API

| Endpoint / thao tác | USER | ADMIN | SUPER_ADMIN |
|---|---|---|---|
| POST /predict: đăng nhập khuôn mặt | Có | Có | Có |
| GET /auth/me, POST /auth/logout | Phiên hợp lệ | Phiên hợp lệ | Phiên hợp lệ |
| GET /logs, GET /logs/export | Không | Có | Có |
| GET /users | Không | Chỉ USER | Toàn bộ |
| POST /users: tạo USER | Không | Có | Có |
| POST /admins: tạo ADMIN | Không | Không | Có |
| PATCH /users/{id}: name/email | Không | Chỉ USER | USER/ADMIN |
| PATCH /users/{id}/role: USER ↔ ADMIN | Không | Không | USER/ADMIN |
| PATCH /users/{id}/status: ACTIVE/DISABLED | Không | Chỉ USER | USER/ADMIN |
| DELETE /users/{id}: soft delete, 204 | Không | Chỉ USER | USER/ADMIN |
| Tạo/sửa/xóa SUPER_ADMIN qua HTTP | Không | Không | Không |

SUPER_ADMIN được bảo vệ khỏi mọi mutation HTTP, kể cả chính mình. Chỉ có bootstrap chủ đầu tiên bằng CLI cục bộ, chưa có luồng thêm SUPER_ADMIN thứ hai/chuyển giao owner.

Các API bảo vệ yêu cầu `Authorization: Bearer <token>`. Actor lấy từ phiên và SQLite; guard ở backend độc lập với UI. Service kiểm tra lại quyền trong transaction sau inference để ngăn actor bị hạ quyền giữa lúc xử lý vẫn tạo được tài khoản.

## Schema và migration

Tiếp tục dùng `login_logs.db`; `database.init_db()` tự chạy migration version 1 trong transaction.

| Cột users | Vai trò |
|---|---|
| id, username | ID số ổn định; username duy nhất, không đổi qua API |
| name, email | Profile; email duy nhất không phân biệt hoa/thường nếu có |
| password_hash | Nullable dự phòng; **không có đăng nhập mật khẩu** |
| role | SUPER_ADMIN, ADMIN, USER |
| status | ACTIVE, DISABLED, DELETED |
| active | Tương thích schema cũ; bằng 1 chỉ khi ACTIVE |
| enrollment_id | Liên kết manifest đăng ký hợp lệ |
| created_at, updated_at, created_by | Thời gian, ID người tạo; CLI có thể không có created_by |

Không thêm face_embedding vào SQLite: tái dùng manifest/Chroma để chứa nhiều template. Bổ sung `schema_migrations`, `identity_revision`, `user_management_audit`; giữ sessions/login_logs. Trigger kiểm tra role/status/active; unique index bảo vệ email.

Migration viết hoa role cũ, giữ trạng thái hợp lệ, vô hiệu hóa role không hợp lệ, không tự nâng ADMIN thành SUPER_ADMIN. Thu hồi phiên cũ trong lần migration đầu, không lặp lại với phiên mới. Trước nâng cấp: dừng backend và backup SQLite/gallery/Chroma an toàn; nếu email legacy xung đột, kiểm tra bản ghi nguồn, không tự xóa để ép migration chạy.

## Kho dữ liệu và đồng bộ

| Vị trí | Nội dung |
|---|---|
| data/authorization/faces/&lt;username&gt;/ | PNG bỏ EXIF; enrollment.json chứa consent, checksum, enrollment ID, template 512D L2; HTTP enrollment thêm user_id |
| data/authorization/chroma/ | Embedding, metadata user_id; collection vshield_facenet_user_id_v2_r&lt;revision&gt; |
| data/authorization/drafts/&lt;enrollment_id&gt;/ | Bản nháp không tham gia nhận diện, giữ để kiểm tra khi xuất bản lỗi |
| login_logs.db | Account, role/status, hash phiên, audit, revision |
| data/faces/ | 54 ảnh legacy chưa gán tài khoản, không tự dùng đăng nhập |

PNG enrollment giữ toàn khung hình đã kiểm tra, khác ảnh PAD 128×128. Gallery đăng nhập phải tách khỏi dataset train/val/test. Không lấy ảnh dataset công khai tự tạo tài khoản có quyền.

`ManagedIdentityIndex` kiểm tra revision SQLite trước truy vấn, dựng snapshot hợp lệ rồi thay thế. Thay đổi qua API không cần restart. Gallery đổi trong lúc search sẽ thử lại hoặc từ chối an toàn. Chroma lỗi có thể fallback FAISS/NumPy từ gallery đã kiểm tra, không dùng vector cũ để bỏ qua gallery lỗi.

## Bootstrap chủ hệ thống đầu tiên

PowerShell ở gốc project, backend đã dừng. Thay ảnh ví dụ bằng ảnh đúng người có consent:

```powershell
.\.venv\Scripts\python.exe scripts/manage-identities.py enroll --username owner --role ADMIN --images "C:\path\to\consented-owner-1.jpg" "C:\path\to\consented-owner-2.jpg" --consent
.\.venv\Scripts\python.exe scripts/manage-identities.py bootstrap-super-admin --username owner
.\.venv\Scripts\python.exe scripts/manage-identities.py sync
.\.venv\Scripts\python.exe scripts/manage-identities.py list
```

Nếu owner đã enroll thì bỏ lệnh enroll, không ghi đè ảnh. Bootstrap chỉ chạy khi chưa có SUPER_ADMIN, yêu cầu account ACTIVE có enrollment, khóa transaction, thu hồi phiên và ghi audit. Không đoán owner từ username.

CLI enroll nhận USER/ADMIN (chữ thường tương thích), 1–20 ảnh. CLI role/disable dành cho người vận hành có quyền máy; không phải quyền ADMIN trên web. Quy trình CLI sync vẫn yêu cầu dừng backend rồi restart, không chạy song song với API đang ghi. Người có quyền sửa SQLite có thể vượt RBAC nên cần ACL máy chặt chẽ.

## Enrollment USER / ADMIN

ADMIN hoặc SUPER_ADMIN mở quản lý trên dashboard, nhập username/name/email, capture hoặc upload 2–10 ảnh khác nhau và xác nhận consent. Tạo tài khoản USER dùng POST /users; chỉ SUPER_ADMIN được chọn ADMIN và gọi POST /admins. Backend cố định role theo endpoint.

Body minh họa, cần thay chuỗi ảnh bằng base64 thật:

```json
{
  "username": "member01",
  "name": "Nguyen Van A",
  "email": "member01@example.com",
  "images": ["data:image/jpeg;base64,...", "data:image/jpeg;base64,..."],
  "consent": true
}
```

Không gửi role, status, id, created_by, đường dẫn file hoặc embedding. Trường thừa bị từ chối. Endpoint đổi role riêng chỉ nhận USER/ADMIN; không nhận SUPER_ADMIN.

Backend detect đúng một mặt/ảnh, tính template L2, kiểm tra ảnh trùng, cùng danh tính và khuôn mặt đã đăng ký kể cả account inactive. Transaction kiểm tra lại actor và duplicate, tạo account/audit, xuất bản draft, tăng revision. Thành công trả 201 và profile, không trả ảnh/embedding.

Giới hạn: 2–10 ảnh HTTP, 8 MiB PNG chuẩn hóa/ảnh, 20 megapixel; tổng chuỗi ảnh tối đa 30 triệu ký tự. Username bắt đầu chữ ASCII thường, chỉ chữ thường/số/_/-, tối đa 64 ký tự, từ chối tên thiết bị Windows.

Nếu commit account thành công nhưng refresh index lỗi, trả thành công với `vector_sync_status=pending`; không bấm tạo lại. Kiểm tra danh sách và storage; truy vấn sau thử refresh lại. Frontend hiển thị cảnh báo chờ sync, không báo enrollment thất bại giả.

## Nhận diện, phiên và thu hồi quyền

POST /predict chạy PAD trước FaceNet; matcher chỉ trả user_id. SQLite kiểm tra ACTIVE trước cấp token opaque 1 giờ, chỉ lưu SHA-256 token. Mỗi request bảo vệ join phiên với user hiện tại. Đổi role/status hoặc xóa thu hồi phiên target; disabled user không được cấp phiên mới dù matcher trả đúng ID. Đổi role không sửa embedding.

Disable/enable qua status trong đúng phạm vi. Delete đặt DELETED/active=0, thu hồi phiên, tăng revision loại khỏi snapshot hiện hành; không có khôi phục DELETED qua API. Username/email soft-delete vẫn giữ.

**Soft delete không xóa sinh trắc học trên đĩa.** Ảnh, manifest, audit/log, draft, backup và collection revision cũ vẫn tồn tại; retention/xóa hoàn toàn cần quy trình riêng.

Mã lỗi: 401 thiếu/hết hạn/thu hồi phiên hoặc account inactive; 403 sai quyền/phạm vi; 404 target không có/đã xóa khi mutation; 409 xung đột hoặc enrollment không nhất quán; 422 payload/ảnh không hợp lệ; 503 model/dịch vụ/storage không sẵn sàng.

## UI và an toàn vận hành

- React cho ADMIN/SUPER_ADMIN màn hình quản lý, ẩn chức năng tạo ADMIN/đổi role với ADMIN; SUPER_ADMIN/DELETED chỉ đọc. Backend vẫn chặn request giả mạo.
- sessionStorage giữ token, /auth/me xác minh; không ghi token vào log/URL/ảnh chụp/commit. API dùng Cache-Control: no-store.
- Duplicate gate không tiết lộ tên người có mặt trùng. Ngưỡng L2 0.9 là heuristic chưa hiệu chuẩn, có thể bỏ sót hoặc từ chối nhầm; không phải kết quả độ chính xác sinh trắc học.
- Enrollment có giám sát quản trị; consent checkbox không thay thế quy trình đồng ý và xác minh danh tính độc lập. PAD áp dụng ở đăng nhập, không tuyên bố enrollment có liveness riêng.
- CORS mặc định localhost:5173 và 127.0.0.1:5173; tùy chỉnh VSHIELD_CORS_ORIGINS. Triển khai thật cần HTTPS, rate limiting, ACL, bảo vệ backup và mã hóa dữ liệu; ứng dụng chưa tự cung cấp các lớp này.

## Chạy và test

Backend từ thư mục gốc:

```powershell
.\.venv\Scripts\python.exe -m uvicorn vshield.api.app:app --reload --port 8000
```

Terminal frontend:

```powershell
cd frontend
npm.cmd run dev
```

UI http://localhost:5173; API docs http://localhost:8000/docs. Makefile hiện tại cũng hỗ trợ make dev/make frontend.

Test từ thư mục gốc:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rbac_api.py tests/test_rbac_database.py tests/test_rbac_identity.py -q
.\.venv\Scripts\python.exe -m pytest tests/ -q
node --test frontend/tests/auth-api.test.js frontend/tests/role-permissions.test.js
npm.cmd --prefix frontend run build
```

Test bao phủ USER tạo USER/ADMIN bị chặn; ADMIN tạo USER thành công nhưng tạo ADMIN/đổi role thất bại; SUPER_ADMIN tạo USER/ADMIN, đổi USER thành ADMIN thành công; disabled identity bị từ chối. Có thêm migration/bootstrap, session revocation, payload injection, duplicate và index refresh. Fixture test không chứng minh camera/PAD/nhận diện thật.

## File triển khai chính

- Schema/quyền: src/vshield/api/roles.py, migrations.py, database.py, user_store.py, bootstrap.py, sessions.py.
- HTTP: src/vshield/api/app.py, user_routes.py, user_schemas.py.
- Enrollment: src/vshield/services/user_management.py, enrollment.py, enrollment_images.py; scripts/manage-identities.py.
- Recognition: src/vshield/services/authentication.py; src/vshield/core/managed_identity_index.py, authorization_gallery.py, chroma_identity_index.py.
- UI: frontend/src/user-management.jsx, user-enrollment-form.jsx, face-enrollment.jsx, role-permissions.js và auth/dashboard tích hợp.

## Điều kiện chưa hoàn tất

Chưa enroll người thật hoặc gán 54 ảnh legacy, chưa kiểm thử camera end-to-end. Thiếu PAD model hợp lệ tại artifacts/models/face_verify_v1.keras nên đăng nhập trả 503; cần trọng số FaceNet phù hợp. Không bỏ PAD để thử đăng nhập.

Dataset v2 vẫn có 0 mẫu phát hành; RBAC không sửa dataset hoặc cung cấp bằng chứng độ chính xác. Xem [external-datasets.md](external-datasets.md).

Cần quyết định trước vận hành: ảnh/consent đúng người, SUPER_ADMIN đầu tiên, phục hồi/chuyển giao owner, retention/xóa/backup, hiệu chuẩn ngưỡng và benchmark hợp lệ.
