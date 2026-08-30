# Prompt for GPT Plus — V-Shield Capstone Document

## Cách sử dụng an toàn

1. Không upload `data/faces/`, ảnh/video dataset, `login_logs.db`, `.env`, model weights, API keys, embeddings hoặc dữ liệu sinh trắc cá nhân.
2. Upload mã nguồn và tài liệu văn bản cần thiết, ưu tiên danh sách bên dưới.
3. Dán nguyên prompt trong khối code vào GPT Plus.
4. Nếu trường có template, rubric hoặc chuẩn trích dẫn riêng, upload thêm và yêu cầu ưu tiên chúng.

## File nên upload

- `docs/codebase-summary.md`
- `README.md`
- `pyproject.toml`
- `src/vshield/api/app.py`
- `src/vshield/api/database.py`
- `src/vshield/services/authentication.py`
- `src/vshield/core/face_preprocessor.py`
- `src/vshield/core/anti_spoof.py`
- `src/vshield/core/embedder.py`
- `src/vshield/core/identity_index.py`
- `src/vshield/core/verifier.py`
- `src/vshield/evaluation/pad_metrics.py`
- `src/vshield/evaluation/recognition_metrics.py`
- `src/vshield/training/train.py`
- `frontend/src/App.jsx` và ba file trong `frontend/src/pages/`
- `plans/vshield-ai-ml-audit-2026-07-30.md`
- `plans/vshield-data-integrity-remediation/plan.md`
- `artifacts/evaluation/v1/INVALID.md`
- `artifacts/evaluation/v2/data-card.md`
- `artifacts/evaluation/v2/data-release-audit.json`

## Prompt copy-paste

```text
Bạn đóng vai Senior AI Researcher, AI Engineer và giảng viên hướng dẫn capstone. Hãy viết tài liệu capstone bằng tiếng Việt cho dự án V-Shield dựa duy nhất trên các file tôi đã upload.

MỤC TIÊU
Tạo báo cáo có chất lượng học thuật nhưng phản ánh đúng code và bằng chứng hiện có. Báo cáo phải giúp hội đồng hiểu bài toán, kiến trúc, pipeline AI, thiết kế thí nghiệm, mức độ hoàn thiện, giới hạn và lộ trình cải thiện. Văn phong rõ, kỹ thuật, có lập luận; không quảng cáo quá mức.

QUY TẮC BẰNG CHỨNG — BẮT BUỘC
1. Không bịa số liệu, experiment, accuracy, loss curve, latency, throughput, FAR, FRR, EER, APCER, BPCER, ACER, Precision@5 hoặc Recall@5.
2. Chỉ dùng con số có trong file nguồn và ghi nguồn ngay sau claim theo dạng `[Nguồn: đường-dẫn-file, mục hoặc key]`.
3. Nếu README mâu thuẫn với audit/evaluation artifact, ưu tiên audit và artifact; mô tả mâu thuẫn trung thực.
4. Nếu thiếu bằng chứng, ghi `N/A — chưa đủ dữ liệu để kết luận` hoặc `[CẦN BỔ SUNG NGUỒN]`. Không nội suy.
5. Phân biệt rõ ba loại nội dung: `đã triển khai`, `đã kiểm thử bằng unit/integration test`, và `đã chứng minh bằng experiment hợp lệ`.
6. Unit test với vector tổng hợp chỉ chứng minh logic phần mềm, không phải độ chính xác nhận diện thực tế.
7. Không đưa username thật, ảnh mặt, embedding, login record, bí mật hoặc dữ liệu cá nhân vào báo cáo.

SỰ THẬT NỀN CẦN KIỂM TRA LẠI TỪ FILE
- V-Shield là hệ thống xác thực khuôn mặt client-server: React/Vite -> FastAPI -> face preprocessing -> anti-spoof CNN -> FaceNet embedding chuẩn hóa L2 -> ChromaDB persistent local ưu tiên, FAISS/NumPy fallback -> SQLite role/login log.
- SQLite chỉ lưu role và login audit. `data/faces/<username>/*` chỉ seed Chroma khi local collection còn trống; sau đó Chroma là kho được query trực tiếp. Repo chưa có enrollment API hay tự động đồng bộ/revoke khi filesystem gallery thay đổi.
- Runtime fail-closed: anti-spoof không khả dụng hoặc không qua thì không chạy FaceNet/identity search.
- Dataset v1 không hợp lệ để chọn model, threshold hoặc công bố performance.
- Các số cần đối chiếu nguồn trước khi dùng: 2.415 mẫu lịch sử bị quarantine; 264 nhóm exact duplicate xuyên split với 746 references; audit lịch sử có 1.567 cặp near-duplicate dHash <= 4; scan OpenCV bảo thủ mới có 5.998 candidates chờ review, không phải 5.998 confirmed duplicates; metadata-only baseline đạt 99,59% accuracy và ROC-AUC 1,0.
- Dataset v2 hiện có 0 released samples; retraining và locked evaluation đang bị block bởi provenance/release gates.
- Tại snapshot hiện tại chưa có versioned recognition gallery/probe protocol và chưa có enrollment identity hợp lệ để công bố Precision@5/Recall@5 thực tế. Kết quả repo-level phải ghi N/A.

PHÂN BIỆT HAI BÀI TOÁN EVALUATION
A. PAD/anti-spoof là binary classification real/fake. Trình bày ROC-AUC, PR-AUC, confusion matrix, binary precision/recall/F1, APCER, BPCER, ACER, threshold chọn trên validation và locked test — nhưng chỉ như protocol đề xuất nếu chưa có data v2 hợp lệ.
B. Identity recognition là closed-set retrieval trên gallery enrollment. Trình bày Precision@5, Recall@5 và Hit Rate@5; không trộn các metric này với binary PAD precision/recall.

ĐỊNH NGHĨA IDENTITY RETRIEVAL
- Mỗi identity có thể có nhiều template. Score identity là khoảng cách L2 nhỏ nhất giữa probe đã normalize và các template của identity đó.
- Top-5 phải gồm tối đa năm identity duy nhất; một người có nhiều ảnh không được chiếm nhiều hạng.
- Ranking dùng để evaluate nên không áp dụng authentication threshold hoặc runner-up margin.
- Với probe q, ground truth G(q), và R5(q) là năm identity đầu:
  Precision@5(q) = |R5(q) giao G(q)| / 5
  Recall@5(q) = |R5(q) giao G(q)| / |G(q)|
- Macro-average theo probe, không weight theo số template.
- Với một ground-truth identity/probe, Recall@5 bằng Hit Rate@5 và Precision@5 tối đa là 0,2. Phải giải thích để tránh diễn giải sai.
- Open-set unknown phải dùng FPIR/FNIR riêng, không ép vào closed-set Precision@5/Recall@5.

PROTOCOL CẦN ĐỀ XUẤT, KHÔNG ĐƯỢC GIẢ VỜ ĐÃ CHẠY
1. Tạo gallery và probe độc lập nhưng có cùng closed-set identities; không dùng cùng ảnh ở hai phía.
2. Tách theo subject/session/clip/device phù hợp mục tiêu, loại exact/near duplicates và ghi sample ID/hash/group provenance.
3. Freeze gallery, probe manifest, preprocessing version, model hash, code commit và seed.
4. Báo query count, gallery identity/template count, Precision@5, Recall@5/Hit Rate@5, confidence interval theo independent group, slice theo session/device/lighting nếu đủ dữ liệu.
5. PAD threshold chỉ chọn trên validation; test v2 khóa và chạy một lần sau khi model/threshold freeze.
6. Nếu protocol hoặc sample size chưa đủ, ghi limitation và kế hoạch thu thập; không tạo bảng kết quả giả.

CẤU TRÚC TÀI LIỆU BẮT BUỘC
1. Trang thông tin đề tài — dùng placeholder cho trường, khoa, sinh viên, giảng viên, niên khóa.
2. Tóm tắt và Abstract tiếng Anh.
3. Chương 1: Bài toán, bối cảnh, mục tiêu, phạm vi, yêu cầu chức năng/phi chức năng, đóng góp thực tế.
4. Chương 2: Cơ sở lý thuyết — PAD, CNN, FaceNet, embedding, L2 normalization/distance, FAISS, threshold/margin, top-k retrieval và metric definitions.
5. Chương 3: Phân tích và thiết kế hệ thống — use cases, kiến trúc client-server, component diagram, sequence/data flow, failure states.
6. Chương 4: Dữ liệu và governance — collection, manifest, deduplication, grouped split, shortcut/confound, versioning, privacy. Nêu rõ v1 invalid và v2 blocked.
7. Chương 5: Cài đặt — backend, frontend, model lifecycle, enrollment, identity index, SQLite, API, error handling, fail-closed behavior.
8. Chương 6: Thiết kế đánh giá — tách PAD và identity retrieval; protocol, formulas, acceptance gates, statistical uncertainty, reproducibility.
9. Chương 7: Kết quả hiện có — chỉ trình bày evidence thật. Tạo bảng `Metric | Value | Evidence status | Source`. Giá trị chưa có phải là N/A.
10. Chương 8: Kiểm thử phần mềm, triển khai và vận hành — pytest, Ruff, Docker/uv, hardware assumptions, logging; không biến test pass thành model accuracy.
11. Chương 9: Bảo mật, privacy, ethics và threat model — biometric data, consent, retention/deletion, access control, replay/presentation attacks, endpoint security, CORS, localStorage, audit logs.
12. Chương 10: Hạn chế, threats to validity, roadmap và kết luận.
13. Tài liệu tham khảo — không bịa citation. Khi cần nguồn học thuật chưa được upload, chèn `[CẦN BỔ SUNG TÀI LIỆU THAM KHẢO]` và gợi ý loại nguồn cần tìm.
14. Phụ lục — API contract, cấu trúc repo, config, lệnh chạy, glossary và checklist tái lập.

YÊU CẦU TRÌNH BÀY
- Dùng heading đánh số, bảng ngắn và công thức rõ ràng.
- Tạo Mermaid diagrams hợp lệ cho system architecture, authentication sequence, dataset governance và evaluation flow.
- Mỗi sơ đồ phải có đoạn giải thích và liên kết tới module nguồn.
- Khi mô tả code, dẫn đường dẫn file/function; không chép nguyên file dài.
- Tạo bảng đối chiếu `Claim -> Evidence -> Limitation` cho các claim AI quan trọng.
- Tạo bảng tách `Implemented`, `Tested`, `Experimentally validated`, `Blocked`.
- Kết thúc bằng danh sách việc cần làm theo P0/P1/P2 và tiêu chí nghiệm thu định lượng nhưng không tự đặt kết quả hiện tại.

QUY TRÌNH OUTPUT
1. Trước tiên xuất: danh sách file đã đọc, conflict/missing evidence, và outline chi tiết.
2. Sau đó viết báo cáo theo từng chương hoàn chỉnh.
3. Nếu vượt giới hạn token, dừng ở cuối một mục hoàn chỉnh, ghi mục tiếp theo, và chờ tôi nói `tiếp tục`; không rút gọn bằng cách bỏ phần evaluation/limitations.
4. Cuối cùng chạy self-review: liệt kê mọi con số đã dùng cùng nguồn; liệt kê mọi N/A; kiểm tra không có claim vượt bằng chứng.

Bắt đầu bằng evidence inventory và outline. Chưa viết số liệu nào cho đến khi đã đối chiếu các file upload.
```

## Kết quả mong đợi

GPT Plus phải tạo outline trước, chỉ viết claim sau khi kiểm kê nguồn, và giữ các metric chưa có ở trạng thái `N/A`. Nếu GPT Plus đưa ra accuracy hoặc Precision@5/Recall@5 thực tế không có nguồn, yêu cầu nó xóa và chạy lại bước self-review.

## Câu hỏi chưa giải quyết

- Trường yêu cầu IEEE, APA hay chuẩn trích dẫn khác?
- Có template Word, số trang, rubric hoặc chương bắt buộc không?
