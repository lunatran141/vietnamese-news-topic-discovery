# TV3 — Validation notes cho dữ liệu LDA

## 1. Thông tin task

- Thành viên: TV3 — Tuấn.
- Task: 3.1 — Đọc Dataset v1 và preprocessing variants.
- Output yêu cầu: Validation notes.
- Tiêu chí hoàn thành: Dùng đúng snapshot/config.
- Trạng thái: Đang chuẩn bị; chờ TV1 bàn giao Dataset v1.
- Ngày cập nhật: 04/10/2026.

## 2. Mục đích kiểm tra

Xác nhận dữ liệu dùng cho LDA thuộc đúng Dataset v1
đã được nhóm duyệt và được tạo bằng cấu hình tiền xử lý
tương ứng.

Không tự thay đổi dữ liệu gốc, article_id hoặc cấu hình
tiền xử lý của nhóm.

## 3. Dữ liệu S01 đã lấy về máy

- Repository:
  https://github.com/lunatran141/vietnamese-news-topic-discovery

- Branch nguồn:
  feature/S01-news-crawler

- File raw trên máy:
  D:\NLP\news-s01-reference\data\raw\news_articles.csv

- Đã xác nhận file tồn tại: Có.
- Dung lượng theo lệnh dir: 42.506.167 bytes.
- Commit của bản tham chiếu: 5c215c0130e00c7f80ffad021a7838fdd2ddc373
- SHA256 của raw CSV: 548b35953368857373e8f26639531b96ab83b742ed6432bff0b6462661b5077a
- Số bài và schema: Chưa kiểm tra trên máy.

Lưu ý: Raw S01 chưa phải Dataset v1 đã được duyệt.

## 4. Gói dữ liệu cần nhận từ TV1

| Thành phần | Trạng thái | Đường dẫn/link |
|---|---|---|
| Dataset v1 đã được nhóm duyệt | Chờ bàn giao | |
| Manifest mô tả Dataset v1 | Chờ bàn giao | |
| P0 — làm sạch tối thiểu | Chờ bàn giao | |
| P1 — chuẩn hóa | Chờ bàn giao | |
| P2 — tách từ tiếng Việt | Chờ bàn giao | |
| P3 — loại stopword | Chờ bàn giao | |
| Cấu hình preprocessing | Chờ bàn giao | |
| Tokenizer và phiên bản | Chờ bàn giao | |
| Danh sách stopword | Chờ bàn giao | |
| Hướng dẫn tái tạo dữ liệu | Chờ bàn giao | |

## 5. Thông tin snapshot/config sẽ xác nhận

- Dataset version:
- Manifest path:
- Commit hoặc link bàn giao:
- Hash các file dữ liệu:
- Preprocessing config path:
- Config hash nếu có:
- Tokenizer và phiên bản:
- Stopword list và phiên bản:
- Quy tắc tạo Title-only/Title+Content:
- Input và preprocessing cho LDA:
  Chưa chốt; cần thống nhất với nhóm.

## 6. Checklist kiểm tra sau khi nhận Dataset v1

| Kiểm tra | Kết quả | Bằng chứng/ghi chú |
|---|---|---|
| File thuộc đúng Dataset v1 được duyệt | Chờ kiểm tra | |
| Hash/config khớp manifest | Chờ kiểm tra | |
| Có đầy đủ cột theo schema chung | Chờ kiểm tra | |
| article_id không thiếu hoặc trùng | Chờ kiểm tra | |
| P0–P3 có cùng tập article_id | Chờ kiểm tra | |
| Thứ tự ID nhất quán hoặc join bằng ID an toàn | Chờ kiểm tra | |
| Không có processed text rỗng chưa xử lý | Chờ kiểm tra | |
| Metadata cần thiết được giữ | Chờ kiểm tra | |
| Không nối category_gold/source vào model text | Chờ kiểm tra | |
| Đã đọc mẫu trước/sau preprocessing | Chờ kiểm tra | |
| Tokenization phù hợp với input LDA | Chờ kiểm tra | |

## 7. Vấn đề và câu hỏi cần xác nhận

- Dataset v1 chính thức nằm ở đâu?
- Manifest và preprocessing config nào đi cùng dữ liệu?
- P0–P3 đã được review và duyệt chưa?
- Quy tắc tạo Title-only/Title+Content là gì?
- Nhóm chọn input/preprocessing nào cho LDA?
- Nếu có dữ liệu thay đổi, nhóm thông báo phiên bản mới thế nào?

## 8. Kết luận hiện tại

Đã lấy được raw CSV S01 và chuẩn bị biểu mẫu review.

Chưa xác nhận được snapshot/config Dataset v1 vì chưa
nhận đủ gói dữ liệu đã được duyệt.

Task 3.1 chưa hoàn thành.
