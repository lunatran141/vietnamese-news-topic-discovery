# Paper notes — Mẫu nghiên cứu liên quan cho R06

## 1. Người cung cấp
- Thành viên: TV4 — Vân
- Phương pháp phụ trách: Embeddings, HDBSCAN, event detection
- Ngày đọc: 09/10/2026

## 2. Citation info — Thông tin trích dẫn
- Citation key: nakshatri_2023_keyevents
- Tên đầy đủ của tài liệu: Using LLM for Improving Key Event Discovery: Temporal-Guided News Stream Clustering with Event Summaries
- Tác giả: Nishanth Nakshatri; Siyi Liu; Sihao Chen; Dan Roth; Dan Goldwasser; Daniel J. Hopkins
- Năm xuất bản: 2023
- Tạp chí/hội nghị hoặc nơi xuất bản: arXiv preprint
- DOI hoặc URL nguồn gốc: https://aclanthology.org/2023.findings-emnlp.274.pdf
- Loại tài liệu: Scientific paper

## 3. Method — Phương pháp
- Bài toán và mục tiêu: Tự động phát hiện và trích xuất các sự kiện tin tức trọng tâm (key news events) từ luồng tin tức lớn mà không cần gán nhãn thủ công.
- Phương pháp: Kết hợp phân tích xu hướng thời gian để tìm các điểm đột biến (peaks), dùng thuật toán phi tham số HDBSCAN để tạo các cụm ứng viên, sau đó dùng tóm tắt đa văn bản để tinh lọc và hợp nhất cụm.
- Representation: Embedding các tóm tắt sự kiện và văn bản bằng cùng một mô hình encoder ngữ nghĩa để tính cosine similarity.
- Ý tưởng/cơ chế chính: Tận dụng HDBSCAN để gom cụm ban đầu dựa trên mật độ bài viết trong các khoảng thời gian nóng, sau đó lọc bỏ các cụm thiếu nhất quán hoặc trùng lặp bằng mô hình tóm tắt LLM.

## 4. Dataset — Dữ liệu
- Tên/nguồn: NELA Dataset (tập dữ liệu tin tức đa chiều chính trị).
- Ngôn ngữ và miền nội dung: Tiếng Anh, các bài báo chính trị - xã hội từ 207 hãng tin tại Hoa Kỳ.
- Quy mô: Trích xuất từ kho ~1.8 triệu bài viết; tập KeyEvents xây dựng gồm 40,000 bài viết phân bổ trên 611 sự kiện thuộc 11 chủ đề.
- Cách chia dữ liệu/protocol: Trích xuất các bài viết theo từ khóa BM25 cho từng chủ đề, sau đó chạy phân cụm theo cửa sổ thời gian xung quanh điểm nóng (peak ± 1 ngày).

## 5. Metric — Đánh giá
- Chỉ số và cách đánh giá: Độ gắn kết của sự kiện (Event coherence), chất lượng ánh xạ bài viết vào cụm (Document mapping quality) và kiểm tra tính nhất quán thông qua chuyên gia.
- Kết quả/phát hiện chính: Pipeline đề xuất tạo ra các cụm sự kiện tập trung và mạch lạc hơn hẳn so với các baseline phân cụm thông thường, giảm tối đa các bài báo gán nhầm.
- Điều kiện đi kèm số liệu: Áp dụng trên tập 40k tài liệu NELA đã qua trích xuất từ khóa chủ đề và xác định đỉnh thời gian.

## 6. Gap — Hạn chế hoặc khoảng trống
- Hạn chế: Phương pháp phụ thuộc nhiều vào việc phát hiện đúng đỉnh tần suất (peaks) trong luồng tin; các sự kiện diễn ra âm ỉ kéo dài không có đỉnh đột biến dễ bị bỏ qua.
- Do ai nêu: Tác giả nhận định trong phần thảo luận phương pháp.
- Phạm vi nhận định: Áp dụng cho các luồng tin mang tính chu kỳ chính trị trên NELA.

## 7. Liên hệ với đồ án
- Tài liệu hỗ trợ: Minh chứng trực tiếp và rõ ràng nhất cho việc dùng HDBSCAN trong phát hiện sự kiện tin tức (Event Detection).
- Nhóm kế thừa: Kế thừa luận điểm sử dụng HDBSCAN như thuật toán gom cụm mật độ phi tham số không cần ấn định trước số lượng sự kiện K.
- Khác biệt cần lưu ý: Đồ án có thể chưa tích hợp bước lọc lại cụm bằng bộ tóm tắt đa văn bản mà dựa thuần túy vào khoảng cách vector embedding.

## 8. Vị trí bằng chứng
| Nhận định/số liệu | Section/vị trí nguồn |
|---|---|
| Sử dụng HDBSCAN tạo cụm sự kiện | Section 1 (Introduction), trang 2 |
| Cửa sổ thời gian peak ± 1 ngày | Section 2.2 (Inference), trang 3 |
| Quy mô KeyEvents 40k bài, 611 sự kiện | Abstract |

## 9. Điều cần xác minh thêm
- Kiểm tra chi phí tính toán khi chạy bộ mã hóa embedding trên toàn bộ tập dữ liệu tiếng Việt.

## 10. Trạng thái và bàn giao
- Trạng thái: chuẩn bị TV3 review
- Link file hoặc PR bàn giao: paper/notes/tv4_nakshatri_2023_keyevents.md
- Nhận xét của TV3 sau review: Chưa kiểm tra