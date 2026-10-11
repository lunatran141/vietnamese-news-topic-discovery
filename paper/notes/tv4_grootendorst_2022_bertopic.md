# Paper notes — Mẫu nghiên cứu liên quan cho R06

## 1. Người cung cấp
- Thành viên: TV4 — Vân
- Phương pháp phụ trách: Embeddings, HDBSCAN, Topic Modeling
- Ngày đọc: 24/10/2026

## 2. Citation info — Thông tin trích dẫn
- Citation key: grootendorst_2022_bertopic
- Tên đầy đủ của tài liệu: BERTopic: Neural topic modeling with a class-based TF-IDF procedure
- Tác giả: Maarten Grootendorst
- Năm xuất bản: 2022
- Tạp chí/hội nghị hoặc nơi xuất bản: arXiv preprint (arXiv:2203.05794)
- DOI hoặc URL nguồn gốc: https://arxiv.org/pdf/2203.05794
- Loại tài liệu: Scientific paper

## 3. Method — Phương pháp
- Bài toán và mục tiêu: Tự động trích xuất các chủ đề tiềm ẩn có độ mạch lạc cao từ kho văn bản mà không bị ràng buộc bởi giả định cụm dạng hình cầu quanh tâm cụm.
- Phương pháp: Tạo nhúng văn bản bằng mô hình ngôn ngữ dựa trên Transformer tiền huấn luyện, giảm chiều dữ liệu bằng UMAP, gom cụm mật độ bằng HDBSCAN và trích xuất từ khóa chủ đề bằng thuật toán biến thể c-TF-IDF (class-based TF-IDF).
- Representation: Sử dụng vector nhúng cấp tài liệu (document embeddings) từ pretrained language models để mã hóa ngữ cảnh.
- Ý tưởng/cơ chế chính: Tách rời hoàn toàn giai đoạn gom cụm tài liệu khỏi giai đoạn biểu diễn từ khóa đại diện chủ đề. Dùng c-TF-IDF để tính mức độ quan trọng của từ ngữ theo từng cụm thay vì chọn các từ gần tâm cụm hình cầu.

## 4. Dataset — Dữ liệu
- Tên/nguồn: Thử nghiệm trên tập dữ liệu Twitter chính trị (Trump dataset) và các benchmark chuẩn cho topic modeling tĩnh và động.
- Ngôn ngữ và miền nội dung: Tiếng Anh (hỗ trợ đa ngữ khi tích hợp multilingual language models).
- Quy mô: Kích thước từ vựng và số lượng tài liệu thử nghiệm tăng dần từ 1,000 đến 43,000 tài liệu (bước nhảy 2,000 tài liệu).
- Cách chia dữ liệu/protocol: Đánh giá trên dải từ 10 đến 50 chủ đề (bước nhảy 10 chủ đề), tính toán lặp lại qua 3 lượt chạy độc lập cho mỗi cấu hình (trung bình cộng 15 lượt chạy riêng biệt); với tác vụ dynamic topic modeling, điểm số được tính trung bình trên 9 mốc thời gian qua 27 giá trị.

## 5. Metric — Đánh giá
- Chỉ số và cách đánh giá: Topic Coherence (TC), Topic Diversity (TD) và thời gian tính toán thực thi theo giây.
- Kết quả/phát hiện chính: BERTopic học được các mẫu ngôn ngữ mạch lạc, duy trì hiệu năng ổn định và cạnh tranh trên nhiều tác vụ chuẩn so với các mô hình cổ điển và mô hình dựa trên cụm khác. Tốc độ tính toán nhanh và khả năng mở rộng tốt khi tăng quy mô kho ngữ liệu.
- Điều kiện đi kèm số liệu: Kết quả TC và TD được tính trung bình qua 15 lượt chạy riêng biệt trên dải 10–50 chủ đề; thời gian xử lý đo trên dải 1,000 đến 43,000 tài liệu.

## 6. Gap — Hạn chế hoặc khoảng trống
- Hạn chế/vấn đề chưa giải quyết: Hiệu năng phụ thuộc nhiều vào việc lựa chọn mô hình nhúng và cấu hình siêu tham số của thuật toán gom cụm; có thể phân loại một số văn bản thành điểm nhiễu ngoại lai không gán chủ đề.
- Do ai nêu: Tác giả Maarten Grootendorst và các nghiên cứu ứng dụng thực nghiệm liên quan.
- Phạm vi nhận định: Áp dụng trên các tập dữ liệu có độ biến thiên phân phối cao hoặc khi cấu hình tham số HDBSCAN chưa tối ưu.

## 7. Liên hệ với đồ án
- Tài liệu hỗ trợ: Cung cấp nền tảng lý thuyết và quy trình chuẩn kết hợp Sentence Embeddings + UMAP + HDBSCAN + c-TF-IDF cho bài toán khám phá chủ đề và phân cụm văn bản.
- Nhóm kế thừa: Kế thừa pipeline trích xuất vector nhúng kết hợp thuật toán HDBSCAN để gom nhóm văn bản dựa trên mật độ mà không cần xác định trước số cụm.
- Khác biệt cần lưu ý: BERTopic tập trung giải quyết bài toán Topic Modeling tĩnh và động, trong khi đồ án hướng tới Event Detection (phát hiện và gom cụm các sự kiện thời sự/mạng xã hội theo mốc thời gian thực).

## 8. Vị trí bằng chứng
| Nhận định/số liệu | Section/trang/bảng/vị trí nguồn |
|---|---|
| Khung 3 bước: Document embeddings, Dimensionality reduction & Clustering, c-TF-IDF | Section 1 (Introduction) |
| Giới hạn của cách tiếp cận giả định tâm cụm hình cầu | Section 1 (Introduction) |
| Đánh giá TC và TD trung bình qua 15 lượt chạy trên 10-50 chủ đề | Table 1 |
| Thời gian thực thi trên quy mô 1,000 đến 43,000 tài liệu | Figure 1 |
| Đánh giá trên dynamic topic modeling qua 9 mốc thời gian | Table 3 |
| Kết luận về tính linh hoạt do tách rời gom cụm và biểu diễn chủ đề | Section 8 (Conclusion) |

## 9. Điều cần xác minh thêm
- Kiểm tra ảnh hưởng của tỷ lệ loại bỏ dữ liệu ngoại lai (outliers) do HDBSCAN tạo ra đối với độ bao phủ của sự kiện.
- Thử nghiệm việc thay thế mô hình ngôn ngữ tiếng Anh bằng PhoBERT hoặc multilingual sentence transformers khi áp dụng trên ngữ liệu tiếng Việt.

## 10. Trạng thái và bàn giao
- Trạng thái: Complete
- Link file hoặc PR bàn giao: paper/notes/tv4_grootendorst_2022_bertopic.md
- Nhận xét của TV3 sau review: Đã đối chiếu đúng mẫu cấu trúc R06