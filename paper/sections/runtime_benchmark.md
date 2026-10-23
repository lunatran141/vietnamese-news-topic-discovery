# 6.5. Đánh giá hiệu năng thời gian chạy và tiêu thụ tài nguyên (Runtime & Resource Benchmark)

Bên cạnh chất lượng phân cụm và độ mạch lạc chủ đề, chi phí tính toán là tiêu chí cốt lõi để đánh giá tính khả thi khi triển khai các hệ thống xử lý dòng tin tức thời gian thực. Bảng 6.6 tổng hợp kết quả đo lường thời gian thực thi theo từng công đoạn, bộ nhớ đỉnh và điều kiện môi trường của năm phương pháp trên cùng ngữ liệu Dataset v1.0.

**Bảng 6.6.** So sánh thời gian thực thi (giây) và tài nguyên của 5 phương pháp khám phá chủ đề trên ngữ liệu tin tức tiếng Việt.

| Phương pháp | Cấu hình tối ưu | Tập dữ liệu huấn luyện | Biểu diễn (Representation) | Huấn luyện (Fit) | Dự đoán Test (Predict) | Đánh giá (Evaluation) | Tổng pipeline (s) | Khởi tạo mô hình (Setup) | RAM đỉnh (MB) | Dung lượng đĩa (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **K-Means** | TF-IDF ($K=9$, seed 42) | Train (6.493 bài) | $22{,}09 \pm 1{,}72$ | $4{,}87 \pm 0{,}52$ | $0{,}015 \pm 0{,}003$ | $0{,}004 \pm 0{,}001$ | **$26{,}98 \pm 1{,}72$** | 0,00 | **58,9** | **2,25** |
| **NMF** | TF-IDF ($K=10$, seed 42) | Train+Val (7.884 bài) | 18,45 | 5,40 | 0,79 | 0,11 | **25,59** | 0,00 | N/A | N/A |
| **LDA** | BoW ($K=15$, seed 42) | Train+Val (7.884 bài) | 2,33 | 19,76 | 0,23 | 0,01 | **$22{,}34 \pm 0{,}35$** | 0,00 | N/A | N/A |
| **HDBSCAN** | MiniLM + UMAP 5D | Train (6.493 bài) | 38,23 | 23,63 | 6,64 | 0,03 | **44,86** | 16,97 | N/A | N/A |
| **BERTopic** | VN-SBERT + UMAP + HDBSCAN | Train (6.493 bài) | 1.084,73 | 57,50 | 1,38 | 21,54 | **1.191,65** | 24,56 | N/A | N/A |

*Ghi chú: Toàn bộ thời gian được đo bằng `time.perf_counter()` (đơn vị: giây). Các giá trị thiếu hoặc chưa được hỗ trợ đo đạc độc lập được ký hiệu là N/A theo đúng quy chuẩn E08. Thời gian tải pre-trained language model từ đĩa được tách riêng ở cột Khởi tạo mô hình (Setup) và không gộp vào tổng pipeline suy luận.*

---

## Phân tích kết quả và thảo luận

1. **Sự vượt trội về tốc độ của nhóm mô hình từ vựng truyền thống:** 
   Các phương pháp dựa trên ma trận tần suất từ (TF-IDF + K-Means, NMF và BoW + LDA) thể hiện ưu thế áp đảo về tốc độ thực thi toàn trình trên CPU tiêu chuẩn (đều dưới 30 giây). Trong đó, K-Means có tốc độ suy luận trên 1.390 bài báo của tập Test cực kỳ nhanh chóng ($0{,}015$ giây, tương đương hơn 90.000 bài/giây) và bộ nhớ RAM đỉnh chỉ tiêu tốn 58,9 MB. Mô hình K-Means cùng bộ vectorizer chỉ chiếm 2,25 MB dung lượng lưu trữ trên đĩa, rất phù hợp cho các thiết bị biên hoặc dịch vụ tin tức tải cao.

2. **Chi phí tính toán của biểu diễn ngữ nghĩa dày đặc:** 
   Việc chuyển dịch từ biểu diễn túi từ sang biểu diễn vector nhúng ngữ cảnh sâu (sentence embeddings) khiến chi phí tính toán tăng vọt theo cấp số nhân:
   - Với **HDBSCAN** (sử dụng multilingual MiniLM), bước trích xuất vector ngữ nghĩa và xấp xỉ dự đoán Test mất 44,86 giây (chưa kể 242,72 giây để mã hóa ngữ liệu Train+Validation).
   - Với **BERTopic** (sử dụng mô hình ngôn ngữ sâu Vietnamese SBERT), tổng thời gian thực thi lên tới **1.191,65 giây** (gần 20 phút), trong đó riêng công đoạn tạo embedding chiếm tới 91,0% tổng thời gian (1.084,73 giây). Điều này phản ánh rõ ràng sự đánh đổi lớn giữa độ bao quát ngữ nghĩa và năng lực phần cứng: BERTopic đòi hỏi tài nguyên tính toán GPU chuyên dụng để có thể vận hành khả thi trong môi trường sản xuất thực tế.

3. **Cảnh báo về tính không đồng nhất của điều kiện đo:** 
   Khi so sánh các con số runtime, nghiên cứu lưu ý hai yếu tố phương pháp luận quan trọng:
   - *Khác biệt về phạm vi dữ liệu huấn luyện:* NMF và LDA tái huấn luyện trên tập gộp Train+Validation (7.884 bài), trong khi K-Means, HDBSCAN và BERTopic chỉ huấn luyện trên tập Train (6.493 bài).
   - *Khác biệt về điều kiện môi trường:* Các số liệu của K-Means được đo lặp lại 3 lần trên máy tính 16 vCPUs (13,76 GB RAM, Windows), trong khi BERTopic được thực thi trên môi trường CPU tiêu chuẩn. Do đó, các giá trị này phản ánh đặc tính tương đối về độ phức tạp thuật toán thay vì một phép đối đầu tuyệt đối trên cùng một phần cứng vật lý duy nhất.

---
