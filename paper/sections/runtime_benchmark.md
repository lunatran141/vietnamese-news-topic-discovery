# 6.5. Đánh giá hiệu năng thời gian chạy và tiêu thụ tài nguyên (Runtime & Resource Benchmark)

Bên cạnh chất lượng phân cụm và độ mạch lạc chủ đề, chi phí tính toán và mức độ chiếm dụng tài nguyên phần cứng là tiêu chí cốt lõi để đánh giá tính khả thi khi triển khai các hệ thống xử lý dòng tin tức thời gian thực. Để đảm bảo tính so sánh công bằng và triệt tiêu hoàn toàn sự sai lệch do cấu hình máy tính khác nhau, toàn bộ thực nghiệm đánh giá tài nguyên và thời gian chạy (Task E08) được thực thi và chuẩn hóa trên **cùng một máy tính vật lý duy nhất** (16 vCPUs, 13,76 GB RAM, hệ điều hành Windows 11 x64, Python 3.13.7) theo đúng quy chuẩn tại Mục 7 trong tài liệu đánh giá của dự án (`docs/evaluation_protocol.md`).

Bảng 6.6 tổng hợp kết quả đo lường chi tiết theo từng công đoạn độc lập (representation, fit, predict trên tập Test, evaluation), bộ nhớ RAM đỉnh và dung lượng lưu trữ mô hình trên đĩa của năm phương pháp.

**Bảng 6.6.** So sánh hiệu năng thời gian thực thi (giây) và tài nguyên của 5 phương pháp khám phá chủ đề trên cùng một môi trường máy tính (Single-Machine Benchmark).

| Phương pháp | Cấu hình tối ưu | Tập dữ liệu huấn luyện | Biểu diễn (Representation) | Huấn luyện (Fit) | Dự đoán Test (Predict) | Đánh giá (Evaluation) | Tổng pipeline (s) | Khởi tạo mô hình (Setup) | RAM đỉnh (MB) | Dung lượng đĩa (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **K-Means** | TF-IDF ($K=9$, seed 42) | Train (6.493 bài) | $17{,}83 \pm 0{,}37$ | $3{,}92 \pm 0{,}06$ | $0{,}008 \pm 0{,}000$ | $0{,}004$ | **$21{,}76 \pm 0{,}43$** | 0,00 | **58,9** | **2,25** |
| **NMF** | TF-IDF ($K=10$, seed 42) | Train+Val (7.884 bài) | $34{,}30 \pm 0{,}57$ | $5{,}02 \pm 0{,}18$ | $0{,}055 \pm 0{,}012$ | $0{,}146$ | **$39{,}53 \pm 0{,}45$** | 0,00 | **363,1** | **22,26** |
| **LDA** | BoW ($K=15$, seed 42) | Train+Val (7.884 bài) | $8{,}96 \pm 1{,}41$ | $16{,}74 \pm 2{,}97$ | $0{,}201 \pm 0{,}008$ | $0{,}045$ | **$25{,}95 \pm 1{,}58$** | 0,00 | **77,0** | **3,87** |
| **HDBSCAN** | MiniLM + UMAP 5D | Train (6.493 bài) | 139,25 | 68,26 | 37,02 | 0,016 | **244,54** | 3,85 | **1.059,2** | **36,39** |
| **BERTopic** | VN-SBERT + UMAP + HDBSCAN | Train (6.493 bài) | 1.179,08 | 1.299,80 | 281,31 | 2,500 | **2.762,69** | 1,62 | **148,4** | **540,00** |

*Ghi chú: Toàn bộ thời gian được đo bằng `time.perf_counter()` (đơn vị: giây) trên cùng cấu hình 16 vCPUs (13,76 GB RAM, Windows 11). Các mô hình truyền thống (K-Means, NMF, LDA) được chạy lặp lại 3 lần và báo cáo giá trị trung bình $\pm$ độ lệch chuẩn. Thời gian tải mô hình ngôn ngữ từ cache cục bộ được tách riêng ở cột Khởi tạo mô hình (Setup) và không gộp vào tổng pipeline suy luận. Thời gian đọc ghi đĩa và vẽ đồ thị bị loại trừ khỏi phạm vi đo.*

---

## Phân tích kết quả và thảo luận

1. **Ưu thế áp đảo về tốc độ và chi phí của nhóm mô hình từ vựng truyền thống:** 
   Các phương pháp dựa trên không gian vector thưa (TF-IDF và Bag-of-Words) thể hiện sự vượt trội về cả tốc độ thực thi và mức độ tiết kiệm tài nguyên trên CPU:
   - **K-Means (TF-IDF):** Đạt tốc độ suy luận nhanh nhất trong toàn bộ nghiên cứu với chỉ $0{,}008$ giây để gán nhãn cho toàn bộ 1.390 bài báo của tập Test (tương đương tốc độ xử lý hơn 170.000 bài báo/giây). Bộ nhớ RAM đỉnh của K-Means chỉ dừng ở mức 58,9 MB và tổng dung lượng lưu trữ trên đĩa là 2,25 MB (bao gồm từ vựng TF-IDF 20.000 chiều và trọng số tâm cụm), chứng minh tính khả thi cao khi tích hợp vào các dịch vụ streaming có độ trễ cực thấp.
   - **NMF (TF-IDF):** Quá trình phân rã ma trận không âm ($K=10$) trên tập Train+Val hoàn thành trong $5{,}02$ giây, và thời gian dự đoán trên tập Test chỉ mất $0{,}055$ giây. Bộ nhớ RAM đỉnh tiêu thụ 363,1 MB do lưu giữ ma trận biểu diễn 50.000 đặc trưng, với kích thước file pipeline nén là 22,26 MB.
   - **LDA (Bag-of-Words):** Nhờ cơ chế lấy mẫu Gibbs trên không gian BoW 15.000 từ, bước biểu diễn chỉ mất $8{,}96$ giây và huấn luyện 20 vòng lặp tốn $16{,}74$ giây. LDA chỉ tiêu tốn 77,0 MB RAM và 3,87 MB bộ nhớ đĩa, biến nó thành giải pháp probabilistic topic modeling rất kinh tế.

2. **Chi phí tính toán và đánh đổi tài nguyên của biểu diễn ngữ nghĩa dày đặc:** 
   Sự chuyển dịch từ ma trận từ vựng thưa sang không gian nhúng ngữ cảnh sâu (dense sentence embeddings) kéo theo sự gia tăng rõ rệt về thời gian thực thi và tài nguyên phần cứng:
   - **HDBSCAN (Multilingual MiniLM-L12-v2):** Việc mã hóa ngữ liệu văn bản bằng mô hình 12-layer MiniLM mất 139,25 giây cho toàn bộ 7.883 bài (khoảng 0,0177 giây/bài). Tuy nhiên, bước giảm chiều UMAP 5D và xây dựng Minimum Spanning Tree (MST) trong HDBSCAN đòi hỏi $68{,}26$ giây huấn luyện và đẩy mức chiếm dụng RAM đỉnh lên **1.059,2 MB** — mức tiêu thụ RAM cao nhất trong 5 phương pháp. Bù lại, kích thước mô hình lưu trữ trên đĩa rất gọn gàng (36,39 MB).
   - **BERTopic (Vietnamese-SBERT):** Sử dụng mạng nơ-ron sâu RoBERTa (768 chiều), tổng thời gian thực thi pipeline trên CPU lên tới **2.762,69 giây** (~46 phút). Trong đó, bước tạo embedding mất 1.179,08 giây (~19,7 phút, tốc độ ~0,1496 giây/bài), bước huấn luyện UMAP, HDBSCAN và tính toán c-TF-IDF toàn thể chiếm 1.299,80 giây (~21,7 phút), và suy luận trên 1.390 bài Test mất 281,31 giây. Về không gian đĩa, checkpoint mô hình Vietnamese-SBERT chiếm tới **540,0 MB**, gấp hơn 240 lần so với mô hình K-Means.

3. **Ý nghĩa phương pháp luận và chuẩn hóa điều kiện đo:** 
   Việc hoàn thành toàn bộ phép đo trên một máy tính vật lý duy nhất đã giải quyết triệt để vấn đề "so sánh khập khiễng" thường gặp khi tổng hợp số liệu phân tán từ máy tính cá nhân của nhiều thành viên:
   - Tất cả các bước trích xuất biểu diễn, huấn luyện, suy luận và đánh giá đều được đo độc lập bằng đồng hồ thực `time.perf_counter()`, loại trừ hoàn toàn thời gian nạp mô hình từ mạng hoặc I/O đĩa.
   - Kết quả này khẳng định rõ rệt quy luật đánh đổi trong kỹ nghệ xử lý ngôn ngữ tự nhiên: Các mô hình ngữ nghĩa sâu (BERTopic, HDBSCAN) có khả năng nắm bắt ngữ cảnh tốt hơn nhưng đòi hỏi tài nguyên tính toán gấp từ 10 đến hơn 100 lần so với các mô hình truyền thống (K-Means, NMF, LDA). Đối với các hệ thống phân loại tin tức thực tế yêu cầu xử lý hàng triệu bài báo mỗi ngày mà không có hạ tầng GPU, K-Means và NMF vẫn là những lựa chọn thực dụng và kinh tế nhất.

---
