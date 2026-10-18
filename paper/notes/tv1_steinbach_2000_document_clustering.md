# Paper notes — Steinbach et al. (2000): A Comparison of Document Clustering Techniques

## 1. Người cung cấp

- Thành viên: TV1 — Khôi.
- Phương pháp phụ trách: Biểu diễn TF-IDF, thuật toán phân cụm K-Means và tiền xử lý dữ liệu văn bản.
- Ngày đọc/đối chiếu: 18/10/2026.

## 2. Citation info — Thông tin trích dẫn

- Citation key: `steinbach_2000_document_clustering`.
- Tên đầy đủ của tài liệu: *A Comparison of Document Clustering Techniques*.
- Tác giả: Michael Steinbach, George Karypis, Vipin Kumar.
- Năm xuất bản: 2000.
- Tạp chí/hội nghị hoặc nơi xuất bản: *KDD Workshop on Text Mining* (Hội nghị ACM SIGKDD lần thứ 4) và báo cáo kỹ thuật *Technical Report TR 00-034*, Department of Computer Science and Engineering, University of Minnesota, 20 trang.
- DOI hoặc URL nguồn gốc: https://hdl.handle.net/11299/215421
- Loại tài liệu: `Scientific paper` (bài báo workshop tại KDD Text Mining kèm báo cáo kỹ thuật đầy đủ tại kho lưu trữ học thuật của Đại học Minnesota).

## 3. Method — Phương pháp

- **Bài toán và mục tiêu:** Nghiên cứu thực nghiệm và đối chiếu hai nhánh tiếp cận phân cụm văn bản phổ biến nhất là phân cụm phân cấp tích tụ (chi phí thời gian bậc hai $O(n^2)$ theo số lượng văn bản) và phân cụm phân hoạch dựa trên K-Means gồm K-Means tiêu chuẩn và biến thể Bisecting K-Means (chi phí thời gian tuyến tính $O(n)$).
- **Phương pháp:** So sánh ba kỹ thuật phân cụm phân cấp tích tụ gồm Intra-Cluster Similarity Technique (IST), Centroid Similarity Technique (CST) và UPGMA với thuật toán K-Means tiêu chuẩn và Bisecting K-Means, đồng thời khảo sát hiệu quả của bước tinh chỉnh kết quả phân cụm bằng K-Means cơ bản.
- **Representation — cách biểu diễn dữ liệu:** Văn bản được biểu diễn theo mô hình không gian vector. Mỗi văn bản $d$ ban đầu là vector tần suất từ $d_{tf} = (tf_1, tf_2, \dots, tf_n)$ sau khi đã loại bỏ các từ dừng phổ biến và đưa các biến thể từ về gốc từ chung. Mỗi từ được nhân với trọng số tần suất nghịch đảo văn bản (IDF) nhằm giảm ảnh hưởng của các từ xuất hiện ở quá nhiều văn bản và ít có khả năng phân biệt chủ đề. Cuối cùng, mỗi vector văn bản được chuẩn hóa về độ dài đơn vị ($\|d\| = 1$) để xử lý sự chênh lệch độ dài giữa các bài viết.
- **Ý tưởng/cơ chế chính:**
  - Độ tương đồng giữa hai văn bản được đo bằng độ tương đồng cosine: $\text{cosine}(d_1, d_2) = (d_1 \bullet d_2) / (\|d_1\| \|d_2\|)$. Với các vector văn bản đã chuẩn hóa độ dài bằng 1, tích vô hướng $d_1 \bullet d_2$ chính là độ tương đồng cosine.
  - Tâm cụm $c$ của tập văn bản $S$ là trung bình cộng các vector văn bản trong cụm: $c = \frac{1}{|S|} \sum_{d \in S} d$. Tác giả chứng minh hai tính chất toán học nền tảng: tích vô hướng giữa một văn bản $d_1$ với tâm cụm $c$ bằng độ tương đồng cosine trung bình giữa $d_1$ với toàn bộ văn bản trong cụm ($d_1 \bullet c = \frac{1}{|S|} \sum_{d \in S} \text{cosine}(d_1, d)$), và bình phương độ dài tâm cụm $\|c\|^2$ bằng độ tương đồng từng cặp trung bình giữa mọi điểm trong cụm ($\frac{1}{|S|^2} \sum_{d \in S} \sum_{d' \in S} \text{cosine}(d', d) = \|c\|^2$).
  - Trong K-Means tiêu chuẩn, tác giả chọn ngẫu nhiên $K$ văn bản làm tâm cụm ban đầu, gán từng văn bản vào tâm cụm gần nhất theo thước đo cosine và cập nhật tâm cụm tăng dần ngay khi điểm mới được gán vào cụm vì cách cập nhật này mang lại độ tương đồng nội cụm cao hơn và entropy thấp hơn.
  - Trong Bisecting K-Means, thuật toán bắt đầu từ một cụm chứa toàn bộ dữ liệu, ở mỗi bước chọn một cụm (mặc định chọn cụm lớn nhất còn lại) và dùng K-Means cơ bản chia đôi cụm đó qua $\text{ITER}$ lần thử, giữ lại phép chia có độ tương đồng tổng thể cao nhất và lặp lại cho đến khi đạt đủ $K$ cụm.
  - Lý giải bản chất dữ liệu văn bản: Do văn bản có từ vựng chiều cao nhưng mỗi bài chỉ chứa một tập con từ vựng và các chủ đề có thể chia sẻ từ vựng chung, hai văn bản thuộc hai lớp khác nhau vẫn thường trở thành láng giềng gần nhất của nhau. Phân cụm phân cấp tích tụ ghép cặp cục bộ nên khi ghép nhầm ở giai đoạn đầu sẽ không thể sửa sai về sau; ngược lại, K-Means đánh giá độ tương đồng của văn bản với tâm cụm (tức dựa trên tính chất toàn cục của cả cụm) nên vượt qua được hiện tượng láng giềng cục bộ bị lẫn lớp.

## 4. Dataset — Dữ liệu

- **Tên/nguồn:** Nghiên cứu thực nghiệm trên 8 bộ dữ liệu văn bản chuẩn lấy từ ba nguồn: bộ phân loại tin tức **Reuters-21578** (`re0`, `re1`), dự án phân mục trang web Yahoo! **WebACE** (`wap`) và các tập tin tức thuộc hội nghị **TREC-5, TREC-6, TREC-7** (`tr31`, `tr45`, `fbis` từ Foreign Broadcast Information Service, `la1` và `la2` từ báo *Los Angeles Times*).
- **Ngôn ngữ và miền nội dung:** Tiếng Anh; bao gồm tin tức báo chí, bản tin thời sự và trang web phân loại theo danh mục. Đặc biệt, nhãn lớp của hai tập `la1` và `la2` được tạo trực tiếp từ tên chuyên mục báo *Los Angeles Times* như *Entertainment*, *Financial*, *Foreign*, *Metro*, *National* và *Sports*.
- **Quy mô (Table 1, trang 10 bản in / trang 12 PDF):**

| Bộ dữ liệu | Nguồn | Số văn bản | Số lớp | Kích thước lớp nhỏ nhất | Kích thước lớp lớn nhất | Kích thước lớp trung bình | Số lượng từ |
|---|---|---:|---:|---:|---:|---:|---:|
| `re0` | Reuters-21578 | 1.504 | 13 | 11 | 608 | 115,7 | 11.465 |
| `re1` | Reuters-21578 | 1.657 | 25 | 10 | 371 | 66,3 (bảng gốc in `663`) | 3.758 |
| `wap` | WebACE | 1.560 | 20 | 5 | 341 | 78,0 | 8.460 |
| `tr31` | TREC | 927 | 7 | 2 | 352 | 132,4 | 10.128 |
| `tr45` | TREC | 690 | 10 | 14 | 160 | 69,0 | 8.261 |
| `fbis` | TREC | 2.463 | 17 | 38 | 506 | 144,9 | 2.000 |
| `la1` | TREC | 3.204 | 6 | 273 | 943 | 534,0 | 31.472 |
| `la2` | TREC | 3.075 | 6 | 248 | 905 | 512,5 | 31.472 |

- **Cách chia dữ liệu/protocol — quy trình thí nghiệm:**
  - Tiền xử lý thống nhất trên cả 8 tập dữ liệu: loại bỏ từ dừng phổ biến (như *"a"*, *"are"*, *"do"*, *"for"*) và rút gọn gốc từ bằng thuật toán hậu tố Porter. Với các tập TREC, tác giả chỉ chọn những văn bản có đúng một nhãn đánh giá liên quan duy nhất.
  - Đây là bài toán phân cụm không giám sát nên không chia tập huấn luyện/kiểm thử kiểu học có giám sát; toàn bộ văn bản của từng tập được đưa vào phân cụm, sau đó dùng nhãn lớp chuẩn để tính các thước đo đánh giá ngoại tại.
  - Để đảm bảo tính công bằng khi so sánh tại các mức số cụm $K \in \{16, 32, 64\}$, tác giả đặt $\text{ITER} = 5$, chạy 10 lượt Bisecting K-Means và với mỗi lượt Bisecting K-Means thì chạy tương đương $\log_2(K) \times \text{ITER}$ lượt K-Means tiêu chuẩn. UPGMA có tính tất định nên chỉ cần chạy 1 lần trên mỗi tập dữ liệu.

## 5. Metric — Đánh giá

- **Chỉ số và cách đánh giá:** Bài báo kết hợp hai thước đo ngoại tại (đối chiếu với nhãn lớp có sẵn) và một thước đo nội tại (không dùng nhãn lớp):
  - **Entropy (thước đo ngoại tại, càng thấp càng tốt):** Đo mức độ đồng nhất nhãn lớp bên trong từng cụm. Với cụm $j$ có xác suất phân bố lớp $p_{ij}$, entropy của cụm là $E_j = -\sum_i p_{ij} \log(p_{ij})$, và tổng entropy toàn cục có trọng số theo kích thước cụm là $E_{CS} = \sum_{j=1}^m \frac{n_j \cdot E_j}{n}$.
  - **F-measure (thước đo ngoại tại, càng cao càng tốt):** Kết hợp $\text{Recall}(i, j) = n_{ij} / n_i$ và $\text{Precision}(i, j) = n_{ij} / n_j$ giữa lớp $i$ và cụm $j$ theo công thức $F(i, j) = \frac{2 \cdot \text{Recall}(i, j) \cdot \text{Precision}(i, j)}{\text{Precision}(i, j) + \text{Recall}(i, j)}$; điểm tổng hợp được tính bằng trung bình có trọng số của giá trị $F(i, j)$ lớn nhất mà mỗi lớp đạt được.
  - **Overall Similarity (thước đo nội tại, càng cao càng tốt):** Đo độ gắn kết nội cụm dựa trên tổng có trọng số của độ tương đồng trung bình giữa các văn bản trong từng cụm, tương đương với việc tính qua bình phương độ dài tâm cụm $\|c\|^2$.
- **Kết quả/phát hiện chính:**
  - Trong 3 kỹ thuật phân cấp tích tụ (Section 7.2, Table 2), UPGMA đạt F-measure cao nhất trên cả 8 tập dữ liệu (ví dụ trên `la1` đạt 0,6963 so với 0,4557 của Centroid Similarity và 0,5977 của Cluster Similarity) và được chọn làm đại diện tốt nhất của nhánh phân cấp để so sánh với K-Means.
  - Khi so sánh ở $K \in \{16, 32, 64\}$ (Section 8.2, Tables 3–8), Bisecting K-Means (có hoặc không có tinh chỉnh) vượt qua cả K-Means tiêu chuẩn lẫn UPGMA trong phần lớn trường hợp trên cả Entropy và Overall Similarity.
  - Đáng chú ý, **K-Means tiêu chuẩn** tuy kém hơn Bisecting K-Means nhưng **nhìn chung vẫn đạt kết quả tốt hơn UPGMA ngay cả khi UPGMA đã được tinh chỉnh thêm bằng K-Means**. Ví dụ tại $K = 16$ trên tập báo chí `la1` (Table 3 và Table 6), K-Means tiêu chuẩn đạt Entropy 1,0710 và Overall Similarity 0,2198, tốt hơn rõ rệt so với UPGMA gốc (Entropy 2,4046; Overall Similarity 0,1420) và UPGMA có tinh chỉnh (Entropy 1,2390; Overall Similarity 0,1995).
  - Về F-measure (Table 9, trang 16 bản in), Bisecting K-Means vượt UPGMA trên 7/8 tập dữ liệu (ví dụ trên `la1`: 0,7856 so với 0,6963; trên `la2`: 0,7882 so với 0,7168) và chỉ thấp hơn nhẹ trên `tr45` (0,8080 so với 0,8528).
  - Về thời gian thực thi (Section 8.2, trang 14 bản in): Trên tập `la1` gồm 3.204 bài báo và 31.472 từ, một lần chạy phân cụm phân cấp mất hơn 1 giờ trên máy tính Pentium, trong khi một lần chạy Bisecting K-Means tìm 32 cụm mất chưa đầy 1 phút trên cùng hệ thống.
- **Điều kiện đi kèm số liệu:** Các kết quả trên được đo trên 8 tập văn bản tiếng Anh đã lọc chỉ giữ bài đơn nhãn, áp dụng chuẩn hóa độ dài vector đơn vị, chạy nhiều lần khởi tạo ngẫu nhiên với cập nhật tâm cụm tăng dần ở $K \in \{16, 32, 64\}$. Không được dùng trực tiếp các con số Entropy hay F-measure này để suy diễn kết quả trên tập dữ liệu tin tức tiếng Việt Dataset v1.0 của đồ án.

## 6. Gap — Hạn chế hoặc khoảng trống

- **Hạn chế do tác giả nêu trực tiếp trong bài báo:**
  - **Độ nhạy với khởi tạo và biến thiên giữa các lần chạy của K-Means tiêu chuẩn:** Tác giả ghi nhận kết quả của K-Means tiêu chuẩn biến thiên khá lớn giữa các lần chạy đơn lẻ do chọn tâm cụm ban đầu ngẫu nhiên; một lần chạy đơn lẻ của K-Means tiêu chuẩn có thể cho kết quả kém hơn UPGMA nếu không thực hiện nhiều lần chạy (Section 8.2, trang 14 bản in; Section 9.1, trang 17 bản in; Section 10, trang 18 bản in).
  - **Sự mất cân bằng kích thước cụm của K-Means tiêu chuẩn:** Tác giả chỉ ra K-Means tiêu chuẩn thường sinh ra các cụm có kích thước chênh lệch rất lớn; các cụm quá lớn thường lẫn nhiều lớp và kéo chất lượng tổng thể (như Entropy có trọng số) đi xuống (Section 9.2, trang 17–18 bản in).
  - **Không phải thuật toán hoàn hảo cho mọi trường hợp:** Trong thực tế, đôi khi K-Means không tìm được các cụm tương ứng với các lớp văn bản mong muốn, và các giải thích lý thuyết ở Section 9 mới mang tính sơ bộ, cần được kiểm chứng thêm bằng các mô hình sinh dữ liệu văn bản nhân tạo (Section 9, trang 16–17 bản in).
- **Nhận xét của TV1 về khoảng trống đối với đồ án:**
  - **Ngôn ngữ và đặc thù tách từ:** Nghiên cứu gốc chỉ khảo sát văn bản tiếng Anh với bước stemming bằng thuật toán Porter, chưa xét đến đặc thù ngôn ngữ đơn lập như tiếng Việt — nơi từ ghép gồm nhiều âm tiết phân cách bằng khoảng trắng và cần bước tách từ chuyên biệt thay vì stemming.
  - **Giả định văn bản đơn nhãn:** Các tập dữ liệu TREC trong bài báo đã được lọc để mỗi văn bản chỉ có đúng một nhãn chủ đề liên quan (Section 6, trang 9 bản in), trong khi báo chí thực tế thường có hiện tượng giao thoa chủ đề giữa các chuyên mục.
  - **Chưa khảo sát khởi tạo k-means++ và cách chọn số cụm $K$:** Bài báo dùng khởi tạo ngẫu nhiên và cố định trước $K \in \{16, 32, 64\}$, chưa đánh giá cơ chế khởi tạo `k-means++` hay quy trình quét dải $K$ kết hợp nhiều chỉ số nội tại (Silhouette, Davies-Bouldin, Calinski-Harabasz) và độ mạch lạc từ khóa (NPMI).

## 7. Liên hệ với đồ án

- **Tài liệu hỗ trợ phương pháp/thí nghiệm nào?**
  - **M01 — Biểu diễn TF-IDF:** Cung cấp cơ sở toán học trực tiếp cho việc xây dựng vector TF-IDF kết hợp chuẩn hóa độ dài đơn vị $L_2$ (`norm="l2"`) để phép tính khoảng cách tới tâm cụm phản ánh đúng độ tương đồng cosine giữa các bài báo có độ dài ngắn dài khác nhau.
  - **M02 — Phân cụm K-Means và lựa chọn số cụm $K$:** Giải thích cơ chế đại diện cụm bằng vector tâm cụm trung bình, lý do K-Means hoạt động hiệu quả và có thời gian chạy tuyến tính $O(n)$ trên không gian TF-IDF chiều cao, đồng thời làm cơ sở cho việc trích xuất từ khóa đại diện (top terms) từ các chiều có trọng số lớn nhất của tâm cụm.
  - **M03 & M03A — Thí nghiệm cắt giảm tiền xử lý và nguồn văn bản:** Nhấn mạnh vai trò của việc loại bỏ từ dừng và lọc nhiễu từ vựng để giảm hiện tượng hai bài báo khác chủ đề chia sẻ nhiều từ chung ngẫu nhiên.
  - **Đánh giá nội tại và ngoại tại:** Cách bài báo tách biệt thước đo nội tại (độ gắn kết cụm) và thước đo ngoại tại (đối chiếu với chuyên mục báo *Los Angeles Times* ở tập `la1`, `la2`) hỗ trợ trực tiếp cho thiết kế đánh giá của TV1: dùng Silhouette, Davies-Bouldin, Calinski-Harabasz để chọn $K$ không giám sát, và chỉ dùng Purity, NMI, ARI đối chiếu với chuyên mục báo `category_gold` như chỉ báo tham khảo.
- **Nhóm kế thừa hoặc khảo sát điều gì?**
  - Kế thừa mô hình không gian vector TF-IDF chuẩn hóa $L_2$ và phân cụm K-Means, đồng thời khắc phục hạn chế biến thiên khởi tạo của bài báo gốc bằng cách sử dụng khởi tạo `k-means++` với nhiều lượt khởi tạo (`n_init=10`) và kiểm tra độ ổn định qua nhiều hạt giống ngẫu nhiên.
  - Kiểm tra phân bố kích thước cụm (cluster size distribution) trong các thực nghiệm `M02`, `M03`, `M03A` đúng như cảnh báo ở Section 9.2 của bài báo về hiện tượng K-Means tạo cụm mất cân bằng.
- **Khác biệt về dữ liệu/protocol cần lưu ý:**
  - Đồ án thực hiện trên Dataset v1.0 gồm 9.274 bài báo tiếng Việt sau khử trùng lặp, sử dụng tách từ tiếng Việt bằng PyVi thay cho Porter stemming, và đánh giá trên các số cụm $K \in \{5, 8, 10, 12, 15, 20\}$ thay vì $K \in \{16, 32, 64\}$.

## 8. Vị trí bằng chứng

| Nhận định/số liệu | Section/trang/bảng/vị trí nguồn (theo số trang bản in / số trang PDF) |
|---|---|
| So sánh độ phức tạp $O(n^2)$ của phân cấp tích tụ và $O(n)$ của K-Means | Abstract, trang 1 bản in (trang 3 PDF); Section 5, trang 9 bản in (trang 11 PDF); Section 10, trang 18 bản in |
| Quy trình 4 bước của thuật toán K-Means cơ bản và khái niệm tâm cụm | Section 2, trang 4 bản in (trang 6 PDF) |
| Mô hình không gian vector, trọng số TF-IDF, loại bỏ từ dừng, stemming và chuẩn hóa độ dài đơn vị | Section 3, trang 4–5 bản in (trang 6–7 PDF); Section 6, trang 9 bản in (trang 11 PDF) |
| Công thức độ tương đồng cosine, vector tâm cụm $c$ và hai tính chất toán học của $d_1 \bullet c$ và $\|c\|^2$ | Section 3, trang 5–6 bản in (trang 7–8 PDF) |
| Định nghĩa và công thức Entropy, F-measure và Overall Similarity | Sections 4.1–4.3, trang 6–8 bản in (trang 8–10 PDF) |
| Khởi tạo ngẫu nhiên $K$ văn bản, cập nhật tâm cụm tăng dần và thuật toán Bisecting K-Means | Section 5, trang 8–9 bản in (trang 10–11 PDF) |
| Mô tả 8 bộ dữ liệu (`re0`, `re1`, `wap`, `tr31`, `tr45`, `fbis`, `la1`, `la2`) và lọc bài đơn nhãn | Section 6, trang 9–10 bản in (trang 11–12 PDF); Table 1, trang 10 bản in (trang 12 PDF) |
| Nhãn lớp của `la1` và `la2` lấy từ tên chuyên mục báo *Los Angeles Times* | Section 6, trang 9 bản in (trang 11 PDF) |
| So sánh 3 kỹ thuật phân cấp IST, CST và UPGMA | Section 7.1–7.2, trang 10–11 bản in (trang 12–13 PDF); Table 2, trang 11 bản in; Figures 1–8, trang 12 bản in |
| Thiết lập số lượt chạy tương đương ($\text{ITER}=5$, 10 lần chạy) và cơ chế tinh chỉnh bằng K-Means | Section 8.1, trang 13 bản in (trang 15 PDF) |
| Kết quả Entropy và Overall Similarity ở $K = 16, 32, 64$ giữa Bisecting K-Means, K-Means tiêu chuẩn và UPGMA | Section 8.2, trang 13–15 bản in (trang 15–17 PDF); Tables 3–8, trang 14–15 bản in |
| So sánh thời gian chạy thực tế trên tập `la1` (hơn 1 giờ so với dưới 1 phút) và độ biến thiên của K-Means tiêu chuẩn | Section 8.2, trang 14 bản in (trang 16 PDF) |
| So sánh F-measure giữa Bisecting K-Means và UPGMA | Table 9, trang 16 bản in (trang 18 PDF) |
| Lý giải hiện tượng láng giềng gần nhất khác lớp (Figure 10) và ưu thế toàn cục của tâm cụm K-Means | Section 9.1, trang 16–17 bản in (trang 18–19 PDF); Figure 10, trang 16 bản in |
| Lý giải ảnh hưởng của sự chênh lệch kích thước cụm trong K-Means tiêu chuẩn | Section 9.2, trang 17–18 bản in (trang 19–20 PDF) |
| Các lưu ý về giới hạn của K-Means (khởi tạo, biến thiên, bản chất sơ bộ của giải thích) | Section 9, trang 16–17 bản in; Section 10, trang 18–19 bản in (trang 20–21 PDF) |

Nguồn chính thức: https://hdl.handle.net/11299/215421

## 9. Điều cần xác minh thêm

- Đối chiếu rõ sự khác biệt giữa cài đặt K-Means trong bài báo gốc (khởi tạo ngẫu nhiên $K$ tài liệu, cập nhật tâm cụm tăng dần, tối đa hóa độ tương đồng cosine) với cài đặt `KMeans` trong `scikit-learn` mà nhóm sử dụng ở `M02`, `M03`, `M03A` (khởi tạo `k-means++`, cập nhật theo batch, tối thiểu hóa tổng bình phương khoảng cách Euclid trên các vector đã chuẩn hóa $L_2$ — vốn tương đương đơn điệu với việc tối đa hóa độ tương đồng cosine).
- Lưu ý chi tiết lỗi in ấn nhỏ trong bài báo gốc để tránh nhầm lẫn khi trích dẫn: trong Table 1 (trang 10 bản in), kích thước lớp trung bình của `re1` in là `663` (thực tế $1657 / 25 = 66,28 \approx 66,3$), và trong văn bản trang 11, 13 tác giả gõ nhầm tham chiếu "table 8" khi nhắc tới bảng kết quả F-measure (thực tế là Table 2 ở trang 11 và Table 9 ở trang 16 bản in).

## 10. Trạng thái và bàn giao

- Trạng thái: `Ready for TV3 review`.
- Link file hoặc PR bàn giao: `paper/notes/tv1_steinbach_2000_document_clustering.md` (bổ sung link PR sau khi push nhánh).
- Nhận xét của TV3 sau review: Chưa có.

