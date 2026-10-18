# Paper notes — Vu et al. (2018): VnCoreNLP: A Vietnamese Natural Language Processing Toolkit

## 1. Người cung cấp

- Thành viên: TV1 — Khôi.
- Phương pháp phụ trách: Tiền xử lý văn bản tiếng Việt (tách từ tiếng Việt), biểu diễn TF-IDF và phân cụm K-Means.
- Ngày đọc/đối chiếu: 18/10/2026.

## 2. Citation info — Thông tin trích dẫn

- Citation key: `vu_2018_vncorenlp`.
- Tên đầy đủ của tài liệu: *VnCoreNLP: A Vietnamese Natural Language Processing Toolkit*.
- Tác giả: Thanh Vu, Dat Quoc Nguyen, Dai Quoc Nguyen, Mark Dras, Mark Johnson.
- Năm xuất bản: 2018.
- Tạp chí/hội nghị hoặc nơi xuất bản: *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Demonstrations (NAACL-HLT 2018)*, trang 56–60.
- DOI hoặc URL nguồn gốc: https://doi.org/10.18653/v1/N18-5012 · [ACL Anthology](https://aclanthology.org/N18-5012/) · [Bản PDF gốc](https://aclanthology.org/N18-5012.pdf).
- Loại tài liệu: `Scientific paper` — bài báo hội nghị khoa học có phản biện đồng cấp (peer-reviewed conference paper).

## 3. Method — Phương pháp

- **Bài toán và mục tiêu:** Xây dựng bộ công cụ xử lý ngôn ngữ tự nhiên tiếng Việt mã nguồn mở, dễ sử dụng, có tốc độ xử lý nhanh và độ chính xác cao nhằm phục vụ trực tiếp cho các bài toán xử lý dữ liệu văn bản quy mô lớn và các tác vụ hạ nguồn trong tiếng Việt.
- **Phương pháp:** Tích hợp bốn thành phần cốt lõi vào một đường ống xử lý thống nhất trên một đối tượng `Annotation` (Figure 1, trang 56): tách từ tiếng Việt (`wseg`), gán nhãn từ loại (`pos`), nhận diện thực thể có tên (`ner`) và phân tích cú pháp phụ thuộc (`parse`). Tác giả lựa chọn các mô hình dựa trên đặc trưng truyền thống thay vì các mô hình mạng nơ-ron sâu nặng nề nhằm tối ưu đồng thời độ chính xác và tốc độ xử lý trên phần cứng phổ thông.
- **Representation — cách biểu diễn dữ liệu:**
  - Giải quyết đặc thù cốt lõi của chữ viết tiếng Việt (Section 3, mục `wseg`, trang 57): Khác với tiếng Anh — nơi khoảng trắng là dấu hiệu trực tiếp xác định ranh giới từ — trong văn bản tiếng Việt, khoảng trắng còn được dùng để phân tách các âm tiết cấu thành nên một từ ghép. Do đó, tách từ tiếng Việt được xác định là bước đầu tiên mang tính quyết định trong mọi đường ống xử lý ngôn ngữ tự nhiên tiếng Việt.
  - Đầu ra của bước tách từ biểu diễn các từ ghép nhiều âm tiết và tên riêng bằng cách nối các âm tiết liên tiếp thông qua dấu gạch dưới `_` để tạo thành một đơn vị từ vựng duy nhất (ví dụ câu *"Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội."* được tách thành các đơn vị từ `Ông`, `Nguyễn_Khắc_Chúc`, `đang`, `làm_việc`, `tại`, `Đại_học`, `Quốc_gia`, `Hà_Nội` ở Figure 2 và Table 1, trang 57).
- **Ý tưởng/cơ chế chính:**
  - Thành phần tách từ (`wseg`) sử dụng mô hình học dựa trên luật chuyển đổi (transformation rule-based learning model) do nhóm tác giả phát triển riêng cho đặc thù tiếng Việt (chi tiết thuật toán gốc được dẫn chiếu tới bài báo đồng hành *Nguyen et al., 2018* tại LREC).
  - Các thành phần tiếp theo trong đường ống sử dụng kết quả đã tách từ: gán nhãn từ loại (`pos`) bằng mô hình MarMoT (CRF bậc cao), nhận diện thực thể (`ner`) bằng mô hình quy nạp đặc trưng động kết hợp gộp âm tiết tên người liên tiếp thành một từ đơn (Footnote 5, trang 58), và phân tích cú pháp phụ thuộc (`parse`) bằng mô hình chuyển trạng thái tham lam có phân nhánh lựa chọn.

## 4. Dataset — Dữ liệu

- **Tên/nguồn:** Bài báo sử dụng các bộ dữ liệu chuẩn quốc gia thuộc chương trình **VLSP** (Vietnamese Language and Speech Processing) và ngân hàng cây cú pháp tiếng Việt **VnDT**:
  - **Tách từ (`wseg`) và gán nhãn từ loại (`pos`):** Sử dụng bộ dữ liệu chuẩn của chiến dịch đánh giá VLSP 2013 đầu tiên (Section 1, trang 56; Section 4, trang 58).
  - **Nhận diện thực thể có tên (`ner`):** Sử dụng bộ dữ liệu VLSP 2016 NER shared task gồm 4 nhãn `PER`, `LOC`, `ORG`, `MISC` (Section 1, trang 56; Section 4.1, trang 58).
  - **Phân tích cú pháp phụ thuộc (`parse`):** Sử dụng ngân hàng cây cú pháp phụ thuộc tiếng Việt VnDT (Section 1, trang 56; Section 4.2, trang 59).
- **Ngôn ngữ và miền nội dung:** Văn bản tiếng Việt chuẩn (chủ yếu từ nguồn báo chí tiếng Việt thuộc các đợt đánh giá chuẩn VLSP).
- **Quy mô và cách chia dữ liệu/protocol — quy trình thí nghiệm:**
  - **Bộ dữ liệu VLSP 2016 cho NER (Section 4.1, trang 58):** Gồm 16.861 câu gán nhãn thủ công cho huấn luyện và phát triển, cùng 2.831 câu gán nhãn thủ công cho kiểm thử (Test). Từ 16.861 câu ban đầu, tác giả lấy mẫu 2.000 câu làm tập phát triển (Development) và dùng 14.861 câu còn lại làm tập huấn luyện (Training). Để phản ánh đúng kịch bản đường ống thực tế (khi không có nhãn từ loại chuẩn có sẵn và các bộ tách từ chuẩn đều gộp tên người đầy đủ thành một token), tác giả gộp các âm tiết liên tiếp của tên người (`PER`) thành một từ ghép nối bằng dấu gạch dưới và thay nhãn từ loại chuẩn bằng nhãn từ loại dự đoán tự động.
  - **Bộ dữ liệu VnDT cho phân tích cú pháp phụ thuộc (Section 4.2, trang 59):** Gồm 10.200 câu; tác giả dùng 1.020 câu cuối làm tập kiểm thử (Test) và 9.180 câu còn lại làm tập huấn luyện (Training), đánh giá trong cả hai điều kiện: dùng nhãn từ loại chuẩn (Gold POS) và dùng nhãn từ loại dự đoán tự động từ đường ống (Auto POS).
  - **Điều kiện phần cứng đo tốc độ (Footnote 4, trang 58):** Toàn bộ tốc độ xử lý trong bài báo được đo trên máy tính cá nhân sử dụng vi xử lý Intel Core i7 2.2 GHz.

## 5. Metric — Đánh giá

- **Chỉ số và cách đánh giá:**
  - **Tách từ (`wseg`):** Điểm $F_1$ (%) và tốc độ xử lý tính bằng số từ trên giây (words per second).
  - **Gán nhãn từ loại (`pos`):** Độ chính xác (accuracy, %) và tốc độ xử lý (từ/giây).
  - **Nhận diện thực thể (`ner`):** Điểm $F_1$ (%) trên tập kiểm thử ứng với kết quả tách từ chuẩn và tốc độ xử lý (từ/giây, đã bao gồm thời gian chạy gán nhãn từ loại).
  - **Phân tích cú pháp phụ thuộc (`parse`):** Điểm Labeled Attachment Score (LAS, %), Unlabeled Attachment Score (UAS, %) tính trên toàn bộ token (kể cả dấu câu) và tốc độ xử lý (từ/giây).
- **Kết quả/phát hiện chính:**
  - **Kết quả tách từ và gán nhãn từ loại (Section 4, trang 58):** Thành phần tách từ (`wseg`) của VnCoreNLP đạt điểm **$F_1$ cao nhất là 97,90%** với tốc độ xử lý **62.000 từ/giây**; thành phần gán nhãn từ loại (`pos`) đạt độ chính xác **95,88%** với tốc độ **25.000 từ/giây**, vượt qua các bộ công cụ tiếng Việt công bố trước đó (như JVnSegmenter, vnTokenizer, NNVLP chỉ đạt khoảng 300 từ/giây ở Section 1, trang 56).
  - **Kết quả nhận diện thực thể có tên (Section 4.1, Table 2, trang 58–59):** VnCoreNLP đạt điểm $F_1$ cao nhất là **88,55%** với tốc độ **18.000 từ/giây**, cao hơn và nhanh gấp 10 lần so với mô hình mạng nơ-ron đứng thứ hai là `BiLSTM-CRF + CNN-char` ($F_1$ đạt 88,28%, tốc độ 1.800 từ/giây). Tác giả cũng phân tích việc ghép các âm tiết tên riêng thành một đơn vị từ vựng (`Nguyễn_Khắc_Chúc`) và vai trò của biểu diễn cấp độ ký tự đối với các tên riêng chưa gặp.
  - **Kết quả phân tích cú pháp phụ thuộc (Section 4.2, Table 3, trang 59):** Với nhãn từ loại tự động, VnCoreNLP đạt **70,23% LAS** và **76,93% UAS** ở tốc độ **8.000 từ/giây** (cao hơn và nhanh hơn nhiều so với mô hình mạng nơ-ron jPTDP đạt 69,49% LAS, 77,68% UAS ở tốc độ 700 từ/giây). Với nhãn từ loại chuẩn, VnCoreNLP đạt **73,39% LAS** và **79,02% UAS**.
- **Điều kiện đi kèm số liệu:** Các chỉ số $F_1$, Accuracy, LAS, UAS được đo trên các tác vụ học có giám sát thuộc các bộ dữ liệu chuẩn VLSP và VnDT. Kết quả $F_1 = 97,90\%$ của bước tách từ phản ánh độ khớp với ranh giới từ gán nhãn thủ công trên tập VLSP, không phải điểm chất lượng phân cụm chủ đề không giám sát trên Dataset v1.0 của đồ án.

## 6. Gap — Hạn chế hoặc khoảng trống

- **Hạn chế do tác giả nêu trực tiếp trong bài báo:**
  - **Chưa tối ưu hóa đặc trưng ngôn ngữ học chuyên sâu cho các bước sau tách từ (Section 5, trang 59):** Tác giả nêu rõ phiên bản hiện tại của VnCoreNLP mới chỉ áp dụng các khuôn mẫu đặc trưng định sẵn từ tiếng Anh cho các mô hình gán nhãn từ loại, nhận diện thực thể và phân tích cú pháp mà chưa thiết kế các đặc trưng ngôn ngữ học chuyên biệt của tiếng Việt cho ba thành phần này.
  - **Sự phụ thuộc giữa quy ước tách từ và các tác vụ hạ nguồn (Section 4.1, trang 58):** Khi chuyển từ đánh giá từng thành phần độc lập sang đường ống thực tế, sự khác biệt trong quy ước tách từ (ví dụ cách tách hay gộp các âm tiết trong họ tên đầy đủ của thực thể `PER`) và sai số lan truyền từ bước trước sang bước sau có ảnh hưởng trực tiếp tới kết quả đầu ra.
- **Nhận xét của TV1 về khoảng trống đối với đồ án:**
  - **Chưa đánh giá tác động trực tiếp của tách từ lên bài toán phân cụm chủ đề không giám sát:** Bài báo đánh giá chất lượng tách từ bằng điểm $F_1$ có giám sát và đánh giá các tác vụ hạ nguồn có giám sát (POS, NER, Dependency Parsing), nhưng chưa khảo sát xem việc nối từ ghép bằng dấu gạch dưới (`_`) kết hợp loại bỏ từ dừng sẽ làm thay đổi ma trận TF-IDF, độ thưa từ vựng, các chỉ số hình học của K-Means (Silhouette, Davies-Bouldin, Calinski-Harabasz) và độ mạch lạc chủ đề (NPMI) như thế nào.
  - **Yêu cầu môi trường thực thi Java:** VnCoreNLP được đóng gói dưới dạng thư viện Java (`.jar`, yêu cầu JVM và cấp phát bộ nhớ như `-Xmx2g` ở Figure 2, trang 57), trong khi đường ống thực nghiệm của đồ án được xây dựng thuần Python cần tích hợp nhẹ, đồng bộ và tái lập nhanh cho 9.274 bài báo qua nhiều biến thể tiền xử lý (`P0`–`P3`).

## 7. Liên hệ với đồ án

- **Tài liệu hỗ trợ phương pháp/thí nghiệm nào?**
  - **D01 — Đường ống tiền xử lý dữ liệu tiếng Việt:** Cung cấp cơ sở học thuật giải thích vì sao văn bản báo chí tiếng Việt không thể chỉ tách theo khoảng trắng như tiếng Anh (vì sẽ làm vỡ các từ ghép như `kinh_tế`, `chính_phủ`, `bất_động_sản` thành các âm tiết đơn lẻ mang nghĩa mơ hồ), mà phải đi qua bước tách từ tiếng Việt nối âm tiết bằng dấu gạch dưới `_`.
  - **M01 — Xây dựng không gian từ vựng TF-IDF:** Giải thích cách biểu diễn mỗi từ ghép đã tách (`word_form` có dấu `_`) thành một token nguyên vẹn khi đưa vào `TfidfVectorizer` để tránh sinh ra các n-gram âm tiết vô nghĩa.
  - **M03 — Thí nghiệm cắt giảm tiền xử lý (P0, P1, P2, P3):** Đặt nền móng cho giả thuyết khoa học của `M03` khi so sánh trực tiếp giữa văn bản chưa tách từ (chỉ tách theo khoảng trắng ở P0, P1) với văn bản đã tách từ tiếng Việt (ở P2, P3) kết hợp loại bỏ từ dừng tiếng Việt.
  - **M03A — So sánh tiêu đề với tiêu đề cộng nội dung:** Hỗ trợ phân tích chất lượng từ khóa tâm cụm khi áp dụng tách từ và lọc từ dừng trên văn bản ngắn (chỉ tiêu đề) so với văn bản đầy đủ (tiêu đề cộng toàn văn bài báo).
- **Nhóm kế thừa hoặc khảo sát điều gì?**
  - Kế thừa quy ước biểu diễn từ ghép tiếng Việt bằng dấu gạch dưới `_` (chuẩn VLSP được mô tả trong Section 2 và Section 3 của bài báo). Trong triển khai thực tế của đồ án, nhóm sử dụng thư viện thuần Python `pyvi` để thực hiện quy ước tách từ này trong đường ống `D01` và kiểm chứng định lượng tác động của nó lên phân cụm K-Means ở `M03`.
  - Khảo sát thực nghiệm sự đánh đổi giữa các chỉ số hình học không gian vector (vốn nhạy cảm với kích thước và độ thưa từ vựng sau khi tách từ) và chất lượng ngữ nghĩa của bộ từ khóa chủ đề (`NPMI`, `Purity`, `NMI`, `ARI`).
- **Khác biệt về dữ liệu/protocol cần lưu ý:**
  - Bài báo gốc đánh giá trên tập câu gán nhãn VLSP 2013/2016 và VnDT cho các bài toán gán nhãn chuỗi và phân tích cú pháp, trong khi đồ án áp dụng tiền xử lý tách từ trên 9.274 bài báo tin tức tiếng Việt (Dataset v1.0) cho bài toán khám phá chủ đề không giám sát.

## 8. Vị trí bằng chứng

| Nhận định/số liệu | Section/trang/bảng/vị trí nguồn |
|---|---|
| Bối cảnh các bộ dữ liệu chuẩn VLSP 2013 (tách từ, POS), VnDT 2014 (cú pháp) và VLSP 2016 (NER) | Section 1, trang 56 |
| So sánh với các đường ống cũ (JVnSegmenter, vnTokenizer) và tốc độ của NNVLP (khoảng 300 từ/giây) | Section 1, trang 56 |
| Kiến trúc đường ống 4 bước (`wseg`, `pos`, `ner`, `parse`) trên đối tượng `Annotation` | Abstract & Section 1, Figure 1, trang 56; Section 2, Listing 1–2, trang 57 |
| Ví dụ câu tiếng Việt được tách từ ghép nối bằng dấu gạch dưới `_` (`Nguyễn_Khắc_Chúc`, `làm_việc`, `Đại_học`, `Quốc_gia`, `Hà_Nội`) | Section 2, Figure 2 và Table 1, trang 57 |
| Giải thích sự khác biệt của khoảng trắng trong tiếng Việt (phân tách âm tiết thay vì từ) và vai trò bước đầu tiên của tách từ | Section 3, mục `wseg`, trang 57 |
| Kết quả tách từ đạt $F_1 = 97,90\%$ ở tốc độ 62K từ/giây và POS tagging đạt 95,88% ở tốc độ 25K từ/giây | Section 4, trang 58 |
| Cấu hình máy đo tốc độ thực nghiệm (Intel Core i7 2.2 GHz) | Footnote 4, trang 58 |
| Quy mô tập dữ liệu VLSP 2016 NER (16.861 câu chia thành 14.861 train / 2.000 dev, và 2.831 câu test) cùng quy ước gộp âm tiết tên riêng | Section 4.1 và Footnote 5, trang 58 |
| Kết quả so sánh NER ($F_1 = 88,55\%$, tốc độ 18K từ/giây, nhanh gấp 10 lần BiLSTM-CRF + CNN-char) | Section 4.1, trang 58–59; Table 2, trang 59 |
| Quy mô tập VnDT (10.200 câu: 1.020 câu test, 9.180 câu train) và kết quả LAS/UAS với Gold POS và Auto POS | Section 4.2 và Table 3, trang 59 |
| Hạn chế về việc chưa tích hợp đặc trưng ngôn ngữ học tối ưu riêng cho tiếng Việt ở các mô hình POS, NER, parsing | Section 5, trang 59 |

Nguồn chính thức: https://aclanthology.org/N18-5012.pdf

## 9. Điều cần xác minh thêm

- Khi viết phần Methodology và Related Work của báo cáo (`R01`, `R06`), cần nêu rõ: bài báo *Vu et al. (2018)* được trích dẫn để làm cơ sở khoa học cho bài toán tách từ tiếng Việt và quy ước ghép âm tiết từ ghép bằng dấu `_`, còn công cụ thực thi được nhóm cài đặt trong code (`src/data/` và `configs/preprocessing.yaml`) là thư viện Python `pyvi` (`ViteTokenizer.tokenize`).
- Không lấy con số $F_1 = 97,90\%$ hoặc tốc độ 62K từ/giây của VnCoreNLP gán cho thư viện `pyvi` trong đồ án; thời gian chạy và tác động của bước tách từ bằng `pyvi` phải được báo cáo từ kết quả thực nghiệm `M03` của nhóm.

## 10. Trạng thái và bàn giao

- Trạng thái: `Ready for TV3 review`.
- Link file hoặc PR bàn giao: `paper/notes/tv1_vu_2018_vncorenlp.md` (bổ sung link PR sau khi push nhánh).
- Nhận xét của TV3 sau review: Chưa có.

