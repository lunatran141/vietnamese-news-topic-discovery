# Paper notes — Grootendorst (2022): BERTopic

## 1. Người cung cấp
- Thành viên: TV4 — Vân
- Phương pháp phụ trách: Embeddings, HDBSCAN, Topic Modeling
- Ngày đọc ban đầu: chưa xác minh (ngày 24/10/2026 trong bản cũ nằm sau ngày đối chiếu).
- Ngày đối chiếu với bài báo: 17/10/2026.

## 2. Citation info — Thông tin trích dẫn
- Citation key: grootendorst_2022_bertopic
- Tên bài: *BERTopic: Neural topic modeling with a class-based TF-IDF procedure*
- Tác giả: Maarten Grootendorst
- Năm xuất bản: 2022
- Nguồn: arXiv preprint, [`arXiv:2203.05794`](https://arxiv.org/abs/2203.05794); [PDF gốc](https://arxiv.org/pdf/2203.05794). Không gán venue hoặc DOI hội nghị cho bản này.

## 3. Method — Phương pháp
- **Bài toán:** Khám phá chủ đề tiềm ẩn từ tập văn bản. BERTopic tách việc **gom bài thành cụm** khỏi việc **diễn giải cụm bằng từ khóa**; không cần giả định từ khóa đại diện phải nằm gần tâm cụm hình cầu (Sections 1, 3).
- **Pipeline:** Sentence-BERT/SBERT tạo embedding tài liệu → UMAP giảm chiều → HDBSCAN tạo cụm và cho phép điểm nhiễu → **c-TF-IDF** chọn từ đại diện cho mỗi cụm/chủ đề (Sections 3.1–3.3). Mô hình embedding có thể được thay thế; thí nghiệm dùng cả MPNet, MiniLM, USE và Doc2Vec (Section 5.2; Table 2).
- **c-TF-IDF:** Gộp văn bản trong mỗi cụm thành một “tài liệu lớp”, rồi tính tầm quan trọng của từ theo lớp. Đây là biểu diễn **Bag-of-Words cho topic words**, mặc dù embedding dùng ngữ cảnh (Section 3.3).
- **Mở rộng theo thời gian:** Bài báo còn mô tả cách theo dõi biểu diễn chủ đề qua các mốc thời gian (Section 4); đây là *dynamic topic modeling*, không tự động đồng nghĩa với phát hiện sự kiện cụ thể.

## 4. Dataset — Dữ liệu
Ba tập chính để so sánh topic modeling (Section 5.1):

| Tập dữ liệu | Quy mô | Vai trò/đặc điểm |
|---|---:|---|
| 20 Newsgroups | 16.309 bài | 20 chuyên mục; tiền xử lý kỹ |
| BBC News | 2.225 bài | Tin BBC giai đoạn 2004–2005; tiền xử lý kỹ |
| Trump Tweets | 44.253 tweet | Tweet giai đoạn 2009–2021, bỏ retweet; tiền xử lý nhẹ hơn |

- Thí nghiệm *dynamic topic modeling* còn dùng các bản chép phát biểu tại Đại hội đồng Liên Hợp Quốc; Trump Tweets được chia 10 mốc thời gian, UN chia 9 mốc (Section 5.1). Bảng kết quả động cần được đọc riêng với Tables 1–2 của ba tập chính.
- Dải **1.000–43.000 tài liệu**, tăng 2.000 mỗi mức, là các **tập con Trump Tweets dùng khảo sát wall time**, không phải quy mô gốc của ba dataset và cũng không phải “kích thước từ vựng” (Section 6.4; Figure 1).

## 5. Metric — Đánh giá
- **Topic Coherence (TC):** Dùng **NPMI**; giá trị từ -1 đến 1, cao hơn biểu thị sự liên hệ giữa từ chủ đề tốt hơn theo metric này.
- **Topic Diversity (TD):** Tỷ lệ từ **khác nhau** trong các danh sách từ đại diện của tất cả chủ đề; từ 0 đến 1, cao hơn nghĩa là ít từ khóa lặp giữa các chủ đề hơn (Section 5.3).
- **Protocol Tables 1–2:** Xét **K = 10, 20, 30, 40, 50** chủ đề; **3 lần chạy ở mỗi K**; điểm báo cáo trung bình trên **15 lượt chạy** cho từng phương pháp/cấu hình và dataset. Đây là protocol của bài báo, chưa phải protocol đánh giá của đồ án (Section 5.3; Tables 1–2).
- **Runtime:** Đo **wall time bằng giây** trên tập con Trump Tweets từ 1.000 đến 43.000 bài; tác giả dùng máy 2 CPU core Intel Xeon 2.00 GHz và GPU Tesla P100. Kết quả phụ thuộc phần cứng, đặc biệt việc tạo embedding có GPU (Section 5.3, Section 6.4; Figure 1).

**Kết quả cần diễn giải đúng:** Table 1 cho thấy BERTopic-MPNet có TC lần lượt **0,166 / 0,167 / 0,066** trên 20 Newsgroups / BBC News / Trump Tweets. CTM có TD **0,886 / 0,819 / 0,855**, cao hơn BERTopic-MPNet **0,851 / 0,794 / 0,663** trên cả ba tập. Table 2 cho thấy các cấu hình embedding của BERTopic có kết quả khác nhau; MiniLM là lựa chọn đáng cân nhắc khi giới hạn tài nguyên. **Section 6.4 nói LDA và NMF nhanh hơn các mô hình neural trong thí nghiệm runtime**; không viết rằng BERTopic nhanh hơn mô hình cổ điển. TC/TD là chỉ báo tham khảo cho chất lượng chủ đề, không thay thế đánh giá thủ công (Section 5.3).

## 6. Gap — Hạn chế và nhận xét riêng

**Tác giả nêu trực tiếp (Section 7.2):**

- BERTopic gán một chủ đề chính cho mỗi tài liệu, trong khi một bài có thể chứa **nhiều chủ đề**. Có thể dùng xác suất soft clustering của HDBSCAN như chỉ báo phân bố chủ đề, nhưng cách này không làm mô hình học đầy đủ tính đa chủ đề trong huấn luyện.
- Embedding tài liệu có ngữ cảnh, nhưng biểu diễn **top words bằng c-TF-IDF dựa trên Bag-of-Words**. Các từ gần nghĩa có thể xuất hiện đồng thời, gây **dư thừa từ khóa** khi giải thích chủ đề. Tác giả nêu maximal marginal relevance là khả năng khắc phục về mặt lý thuyết nhưng chưa thử trong nghiên cứu.

**Nhận xét/việc cần kiểm chứng cho đồ án:** Tỷ lệ outlier của HDBSCAN và độ nhạy với embedding/siêu tham số có thể ảnh hưởng coverage hoặc chất lượng chủ đề tiếng Việt. Đây là câu hỏi thực nghiệm của đồ án, **không gán cho tác giả như kết luận đã chứng minh ở Section 7.2**. Section 6.4 còn cho thấy chi phí runtime phụ thuộc việc có GPU.

## 7. Liên hệ với đồ án

- **Trọng tâm Vietnamese News Topic Discovery:** Bài báo làm cơ sở cho pipeline biểu diễn ngữ nghĩa → UMAP → HDBSCAN → từ khóa chủ đề bằng c-TF-IDF. Đồ án cần kiểm tra chất lượng cụm và từ khóa trên **báo tiếng Việt** thay vì suy kết quả từ ba tập tiếng Anh.
- **M10–M12 và BERTopic của nhóm:** Đối chiếu embedding MiniLM/MPNet, tỷ lệ outlier, NPMI topic coherence, topic diversity và ví dụ chủ đề thực tế theo cùng protocol đánh giá. NPMI/TD giúp đọc chất lượng topic; outlier analysis cho biết bài nào chưa được gán cụm.
- **Phân biệt với event detection:** Một topic rộng như “thể thao” có thể chứa nhiều sự kiện cụ thể khác thời điểm. BERTopic/dynamic topic modeling ở bài này không tự động xác định các sự kiện tin tức riêng lẻ; phần event detection của đồ án cần thêm tiêu chí thời gian và đánh giá sự kiện.

## 8. Vị trí bằng chứng

| Nhận định/số liệu | Vị trí trong [PDF gốc](https://arxiv.org/pdf/2203.05794) |
|---|---|
| SBERT → UMAP → HDBSCAN → c-TF-IDF; biểu diễn cụm bằng từ khóa | Sections 3.1–3.3 |
| Ba dataset: 16.309 / 2.225 / 44.253 tài liệu; xử lý dữ liệu động | Section 5.1 |
| Baselines LDA, NMF, CTM, Top2Vec và bốn embedding của BERTopic | Section 5.2 |
| Định nghĩa NPMI/TD, K = 10–50 bước 10, ba lần chạy mỗi K; điều kiện máy đo runtime | Section 5.3; Tables 1–2 |
| Kết quả TC/TD giữa các mô hình; CTM có TD cao hơn BERTopic | Table 1; Section 6.1 |
| So sánh bốn embedding của BERTopic | Table 2; Section 6.2 |
| Dữ liệu động và các mốc thời gian | Section 5.1; Section 6.3; Table 3 |
| Khảo sát wall time 1.000–43.000 bài; LDA/NMF nhanh hơn | Section 6.4; Figure 1 |
| Hai hạn chế: tài liệu đa chủ đề; top words dư thừa do Bag-of-Words | Section 7.2 |

## 9. Điều cần xác minh thêm khi áp dụng vào đồ án

- Kiểm tra xem preprocessing tiếng Việt, lựa chọn embedding và c-TF-IDF tạo từ khóa chủ đề dễ hiểu hay từ gần nghĩa bị lặp; đọc ví dụ bài thực tế cùng với NPMI/TD.
- Báo tỷ lệ outlier và coverage theo cấu hình HDBSCAN, đồng thời xem các bài đa chủ đề thay vì coi mọi bài `-1` là dữ liệu lỗi.
- Nếu so runtime của MiniLM/MPNet, ghi rõ máy, GPU, thời gian tạo embedding và cùng quy mô dữ liệu; không chuyển nguyên kết quả Figure 1 sang môi trường của nhóm.
- Nếu làm event detection, cần kiểm tra riêng việc gom các bài nói về **cùng một sự kiện theo thời gian**, ngoài đánh giá topic coherence/diversity.

## 10. Trạng thái và bàn giao

- Trạng thái: đã đối chiếu với PDF gốc và chỉnh theo nhận xét; chờ review/nghiệm thu bản cập nhật.
- File bàn giao: `paper/notes/tv4_grootendorst_2022_bertopic.md`.
