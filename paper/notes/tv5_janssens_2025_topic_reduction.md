# Paper notes — Janssens et al. (2025): BERTopic topic reduction

## 1. Người cung cấp

- Thành viên: TV5 — Ngân
- Phương pháp phụ trách: BERTopic / topic representation / topic reduction
- Ngày đọc: 11/10/2026

## 2. Citation info — Thông tin trích dẫn

- Citation key: `janssens_2025_topic_reduction`
- Tên đầy đủ của tài liệu: *A Comparative Analysis of Topic Reduction Techniques for BERTopic*
- Tác giả: Wannes Janssens, Matthias Bogaert, Dirk Van den Poel
- Năm xuất bản: 2025
- Tạp chí/hội nghị hoặc nơi xuất bản: IEEE Access, Volume 13, pages 204087–204103
- DOI hoặc URL nguồn gốc: https://doi.org/10.1109/ACCESS.2025.3638956
- Loại tài liệu: `Scientific paper`.

## 3. Method — Phương pháp

- Bài toán và mục tiêu: So sánh các cách giảm số topic do BERTopic/HDBSCAN tạo ra và phân tích tác động của chúng đến coherence, diversity, outlier, runtime và chi phí tính toán.
- Phương pháp: Hai nhóm chính gồm giảm trực tiếp trong quá trình fit bằng cách tăng `min_cluster_size` của HDBSCAN, và giảm gián tiếp sau khi fit bằng Agglomerative Clustering hoặc LLM prompting để gộp các topic tương tự.
- Representation — cách biểu diễn dữ liệu: BERTopic mặc định dùng `all-MiniLM-L6-v2` tạo embedding 384 chiều, UMAP giảm còn 5 chiều, HDBSCAN gom cụm, CountVectorizer + c-TF-IDF biểu diễn topic. Với giảm gián tiếp, paper so sánh c-TF-IDF embedding, average reduced SBERT embedding, SBERT embedding của top-10 words và SBERT embedding của nhãn do LLM tạo.
- Ý tưởng/cơ chế chính: Giảm trực tiếp thay đổi quá trình tạo cluster nên không chỉ gộp topic mà còn ảnh hưởng outlier. Giảm gián tiếp giữ mô hình đã fit rồi lặp lại việc gộp hai topic gần nhất đến khi đạt số topic mong muốn. Paper còn thử GPT-4o-mini và Llama-3-8B-Instruct để nhận diện các topic chồng lấn.

## 4. Dataset — Dữ liệu

- Tên/nguồn: Trump Tweets, #Covid19 Tweets và Yelp Reviews; các bộ dữ liệu mở được dẫn nguồn từ Kaggle trong paper.
- Ngôn ngữ và miền nội dung: Tiếng Anh; bài đăng mạng xã hội/ngôn ngữ ngắn và đánh giá doanh nghiệp/ngôn ngữ dài hơn.
- Quy mô: Trump Tweets gồm 43.352 tweet từ 05/2009–06/2020; #Covid19 Tweets gồm 178.683 tweet từ 06/2020–08/2020; Yelp Reviews gồm 10.000 review của 4.171 doanh nghiệp từ 04/2005–05/2013.
- Cách chia dữ liệu/protocol — quy trình thí nghiệm — nếu có: Dữ liệu không được preprocessing trước để giữ ngữ cảnh cho SBERT và đảm bảo so sánh các phương pháp reduction trên cùng đầu vào. Paper chạy nhiều cấu hình HDBSCAN, thay đổi `min_cluster_size` và `cluster_selection_method`, rồi so sánh các phương pháp khi giảm về các mức số topic định trước. Train/validation/test split kiểu supervised không được báo cáo vì đây là thí nghiệm unsupervised topic modeling.

## 5. Metric — Đánh giá

- Chỉ số và cách đánh giá: NPMI Topic Coherence, Topic Diversity, số outlier, runtime và token usage đối với phương pháp dùng LLM.
- Kết quả/phát hiện chính: Tăng `min_cluster_size` thường giữ coherence tốt; giảm gián tiếp thường tăng diversity, đặc biệt trên dữ liệu ngắn, nhưng có thể đánh đổi coherence trên văn bản dài. Trong các embedding cho Agglomerative Clustering, c-TF-IDF có hiệu quả tốt và chi phí thấp. LLM prompting có thể đạt coherence/diversity tốt hơn nhưng chậm hơn và tốn tài nguyên/token.
- Giới hạn diễn giải kết quả LLM: Kết quả thuận lợi của LLM prompting chỉ được ghi nhận trên ba dataset, các cấu hình reduction và các metric NPMI/topic diversity/runtime/token usage của nghiên cứu này. Kết quả đó không chứng minh LLM luôn tốt hơn c-TF-IDF, Agglomerative Clustering hoặc các phương pháp topic modeling khác trên mọi corpus. Hiệu quả có thể thay đổi theo ngôn ngữ, độ dài văn bản, prompt, mô hình LLM, số topic mục tiêu, metric và tiêu chí đánh giá của người dùng.
- Điều kiện đi kèm số liệu: Kết luận được tổng hợp trên ba dataset và nhiều cấu hình HDBSCAN. Ví dụ runtime/token trong Section V.D giảm từ 50 xuống 25 topic trên Trump Tweets, chạy trên laptop có NVIDIA RTX 4050, Intel Core Ultra 7 và 32 GB RAM; các LLM reduction trong thí nghiệm này gọi OpenAI API. Không so trực tiếp số liệu này với kết quả đồ án nếu khác dữ liệu hoặc cấu hình.

## 6. Gap — Hạn chế hoặc khoảng trống

- Hạn chế/vấn đề chưa giải quyết: Nghiên cứu chỉ tập trung vào BERTopic và ba dataset chủ yếu gồm social-media text và customer reviews; chưa rõ khả năng khái quát sang văn bản khoa học, pháp lý hoặc kỹ thuật. Paper không thực hiện human evaluation, LLM-based qualitative evaluation hay statistical significance tests. Phương pháp LLM có chi phí, giới hạn token, độ không nhất quán và phụ thuộc prompt/nội dung nhạy cảm.
- Do tác giả nêu hay người đọc nhận xét? Các hạn chế trên do tác giả nêu trong Section VII.
- Bằng chứng và phạm vi của nhận định: Kết luận và flowchart lựa chọn phương pháp dựa trên các dataset/cấu hình của nghiên cứu; tác giả lưu ý kết quả có thể thay đổi theo ứng dụng.

## 7. Liên hệ với đồ án

- Tài liệu hỗ trợ phương pháp/thí nghiệm nào? Hỗ trợ M14 (BERTopic hyperparameter/ablation), M15 (topic representation), E03 (model comparison), E06 (error analysis), P07 (rename/merge topics) và phần thảo luận limitation.
- Nhóm kế thừa hoặc khảo sát điều gì? Khi BERTopic sinh quá nhiều topic, nhóm có thể thử tăng `min_cluster_size` và Agglomerative Clustering dựa trên c-TF-IDF; báo đồng thời số topic, outlier rate, NPMI/coherence nếu phù hợp, diversity, ví dụ định tính và runtime. c-TF-IDF reduction là lựa chọn thực tế để thử trước vì không cần external AI API.
- Khác biệt về dữ liệu/protocol cần lưu ý: Paper dùng tiếng Anh và phần lớn văn bản ngắn, còn đồ án là full-article tiếng Việt. Nhóm phải kiểm chứng lại trade-off coherence/diversity trên corpus của mình. Các thử nghiệm GPT-4o-mini chỉ là related work/khả năng mở rộng, không được biến thành phụ thuộc external AI API của pipeline chính.

## 8. Vị trí bằng chứng

| Nhận định/số liệu | Section/trang/bảng/vị trí nguồn |
|---|---|
| Khoảng trống về so sánh topic reduction | Introduction, trang 204088–204090 |
| Phân loại direct và indirect reduction | Section III, trang 204090–204091; Figure 1 |
| Tăng `min_cluster_size` của HDBSCAN | Section III-A, trang 204090–204091 |
| Agglomerative Clustering và các topic embeddings | Section III-B, trang 204091–204092 |
| LLM prompting để gộp topic | Section III-C, trang 204092–204093; Figure 4 |
| Dataset và quy mô | Section IV-A, trang 204093; Table 4 |
| BERTopic setup: MiniLM, UMAP, HDBSCAN, c-TF-IDF | Section IV-B, trang 204093–204095; Figures 5–6 |
| NPMI, diversity và nhiều cấu hình HDBSCAN | Section IV-C, trang 204095–204096 |
| Ảnh hưởng đến outlier, coherence và diversity | Section V; Figures 7–11 |
| Runtime và token usage | Section V-D, trang 204099–204100; Table 6 |
| Khuyến nghị chọn phương pháp | Section VI, trang 204099–204100; Figure 12 |
| Giới hạn dữ liệu, thiếu human evaluation/statistical tests và hạn chế LLM | Section VII, trang 204100–204101 |

## 9. Điều cần xác minh thêm

- Xác định mức số topic mục tiêu và tiêu chí gộp topic phù hợp cho tin tức tiếng Việt của nhóm.
- Kiểm tra phiên bản BERTopic/HDBSCAN và cách thư viện hiện tại triển khai `reduce_topics`.
- Nếu thử reduction, bổ sung human evaluation hoặc case study thay vì chỉ dựa vào NPMI/diversity.
- Không dùng GPT-4o-mini hoặc external AI API trong pipeline chính nếu chưa được nhóm và yêu cầu môn học cho phép.

## 10. Trạng thái và bàn giao

- Trạng thái: `Revised after TV3 review`.
- Commit bàn giao ban đầu: https://github.com/lunatran141/vietnamese-news-topic-discovery/commit/a0c8866
- PR bàn giao ban đầu: https://github.com/lunatran141/vietnamese-news-topic-discovery/pull/11
- Nhận xét của TV3 sau review: Đã giới hạn kết luận về LLM prompting theo dataset, cấu hình và metric; không diễn giải thành LLM luôn vượt trội hơn các phương pháp khác.

