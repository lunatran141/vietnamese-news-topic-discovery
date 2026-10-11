# Paper notes — Grootendorst (2022): BERTopic

## 1. Người cung cấp

- Thành viên: TV5 — Ngân
- Phương pháp phụ trách: BERTopic / topic representation
- Ngày đọc: 11/10/2026

## 2. Citation info — Thông tin trích dẫn

- Citation key: `grootendorst_2022_bertopic`
- Tên đầy đủ của tài liệu: *BERTopic: Neural topic modeling with a class-based TF-IDF procedure*
- Tác giả: Maarten Grootendorst
- Năm xuất bản: 2022
- Tạp chí/hội nghị hoặc nơi xuất bản: arXiv preprint, arXiv:2203.05794 [cs.CL]
- DOI hoặc URL nguồn gốc: https://doi.org/10.48550/arXiv.2203.05794
- Loại tài liệu: `Scientific paper` (preprint; TV3 xác nhận cách thống kê loại nguồn).

## 3. Method — Phương pháp

- Bài toán và mục tiêu: Khám phá các chủ đề tiềm ẩn trong tập văn bản bằng cách gom cụm các biểu diễn ngữ nghĩa của tài liệu, sau đó tạo biểu diễn từ khóa dễ diễn giải cho từng cụm.
- Phương pháp: BERTopic gồm ba bước chính: tạo document embedding, giảm chiều và gom cụm, rồi tạo topic representation bằng class-based TF-IDF (c-TF-IDF).
- Representation — cách biểu diễn dữ liệu: Sentence-BERT tạo dense document embeddings; UMAP giảm chiều; HDBSCAN gom các tài liệu tương tự và nhận diện outlier; c-TF-IDF tạo phân phối từ đại diện cho từng topic.
- Ý tưởng/cơ chế chính: Mỗi cụm được xem như một lớp. Các tài liệu trong một cụm được nối thành một “tài liệu lớp”, rồi TF-IDF được điều chỉnh để đo độ quan trọng của từ đối với lớp thay vì đối với từng tài liệu. Việc tách bước embedding/clustering khỏi bước biểu diễn topic cho phép thay đổi embedding hoặc tinh chỉnh biểu diễn mà không nhất thiết phải thiết kế lại toàn bộ mô hình.

## 4. Dataset — Dữ liệu

- Tên/nguồn: 20 Newsgroups, BBC News, Trump Tweets; thí nghiệm dynamic topic modeling sử dụng thêm UN General Debate corpus.
- Ngôn ngữ và miền nội dung: Tiếng Anh; tin tức, bài đăng mạng xã hội và diễn văn chính trị.
- Quy mô: 20 Newsgroups có 16.309 bài thuộc 20 nhóm; BBC News có 2.225 tài liệu từ năm 2004–2005; Trump Tweets có 44.253 tweet không gồm retweet, từ năm 2009–2021. Quy mô UN General Debate không được báo cáo trực tiếp trong phần mô tả thí nghiệm.
- Cách chia dữ liệu/protocol — quy trình thí nghiệm — nếu có: Paper không báo cáo train/validation/test split kiểu supervised. Với đánh giá chính, số topic được xét từ 10 đến 50 theo bước 10; mỗi mức chạy 3 lần. Trump Tweets được chia thành 10 mốc thời gian và UN General Debate thành 9 mốc cho dynamic topic modeling.

## 5. Metric — Đánh giá

- Chỉ số và cách đánh giá: Topic Coherence (TC) bằng NPMI, Topic Diversity (TD) và wall time. NPMI nằm trong [-1, 1], giá trị cao hơn biểu thị liên hệ từ tốt hơn; TD nằm trong [0, 1], giá trị cao hơn biểu thị tập từ giữa các topic đa dạng hơn.
- Kết quả/phát hiện chính: BERTopic đạt topic coherence cạnh tranh trên cả ba dataset và có TC cao nhất trên Trump Tweets trong Table 1. CTM có topic diversity cao hơn BERTopic trong so sánh chính. BERTopic tương đối ổn định giữa các embedding model; tác giả nhận xét `all-MiniLM-L6-v2` là lựa chọn cân bằng giữa tốc độ và chất lượng khi tài nguyên GPU hạn chế.
- Điều kiện đi kèm số liệu: Baseline gồm LDA, NMF, CTM và Top2Vec. Kết quả Table 1 là trung bình của 15 lần chạy cho mỗi mô hình (5 mức số topic × 3 lần). Thí nghiệm runtime sử dụng 2 CPU cores Intel Xeon 2.00 GHz và GPU Tesla P100 16 GB. Các số liệu này chỉ mô tả protocol của paper, không được xem là kết quả trên dữ liệu tiếng Việt của nhóm.

## 6. Gap — Hạn chế hoặc khoảng trống

- Hạn chế/vấn đề chưa giải quyết: BERTopic giả định mỗi tài liệu chủ yếu thuộc một topic, nên chưa mô hình hóa đầy đủ tài liệu đa chủ đề. Topic representation vẫn được tạo bằng bag-of-words/c-TF-IDF nên không trực tiếp giữ ngữ cảnh giữa các từ; các từ đại diện có thể gần nghĩa hoặc dư thừa. Paper không khảo sát Maximal Marginal Relevance như một cách giảm trùng lặp từ khóa. NPMI và topic diversity chỉ là proxy cho đánh giá chủ quan về chất lượng topic.
- Do tác giả nêu hay người đọc nhận xét? Các hạn chế trên do tác giả nêu trong Sections 5.3 và 7.2.
- Bằng chứng và phạm vi của nhận định: Nhận định chỉ áp dụng cho thiết kế và thí nghiệm được mô tả trong paper; không đủ cơ sở để khẳng định đây là hạn chế của mọi biến thể BERTopic hoặc mọi corpus.

## 7. Liên hệ với đồ án

- Tài liệu hỗ trợ phương pháp/thí nghiệm nào? Hỗ trợ trực tiếp M13 (BERTopic pipeline), M14 (embedding/hyperparameter experiments), M15 (topic representation), E03 (model comparison) và E06 (error analysis).
- Nhóm kế thừa hoặc khảo sát điều gì? Nhóm có thể kế thừa pipeline embedding → UMAP → HDBSCAN → c-TF-IDF; lưu riêng assignments, outliers, topic keywords và representative documents; khảo sát embedding/config khác nhau và đánh giá bằng quantitative lẫn qualitative evidence.
- Khác biệt về dữ liệu/protocol cần lưu ý: Paper dùng corpus tiếng Anh, còn đồ án dùng tin tức tiếng Việt. Nhóm phải chọn embedding phù hợp tiếng Việt, giữ cùng dữ liệu/split khi so sánh và không trình bày kết quả của paper như kết quả thực nghiệm của nhóm.

## 8. Vị trí bằng chứng

| Nhận định/số liệu | Section/trang/bảng/vị trí nguồn |
|---|---|
| Pipeline embedding → giảm chiều/gom cụm → c-TF-IDF | Section 3, trang 2–3 |
| SBERT cho document embeddings | Section 3.1, trang 2 |
| UMAP và HDBSCAN | Section 3.2, trang 2–3 |
| Công thức và cơ chế c-TF-IDF | Section 3.3, trang 3 |
| Dataset và quy mô | Section 5.1, trang 4 |
| Baseline và embedding models | Section 5.2, trang 4–5 |
| NPMI, topic diversity và protocol | Section 5.3, trang 5 |
| Kết quả so sánh chính | Table 1, trang 5 |
| Ảnh hưởng của embedding model | Table 2 và Section 6.2, trang 6 |
| Runtime và điều kiện phần cứng | Figure 1 và Section 6.4, trang 6–7 |
| Điểm mạnh và khả năng tinh chỉnh representation | Section 7.1, trang 7–8 |
| Hạn chế về tài liệu đa chủ đề và topic representation | Section 7.2, trang 8 |

## 9. Điều cần xác minh thêm

- TV3 xác nhận arXiv preprint có được tính vào chỉ tiêu “scientific paper” của môn hay chỉ dùng như nguồn phương pháp.
- Kiểm tra embedding model phù hợp cho tiếng Việt và tài nguyên máy của nhóm.
- Đối chiếu phiên bản thư viện BERTopic nhóm sử dụng vì API/tính năng hiện tại có thể khác paper năm 2022.

## 10. Trạng thái và bàn giao

- Trạng thái: `Ready for TV3 review`.
- Link file hoặc PR bàn giao: Chưa tạo — bổ sung sau khi push branch và mở PR.
- Nhận xét của TV3 sau review:

