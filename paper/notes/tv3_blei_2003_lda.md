# Paper notes — Blei et al. (2003): Latent Dirichlet Allocation

## 1. Người cung cấp

- Thành viên: TV3 — Tuấn.
- Phương pháp phụ trách: Latent Dirichlet Allocation (LDA), topic modeling và đánh giá LDA.
- Ngày đọc/review: 17/10/2026.

## 2. Citation info — Thông tin trích dẫn

- Citation key: `blei_2003_lda`.
- Tên đầy đủ của tài liệu: *Latent Dirichlet Allocation*.
- Tác giả: David M. Blei, Andrew Y. Ng, Michael I. Jordan.
- Năm xuất bản: 2003.
- Tạp chí: *Journal of Machine Learning Research (JMLR)*, Volume 3, pages 993–1022.
- URL nguồn gốc: https://www.jmlr.org/papers/v3/blei03a.html
- Bản PDF: https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
- Loại tài liệu: Scientific paper — journal article.

## 3. Method — Phương pháp

- **Bài toán và mục tiêu:** Xây dựng mô hình xác suất cho các tập hợp dữ liệu rời rạc, đặc biệt là corpus văn bản, nhằm khám phá các cấu trúc chủ đề tiềm ẩn và biểu diễn tài liệu trong không gian chủ đề.
- **Phương pháp:** Latent Dirichlet Allocation (LDA), một mô hình xác suất sinh (generative probabilistic model) thuộc nhóm hierarchical Bayesian models.
- **Representation:** Văn bản được biểu diễn theo giả định Bag-of-Words, bỏ qua thứ tự từ. Mỗi tài liệu có một phân phối xác suất trên K chủ đề, mỗi chủ đề tương ứng với một phân phối xác suất trên từ vựng.
- **Cơ chế chính:** Với mỗi tài liệu, mô hình sinh phân phối chủ đề θ từ phân phối Dirichlet có tham số α. Với mỗi vị trí từ, chọn topic tiềm ẩn z theo θ, sau đó sinh từ w theo phân phối từ có điều kiện trên topic, được tham số hóa bởi β.
- **Suy luận và học tham số:** Bài báo phát triển phương pháp variational inference để xấp xỉ posterior và sử dụng variational EM cho ước lượng tham số theo empirical Bayes.
- **So sánh lý thuyết:** Khác với mixture of unigrams gán một topic cho toàn bộ tài liệu, LDA cho phép các từ trong cùng tài liệu thuộc nhiều topic. So với pLSI, LDA xây dựng mô hình sinh cho cả tài liệu mới và không cần một bộ tham số topic mixture riêng cho từng tài liệu huấn luyện.
- **Giả định quan trọng:** Số chủ đề K được xác định trước; mô hình cơ bản sử dụng giả định exchangeability đối với các từ trong tài liệu.

LDA là mô hình topic modeling xác suất, không đồng nhất với thuật toán phân cụm K-Means.

## 4. Dataset — Dữ liệu

- **Tên/nguồn:** Bài báo sử dụng corpus scientific abstracts của cộng đồng nghiên cứu *C. elegans* và một tập con của corpus tin tức Associated Press (AP) thuộc TREC để đánh giá document modeling.
- **Ngôn ngữ và miền nội dung:** Văn bản tiếng Anh, gồm abstract nghiên cứu khoa học và tin tức.
- **Quy mô C. elegans:** 5.225 abstracts, gồm 28.414 unique terms.
- **Quy mô TREC AP:** 16.333 bài báo tin tức, gồm 23.075 unique terms.
- **Preprocessing:** Bài báo loại bỏ 50 stopwords; với corpus AP, các từ chỉ xuất hiện một lần cũng được loại bỏ.
- **Experimental protocol:** Trong thí nghiệm document modeling, 90% dữ liệu được dùng để huấn luyện và 10% được giữ lại làm Test.
- **Các tác vụ khác:** Bài báo còn trình bày thực nghiệm về document classification và collaborative filtering, với dữ liệu và thiết lập riêng cho từng tác vụ.
- **Vị trí bằng chứng:** Section 7.1 — Document Modeling, trang 1008–1012; Sections 7.2–7.3.

Dataset của nghiên cứu gốc không phải Dataset v1.0 gồm 9.274 bài báo tiếng Việt của đồ án.

## 5. Metric — Đánh giá

- **Metric chính:** Held-out perplexity, được tính từ xác suất của các tài liệu trong tập Test. Perplexity thấp hơn thể hiện khả năng khái quát hóa tốt hơn trong phép so sánh này.
- **Baseline:** Unigram model, mixture of unigrams và probabilistic Latent Semantic Indexing (pLSI).
- **Kết quả chính:** Figure 9 trình bày perplexity trên hai corpus C. elegans và AP với các số topic khác nhau. Trong những điều kiện khảo sát, LDA đạt perplexity thấp hơn các baseline được so sánh.
- **Hiện tượng overfitting:** Bài báo phân tích vấn đề overfitting của mixture of unigrams và pLSI khi thay đổi số lượng topic. Table 1 minh họa hiện tượng này trên corpus AP, không phải bảng kết quả perplexity của LDA.
- **Đánh giá bổ sung:** Nghiên cứu cũng đánh giá ứng dụng của biểu diễn LDA trong document classification và collaborative filtering.
- **Điều kiện thực nghiệm:** Kết quả phụ thuộc dataset, preprocessing, số topic, cách khởi tạo và kỹ thuật suy luận được sử dụng.
- **Vị trí bằng chứng:** Section 7.1, Figure 9 và Table 1; Sections 7.2–7.3.

Không suy diễn rằng LDA luôn vượt NMF, K-Means hoặc BERTopic về coherence, diversity hay các chỉ số khác. Bài báo này không kiểm chứng những kết luận đó trên tiếng Việt.

## 6. Gap — Hạn chế hoặc khoảng trống

- **Giới hạn Bag-of-Words:** Mô hình cơ bản bỏ qua thứ tự từ và các quan hệ phụ thuộc cú pháp, nên không trực tiếp mô hình hóa ngữ cảnh tuần tự.
- **Số topic cố định:** K được giả định là biết trước trong mô hình cơ bản, do đó việc lựa chọn số lượng chủ đề phù hợp vẫn cần được nghiên cứu thực nghiệm.
- **Suy luận gần đúng:** Việc tính posterior chính xác là khó xử lý trực tiếp; bài báo sử dụng variational inference thay vì suy luận chính xác.
- **Khả năng diễn giải:** Phân phối từ của topic không tự bảo đảm mọi topic đều rõ nghĩa với người đọc; cần xem xét các từ khóa và tài liệu đại diện khi ứng dụng.
- **Giới hạn ngôn ngữ và dữ liệu:** Thí nghiệm gốc không đánh giá corpus tin tức tiếng Việt đã phân đoạn từ, nên kết quả không tự động khái quát sang Dataset v1.0.
- **Giới hạn về stability:** Bài báo không thiết kế thí nghiệm chuyên biệt để đo sự nhất quán của các topic giữa nhiều random seeds.

**Phân biệt nguồn nhận định:** Bag-of-Words, số topic cố định, vấn đề suy luận và các hạn chế của pLSI được trình bày trong nghiên cứu gốc. Các ý về khả năng diễn giải, ứng dụng tiếng Việt và multi-seed stability là nhận xét của TV3 về phạm vi ứng dụng đối với đồ án.

## 7. Liên hệ với đồ án

- **M07 — LDA Pipeline:** Nghiên cứu cung cấp nền tảng lý thuyết cho việc xây dựng CountVectorizer/Bag-of-Words và huấn luyện LDA trên Dataset v1.0.
- **M08 — LDA Experiments:** Giải thích vai trò của K và cơ sở để khảo sát các số lượng topic khác nhau, xuất topic keywords, document-topic distributions và representative documents.
- **M08 — Topic Evaluation:** Bài báo hỗ trợ phần lý thuyết về LDA; các metric coherence và diversity cần được giải thích bằng những nghiên cứu chuyên biệt về đánh giá topic.
- **M09 — LDA Stability:** Cung cấp nền tảng mô hình xác suất, nhưng không được dùng làm bằng chứng trực tiếp cho độ ổn định giữa các random seeds.
- **3.7 — LDA/NMF Comparison:** Hỗ trợ mô tả sự khác biệt về nguyên lý giữa probabilistic topic modeling và matrix factorization.
- **R03 — Paper Writing:** Dùng làm tài liệu nền tảng trong phần Methodology và Related Work.

Đồ án sử dụng Dataset v1.0 với 9.274 bài báo tiếng Việt sau deduplication, preprocessing và giao thức Train/Validation/Test riêng đã được khóa. Kết quả thực nghiệm trên corpus của nhóm phải được báo cáo độc lập.

Implementation LDA trong scikit-learn không được mặc nhiên xem là trùng hoàn toàn với quy trình tối ưu hóa cụ thể của nghiên cứu gốc; cần mô tả theo code và cấu hình nhóm thực sự sử dụng.

## 8. Vị trí bằng chứng

| Nhận định/số liệu | Section/trang/bảng/vị trí nguồn |
|---|---|
| Định nghĩa LDA và các đóng góp chính | Abstract, trang 993 |
| Bag-of-Words và giả định exchangeability | Section 1, trang 994–995; Section 3.1 |
| Quy trình sinh LDA và phân phối Dirichlet | Section 3, trang 996–997; Figure 1 |
| Số chủ đề K được giả định cố định | Section 3, trang 996 |
| So sánh LDA với mixture of unigrams và pLSI | Section 4.2–4.3, trang 1000–1001 |
| Variational inference và parameter estimation | Section 5 |
| Ví dụ biểu diễn các topic trong văn bản | Section 6; Section 7.1, Figure 8 |
| Corpus C. elegans và TREC AP, quy mô dữ liệu | Section 7.1, trang 1008 |
| Train 90% và Test 10% trong document modeling | Section 7.1, trang 1008 |
| Held-out perplexity | Section 7.1, trang 1008 |
| So sánh perplexity trên hai corpus | Section 7.1, Figure 9, trang 1010 |
| Hiện tượng overfitting của baseline | Section 7.1, Table 1 |
| Document classification | Section 7.2 |
| Collaborative filtering | Section 7.3 |

Nguồn chính: https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf

## 9. Điều cần xác minh thêm

- Đối chiếu phiên bản scikit-learn và cách triển khai variational inference thực tế trong pipeline LDA của nhóm khi viết phần Methodology.
- Kiểm tra cấu hình `configs/lda.yaml`, `configs/lda_experiments.yaml` và `configs/lda_stability.yaml` trước khi mô tả tham số huấn luyện.
- Bổ sung nguồn học thuật chuyên biệt nếu cần thảo luận sâu về stability, topic matching và multi-seed evaluation.
- Không sử dụng perplexity của nghiên cứu gốc thay cho coherence, diversity hoặc stability của Dataset v1.0.

Các thông tin thư mục, mô hình, corpus, giao thức 90/10 và kết quả perplexity được đối chiếu với nghiên cứu gốc.

## 10. Trạng thái và bàn giao

- Trạng thái: `Ready for TV3 review`.
- Link file hoặc PR bàn giao: Chưa tạo; bổ sung sau khi push branch.
- Nhận xét của TV3 sau review: Đã đối chiếu nội dung phương pháp, dataset, protocol, metric và các vị trí bằng chứng chính với paper gốc. Chưa nghiệm thu vào Related Work Matrix.