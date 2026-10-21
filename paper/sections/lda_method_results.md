# TV3 — LDA: Methodology, Results và Discussion

## 1. Phương pháp LDA (Methodology)

Nghiên cứu sử dụng Dataset v1.0 gồm 9.274 bài báo tiếng Việt sau khử trùng lặp, với biến thể tiền xử lý P2 (`p2_segmented`) và trường đầu vào `text`. Dữ liệu được chia cố định thành 6.493 bài Train, 1.391 bài Validation và 1.390 bài Test.

Văn bản được biểu diễn bằng Bag-of-Words (BoW) với unigram, tối đa 15.000 đặc trưng, `min_df=3` và `max_df=0.85`. Trong giai đoạn lựa chọn mô hình, bộ vector hóa được học trên Train và áp dụng cho Validation nhằm tránh rò rỉ dữ liệu.

Latent Dirichlet Allocation (LDA) là mô hình chủ đề xác suất, biểu diễn mỗi tài liệu bằng một phân phối trên các chủ đề và mỗi chủ đề bằng một phân phối trên các từ (Blei et al., 2003). Nhờ đó, một bài báo có thể liên quan đến nhiều chủ đề với những mức xác suất khác nhau.

Mô hình sử dụng phương pháp huấn luyện `batch`, tối đa 20 vòng lặp. Nhóm khảo sát bảy giá trị K = {4, 6, 8, 10, 12, 15, 20} trên Train/Validation và đánh giá độ ổn định của K=15 qua năm random seeds. Cấu hình cuối được phê duyệt là **K=15, seed=42**. Sau khi chốt cấu hình, BoW và LDA được huấn luyện lại trên Train+Validation (7.884 bài), sau đó đánh giá trên Test (1.390 bài). Test không tham gia lựa chọn K hoặc điều chỉnh siêu tham số.

## 2. Lựa chọn số lượng chủ đề (M08)

Nhóm khảo sát bảy giá trị K = {4, 6, 8, 10, 12, 15, 20}. LDA được huấn luyện trên Train và đánh giá bằng coherence C_V, C_NPMI và topic diversity trên Validation.

| K | C_V | C_NPMI | Diversity |
|---:|---:|---:|---:|
| 4 | 0,3757 | -0,0088 | 0,6000 |
| 6 | 0,3997 | 0,0039 | 0,6167 |
| 8 | 0,4217 | 0,0216 | 0,6500 |
| 10 | 0,4557 | 0,0406 | 0,6500 |
| 12 | 0,4905 | 0,0521 | 0,6250 |
| **15** | **0,5133** | **0,0606** | **0,6400** |
| 20 | 0,5418 | 0,0748 | 0,6100 |

K=20 đạt coherence cao nhất trong các cấu hình thử nghiệm. Tuy nhiên, nhóm lựa chọn **K=15** vì đạt sự cân bằng giữa coherence, diversity và khả năng diễn giải sơ bộ, đồng thời hạn chế sự phân mảnh chủ đề so với K=20. K=20 được giữ làm cấu hình đối chiếu trên Validation, không sử dụng để đánh giá lại Final Test.

*Nguồn: `docs/m08_lda_experiments.md`; `results/lda/m08/topic_number_results.csv`.*

## 3. Đánh giá độ ổn định của LDA (M09)

Để đánh giá ảnh hưởng của khởi tạo ngẫu nhiên, nhóm huấn luyện LDA với K=15 trên tập Train bằng năm random seeds: 42, 52, 62, 72 và 82. Các mô hình sử dụng cùng cấu hình Bag-of-Words và được đánh giá trên Validation.

Do số thứ tự topic có thể thay đổi giữa các lần huấn luyện, nhóm sử dụng thuật toán Hungarian để ghép các topic dựa trên độ tương đồng Jaccard của 10 từ khóa đứng đầu. Sau khi căn chỉnh topic, độ tương đồng Cosine được tính giữa các vector phân phối chủ đề của cùng bài báo.

Kết quả trung bình trên 10 cặp seed:

| Chỉ số Stability | Giá trị |
|---|---:|
| Mean Keyword Jaccard | **0,455685** |
| Mean Document Cosine | **0,681273** |

Kết quả cho thấy từ khóa đại diện và phân phối chủ đề có sự thay đổi giữa các lần khởi tạo. Tuy nhiên, chưa có ngưỡng đánh giá stability được thống nhất để kết luận mô hình đạt hay không đạt. Thực nghiệm mới khảo sát K=15 nên chưa thể kết luận cấu hình này ổn định hơn K=20.

*Nguồn: `docs/m09_lda_stability.md`; `results/lda/m09/pairwise_stability.csv`.*

## 4. Kết quả đánh giá cuối trên tập Test

Sau khi phê duyệt cấu hình K=15, seed 42, mô hình LDA được huấn luyện lại trên Train+Validation gồm 7.884 bài và đánh giá trên 1.390 bài Test. Tập Test được giữ riêng trong giai đoạn lựa chọn cấu hình.

| Chỉ số | LDA Final |
|---|---:|
| Số chủ đề | 15 |
| NPMI Coherence | 0,147094 |
| Topic Diversity | 0,653333 |
| Adjusted Rand Index (ARI) | 0,406466 |
| Normalized Mutual Information (NMI) | 0,572713 |

NPMI sử dụng cách tính đồng xuất hiện từ ở cấp tài liệu (`document_binary_common`). ARI và NMI được tính bằng cách đối chiếu chủ đề có xác suất cao nhất của mỗi bài với `category_gold`. Do `category_gold` là chuyên mục báo, hai chỉ số này chỉ phản ánh mức độ tương ứng với chuyên mục, không phải độ chính xác phân loại hay nhãn chủ đề thực sự.

Kết quả cho thấy mô hình LDA tạo ra 15 chủ đề với mức độ đa dạng từ khóa là 0,653333. Những chỉ số trên cung cấp cơ sở đánh giá định lượng, nhưng chưa đủ để kết luận về khả năng diễn giải của các chủ đề nếu không xem xét thêm từ khóa và bài báo đại diện.

*Nguồn: `results/lda/final/test_metrics.json`; `experiments/lda/run_lda_final.py`.*

## 5. Phân tích khả năng diễn giải chủ đề

Để phân tích khả năng diễn giải của LDA Final, nhóm trích xuất 10 từ khóa đặc trưng cho mỗi topic và ba bài báo đại diện có xác suất chủ đề cao nhất trên tập Train+Validation. Tổng cộng có 45 bài đại diện cho 15 topic.

Một số nhóm nội dung có thể nhận diện tương đối rõ từ từ khóa và bài đại diện:

| Topic | Nhãn diễn giải sơ bộ | Nội dung đại diện |
|---|---|---|
| 0 | Giáo dục đại học | Thủ khoa, SAT, GPA |
| 3 | Thể thao ASIAD | Thành tích, lịch thi đấu |
| 4 | Pháp luật | Điều tra, xét xử |
| 6 | Bóng đá | ASEAN Cup, bóng đá quốc tế |
| 7 | Y tế | Tim mạch, bệnh hô hấp |
| 8 | Thị trường vàng | Giá vàng, lãi suất |
| 9 | Giáo dục phổ thông | Trường học, học sinh |
| 12 | AI và an toàn thông tin | ChatGPT, Gemini, AI agent |

Kết quả cho thấy LDA có thể phân biệt những chủ đề thuộc cùng lĩnh vực rộng, chẳng hạn giáo dục phổ thông và giáo dục đại học. Tuy nhiên, Topic 1 bao quát nhiều nội dung về tài chính, kinh doanh và tiêu dùng, cho thấy khả năng pha trộn chủ đề. Một số topic như 11 và 12 có từ khóa chưa đặc trưng; việc xem xét bài đại diện giúp làm rõ nội dung hơn so với chỉ đọc danh sách từ khóa.

Các nhãn trên mới là diễn giải sơ bộ, chưa được xác nhận bằng đánh giá thủ công độc lập của nhiều người.

*Nguồn: `results/lda/final/topic_keywords.csv`; `results/lda/final/representative_docs_development.csv`; `docs/lda_nmf_comparison.md`.*

## 6. Thảo luận và hạn chế (Discussion & Limitations)

Trong các cấu hình cuối đã lựa chọn, LDA với K=15 đạt NPMI 0,147094 và topic diversity 0,653333 trên Test, thấp hơn NMF với K=10 (NPMI 0,440680; diversity 1,000000). ARI và NMI của LDA cũng thấp hơn NMF trong cùng tập Test. Tuy nhiên, NMF sử dụng TF-IDF trên P3, còn LDA sử dụng Bag-of-Words trên P2; đồng thời số topic khác nhau. Vì vậy, kết quả phản ánh sự khác biệt giữa hai pipeline, không chứng minh NMF luôn tốt hơn LDA.

LDA có ưu điểm biểu diễn một bài báo bằng phân phối xác suất trên nhiều chủ đề, hỗ trợ phân tích các văn bản có nội dung giao thoa. Phân tích từ khóa và bài báo đại diện cho thấy nhiều chủ đề có thể diễn giải tương đối rõ, nhưng một số topic vẫn chứa từ phổ biến hoặc pha trộn nhiều nội dung.

Thực nghiệm M09 cho thấy LDA nhạy với random seed: Mean Keyword Jaccard đạt 0,455685 và Mean Document Cosine đạt 0,681273 trên 10 cặp seed. Kết quả này cho thấy tính nhất quán giữa các lần huấn luyện còn hạn chế, nhưng chưa đủ cơ sở để kết luận K=15 ổn định hơn hoặc kém hơn các giá trị K khác.

Các hạn chế chính gồm: (1) chỉ khảo sát stability tại K=15 với năm random seeds; (2) sử dụng P2 làm cấu hình chính, chưa thực hiện đầy đủ ablation tiền xử lý riêng cho LDA; (3) nhãn `category_gold` chỉ là chuyên mục tham chiếu, không phải ground-truth topic; và (4) nhãn diễn giải chủ đề chưa được xác nhận bằng đánh giá thủ công độc lập của nhiều người.

Trong nghiên cứu tiếp theo, nhóm có thể khảo sát stability tại các giá trị K khác, đánh giá bổ sung trên dữ liệu mới và tổ chức chấm khả năng diễn giải chủ đề bởi nhiều người đánh giá.