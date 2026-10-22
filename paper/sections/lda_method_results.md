# LDA: Methodology, Results and Discussion

## 1. Phương pháp nghiên cứu (Methodology)

Nghiên cứu sử dụng Dataset v1.0 gồm 9.274 bài báo tiếng Việt sau khử trùng lặp, với biến thể tiền xử lý P2 (`p2_segmented`) và trường đầu vào `text`. Dữ liệu được chia cố định thành Train (6.493 bài), Validation (1.391 bài) và Test (1.390 bài). Văn bản được biểu diễn bằng Bag-of-Words (BoW) với unigram, tối đa 15.000 đặc trưng, `min_df=3` và `max_df=0.85`.

Latent Dirichlet Allocation (LDA) là mô hình xác suất sinh, trong đó mỗi tài liệu được biểu diễn bằng phân phối xác suất trên các chủ đề và mỗi chủ đề bằng phân phối xác suất trên từ vựng (Blei et al., 2003). Mô hình được huấn luyện theo phương pháp `batch`, tối đa 20 vòng lặp. Giai đoạn lựa chọn cấu hình sử dụng Train/Validation và không sử dụng Test. Sau khi lựa chọn **K=15, seed=42**, BoW và LDA được huấn luyện lại trên Train+Validation (7.884 bài), sau đó đánh giá trên Test độc lập.

*Nguồn: Blei et al. (2003); `configs/lda.yaml`; `configs/lda_final.yaml`; `experiments/lda/run_lda_final.py`.*

## 2. Lựa chọn số lượng chủ đề (M08)

Số lượng chủ đề được khảo sát với K ∈ {4, 6, 8, 10, 12, 15, 20}, sử dụng coherence C_V, C_NPMI và topic diversity để đánh giá trên Validation. Các thước đo coherence cung cấp cơ sở định lượng về mức độ liên hệ giữa các từ khóa mô tả chủ đề (Röder et al., 2015).

K=20 đạt coherence cao nhất trong dải thử nghiệm (C_V=0,5418; C_NPMI=0,0748), nhưng diversity đạt 0,6100. Trong khi đó, K=15 đạt C_V=0,5133; C_NPMI=0,0606 và diversity=0,6400. Vì vậy, K=15 được lựa chọn dựa trên sự đánh đổi giữa độ mạch lạc, độ đa dạng và khả năng diễn giải sơ bộ, thay vì tối ưu một chỉ số đơn lẻ. Coherence trong M08 phục vụ lựa chọn K trên Validation và không được so sánh trực tiếp với NPMI document-level ở Final Test.

*Nguồn: Röder et al. (2015); `docs/m08_lda_experiments.md`; `results/lda/m08/topic_number_results.csv`.*

## 3. Đánh giá độ ổn định (M09)

Độ ổn định của LDA được đánh giá tại K=15 với năm random seeds {42, 52, 62, 72, 82}. Do topic ID có thể thay đổi giữa các lần huấn luyện, các chủ đề được căn chỉnh bằng thuật toán Hungarian dựa trên Jaccard similarity của 10 từ khóa đặc trưng. Sau đó, Cosine similarity được tính giữa các vector phân phối chủ đề của cùng bài báo Validation.

Trung bình trên 10 cặp seed, **Mean Keyword Jaccard đạt 0,455685** và **Mean Document Cosine đạt 0,681273**. Kết quả cho thấy từ khóa đại diện và phân phối chủ đề thay đổi giữa các lần khởi tạo. Tuy nhiên, thực nghiệm mới khảo sát K=15 nên chưa đủ cơ sở kết luận độ ổn định tương đối so với những giá trị K khác.

*Nguồn: Kuhn (1955), cơ sở thuật toán Hungarian; `docs/m09_lda_stability.md`; `results/lda/m09/pairwise_stability.csv`.*

## 4. Kết quả đánh giá cuối (Final Test)

Mô hình LDA Final (K=15, seed=42) được đánh giá trên 1.390 bài báo Test độc lập. Kết quả được trình bày trong Bảng 1.

**Bảng 1. Kết quả đánh giá LDA Final trên Test**

| Chỉ số | Giá trị |
|---|---:|
| NPMI Coherence | 0,147094 |
| Topic Diversity | 0,653333 |
| Adjusted Rand Index (ARI) | 0,406466 |
| Normalized Mutual Information (NMI) | 0,572713 |

NPMI được tính dựa trên đồng xuất hiện từ ở cấp tài liệu (`document_binary_common`). ARI và NMI được xác định bằng cách đối chiếu topic có xác suất cao nhất với chuyên mục tham chiếu `category_gold`. Do đó, hai chỉ số này phản ánh mức độ tương ứng với chuyên mục báo, không phải độ chính xác của nhãn chủ đề. Các thước đo định lượng cung cấp thông tin về chất lượng mô hình nhưng cần được xem xét cùng phân tích khả năng diễn giải (Röder et al., 2015).

*Nguồn: `results/lda/final/test_metrics.json`; `experiments/lda/run_lda_final.py`; Hubert và Arabie (1985), Vinh et al. (2010) về các thước đo đối chiếu phân hoạch.*

## 5. Phân tích khả năng diễn giải chủ đề (Topic Interpretability)

Khả năng diễn giải được xem xét thông qua 10 từ khóa có trọng số cao nhất và 10 bài báo đại diện có xác suất chủ đề chính cao nhất cho mỗi topic, trích xuất từ Train và Validation. Tổng cộng 150 bài báo đại diện được sử dụng để hỗ trợ đề xuất nhãn cho 15 chủ đề. Các nhãn này thuộc mô hình LDA Final sau khi huấn luyện lại trên Train+Validation, không kế thừa trực tiếp topic ID từ thực nghiệm M08.

**Bảng 2. Nhãn diễn giải đề xuất của 15 topic LDA Final**

| Topic | Nhãn diễn giải đề xuất |
|---|---|
| 0 | Giáo dục đại học và tuyển sinh |
| 1 | Tài chính, ngân hàng và thị trường tiêu dùng |
| 2 | Dinh dưỡng và chăm sóc sức khỏe |
| 3 | Thể thao Việt Nam tại ASIAD |
| 4 | Tội phạm và điều tra hình sự |
| 5 | Giải chạy Marathon và sự kiện chạy bộ |
| 6 | Bóng đá trong nước và quốc tế |
| 7 | Y tế, bệnh lý và điều trị |
| 8 | Thị trường vàng và biến động giá |
| 9 | Giáo dục phổ thông và quản lý trường học |
| 10 | Thời trang và phong cách người nổi tiếng |
| 11 | Âm nhạc và hoạt động biểu diễn |
| 12 | Trí tuệ nhân tạo và rủi ro an toàn AI |
| 13 | Phát triển doanh nghiệp và công nghệ |
| 14 | Quản lý nhà nước và chính sách pháp lý |

Phân tích định tính cho thấy nhiều chủ đề có nội dung tương đối rõ, như thể thao ASIAD (Topic 3), chạy bộ (Topic 5), bóng đá (Topic 6) và thị trường vàng (Topic 8). Mô hình cũng hình thành các chủ đề có trọng tâm khác nhau trong cùng lĩnh vực, chẳng hạn giáo dục đại học (Topic 0) so với giáo dục phổ thông (Topic 9), hoặc dinh dưỡng (Topic 2) so với điều trị y tế (Topic 7).

Tuy nhiên, khả năng diễn giải chưa đồng đều. Topic 1 pha trộn nội dung tài chính, ngân hàng và tiêu dùng; Topic 13 bao quát doanh nghiệp, công nghệ và một số nội dung khác biệt. Một bài đại diện của Topic 0 liên quan đến cuộc thi sắc đẹp cũng không phù hợp với nội dung chiếm ưu thế. Những trường hợp này cho thấy nhãn diễn giải mang tính khái quát và không bảo đảm mọi tài liệu trong cùng topic đều đồng nhất về ngữ nghĩa.

Toàn bộ nhãn được lưu ở trạng thái `proposed`, chưa trải qua kiểm tra chéo độc lập. Vì vậy, đây là kết quả phân tích định tính hỗ trợ diễn giải, không phải phép đo độ chính xác của chủ đề.

*Nguồn: `artifacts/common/lda/lda_k15_seed42/topic_keywords.csv`; `results/lda/topic_label_review/representative_docs_top10.csv`; `results/lda/topic_label_review/topic_label_proposals.csv`; `artifacts/common/lda/lda_k15_seed42/topic_display_mapping.json` (commit `b1f17cd`, nhánh `feature/lda-dashboard-handoff`).*

## 6. Thảo luận và hạn chế (Discussion & Limitations)

Trong các cấu hình cuối được lựa chọn, LDA đạt NPMI=0,147094 và topic diversity=0,653333 trên Test, thấp hơn NMF với K=10 (lần lượt 0,440680 và 1,000000). ARI và NMI của LDA cũng thấp hơn NMF. Tuy nhiên, NMF sử dụng TF-IDF trên P3 và K=10, trong khi LDA sử dụng BoW trên P2 và K=15. Do đó, sự khác biệt không thể được quy hoàn toàn cho bản thân thuật toán.

LDA có ưu điểm mô hình hóa phân phối xác suất trên nhiều chủ đề (Blei et al., 2003), nhưng thực nghiệm ghi nhận sự thay đổi giữa các random seeds và một số topic pha trộn nội dung. Các hạn chế khác gồm việc chưa khảo sát stability tại nhiều giá trị K, chưa thực hiện đầy đủ ablation tiền xử lý riêng cho LDA và chưa có đánh giá khả năng diễn giải độc lập. Những vấn đề này có thể được xem xét trong các nghiên cứu tiếp theo.

*Nguồn: Blei et al. (2003); O'Callaghan et al. (2015); `docs/lda_nmf_comparison.md`; `results/lda/final/test_metrics.json`; `results/nmf/nmf_p3_k10_seed42_v1/test_metrics.json`.*