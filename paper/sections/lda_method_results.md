# LDA: Methodology, Results and Discussion

## 1. Phương pháp nghiên cứu (Methodology)

Nghiên cứu sử dụng Dataset v1.0 gồm 9.274 bài báo tiếng Việt sau khử trùng lặp, với biến thể tiền xử lý P2 (p2_segmented) và trường đầu vào text. Dữ liệu được chia cố định thành Train (6.493 bài), Validation (1.391 bài) và Test (1.390 bài). Văn bản được biểu diễn bằng Bag-of-Words (BoW) với unigram, tối đa 15.000 đặc trưng, min_df=3 và max_df=0.85.

Latent Dirichlet Allocation (LDA) là mô hình xác suất sinh, trong đó mỗi tài liệu được biểu diễn bằng phân phối xác suất trên các chủ đề và mỗi chủ đề bằng phân phối xác suất trên từ vựng (Blei et al., 2003). Mô hình được huấn luyện theo phương pháp batch, tối đa 20 vòng lặp. Giai đoạn lựa chọn cấu hình sử dụng Train/Validation và không sử dụng Test. Sau khi lựa chọn **K=15, seed=42**, BoW và LDA được huấn luyện lại trên Train+Validation (7.884 bài), sau đó đánh giá trên Test độc lập.

*Nguồn thực nghiệm: configs/lda.yaml; configs/lda_final.yaml; experiments/lda/run_lda_final.py.*

## 2. Lựa chọn cấu hình (Model Selection)

Số lượng chủ đề được khảo sát với K = 4, 6, 8, 10, 12, 15 và 20, sử dụng coherence C_V, C_NPMI và topic diversity để đánh giá trên Validation. Các thước đo coherence cung cấp cơ sở định lượng về mức độ liên hệ giữa các từ khóa mô tả chủ đề (Röder et al., 2015).

K=20 đạt coherence cao nhất trong dải thử nghiệm (C_V=0,5418; C_NPMI=0,0748), nhưng diversity đạt 0,6100. Trong khi đó, K=15 đạt C_V=0,5133; C_NPMI=0,0606 và diversity=0,6400. Vì vậy, K=15 được lựa chọn dựa trên sự đánh đổi giữa độ mạch lạc, độ đa dạng và khả năng diễn giải sơ bộ, thay vì tối ưu một chỉ số đơn lẻ. Coherence trong M08 phục vụ lựa chọn K trên Validation và không được so sánh trực tiếp với NPMI document-level ở Final Test.

*Nguồn thực nghiệm: docs/m08_lda_experiments.md; results/lda/m08/topic_number_results.csv.*

## 3. Đánh giá độ ổn định và phân tích bổ sung (Stability & Additional Analysis)

Độ ổn định của LDA được đánh giá tại K=15 với năm random seeds 42, 52, 62, 72 và 82. Từ vựng BoW và mô hình được học trên Train; các phân phối chủ đề được so sánh trên Validation. Do topic ID có thể thay đổi giữa các lần huấn luyện, các chủ đề được căn chỉnh bằng thuật toán Hungarian dựa trên Jaccard similarity của 10 từ khóa đặc trưng. Sau đó, Cosine similarity được tính giữa các vector phân phối chủ đề của cùng bài báo Validation.

Trung bình trên 10 cặp seed khác nhau, Mean Keyword Jaccard đạt 0,455685 và Mean Document Cosine đạt 0,681273. Các dòng tự so sánh không được đưa vào trung bình này. Kết quả cho thấy từ khóa đại diện và phân phối chủ đề thay đổi giữa các lần khởi tạo. Tuy nhiên, thực nghiệm mới khảo sát K=15 nên chưa đủ cơ sở kết luận độ ổn định tương đối so với những giá trị K khác.

*Nguồn thực nghiệm: docs/m09_lda_stability.md; results/lda/m09/pairwise_stability.csv.*

## 4. Kết quả đánh giá cuối (Final Test)

Mô hình LDA Final (K=15, seed=42) được đánh giá trên 1.390 bài báo Test độc lập, đạt NPMI 0,147094, topic diversity 0,653333, Adjusted Rand Index (ARI) 0,406466 và Normalized Mutual Information (NMI) 0,572713.

NPMI được tính dựa trên đồng xuất hiện từ ở cấp tài liệu (document_binary_common). ARI và NMI được xác định bằng cách đối chiếu topic có xác suất cao nhất với chuyên mục tham chiếu category_gold. Do đó, hai chỉ số này phản ánh mức độ tương ứng với chuyên mục báo, không phải độ chính xác của nhãn chủ đề. Các thước đo định lượng cung cấp thông tin về chất lượng mô hình nhưng cần được xem xét cùng phân tích khả năng diễn giải (Röder et al., 2015).

*Nguồn thực nghiệm: results/lda/final/test_metrics.json; experiments/lda/run_lda_final.py.*

## 5. Phân tích khả năng diễn giải chủ đề (Topic Interpretability)

Khả năng diễn giải được xem xét thông qua 10 từ khóa có trọng số cao nhất và 10 bài báo đại diện có xác suất chủ đề chính cao nhất cho mỗi topic, trích xuất từ Train và Validation. Tổng cộng 150 bài báo đại diện được sử dụng để hỗ trợ đề xuất nhãn cho 15 chủ đề. Các nhãn này thuộc mô hình LDA Final sau khi huấn luyện lại trên Train+Validation, không kế thừa trực tiếp topic ID từ thực nghiệm M08.

Phân tích định tính cho thấy nhiều chủ đề có nội dung tương đối rõ, như thể thao ASIAD (Topic 3), chạy bộ (Topic 5), bóng đá (Topic 6) và thị trường vàng (Topic 8). Mô hình cũng hình thành các chủ đề có trọng tâm khác nhau trong cùng lĩnh vực, chẳng hạn giáo dục đại học (Topic 0) so với giáo dục phổ thông (Topic 9), hoặc dinh dưỡng (Topic 2) so với điều trị y tế (Topic 7).

Tuy nhiên, khả năng diễn giải chưa đồng đều. Topic 1 pha trộn nội dung tài chính, ngân hàng và tiêu dùng; Topic 13 bao quát doanh nghiệp, công nghệ và một số nội dung khác biệt. Một bài đại diện của Topic 0 liên quan đến cuộc thi sắc đẹp cũng không phù hợp với nội dung chiếm ưu thế. Những trường hợp này cho thấy nhãn diễn giải mang tính khái quát và không bảo đảm mọi tài liệu trong cùng topic đều đồng nhất về ngữ nghĩa.

Toàn bộ nhãn được lưu ở trạng thái proposed, chưa trải qua kiểm tra chéo độc lập. Vì vậy, đây là kết quả phân tích định tính hỗ trợ diễn giải, không phải phép đo độ chính xác của chủ đề.

*Nguồn thực nghiệm: artifacts/common/lda/lda_k15_seed42/topic_keywords.csv; results/lda/topic_label_review/representative_docs_top10.csv; results/lda/topic_label_review/topic_label_proposals.csv; artifacts/common/lda/lda_k15_seed42/topic_display_mapping.json.*

## 6. Thảo luận và hạn chế (Discussion & Limitations)

Trong các cấu hình cuối được lựa chọn, LDA đạt NPMI=0,147094 và topic diversity=0,653333 trên Test, thấp hơn NMF với K=10 (lần lượt 0,440680 và 1,000000). ARI và NMI của LDA cũng thấp hơn NMF. Tuy nhiên, NMF sử dụng TF-IDF trên P3 và K=10, trong khi LDA sử dụng BoW trên P2 và K=15. Do đó, sự khác biệt không thể được quy hoàn toàn cho bản thân thuật toán.

LDA mô hình hóa phân phối xác suất trên nhiều chủ đề (Blei et al., 2003), nhưng thực nghiệm ghi nhận sự thay đổi giữa các random seeds và một số topic pha trộn nội dung. Khi đối chiếu với NMF, cần xem xét cả cách biểu diễn văn bản và lựa chọn từ khóa đại diện (O'Callaghan et al., 2015). Các hạn chế khác gồm việc chưa khảo sát stability tại nhiều giá trị K, chưa thực hiện đầy đủ ablation tiền xử lý riêng cho LDA và chưa có đánh giá khả năng diễn giải độc lập. Những vấn đề này có thể được xem xét trong các nghiên cứu tiếp theo.

*Nguồn thực nghiệm: docs/lda_nmf_comparison.md; results/lda/final/test_metrics.json; results/nmf/nmf_p3_k10_seed42_v1/test_metrics.json.*

## References

Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. *Journal of Machine Learning Research, 3*, 993–1022. https://www.jmlr.org/papers/v3/blei03a.html

O'Callaghan, D., Greene, D., Carthy, J., & Cunningham, P. (2015). An analysis of the coherence of descriptors in topic modeling. *Expert Systems with Applications, 42*(13), 5645–5657. https://doi.org/10.1016/j.eswa.2015.02.055

Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. In *Proceedings of the Eighth ACM International Conference on Web Search and Data Mining* (pp. 399–408). https://doi.org/10.1145/2684822.2685324
