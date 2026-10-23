# Embeddings kết hợp HDBSCAN trong khám phá chủ đề tin tức tiếng Việt

## Dữ liệu và phương pháp

Nghiên cứu sử dụng Dataset v1.0 gồm **9.274 bài báo tiếng Việt** sau khử trùng lặp, chia thành Train (**6.493 bài**), Validation (**1.391 bài**) và Test (**1.390 bài**).

Với dữ liệu tiền xử lý P0, nhóm so sánh hai mô hình embedding MiniLM và MPNet trên hai dạng đầu vào: chỉ tiêu đề, hoặc tiêu đề kết hợp tối đa 60 từ đầu nội dung. Embedding được chuẩn hóa, giảm xuống năm chiều bằng UMAP với khoảng cách cosine, sau đó phân cụm bằng HDBSCAN với khoảng cách Euclidean. Các tham số khảo sát gồm kích thước cụm tối thiểu, số điểm tối thiểu và cách chọn cụm EOM hoặc Leaf.

UMAP và HDBSCAN chỉ được fit trên Train; bài Validation và Test được gán cụm bằng mô hình đã học. HDBSCAN có thể gán nhãn `-1` cho bài chưa thuộc một vùng đủ mật độ. Nhóm phân tích riêng các trường hợp này khi đánh giá kết quả (McInnes et al., 2017; McInnes et al., 2018).

**Nguồn thực nghiệm:** `configs/embeddings.yaml`; `configs/hdbscan.yaml`; `results/hdbscan/m11_summary.csv`.

## Khảo sát cấu hình và chọn run đại diện

Nhóm kết hợp bốn dạng biểu diễn (MiniLM và MPNet, mỗi mô hình dùng chỉ tiêu đề hoặc tiêu đề kết hợp nội dung) với tám cấu hình HDBSCAN, tạo thành **32 run trên Train**. Theo cặp tham số (`min_cluster_size`, `min_samples`), nhóm thử sáu cấu hình EOM: `(15, 5)`, `(15, 10)`, `(30, 5)`, `(30, 10)`, `(50, 5)`, `(50, 10)`; và hai cấu hình Leaf: `(15, 5)`, `(30, 10)`.

Sau đó, nhóm đối chiếu trên Validation bốn run tiêu đề kết hợp nội dung của MiniLM và MPNet, mỗi mô hình dùng EOM hoặc Leaf ở cùng tham số 15/5:

| Run | Coverage | ARI | NMI |
|---|---:|---:|---:|
| MiniLM + EOM | 65,20% | 0,1303 | 0,4218 |
| MiniLM + Leaf | 50,97% | 0,0093 | 0,3592 |
| MPNet + EOM | 62,04% | 0,0464 | 0,4101 |
| MPNet + Leaf | 51,33% | 0,0098 | 0,3562 |

**MiniLM Title+Content + EOM** có kết quả tốt nhất trong bốn run được đối chiếu nên được chọn làm run cuối. ARI và NMI được tính trên cả bài mang nhãn `-1` và đối chiếu với `category_gold` — chuyên mục báo rộng, không phải nhãn chuẩn của từng chủ đề.

**Nguồn thực nghiệm:** `results/hdbscan/validation/m11_validation_metrics.csv`; `docs/e03_hdbscan_contribution.md`; `configs/hdbscan.yaml`.

## Độ ổn định và đánh giá thủ công

Với run được chọn, nhóm thay seed UMAP từ 42 đến 51 và giữ nguyên dữ liệu cùng các tham số khác. Số cụm Train dao động từ **99 đến 117**. ARI trung bình giữa các cặp lần chạy là **0,5165 trên Train** và **0,4774 trên Validation**. Trung bình **19,96%** bài Validation chuyển giữa trạng thái `-1` và trạng thái được gán cụm khi so sánh các cặp seed.

Đây là ARI **giữa các lần chạy**, khác ARI đối chiếu `category_gold`. Seed 42 của run gốc vẫn được giữ.

Đánh giá thủ công trên **năm cụm và 25 cặp bài–cụm** cho điểm ranh giới trung bình **2,5/5**, thấp hơn mức phù hợp bài–cụm **3,9/5**. Phân tích outlier M12 trên MPNet Train còn phân biệt đầu vào thiếu ngữ cảnh, chủ đề nhỏ, bài nhiều trọng tâm và bài hợp lệ vẫn bị gán `-1`. Các mẫu này không đại diện cho toàn bộ dữ liệu.

**Nguồn thực nghiệm:** `docs/e04_hdbscan_stability.md`; `docs/hdbscan_human_eval.md`; `docs/m12_outlier_analysis.md`.

## Kết quả trên Test

Trên tập Train, cấu hình `minilm_title_content_eom_15_5` tạo ra **101 cụm**. Khi áp dụng mô hình đã học cho **1.390 bài Test độc lập**, có **511 bài** nhận nhãn `-1`, tương ứng tỷ lệ outlier **36,76%** và coverage **63,24%**. ARI đạt **0,1244** và NMI đạt **0,4214**. Hai chỉ số này đo mức tương ứng với chuyên mục tham chiếu, không phải độ chính xác của tên chủ đề.

Topic diversity của run là **0,9327**. Coherence NPMI đạt **0,2063** khi tính trên toàn bộ **9.274 bài**. Do bộ từ khóa và corpus dùng cho phép đo này bao gồm cả Train, Validation và Test, coherence chỉ mô tả kết quả sau huấn luyện, **không phải điểm đánh giá Test độc lập**.

**Nguồn thực nghiệm:** `results/hdbscan/test/minilm_title_content_eom_15_5_test_metrics.json`; `artifacts/common/hdbscan/minilm_title_content_eom_15_5/coherence_summary.json`.

## Diễn giải cụm và các ca lỗi

Nhóm dùng từ khóa, bài đại diện và phiếu rà tên để mô tả 101 cụm. Các cụm có thể nhận diện nội dung như y tế, bóng đá câu lạc bộ và thị trường chứng khoán. **Mười sáu cụm** được xếp vào **bảy nhóm hiển thị** để giao diện trình bày thuận tiện; việc gộp này không thay đổi nhãn gán bài hoặc metric HDBSCAN.

Phiếu tên nội bộ đánh dấu `approved`, nhưng đánh giá độc lập E05 chỉ khảo sát năm cụm. Các ca HS001/HS004 cho thấy một cụm có thể gộp những sự kiện khác nhau cùng nhắc Apple. HS024 là tin xã hội chỉ nhắc bệnh viện trong diễn biến nhưng bị đưa vào cụm y tế. Bài laptop `VNN_707b880b93fd` cũng lạc vào cụm y tế trên Train; ca này không thuộc 25 bài được hai người chấm.

**Nguồn thực nghiệm:** `results/hdbscan/topic_naming/topic_name_review.csv`; `artifacts/common/hdbscan/minilm_title_content_eom_15_5/topic_display_mapping.json`; `results/hdbscan/error_analysis/hdbscan_error_cases.csv`; `docs/e03_hdbscan_contribution.md`; `docs/e06_hdbscan_error_analysis.md`.

## Tài liệu tham khảo

- McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density based clustering. *Journal of Open Source Software, 2*(11), Article 205. https://doi.org/10.21105/joss.00205
- McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform manifold approximation and projection for dimension reduction* [Preprint]. arXiv. https://arxiv.org/abs/1802.03426
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing* (pp. 3982–3992). Association for Computational Linguistics. https://aclanthology.org/D19-1410/
- Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. In *Proceedings of the Eighth ACM International Conference on Web Search and Data Mining* (pp. 399–408). Association for Computing Machinery. https://doi.org/10.1145/2684822.2685324