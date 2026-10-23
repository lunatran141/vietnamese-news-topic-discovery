# Kết quả thực nghiệm và phân tích HDBSCAN

## 1. Dữ liệu và quy trình thực nghiệm

Nghiên cứu sử dụng Dataset v1.0 gồm **9.274 bài báo tiếng Việt** sau khử trùng lặp, chia thành Train (**6.493 bài**), Validation (**1.391 bài**) và Test (**1.390 bài**).

Với dữ liệu tiền xử lý **P0**, nhóm so sánh hai mô hình embedding **MiniLM** và **MPNet** trên hai dạng đầu vào: chỉ tiêu đề và tiêu đề kết hợp tối đa 60 từ đầu nội dung. Embedding được chuẩn hóa, giảm xuống năm chiều bằng UMAP với khoảng cách cosine, sau đó phân cụm bằng HDBSCAN với khoảng cách Euclidean. Các tham số khảo sát gồm kích thước cụm tối thiểu, số điểm tối thiểu và cách chọn cụm EOM hoặc Leaf.

UMAP và HDBSCAN chỉ được fit trên Train; bài Validation và Test được gán cụm bằng mô hình đã học. HDBSCAN có thể gán nhãn `-1` cho bài chưa thuộc một vùng đủ mật độ; nhóm phân tích riêng các trường hợp này khi đánh giá kết quả (McInnes et al., 2017; McInnes et al., 2018).

**Nguồn thực nghiệm:** `configs/embeddings.yaml`; `configs/hdbscan.yaml`; `results/hdbscan/m11_summary.csv`.

## 2. Khảo sát cấu hình và lựa chọn run

Nhóm kết hợp bốn dạng biểu diễn (MiniLM và MPNet, mỗi mô hình dùng chỉ tiêu đề hoặc tiêu đề kết hợp nội dung) với tám cấu hình HDBSCAN, tạo thành **32 run trên Train**. Theo cặp tham số `(min_cluster_size, min_samples)`, nhóm thử sáu cấu hình EOM là `(15, 5)`, `(15, 10)`, `(30, 5)`, `(30, 10)`, `(50, 5)`, `(50, 10)` và hai cấu hình Leaf là `(15, 5)`, `(30, 10)`. Sau đó, nhóm đối chiếu trên Validation bốn run tiêu đề kết hợp nội dung của MiniLM và MPNet, mỗi mô hình dùng EOM hoặc Leaf ở cùng tham số 15/5.

Kết quả trên Validation:

- **MiniLM + EOM:** coverage 65,20%; ARI 0,1303; NMI 0,4218.
- **MiniLM + Leaf:** coverage 50,97%; ARI 0,0093; NMI 0,3592.
- **MPNet + EOM:** coverage 62,04%; ARI 0,0464; NMI 0,4101.
- **MPNet + Leaf:** coverage 51,33%; ARI 0,0098; NMI 0,3562.

**MiniLM Title+Content + EOM** có kết quả tốt nhất trong bốn run được đối chiếu nên được chọn làm run cuối. ARI và NMI được tính trên cả bài mang nhãn `-1` và đối chiếu với `category_gold` — chuyên mục báo rộng, không phải nhãn chuẩn của từng chủ đề.

**Nguồn thực nghiệm:** `results/hdbscan/validation/m11_validation_metrics.csv`; `docs/e03_hdbscan_contribution.md`; `configs/hdbscan.yaml`.

## 3. Độ ổn định và đánh giá thủ công

Với run được chọn, nhóm thay seed UMAP từ 42 đến 51 và giữ nguyên dữ liệu, tham số khác. Số cụm Train dao động từ **99 đến 117**, ARI trung bình giữa các cặp lần chạy là **0,5165 trên Train** và **0,4774 trên Validation**. Trung bình **19,96%** bài Validation chuyển giữa trạng thái `-1` và được gán cụm khi so sánh các cặp seed. Đây là ARI giữa các lần chạy, khác ARI đối chiếu `category_gold`; seed 42 của run gốc vẫn được giữ.

Đánh giá thủ công trên năm cụm, 25 cặp bài–cụm cho điểm ranh giới trung bình **2,5/5**, thấp hơn mức phù hợp bài–cụm **3,9/5**. Phân tích outlier M12 trên MPNet Train còn phân biệt đầu vào thiếu ngữ cảnh, chủ đề nhỏ, bài nhiều trọng tâm và bài hợp lệ vẫn bị gán `-1`. Các mẫu này không đại diện cho toàn bộ dữ liệu.

**Nguồn thực nghiệm:** `docs/e04_hdbscan_stability.md`; `docs/hdbscan_human_eval.md`; `docs/m12_outlier_analysis.md`.

## 4. Đánh giá trên tập Test

Cấu hình `minilm_title_content_eom_15_5` được chọn trên Validation và tạo **101 cụm** khi fit trên Train. Để đánh giá cuối, nhóm dùng lại mô hình MiniLM, UMAP và HDBSCAN đã lưu; **1.390 bài Test** được mã hóa, biến đổi bằng UMAP rồi gán vào các cụm đã học, không fit lại mô hình trên Test. Chương trình kiểm tra các bài Test không trùng `article_id` với Train hoặc Validation và không tạo ra `cluster_id` mới ngoài các nhãn của mô hình đã fit.

Kết quả có **879 bài được gán cụm** và **511 bài mang nhãn `-1`**, tương ứng độ bao phủ **63,24%** và tỷ lệ outlier **36,76%**.

ARI đạt **0,1244** và NMI đạt **0,4214** trên toàn bộ 1.390 bài Test, bao gồm cả những bài mang nhãn `-1`. `category_gold` chỉ được đưa vào sau bước dự đoán để đối chiếu kết quả với chuyên mục báo. Vì chuyên mục rộng hơn chủ đề hoặc sự kiện cụ thể, ARI và NMI không phải độ chính xác của tên chủ đề.

## 5. Chất lượng artifact cuối

Artifact sau cùng có topic diversity **0,9327**, được tính từ 942 từ khóa khác nhau trong 1.010 vị trí từ khóa của 101 cụm. Coherence NPMI ở cấp tài liệu đạt **0,2063** khi dùng toàn bộ 9.274 bài làm corpus tham chiếu.

Phép đo này sử dụng mười từ khóa mỗi cụm và văn bản gồm tiêu đề cùng tối đa 60 từ đầu nội dung; cụm outlier `-1` không được xem là một chủ đề để tính điểm. Do bộ từ khóa được trích từ các bài thuộc cả Train, Validation và Test sau khi đã chốt cấu hình, coherence mô tả artifact cuối chứ không phải kết quả đánh giá Test độc lập. Điểm này cần được diễn giải theo đúng corpus và cách tính đồng xuất hiện từ, thay vì so trực tiếp với coherence của phương pháp dùng giao thức khác (Röder et al., 2015).

**Nguồn thực nghiệm:** `experiments/hdbscan/evaluate_p01_hdbscan_test.py`; `experiments/hdbscan/measure_hdbscan_coherence.py`; `results/hdbscan/test/minilm_title_content_eom_15_5_test_metrics.json`; `artifacts/common/hdbscan/minilm_title_content_eom_15_5/coherence_summary.json`.

## 6. Đặt tên và diễn giải chủ đề

Sau khi chốt run `minilm_title_content_eom_15_5`, nhóm tạo artifact diễn giải cho 101 cụm từ các bài đã được gán cụm thuộc Train, Validation và Test. Với mỗi cụm, chương trình gộp tiêu đề đã xử lý và tối đa 60 từ đầu nội dung của các bài, rồi dùng TF-IDF để lấy 10 từ khóa. Tên tự động ban đầu được ghép từ ba từ khóa đứng đầu. Chương trình cũng chọn năm bài đại diện mỗi cụm theo độ mạnh thành viên; phiếu rà tên bổ sung năm bài Train lấy ngẫu nhiên mỗi cụm để người đọc kiểm tra những bài ít điển hình hơn. Các artifact này được tạo sau khi chọn cấu hình, phục vụ diễn giải và hiển thị, không dùng để chọn lại mô hình.

Phiếu đặt tên ghi `suggest_label` là tên được lưu khi lập phiếu và `display_name` là tên hiển thị đang sử dụng. Cả 101 dòng có trạng thái rà nội bộ `approved`, nhưng điều đó không có nghĩa toàn bộ tên đã được đánh giá độc lập: lượt chấm E05 của hai thành viên chỉ khảo sát năm cụm và 25 cặp bài–cụm trên Validation. Nhóm còn xếp 16 cụm vào bảy nhóm hiển thị để giao diện trình bày các nội dung liên quan cùng nhau. Việc đặt tên và nhóm hiển thị không thay đổi cụm gốc, nhãn gán bài hay chỉ số HDBSCAN.

Các bài được đọc cho thấy giới hạn của từ khóa và tên cụm. HS001 và HS004 cùng nói về vụ kiện Taptic Engine của Apple, nhưng cụm chứa chúng còn có tin về iPhone Duo — một sự kiện khác cùng nhắc Apple. HS024 chủ yếu là tin xã hội về trẻ sơ sinh bị bỏ rơi; bệnh viện chỉ xuất hiện trong diễn biến dù bài được gán vào cụm y tế. Bài Train `VNN_707b880b93fd` nói về laptop cũng lạc vào cụm này. Ca laptop được phát hiện khi rà ví dụ đặt tên và không thuộc 25 bài được chấm trong E05. Những trường hợp này cho thấy cần đối chiếu từ khóa với nội dung bài trước khi kết luận một cụm có ranh giới chủ đề rõ ràng.

**Nguồn thực nghiệm:** `results/hdbscan/topic_naming/topic_name_review.csv`; `artifacts/common/hdbscan/minilm_title_content_eom_15_5/topic_display_mapping.json`; `results/hdbscan/error_analysis/hdbscan_error_cases.csv`.

## 7. Hướng phát triển

Hướng phát triển là xây dựng hệ thống theo dõi chủ đề báo chí theo thời gian: định kỳ phân cụm dữ liệu mới, đối chiếu các cụm giữa những giai đoạn để phát hiện chủ đề mới xuất hiện, chủ đề đang thay đổi và các sự kiện tách ra từ một chủ đề rộng. Các bài mang nhãn `-1` sẽ được rà soát để phân biệt tin ngoài phạm vi, chủ đề hiếm và dấu hiệu của chủ đề mới; kết quả rà soát của con người được dùng để cập nhật tên và nhóm hiển thị.

Hướng này cần được đánh giá trên dữ liệu thu thập ở thời điểm muộn hơn, với tiêu chí về độ ổn định chủ đề, chất lượng gán bài và khả năng phát hiện chủ đề mới, trước khi áp dụng vào hệ thống theo dõi tin tức.

**Nguồn thực nghiệm:** `docs/e03_hdbscan_contribution.md`; `docs/e06_hdbscan_error_analysis.md`.

## Tài liệu tham khảo

- McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform manifold approximation and projection for dimension reduction* [Preprint]. arXiv. [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing* (pp. 3982–3992). Association for Computational Linguistics. [https://aclanthology.org/D19-1410/](https://aclanthology.org/D19-1410/)
- Eklund, A., & Forsman, M. (2022). Topic modeling by clustering language model embeddings: Human validation on an industry dataset. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: Industry Track* (pp. 635–643). Association for Computational Linguistics. [https://doi.org/10.18653/v1/2022.emnlp-industry.65](https://doi.org/10.18653/v1/2022.emnlp-industry.65)
- Eklund, A., Forsman, M., & Drewes, F. (2023). An empirical configuration study of a common document clustering pipeline. *Northern European Journal of Language Technology, 9*(1). [https://doi.org/10.3384/nejlt.2000-1533.2023.4396](https://doi.org/10.3384/nejlt.2000-1533.2023.4396)
- Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. In *Advances in Neural Information Processing Systems* (Vol. 33). [https://proceedings.neurips.cc/paper_files/paper/2020/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)