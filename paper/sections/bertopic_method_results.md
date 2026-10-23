# Khám phá chủ đề bằng BERTopic

## 1. Phương pháp nghiên cứu

Nghiên cứu sử dụng Dataset v1.0 gồm 9.274 bài báo tiếng Việt, được chia cố
định thành Train (6.493 bài), Validation (1.391 bài) và Test (1.390 bài). Dữ
liệu đầu vào của BERTopic là biến thể tiền xử lý P1 (`p1_normalized`); mỗi văn
bản được tạo bằng cách nối tiêu đề với nội dung bài báo. Việc giữ nguyên phép
chia dữ liệu và cấu hình thực nghiệm giúp hạn chế rò rỉ dữ liệu, đồng thời bảo
đảm khả năng tái lập kết quả.

Quy trình BERTopic gồm bốn thành phần chính. Trước tiên, Sentence-BERT chuyển
mỗi bài báo thành một vector ngữ nghĩa. UMAP sau đó giảm số chiều của vector để
hỗ trợ phân cụm. HDBSCAN phát hiện các vùng có mật độ cao và gán nhãn `-1` cho
những bài không thuộc cụm đủ rõ. Cuối cùng, c-TF-IDF trích xuất các từ khóa nổi
bật để diễn giải nội dung của từng topic (Grootendorst, 2022; McInnes et al.,
2017; McInnes et al., 2018; Reimers & Gurevych, 2019).

Cấu hình E1 sử dụng mô hình biểu diễn `keepitreal/vietnamese-sbert`; UMAP với
`n_neighbors=15`, `n_components=5`, `min_dist=0`, độ đo cosine và seed 42;
HDBSCAN với `min_cluster_size=15`, `min_samples=5`, độ đo Euclidean và phương
pháp chọn cụm EOM. Mô hình chỉ được fit trên Train, Validation được dùng để
chọn cấu hình và Test chỉ được mở sau khi E1 đã được chốt.

*Nguồn thực nghiệm:* `results/bertopic_experiments/e1_vietnamese_sbert/config_used.yaml`;
`src/models/bertopic_pipeline.py`; `experiments/run_bertopic_experiments.py`.

## 2. Lựa chọn cấu hình

Bốn cấu hình E0–E3 được so sánh trên Validation. E0 sử dụng MiniLM đa ngôn
ngữ; E1 thay mô hình biểu diễn bằng Vietnamese SBERT; E2 tăng UMAP
`n_neighbors` lên 30; E3 tăng HDBSCAN `min_cluster_size` lên 30. Tỷ lệ outlier
tương ứng là 42,27%, 35,87%, 51,91% và 39,11%.

E3 giảm số topic từ 96 xuống 57 và đạt NMI = 0,4266, NPMI = -0,0494. Tuy
nhiên, E1 có tỷ lệ outlier thấp nhất và coverage cao nhất, bằng 64,13%, nên giữ
lại được nhiều bài báo có topic hơn. Vì mục tiêu của hệ thống là bao phủ phần
lớn kho tin trong khi vẫn duy trì chất lượng ngữ nghĩa chấp nhận được, E1 được
chọn làm cấu hình cuối. Kết quả này không có nghĩa E1 vượt trội ở mọi chỉ số,
mà phản ánh sự đánh đổi giữa số topic, coverage, coherence và khả năng diễn
giải. Cách diễn giải theo nhiều tiêu chí cũng phù hợp với nhận xét của Janssens
et al. (2025) về sự đánh đổi khi điều chỉnh số lượng topic trong BERTopic.

*Nguồn thực nghiệm:* `results/bertopic_experiments/summary.csv`;
`docs/bertopic_final_evaluation.md`.

## 3. Nghiên cứu loại bỏ thành phần và độ ổn định

Thử nghiệm E4 sử dụng cùng mô hình Vietnamese SBERT và cùng tham số
UMAP–HDBSCAN như E1, nhưng chỉ dùng tiêu đề bài báo thay vì tiêu đề kết hợp nội
dung. E4 tạo 93 topic, đạt topic diversity = 0,8602 và runtime = 139,31 giây.
Tuy nhiên, tỷ lệ outlier trên Validation tăng lên 46,66%, trong khi ARI =
0,0288, NMI = 0,3088 và NPMI = -0,1066. Kết quả cho thấy tiêu đề đơn lẻ chưa
cung cấp đủ ngữ cảnh để phân biệt nhiều chủ đề tin tức có từ ngữ tương tự; vì
vậy, đầu vào tiêu đề kết hợp nội dung của E1 được giữ lại.

Các thử nghiệm E2 và E3 cũng cho thấy số topic và tỷ lệ outlier nhạy với tham
số của UMAP và HDBSCAN. Nghiên cứu chưa chạy lặp BERTopic với nhiều random
seed, nên chưa thể đưa ra kết luận thống kê đầy đủ về độ ổn định. Đây là một
giới hạn cần được bổ sung trong nghiên cứu tiếp theo.

*Nguồn thực nghiệm:* `results/bertopic_experiments/summary.csv`;
`results/bertopic_experiments/e4_vietnamese_sbert_title_only/metrics.json`.

## 4. Kết quả đánh giá cuối

Sau khi E1 được lựa chọn bằng Validation, mô hình được đánh giá một lần trên
Test độc lập gồm 1.390 bài. Trong đó, 875 bài được gán vào topic và 515 bài
nhận nhãn outlier, tương ứng coverage = 62,95% và tỷ lệ outlier = 37,05%. Mô
hình tạo 96 topic trên tập Train.

So với chuyên mục tham chiếu, mô hình đạt ARI = 0,0559 và NMI = 0,4072. Hai
chỉ số này đo mức tương ứng giữa cấu trúc cụm và chuyên mục có sẵn, không phải
độ chính xác tuyệt đối của tên topic. NPMI trên Test bằng -0,0767, cho thấy một
số nhóm từ khóa chưa thường xuyên đồng xuất hiện trong cùng ngữ cảnh; do đó,
khả năng bao phủ tương đối tốt chưa đồng nghĩa mọi topic đều mạch lạc về ngữ
nghĩa (Röder et al., 2015).

Tổng runtime được ghi nhận là 1.191,65 giây, gồm tải mô hình 24,56 giây, tạo
biểu diễn 1.084,73 giây, fit 57,50 giây, dự đoán Validation 23,48 giây và dự
đoán Test 1,38 giây. Thời gian đánh giá 21,54 giây được ghi riêng. Kết quả cho
thấy bước tạo embedding chiếm phần lớn chi phí tính toán của pipeline; tổng
runtime không nên được diễn giải là riêng thời gian huấn luyện BERTopic.

*Nguồn thực nghiệm:* `results/bertopic_experiments/e1_vietnamese_sbert/metrics.json`;
`results/bertopic_experiments/e1_vietnamese_sbert/test_assignments.csv`.

## 5. Khả năng diễn giải topic

Mỗi topic được mô tả bằng 10 từ khóa c-TF-IDF và tối đa năm bài báo đại diện.
Một số topic có nội dung tương đối rõ: Topic 0 liên quan đến doanh nghiệp, đầu
tư và phát triển; Topic 1 tập trung vào điều tra, tội phạm và công an; Topic 2
mô tả dinh dưỡng, thực phẩm và sức khỏe.

Tuy nhiên, một số cụm vẫn pha trộn nhiều nội dung. Chẳng hạn, Topic 3 chủ yếu
nói về iPhone và Apple nhưng có bài liên quan đến CT; Topic 4 nghiêng về y tế
nhưng chứa bài về thuốc lá điện tử. Các trường hợp này cho thấy sự tương đồng
từ vựng hoặc ngữ nghĩa cục bộ có thể khiến bài báo được xếp vào cụm chưa hoàn
toàn phù hợp.

Tên topic hiện được đề xuất tự động từ từ khóa và bài đại diện, sau đó lưu
trong file mapping để hiển thị trên giao diện. Việc đổi tên hoặc gộp nhóm hiển
thị không thay đổi nhãn gốc hay các metric của BERTopic. Các tên vẫn được đánh
dấu `pending_human_review`; vì vậy, chúng là nhãn hỗ trợ diễn giải, chưa phải
nhãn chuẩn đã được xác nhận.

*Nguồn thực nghiệm:* `artifacts/common/bertopic/e1_vietnamese_sbert/topic_keywords.csv`;
`representative_docs.csv`; `topic_display_mapping.json`;
`results/bertopic_experiments/e1_vietnamese_sbert/topic_name_review.csv`.

## 6. Thảo luận và hạn chế

BERTopic có ưu điểm là khai thác biểu diễn ngữ nghĩa tiếng Việt, không yêu cầu
ấn định trước số topic và có khả năng nhận diện outlier. Việc kết hợp từ khóa
c-TF-IDF với bài báo đại diện cũng hỗ trợ người dùng hiểu nội dung từng cụm.
Tuy nhiên, kết quả cho thấy ba đánh đổi chính.

Thứ nhất, tham số phân cụm nhỏ tạo nhiều topic chi tiết nhưng làm giao diện khó
theo dõi; tăng `min_cluster_size` giúp giảm số topic nhưng có thể làm tăng số
bài bị loại. Thứ hai, coverage và coherence không nhất thiết tăng đồng thời:
mô hình có thể gán được nhiều bài nhưng từ khóa trong một số topic vẫn thiếu
tính gắn kết. Thứ ba, tên topic tự động có thể chưa phản ánh đầy đủ các bài nằm
ở ranh giới giữa nhiều chủ đề.

Các hướng phát triển tiếp theo gồm chạy nhiều random seed để đo độ ổn định,
thử các giá trị trung gian của `min_cluster_size`, hoàn thiện đánh giá tên topic
bởi nhiều người chấm, phân tích sâu các bài đa chủ đề và chuẩn hóa cùng một
giao thức benchmark cho cả năm phương pháp. Các cải tiến này cần được thực
hiện trên Validation trước khi đánh giá lại trên Test.

*Nguồn thực nghiệm:* `docs/bertopic_final_evaluation.md`;
`results/bertopic_experiments/e1_vietnamese_sbert/error_analysis_cases.csv`;
`results/bertopic_experiments/e1_vietnamese_sbert/topic_name_review.csv`.

## Tài liệu tham khảo

Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based
TF-IDF procedure* [Preprint]. arXiv.
https://doi.org/10.48550/arXiv.2203.05794

Janssens, W., Bogaert, M., & Van den Poel, D. (2025). A comparative analysis
of topic reduction techniques for BERTopic. *IEEE Access, 13*, 204087–204103.
https://doi.org/10.1109/ACCESS.2025.3638956

McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density
based clustering. *Journal of Open Source Software, 2*(11), Article 205.
https://doi.org/10.21105/joss.00205

McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform manifold
approximation and projection for dimension reduction* [Preprint]. arXiv.
https://arxiv.org/abs/1802.03426

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using
Siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing* (pp. 3982–3992). Association for
Computational Linguistics. https://doi.org/10.18653/v1/D19-1410

Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic
coherence measures. In *Proceedings of the Eighth ACM International Conference
on Web Search and Data Mining* (pp. 399–408). Association for Computing
Machinery. https://doi.org/10.1145/2684822.2685324
