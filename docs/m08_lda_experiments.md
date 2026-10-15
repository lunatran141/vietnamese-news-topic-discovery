
# M08 — LDA Topic Experiments

## 1. Mục tiêu

Thực nghiệm nhiều số lượng topic K để đánh giá chất lượng
mô hình LDA trên Dataset v1.0.

Đánh giá bằng:
- Topic Coherence c_v.
- Topic Coherence c_npmi.
- Topic Diversity.
- Top Keywords và Representative Documents.
- Khả năng diễn giải thủ công.

## 2. Dữ liệu và phương pháp

- Dataset: v1.0.
- Preprocessing: p2_segmented.
- Train: 6493 bài.
- Validation: 1391 bài.
- Model input: text.
- Random seed: 42.
- K: [4, 6, 8, 10, 12, 15, 20].
- Top keywords: 10 từ/topic.

Bag-of-Words được fit trên Train.
LDA được huấn luyện trên Train.
Coherence được tính trên Validation.
Không sử dụng Test để lựa chọn K.

Các cấu hình BoW và LDA kế thừa từ M07.

## 3. Cách chạy

Cài đặt dependencies:

    python -m pip install -r requirements.txt

Chạy thực nghiệm:

    python -m experiments.lda.run_m08_lda_experiments

Chạy unit tests:

    python -m pytest tests/test_lda_pipeline.py tests/test_lda_metrics.py -q

## 4. Kết quả định lượng

| K | c_v | c_npmi | Diversity | Runtime (s) |
|---|---:|---:|---:|---:|
| 4 | 0.3757 | -0.0088 | 0.6000 | 28.02 |
| 6 | 0.3997 | 0.0039 | 0.6167 | 36.40 |
| 8 | 0.4217 | 0.0216 | 0.6500 | 78.69 |
| 10 | 0.4557 | 0.0406 | 0.6500 | 47.76 |
| 12 | 0.4905 | 0.0521 | 0.6250 | 55.08 |
| 15 | 0.5133 | 0.0606 | 0.6400 | 65.28 |
| 20 | 0.5418 | 0.0748 | 0.6100 | 70.32 |

K = 20 đạt Coherence cao nhất trong dải thử nghiệm.
K = 8 và K = 10 đạt Diversity cao nhất.

Runtime là thời gian quan sát trong một lần chạy,
chưa phải benchmark nhiều lần.

## 5. Đánh giá khả năng diễn giải

### K = 15

Các topic tương đối rõ:
- Giá cả và tài chính.
- Doanh nghiệp và công nghệ.
- Y tế và bệnh lý.
- Dinh dưỡng, sức khỏe.
- Apple và thiết bị công nghệ.
- Giáo dục phổ thông.
- Đại học và đào tạo.
- Công an và điều tra.
- Thể thao ASIAD.
- Bóng đá.
- Chính sách và doanh nghiệp.

Topic 5 chứa nhiều từ phổ biến như
"một", "không", "những", "người",
nên khó xác định chủ đề cụ thể.

### K = 20

Các topic được phân tách chi tiết hơn:
- Y tế chia thành bệnh lý và dịch vụ bệnh viện.
- Bóng đá có thể tách thành bóng đá Việt Nam
  và bóng đá quốc tế.
- Doanh nghiệp có các hướng công nghệ,
  sản xuất, đầu tư và vốn.

Tuy nhiên, một số topic vẫn chung chung,
có keywords trùng lặp hoặc khó đặt tên.

Đánh giá này dựa trên 10 keywords và
một bài báo đại diện cho mỗi topic,
chưa phải đánh giá nhiều người chấm.

## 6. Đề xuất lựa chọn K

Đề xuất K = 15 làm ứng viên chính vì:

1. Coherence tương đối cao.
2. Topic Diversity đạt 0.64, cao hơn K = 20.
3. Số topic vừa phải và nhiều chủ đề dễ diễn giải.
4. Hạn chế một phần sự phân mảnh so với K = 20.

Giữ K = 20 làm cấu hình đối chiếu
vì có Coherence cao nhất trong dải đã thử.

Đây là đề xuất sơ bộ, không khẳng định K = 15
là số topic tối ưu tuyệt đối.

## 7. Kết quả đầu ra

Thư mục results/lda/m08/:

- topic_number_results.csv
- topic_keywords.csv
- representative_docs.csv

Các file kết quả hiện được tạo cục bộ
bằng script M08.

## 8. Kiểm thử

M08: 7/7 tests passed.

M07 + M08: 13/13 tests passed trong 6.24 giây.

## 9. Giới hạn

- Chưa thử K lớn hơn 20.
- Chưa đánh giá thủ công theo nhiều người chấm.
- Chưa thử ablation p3_no_stopwords.
- Chưa đánh giá đa seed; thuộc M09.
- Chưa đánh giá Test; giữ Test cho final evaluation.

## 10. Công việc tiếp theo

M09 — LDA Stability:
- Huấn luyện với nhiều random seeds.
- Đánh giá mức độ ổn định của topics.
- Phân tích tác động của tính ngẫu nhiên.
