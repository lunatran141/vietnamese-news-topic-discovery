
# M07 — Bag-of-Words + LDA Pipeline

## 1. Mục tiêu

Task M07 triển khai mô hình Latent Dirichlet Allocation (LDA)
để khám phá chủ đề trong tập tin tức tiếng Việt.

Pipeline:
1. Đọc Dataset v1.0 đã preprocessing.
2. Chia Train/Validation theo split manifest.
3. Xây dựng Bag-of-Words bằng CountVectorizer.
4. Huấn luyện LDA chỉ trên Train.
5. Suy luận topic distributions cho Train và Validation.
6. Lưu model, vectorizer, BoW matrices và kết quả.

## 2. Dataset và Split

| Thành phần | Giá trị |
|---|---|
| Dataset version | v1.0 |
| Default preprocessing | p2_segmented |
| Ablation preprocessing | p3_no_stopwords |
| Dataset path | data/processed/p2_segmented.csv |
| Input column | text |
| ID column | article_id |
| Split manifest | data/splits/split_manifest.csv |
| Train documents | 6493 |
| Validation documents | 1391 |
| Test documents | 1390 (không sử dụng trong M07) |

Vocabulary và LDA model chỉ fit trên Train.
Validation chỉ được transform bằng các thành phần đã fit.

Test được khóa, không dùng để tuning hoặc chọn mô hình.

## 3. Cấu trúc code

| File | Chức năng |
|---|---|
| configs/lda.yaml | Cấu hình M07 |
| src/features/bow.py | Xây dựng Bag-of-Words |
| src/models/lda_pipeline.py | Huấn luyện và suy luận LDA |
| experiments/lda/run_m07_lda.py | Script chạy M07 |
| tests/test_lda_pipeline.py | Unit tests |

## 4. Cấu hình Baseline

Bag-of-Words:
- max_features: 15000
- min_df: 3
- max_df: 0.85
- ngram_range: [1, 1]

LDA:
- n_components: 10
- learning_method: batch
- max_iter: 20
- random_state: 42

Đây là cấu hình baseline, chưa phải cấu hình tối ưu.

## 5. Cách chạy

Tại thư mục gốc repository:

```powershell
python -m pip install -r requirements.txt
python -m experiments.lda.run_m07_lda --config configs/lda.yaml
```

## 6. Kết quả thực nghiệm

Lần chạy baseline đã ghi nhận:

| Metric | Kết quả |
|---|---|
| Train documents | 6493 |
| Validation documents | 1391 |
| Vocabulary size | 15000 |
| Number of topics | 10 |
| Train topic matrix | (6493, 10) |
| Validation topic matrix | (1391, 10) |
| Runtime | 30.75 giây |

Runtime là giá trị quan sát được trên môi trường chạy,
không phải kết quả benchmark chính thức.

## 7. Artifacts và Results

artifacts/lda/:
- bow_vectorizer.joblib
- lda_model.joblib
- bow_train.npz
- bow_val.npz
- feature_names.json
- topic_word_distribution.npy
- run_metadata.json

results/lda/:
- document_topics_train.csv
- document_topics_val.csv

Document-topic CSV chứa:
- article_id
- topic_0 đến topic_9
- dominant_topic
- dominant_probability

Các artifacts sinh ra có thể được tạo lại bằng lệnh chạy M07.
Model và ma trận lớn hiện chưa được đưa vào Git thường.

## 8. Automated Tests

Chạy:

```powershell
python -m pytest tests/test_lda_pipeline.py -v
```

Kết quả: 6/6 tests PASSED.

Các test kiểm tra:
- BoW matrix
- Vocabulary không bị thay đổi khi transform
- Topic probability distributions
- Khả năng tái lập với random_state cố định
- Save/load model
- Từ chối ma trận chứa giá trị âm

## 9. Reproducibility Verification

Ngày kiểm tra: 2026-10-15.

Pipeline được chạy hai lần với cùng Dataset v1.0,
cấu hình BoW/LDA và random_state = 42.

| File | SHA256 |
|---|---|
| lda_model.joblib | DF06D5AE222E55F4D5E1E2684F4209C74BB147398A9E6D84969038216758D96B |
| document_topics_train.csv | 8E07DC5C38DC5946EB9501ECACF1BD0B1FDF00A7B847242D4CD65921FF3A8C0E |
| document_topics_val.csv | C68E48BB6EA7339DC7E592EAF837A01E099BA7BE71D9561AAF7D068455AB26FC |

Cả ba SHA256 giống nhau giữa hai lần chạy.

Kết luận: Pipeline tái lập được trong cùng môi trường
với cùng dữ liệu và cấu hình.

Giới hạn: Chưa kiểm thử trên môi trường sạch hoặc máy khác.

## 10. Known Issues

Kiểm thử tổng thể repository phát hiện hai test D04
thất bại vì checksum của configs/split.yaml không khớp
với Dataset v1 Manifest.

Nguyên nhân đã truy vết: D05 cập nhật split config
sau khi D04 freeze manifest.

Vấn đề cần chủ sở hữu D04/D05 xác nhận.
Không tự thay đổi manifest trong M07.
