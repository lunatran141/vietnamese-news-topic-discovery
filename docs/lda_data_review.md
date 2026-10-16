
# TV3 – Validation Notes cho Dataset v1.0 và LDA

## 1. Thông tin task

- Thành viên: TV3 – Tuấn
- Task: 3.1 – D04: Rà soát Dataset v1 và preprocessing variants
- Output yêu cầu: Validation Notes
- Tiêu chí hoàn thành: Sử dụng đúng Dataset snapshot và cấu hình đã được nhóm phê duyệt
- Dataset: v1.0
- Ngày cập nhật: 16/10/2026
- Trạng thái: Đã hoàn thành kiểm tra kỹ thuật, chờ review tài liệu

## 2. Mục tiêu

Xác nhận pipeline LDA sử dụng đúng Dataset v1.0, đúng biến thể tiền xử lý và đúng phân chia Train/Validation/Test.

Các nội dung rà soát bao gồm:

1. Nguồn dữ liệu và tính toàn vẹn.
2. Schema và số lượng bài báo.
3. Các biến thể tiền xử lý P0–P3.
4. Biến thể đầu vào mặc định của LDA.
5. Phân chia Train/Validation/Test và phòng tránh data leakage.
6. Khả năng tái lập và kết quả kiểm thử.

Không chỉnh sửa dữ liệu nguồn, article_id hoặc cấu hình chính thức của nhóm trong task này.

## 3. Thông tin Dataset Snapshot

Nguồn thông tin:
`data/processed/dataset_manifest.json`

| Thuộc tính | Giá trị |
|---|---|
| Dataset version | v1.0 |
| Ngày tạo | 11/10/2026 |
| Review status | approved |
| Quality gate | PASSED |
| Source Git commit | 8e36cc4adeab60cbeea6a3b4360364e08b330154 |
| Random seed | 42 |

### 3.1. Dữ liệu gốc

- File: `data/raw/news_articles.csv`
- Số bài báo: 9.304
- Số cột: 7
- Dung lượng: 42.506.167 bytes
- SHA256:
  `548b35953368857373e8f26639531b96ab83b742ed6432bff0b6462661b5077a`

### 3.2. Dữ liệu sau Deduplication

- File: `data/interim/news_articles_deduplicated.csv`
- Số bài báo: 9.274
- Số bài bị loại: 30
- SHA256:
  `8d4ebfade830c4403d90eba8389407232437327f30e51125938b4d7d6e8ce9fa`

Deduplication giúp giảm sự lặp lại nội dung, hạn chế ảnh hưởng của các bài trùng đến thống kê từ vựng và mô hình chủ đề.

## 4. Schema dữ liệu

Dataset sau tiền xử lý có 10 cột:

- `article_id`
- `title`
- `content`
- `processed_title`
- `processed_content`
- `text`
- `published_at`
- `source`
- `url`
- `category_gold`

Trong đó:

- `article_id`: Định danh duy nhất của bài báo.
- `text`: Cột văn bản đầu vào được khai báo cho mô hình.
- `category_gold`: Nhãn tham chiếu, không sử dụng để huấn luyện LDA theo cơ chế supervised learning.
- `source`, `published_at`: Metadata phục vụ kiểm tra và phân tích.

Theo manifest, công thức trường `text` được khai báo:

`title + ' ' + processed_content`

Pipeline M07 sử dụng `load_split_dataset()` và `build_article_text()` để nạp và chuẩn bị văn bản nhất quán.

## 5. Rà soát các biến thể tiền xử lý

| Biến thể | Số bài | Mô tả |
|---|---:|---|
| P0 – Minimal | 9.274 | Làm sạch HTML và khoảng trắng, giữ nguyên phần lớn văn bản |
| P1 – Normalized | 9.274 | Chuẩn hóa chữ thường, Unicode và loại một số thông tin nhiễu |
| P2 – Segmented | 9.274 | Phân đoạn từ tiếng Việt bằng PyVi |
| P3 – No Stopwords | 9.274 | Loại từ dừng theo danh sách của nhóm |

### 5.1. P0 – Minimal

File: `data/processed/p0_minimal.csv`

SHA256:
`8565456f294e2fdec1535070105989b43ecd15d42abcde8e80e67eafa1ccc31e`

### 5.2. P1 – Normalized

File: `data/processed/p1_normalized.csv`

SHA256:
`5fbacd7b0d23b21eeba5f1a4b4f9a4ae779b0a13b57b9bb0dadd23c29cba0f86`

### 5.3. P2 – Segmented

File: `data/processed/p2_segmented.csv`

SHA256:
`be99ead1f6608976c86b172ad885e2676a0d25f6f55e2d732a9f15cf62a2add5`

P2 là biến thể mặc định được manifest khuyến nghị cho LDA.

LDA sử dụng biểu diễn Bag-of-Words, nên việc phân đoạn đúng từ tiếng Việt rất quan trọng.

Ví dụ minh họa:

- Trước phân đoạn: `ngân hàng trung ương`
- Sau phân đoạn: `ngân_hàng trung_ương`

Cách biểu diễn này giúp CountVectorizer xem các từ ghép như một đơn vị từ vựng.

Đây là ví dụ về nguyên lý phân đoạn từ, không phải kết quả trích trực tiếp từ một dòng Dataset.

### 5.4. P3 – No Stopwords

File: `data/processed/p3_no_stopwords.csv`

SHA256:
`663a34f309ef1f6d8d2dc61a68fd041d5f7aac71aa930dccb74841847a7d6710`

P3 được quy định là biến thể đối chứng (ablation) cho LDA.

Việc loại từ dừng có thể làm thay đổi Coherence, Diversity và cách diễn giải topic. Cần thực nghiệm để xác định ảnh hưởng thực tế, không mặc định P3 luôn tốt hơn P2.

### 5.5. Tokenizer và Stopwords

- Tokenizer: PyVi
- Phiên bản: 0.1.1
- Module: `pyvi.ViTokenizer`
- Danh sách stopwords: `configs/vietnamese_stopwords.txt`
- Số mục stopwords: 391

## 6. Xác minh cấu hình

Các cấu hình chính thức:

- `configs/preprocessing.yaml`
- `configs/vietnamese_stopwords.txt`
- `configs/split.yaml`

SHA256 của `configs/split.yaml` trong manifest hiện hành:

`2dd5d18e2fbfe1b046aead044dc033a737cdf0f373fa254ea542fcf85ddcc560`

Sau khi nhóm đồng bộ manifest, kiểm tra Dataset Freeze đã xác nhận cấu hình này khớp với manifest hiện tại.

## 7. Rà soát D05 – Dataset Splits

Cấu hình sử dụng:

`configs/split.yaml`

Kết quả phân chia:

| Tập dữ liệu | Số bài | Tỷ lệ cấu hình |
|---|---:|---:|
| Train | 6.493 | 70% |
| Validation | 1.391 | 15% |
| Test | 1.390 | 15% |
| Tổng | 9.274 | 100% |

Các file đầu ra:

- `data/splits/train_ids.csv`
- `data/splits/validation_ids.csv`
- `data/splits/test_ids.csv`
- `data/splits/split_manifest.csv`

### 7.1. Stratified Splitting

D05 sử dụng `category_gold` để duy trì sự phân bố tương đối của các nhóm chuyên mục giữa Train, Validation và Test.

Việc này giúp hạn chế sự mất cân bằng về thành phần chủ đề giữa các tập.

### 7.2. Group-aware Splitting

Cấu hình có bật event grouping:

- Algorithm: `connected_components`
- Content similarity threshold: 0.60
- Title similarity threshold: 0.60

Mục đích là hạn chế các bài báo liên quan cùng sự kiện xuất hiện ở cả Train và Test.

Theo kết quả kiểm thử D05, bài kiểm tra chống event leakage giữa Train và Test đã passed.

### 7.3. Chính sách khóa Test

Cấu hình:

- `lock_test_set: true`
- `allow_hyperparameter_tuning_on_test: false`
- `allow_model_selection_on_test: false`

Quy tắc sử dụng:

- Train: Fit vectorizer và huấn luyện mô hình.
- Validation: Thực nghiệm số topic, đánh giá và lựa chọn cấu hình.
- Test: Chỉ dùng cho đánh giá cuối cùng.

Đây là nguyên tắc phòng tránh Data Leakage và đảm bảo đánh giá khách quan trong Machine Learning.

## 8. Kiểm tra tính toàn vẹn

### 8.1. Dataset Freeze Verification

Lệnh:

`python -m src.preprocessing.freeze_dataset --verify`

Kết quả thực tế:

`TẤT CẢ KIỂM TRA ĐỀU KHỚP! Dataset v1 đạt chuẩn Freeze.`

Bao gồm:

- Raw data checksum: Khớp.
- Preprocessing config: Khớp.
- Stopwords config: Khớp.
- Split config: Khớp.
- P0, P1, P2, P3: Khớp.

### 8.2. D04 Tests

Lệnh:

`python -m pytest tests/test_freeze_manifest.py -v`

Kết quả: 6/6 passed.

### 8.3. D05 Tests

Lệnh:

`python -m pytest tests/test_splits.py -v`

Kết quả: 7/7 passed.

### 8.4. LDA Tests

Lệnh:

`python -m pytest tests/test_lda_pipeline.py tests/test_lda_metrics.py tests/test_lda_stability.py -q`

Kết quả: 25/25 passed.

### 8.5. Kiểm thử toàn repository

Ở lần kiểm tra sau khi cập nhật D04/D05, toàn repository đạt:

`133 passed, 4 warnings`

Các warnings liên quan đến khác biệt phiên bản scikit-learn khi nạp artifacts KMeans và TF-IDF.

Đây là kết quả ghi nhận tại thời điểm kiểm tra, không phải khẳng định trạng thái hiện tại của mọi commit mới.

## 9. Liên hệ với pipeline LDA

### M07 – BoW + LDA Pipeline

- Sử dụng Dataset v1.0 với P2.
- Fit CountVectorizer trên Train.
- Huấn luyện LDA trên Train.
- Suy luận phân phối topic cho Validation.

### M08 – LDA Topic Experiments

- Thử nghiệm K = 4, 6, 8, 10, 12, 15, 20.
- Đánh giá Coherence và Topic Diversity.
- Trích xuất Top-10 keywords và các bài báo đại diện.
- Sử dụng Validation cho đánh giá.

### M09 – LDA Stability

- Sử dụng K = 15.
- Huấn luyện 5 random seeds.
- So sánh 10 cặp seed.
- Áp dụng Hungarian Matching, Jaccard và Cosine Similarity.

Kết quả:

- Mean Keyword Jaccard: 0.455685
- Standard Deviation Jaccard: 0.036020
- Mean Document-topic Cosine: 0.681273
- Standard Deviation Cosine: 0.051569

Các task M07, M08 và M09 đã merge vào `develop`.

## 10. Checklist nghiệm thu TV3

| Tiêu chí | Kết quả |
|---|---|
| Sử dụng Dataset v1.0 đã phê duyệt | Đạt |
| Dùng đúng preprocessing variant cho LDA | Đạt |
| Raw và các biến thể khớp manifest | Đạt |
| Split Train/Validation/Test theo protocol | Đạt |
| Kiểm thử không trùng Train/Validation | Đạt theo kiểm thử LDA |
| D05 kiểm tra chống event leakage Train/Test | Đạt theo kiểm thử |
| Không sử dụng Test để tuning LDA | Đạt theo pipeline đã triển khai |
| Có bằng chứng chạy verification | Đạt |
| Có tài liệu Validation Notes cập nhật | Đã soạn; chờ review và merge |

Một số tiêu chí kiểm tra định tính bằng mẫu bài báo trước/sau preprocessing chưa được ghi nhận bằng bảng ví dụ trực tiếp từ Dataset. Nếu nhóm yêu cầu minh chứng này, cần bổ sung sau khi trích mẫu thực tế.

## 11. Kết luận

Dataset v1.0 hiện hành đã được nhóm phê duyệt theo manifest và vượt qua các kiểm tra tính toàn vẹn D04/D05.

TV3 đã xác minh:

1. Nguồn dữ liệu và metadata Dataset.
2. Các biến thể tiền xử lý P0–P3.
3. Cấu hình P2 dành cho LDA.
4. Phân chia Train/Validation/Test.
5. Chính sách khóa Test và phòng tránh Data Leakage.
6. Sự tương thích với pipeline M07–M09.

Kết quả xác minh kỹ thuật cho thấy Dataset snapshot và các cấu hình đang dùng phù hợp để phục vụ thí nghiệm LDA.

**Trạng thái task 3.1:** Đã thực hiện kiểm tra và chuẩn bị đầy đủ Validation Notes cập nhật, chờ review/merge theo quy trình nhóm.
