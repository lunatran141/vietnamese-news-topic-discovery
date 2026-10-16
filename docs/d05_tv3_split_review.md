# D05 – TV3 Review Train/Validation/Test Split

**Task ID:** 3.D05
**Người thực hiện:** TV3 – Tuấn
**Nhóm việc:** Data / Experimental Setup
**Dataset:** Vietnamese News Dataset v1.0
**Ngày review:** 16/10/2026
**Output:** Review Notes + Protocol Review Decision
**Trạng thái:** Technical Review PASSED – Pending Team Approval

## 1. Mục tiêu review

Task 3.D05 có nhiệm vụ rà soát độc lập giao thức phân chia dữ liệu Train/Validation/Test đã được triển khai trong D05 của nhóm.

Mục tiêu nghiệm thu:

- Xác nhận sử dụng đúng Dataset v1.0 đã được phê duyệt.
- Kiểm tra số lượng và tỷ lệ các tập dữ liệu.
- Kiểm tra phân tầng theo chuyên mục và gom nhóm theo sự kiện.
- Xác nhận không có trùng `article_id` hoặc event group giữa các tập.
- Chứng minh bộ split có thể tái tạo bằng cùng input, config và seed.
- Kiểm tra LDA không sử dụng tập Test cho tuning hoặc lựa chọn mô hình.

Phạm vi review là giao thức D05 và cách pipeline LDA M07–M09 sử dụng giao thức đó; không xây dựng lại thuật toán chia dữ liệu.

## 2. Dataset và cấu hình chính thức

Các thành phần được sử dụng:

| Thành phần | Giá trị |
|---|---|
| Dataset version | v1.0 |
| Dataset review status | approved |
| Quality gate | PASSED |
| Dataset đầu vào | `data/interim/news_articles_deduplicated.csv` |
| Split configuration | `configs/split.yaml` |
| Split manifest | `data/splits/split_manifest.csv` |
| Random seed | 42 |
| Stratification column | `category_gold` |
| ID column | `article_id` |

Tập dữ liệu sau deduplication có 9.274 bài báo.

### 2.1. Phân chia chính thức

| Tập | Số bài | Tỷ lệ thực tế | Mục đích |
|---|---:|---:|---|
| Train | 6.493 | 70,01% | Huấn luyện mô hình và fit vectorizer |
| Validation | 1.391 | 15,00% | Tuning và đánh giá sơ bộ |
| Test | 1.390 | 14,99% | Đánh giá cuối cùng |
| Tổng | 9.274 | 100% | Toàn bộ Dataset sau deduplication |

Các file tương ứng:

- `data/splits/train_ids.csv`
- `data/splits/validation_ids.csv`
- `data/splits/test_ids.csv`
- `data/splits/split_manifest.csv`

## 3. Review code chia dữ liệu

**File implementation:** `src/preprocessing/build_splits.py`

### 3.1. Stratified Group K-Fold

Implementation sử dụng `StratifiedGroupKFold` của scikit-learn.

Quy trình:

1. Tạo event group từ các cặp bài tương đồng.
2. Phân chia thành 10 folds, lấy folds 0–6 làm Train.
3. Chia phần còn lại thành Validation và Test bằng hai folds.
4. Kiểm tra event group không giao nhau giữa Train, Validation và Test.

Cách làm kết hợp hai nguyên tắc:

- **Stratified splitting:** cố gắng giữ phân bố các chuyên mục.
- **Group-aware splitting:** không chia cùng một nhóm sự kiện vào nhiều tập.

Kết quả thực nghiệm phù hợp tỷ lệ 70%/15%/15%.

**Nhận xét:** Code sử dụng cách lựa chọn số fold cố định. Mặc dù có đọc các tỷ lệ từ `configs/split.yaml`, implementation hiện tại chưa tự động chuyển đổi số fold khi thay đổi tỷ lệ. Vì vậy, thay đổi tỷ lệ cấu hình trong tương lai cần sửa và kiểm thử lại thuật toán.

### 3.2. Event grouping

File đầu vào:

`results/dedup_data/dedup_candidates_review.csv`

Đã kiểm tra file tồn tại, dung lượng 237.382 bytes.

Các tham số:

- Content similarity threshold: 0.60.
- Title similarity threshold: 0.60.
- Thuật toán: connected components.

Kết quả chạy thực tế ngày 16/10/2026:

| Chỉ số | Giá trị |
|---|---:|
| Cặp bài tương đồng cao | 168 |
| Tổng event groups | 9.114 |
| Cụm nhiều bài | 141 |

Implementation kiểm tra không có group ID xuất hiện đồng thời ở hai tập dữ liệu.

**Giới hạn:** Nếu thiếu file candidates, code đưa ra cảnh báo và chuyển sang nhóm độc lập. Do đó, file candidates phải được giữ đầy đủ trong những lần tái tạo Dataset v1.0.

Việc không trùng event group xác nhận tính độc lập theo các cặp ứng viên được phát hiện và ngưỡng đã chọn; không chứng minh tuyệt đối rằng mọi bài báo cùng sự kiện trong thực tế đều đã được nhận diện.

## 4. Kiểm thử D05

**File:** `tests/test_splits.py`

Bộ test kiểm tra:

1. Các file split và cấu hình tồn tại.
2. Tỷ lệ cấu hình, random seed và chính sách Test lock.
3. Bao phủ đủ 9.274 bài.
4. Không trùng `article_id` giữa Train, Validation và Test.
5. File split IDs chỉ chứa `article_id`.
6. Phân bố `category_gold` không vượt ngưỡng chênh lệch cho phép theo bài kiểm thử.
7. Không có event group giao nhau giữa ba tập.

Kết quả đã ghi nhận:

- D05: **7/7 tests passed**.
- Chạy gộp D04 và D05: **13/13 tests passed**.

Lệnh kiểm thử:

`python -m pytest tests/test_freeze_manifest.py tests/test_splits.py -q`

Kết quả:

`13 passed in 20.57s`

Đây là kết quả kiểm thử thành công trong môi trường hiện tại.

## 5. Kiểm tra khả năng tái lập

### 5.1. Phương pháp

TV3 đã chạy lại D05 với cùng:

- Dataset sau deduplication.
- `configs/split.yaml`.
- Random seed 42.
- File event candidates chính thức.
- Môi trường Python hiện tại.

Kết quả được ghi sang thư mục tạm:

`$env:TEMP\d05_tv3_repro_check\`

Điều này tránh ghi đè các file split chính thức đã được khóa.

Lệnh thực hiện:

```powershell
$auditRoot = Join-Path $env:TEMP "d05_tv3_repro_check"
$splitOut = Join-Path $auditRoot "splits"
$reportOut = Join-Path $auditRoot "reports"

python -m src.preprocessing.build_splits `
    --config configs/split.yaml `
    --input data/interim/news_articles_deduplicated.csv `
    --output-dir "$splitOut" `
    --results-dir "$reportOut"
```

### 5.2. Kết quả tái tạo

Pipeline chạy thành công và tạo lại:

- Train: 6.493 bài (70,01%).
- Validation: 1.391 bài (15,00%).
- Test: 1.390 bài (14,99%).

### 5.3. So sánh SHA256

| File | SHA256 chính thức | Kết quả |
|---|---|---|
| `train_ids.csv` | `206D15CEE6E9D8565D57988FCDF2F8B43C1E27A0864DF3907477DE2964C3125C` | MATCH |
| `validation_ids.csv` | `F8A93AC72759D0F986BCE35AF5D1B3CF959946BED60B923D67083D0F93FAAF87` | MATCH |
| `test_ids.csv` | `1FA4F4E906F99241806126EBD607DB7EC9DA50CB3225FF4E7375DBF7FCCB9E31` | MATCH |
| `split_manifest.csv` | `387E815AF3CB6C0F20967CFA6AEFFBD61F2B447D201F3EEDA1173EEDDA8E1455` | MATCH |

**Kết luận:** 4/4 file tái tạo có checksum trùng hoàn toàn với bộ split chính thức.

Điều này xác nhận khả năng tái lập trong cùng môi trường, với cùng dữ liệu và cấu hình. Chưa có kết quả kiểm thử độc lập trên hệ điều hành hoặc phiên bản thư viện khác.

## 6. Kiểm tra Test Set Isolation đối với LDA

Các script đã rà soát:

- `experiments/lda/run_m07_lda.py`
- `experiments/lda/run_m08_lda_experiments.py`
- `experiments/lda/run_m09_lda_stability.py`

### 6.1. M07 – LDA Pipeline

Qua kiểm tra code:

- Nạp Train và Validation từ manifest.
- Fit CountVectorizer chỉ trên Train.
- Huấn luyện LDA trên Train.
- Transform Validation bằng vectorizer đã fit.
- Không phát hiện sử dụng Test trong quá trình huấn luyện.

### 6.2. M08 – LDA Experiments

Qua kiểm tra code:

- Fit Bag-of-Words trên Train.
- Huấn luyện các giá trị K trên Train.
- Tính Coherence, Diversity và lựa chọn bài đại diện trên Validation.
- Không phát hiện sử dụng Test để thử hoặc chọn K.

### 6.3. M09 – LDA Stability

Qua kiểm tra code:

- Fit Bag-of-Words trên Train.
- Huấn luyện LDA với nhiều random seeds trên Train.
- Tính độ ổn định document-topic trên Validation.
- Có kiểm tra `evaluation_split` phải là Validation.
- Không phát hiện sử dụng Test để đánh giá stability.

### 6.4. Kết luận Test isolation

**Không phát hiện Test leakage trong ba script M07–M09 được rà soát.**

Phạm vi kết luận chỉ áp dụng cho LDA. Việc xác minh toàn bộ năm phương pháp thuộc quy trình đánh giá chung của nhóm.

## 7. Approved Protocol – Đề xuất phê duyệt

TV3 xác nhận các thành phần kỹ thuật sau phù hợp để sử dụng cho LDA:

| Thành phần | Protocol |
|---|---|
| Dataset | Dataset v1.0 – approved |
| Train IDs | `data/splits/train_ids.csv` |
| Validation IDs | `data/splits/validation_ids.csv` |
| Test IDs | `data/splits/test_ids.csv` |
| Random seed cho split | 42 |
| Stratification | `category_gold` |
| Event grouping | Connected components, threshold 0.60 |
| Fit model/vectorizer | Chỉ trên Train |
| Hyperparameter tuning | Validation |
| Model selection | Validation |
| Final evaluation | Test đã khóa |
| Reproducibility | 4/4 SHA256 MATCH |

**Technical Review Decision:** PASSED.

**Recommendation:** Approve D05 v1.0 protocol for LDA experiments.

**Official Team Approval:** Pending confirmation from the responsible reviewer or group.

Việc Dataset đã có trạng thái `approved` không đồng nghĩa TV1 đã phê duyệt riêng tài liệu review này.

## 8. Những điểm cần theo dõi

| Vấn đề | Mức ảnh hưởng | Hướng xử lý |
|---|---|---|
| Số folds cố định | Khi thay đổi tỷ lệ split | Giữ nguyên cấu hình v1.0; review lại nếu thay tỷ lệ |
| File event candidates có thể thiếu | Ảnh hưởng chống event leakage | Yêu cầu file candidates tồn tại khi tái tạo |
| Test lock chủ yếu được khai báo ở config | Có thể bị script khác sử dụng sai | Audit riêng từng pipeline |
| Chưa thử trên môi trường thứ hai | Chưa chứng minh tái lập đa môi trường | Bổ sung khi nhóm cần kiểm chứng liên máy |
| Matplotlib cảnh báo `set_ticklabels()` | Ảnh hưởng trình bày biểu đồ | Có thể sửa trong task bảo trì, không ảnh hưởng split IDs |

## 9. Kết luận

TV3 đã hoàn thành các hoạt động review kỹ thuật của Task 3.D05:

1. Đọc implementation và unit tests của D05.
2. Kiểm tra file event candidates.
3. Xác minh số lượng, phân bố và tính độc lập của các split theo kiểm thử.
4. Chạy lại D05 ở thư mục tạm.
5. Đối chiếu và xác nhận cả bốn checksum khớp.
6. Rà soát cách LDA M07–M09 sử dụng Train/Validation/Test.
7. Đưa ra quyết định review kỹ thuật và đề xuất phê duyệt protocol.

**Kết luận kỹ thuật:** PASSED.

**Trạng thái Task 3.D05:** Review Notes và Protocol Review Decision đã được chuẩn bị để nộp; chờ review/merge và xác nhận phê duyệt chính thức từ nhóm.

Không thay đổi dữ liệu, cấu hình hoặc bộ split chính thức trong task này.
