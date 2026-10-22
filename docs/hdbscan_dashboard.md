# HDBSCAN Dashboard

## 1. Mục đích

Dashboard HDBSCAN hiển thị kết quả của run đại diện:

```text
minilm_title_content_eom_15_5
```

Common artifact được đọc từ:

```text
artifacts/common/hdbscan/minilm_title_content_eom_15_5
```

Dashboard hỗ trợ:

- Tổng quan chất lượng và runtime.
- Coverage và outlier theo split.
- Full membership vector và bài chồng lấn.
- Bản đồ UMAP 2D.
- So sánh MiniLM–MPNet và EOM–Leaf.
- Stability theo UMAP seed.
- Phân tích hard cases.
- Bài đại diện của từng topic.
- Human evaluation.

## 2. Các file triển khai

Các file chính:

```text
app/hdbscan_dashboard.py
app/data_service.py
app/streamlit_app.py
tests/test_hdbscan_dashboard.py
```

Thư viện Altair được khai báo trong:

```text
requirements.txt
```

## 3. Chạy dashboard

Từ thư mục gốc của dự án, chạy:

```powershell
python -B -m streamlit run app/streamlit_app.py
```

Trong sidebar, chọn:

```text
HDBSCAN · minilm_title_content_eom_15_5
```

## 4. Chất lượng và runtime

Dashboard hiển thị:

- Số bài.
- Số topic.
- Test coverage.
- Test ARI.
- Test NMI.
- Topic diversity.
- Coherence.
- Fit time.
- Pipeline runtime.
- Runtime scope.

Dữ liệu runtime được đọc từ:

```text
artifacts/common/hdbscan/minilm_title_content_eom_15_5/runtime_summary.json
```

### 4.1. Fit time

HDBSCAN không lưu trực tiếp `fit_time_seconds`. Dashboard tính:

```text
fit_time =
    umap_fit_seconds
    + hdbscan_fit_seconds
```

Với run đại diện, fit time xấp xỉ:

```text
23.63 giây
```

### 4.2. Runtime scope

Runtime train/validation và runtime test được đo ở hai lần chạy riêng biệt. Dashboard không cộng hai runtime này thành một phép đo end-to-end.

Dashboard hiển thị nguyên văn `runtime_scope` để người dùng biết các bước được tính trong runtime.

### 4.3. Coherence

Dashboard đọc `coherence_summary.json` của run HDBSCAN. NPMI được tính
riêng trên train, val, test và all, dùng 10 từ khóa cho mỗi trong 101 cụm.
Từ khóa được rút từ toàn bộ train + val + test, nên đây là phép đo mô tả
theo split, không phải đánh giá test độc lập. Từ khóa vắng mặt và điểm
của từng cụm được ghi trong artifact coherence.

Giá trị thiếu không được đổi thành `0`.

## 5. Coverage và outlier

Dashboard tính coverage theo từng split:

```text
coverage =
    assigned_count / document_count
```

Outlier rate được tính:

```text
outlier_rate =
    outlier_count / document_count
```

Một bài được xem là outlier khi:

```text
topic_id = -1
```

Dashboard hiển thị riêng các split:

```text
train
val
test
```

## 6. Outlier explorer

Outlier explorer hỗ trợ:

- Lọc theo split.
- Tìm theo `article_id`.
- Tìm theo tiêu đề.
- Sắp xếp theo `membership_strength`.
- Tải danh sách đang lọc dưới dạng CSV.

Các trường có thể hiển thị:

```text
article_id
split
title
topic_id
membership_strength
soft_primary_topic_id
soft_primary_membership_strength
second_topic_id
second_membership_strength
top_two_share_gap
is_multi_topic
category_gold
```

`membership_strength` nằm trong khoảng `[0, 1]`, nhưng không phải xác suất đã hiệu chuẩn.

## 7. Full membership vector

Full membership vector của 9.274 bài trên 101 topic được lưu tại:

```text
results/hdbscan/membership/minilm_title_content_eom_15_5/membership_vectors.npz
```

Bảng tóm tắt được lưu tại:

```text
results/hdbscan/membership/minilm_title_content_eom_15_5/membership_summary.csv
```

Các trường được đưa vào `document_topics.csv`:

```text
soft_primary_topic_id
soft_primary_membership_strength
second_topic_id
second_membership_strength
top_two_share_gap
is_multi_topic
```

### 7.1. Quy tắc multi-topic

Quy tắc hiện tại:

```text
is_multi_topic =
    top_two_share_gap <= 0.10
```

Đây là heuristic phục vụ phân tích bài biên và bài có khả năng chồng lấn chủ đề. Đây không phải ngưỡng xác suất được HDBSCAN hiệu chuẩn.

Dashboard hỗ trợ:

- Lọc theo split.
- Chỉ hiển thị bài multi-topic.
- Điều chỉnh top-2 gap tối đa.
- Tìm theo article ID hoặc tiêu đề.
- Sắp xếp bài theo top-2 gap tăng dần.

## 8. Bản đồ HDBSCAN 2D

Tọa độ được đọc từ:

```text
artifacts/common/hdbscan/minilm_title_content_eom_15_5/projection_2d.csv
```

`projection_2d.csv` chứa:

```text
article_id
split
topic_id
x
y
```

Khi hiển thị, dashboard ghép thêm metadata theo `article_id` từ `document_topics.csv` và split manifest.

Tooltip của bản đồ gồm:

- Article ID.
- Tiêu đề.
- Nguồn báo.
- Ngày đăng.
- Split.
- Topic.
- Membership strength.
- Top-2 gap.

Quy ước hiển thị:

- Màu biểu thị topic.
- Ký hiệu biểu thị split.
- Outlier có màu xám.

UMAP 2D chỉ phục vụ trực quan hóa. Tọa độ 2D không phải metric chất lượng và không phải không gian embedding dùng trực tiếp để fit HDBSCAN.

## 9. So sánh cấu hình HDBSCAN

Dashboard đọc kết quả từ:

```text
results/hdbscan/m11_summary.csv
results/hdbscan/validation/m11_validation_metrics.csv
```

Bốn cấu hình được so sánh:

```text
MiniLM + EOM
MiniLM + Leaf
MPNet + EOM
MPNet + Leaf
```

Các trường hiển thị:

- Embedding run.
- Experiment.
- Min cluster size.
- Min samples.
- Số topic trên train.
- Train outlier rate.
- Validation coverage.
- Validation outlier rate.
- Validation ARI.
- Validation NMI.

Các metric so sánh thuộc validation. UMAP và HDBSCAN được fit trên train; validation chỉ được transform và dự đoán.

Tập test không được sử dụng để chọn lại cấu hình.

## 10. Stability

Stability được đo trên 10 UMAP seed:

```text
42, 43, 44, 45, 46, 47, 48, 49, 50, 51
```

Bảng tổng hợp được đọc từ:

```text
results/hdbscan/stability/minilm_title_content_eom_15_5_seeds42_51/stability_summary.csv
```

Các trường gồm:

```text
seed
topic_count
outlier_rate
coverage
ari_between_runs
keyword_jaccard
```

Ý nghĩa:

- `topic_count`: số topic của lần chạy.
- `outlier_rate`: tỷ lệ bài có nhãn `-1`.
- `coverage`: tỷ lệ bài được gán vào cluster.
- `ari_between_runs`: mức ổn định assignment giữa các lần chạy.
- `keyword_jaccard`: mức giống nhau giữa các bộ từ khóa sau khi ghép topic.

ARI giữa các run bao gồm nhãn outlier `-1`.

Keyword Jaccard sử dụng top-10 từ khóa TF-IDF sau khi ghép topic giữa các run.

## 11. Hard cases

Dashboard hỗ trợ hai nhóm hard cases.

### 11.1. Error analysis đã duyệt

Dữ liệu được đọc từ:

```text
results/hdbscan/error_analysis/hdbscan_error_cases.csv
```

Dashboard hỗ trợ:

- Lọc theo nhóm lỗi.
- Lọc theo split.
- Xem embedding run.
- Xem experiment.
- Xem topic.
- Xem tiêu đề bài.
- Xem bằng chứng của reviewer.
- Xem diễn giải lỗi.

Các nhóm lỗi có thể gồm bài trùng, topic quá rộng, bài lệch trọng tâm hoặc các trường hợp cần xem xét thủ công.

### 11.2. Bài không ổn định theo seed

Dữ liệu được đọc từ:

```text
results/hdbscan/stability/minilm_title_content_eom_15_5_seeds42_51/train_transition_cases.csv
results/hdbscan/stability/minilm_title_content_eom_15_5_seeds42_51/val_transition_cases.csv
```

Một transition xảy ra khi bài thay đổi giữa:

```text
cluster → outlier
outlier → cluster
```

Các trường tổng hợp gồm:

```text
transition_count
noise_to_cluster
cluster_to_noise
compared_seed_pairs
instability_rate
```

Với 10 seed, tổng số cặp seed là:

```text
10 × 9 / 2 = 45
```

Instability rate được tính:

```text
instability_rate =
    transition_count / compared_seed_pairs
```

Giá trị càng cao cho thấy assignment của bài càng nhạy với seed.

Dashboard hỗ trợ:

- Chọn số lần chuyển trạng thái tối thiểu.
- Lọc theo train hoặc validation.
- Xem title, topic và membership.
- Xem top-2 gap.
- Xem cờ multi-topic và outlier.

## 12. Bài đại diện

Bài đại diện được đọc từ:

```text
artifacts/common/hdbscan/minilm_title_content_eom_15_5/representative_docs.csv
```

Dashboard cho phép chọn topic và hiển thị các bài đại diện được sắp xếp theo `rank`.

Các bài đại diện được chọn từ train, validation và test sau khi cấu hình đại diện đã được khóa.

## 13. Human evaluation

Dashboard dự kiến đọc bảng chuẩn hóa:

```text
results/hdbscan/human_eval/normalized/human_eval_reviews.csv
```

Các trường đánh giá chính:

```text
outlier_correct
topic_consistent
is_boundary_or_mixed
reviewer
review_status
notes
```

Hiện quá trình đánh giá độc lập của R1 và R2 chưa hoàn thành. Dashboard hiển thị:

```text
N/A — đang chờ kết quả đánh giá độc lập từ R1 và R2.
```

Dashboard không tự động suy ra hoặc tạo số giả cho các nhãn đánh giá chủ quan.

Sau khi R1 và R2 hoàn thành, dữ liệu sẽ được chuẩn hóa và tab Human evaluation sẽ hiển thị:

- Số phiếu.
- Số người đánh giá.
- Số phiếu hoàn thành.
- Chi tiết từng phiếu đánh giá.

## 14. Xử lý metric thiếu

Dashboard sử dụng quy ước:

```text
N/A
```

khi metric chưa được đo.

Dashboard không thay metric thiếu bằng:

```text
0
```

Các metric không có ý nghĩa đối với HDBSCAN cần được ghi rõ là chưa đo hoặc không áp dụng.

## 15. Kiểm thử

File kiểm thử:

```text
tests/test_hdbscan_dashboard.py
```

Kiểm thử bao gồm:

- Runtime và fit time.
- Coverage theo split.
- Metadata phục vụ bản đồ 2D.
- Stability schema.
- So sánh bốn cấu hình MiniLM/MPNet và EOM/Leaf.
- Error-analysis hard cases.
- Transition hard cases theo seed.

Chạy riêng test dashboard:

```powershell
python -B -m pytest -p no:cacheprovider `
    tests/test_hdbscan_dashboard.py -v
```

Chạy bộ test tích hợp:

```powershell
python -B -m pytest -p no:cacheprovider `
    tests/test_hdbscan_dashboard.py `
    tests/test_p01_hdbscan_export.py `
    tests/test_e04_hdbscan_stability.py `
    tests/test_artifact_loader.py `
    tests/test_artifact_schemas.py -v
```

Kết quả kiểm tra hiện tại:

```text
33 passed
```

## 16. Trạng thái hoàn thành

Các phần đã hoàn thành:

- Common artifact HDBSCAN.
- Train, validation và test assignments.
- Runtime và runtime scope.
- Test ARI/NMI.
- Topic diversity.
- Stability theo seed.
- Full membership vector.
- Coverage theo split.
- Outlier explorer.
- Bản đồ 2D.
- So sánh MiniLM–MPNet.
- So sánh EOM–Leaf.
- Stability hard cases.
- Error-analysis hard cases.
- Kiểm thử dashboard.

Phần đang chờ:

- Kết quả human evaluation hoàn chỉnh từ R1 và R2.
- Chuẩn hóa kết quả thành `human_eval_reviews.csv`.
