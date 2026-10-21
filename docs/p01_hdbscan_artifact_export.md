# P01 — Xuất common artifact cho HDBSCAN

## 1. Mục tiêu

Xuất kết quả của run HDBSCAN `minilm_title_content_eom_15_5` theo common artifact schema để loader chung và giao diện có thể đọc dữ liệu. Bundle gồm thông tin bài viết, cụm, từ khóa, bài đại diện, tọa độ 2D và metadata.

HDBSCAN được fit trên tập train. Validation dùng để chốt run đại diện. Sau đó, cả validation và test được đưa qua UMAP/HDBSCAN đã fit; test không dùng để chọn lại cấu hình.

## 2. Run và phạm vi dữ liệu

- Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Văn bản đầu vào: tiêu đề và 60 từ đầu của nội dung.
- Cấu hình run: UMAP 5 chiều, `n_neighbors=15`, `metric=cosine`, `random_state=42`; HDBSCAN `min_cluster_size=15`, `min_samples=5`, `cluster_selection_method=eom`.
- Train: 6.493 bài.
- Validation: 1.391 bài.
- Test độc lập: 1.390 bài.
- Tổng số bài trong `document_topics.csv`: 9.274 bài, với 9.274 `article_id` duy nhất.
- Số cụm thường: 101.
- Số bài outlier (`topic_id=-1`): 2.603, gồm 1.608 train, 484 validation và 511 test.
- Số bài được gán vào cụm thường: 6.671.

`topic_probability` chứa **membership strength của HDBSCAN** trong khoảng `[0, 1]`. Đây không phải xác suất đã hiệu chuẩn.

## 3. Mã nguồn và vị trí đầu ra

Exporter:

```text
experiments/hdbscan/export_p01_hdbscan.py
```

Benchmark runtime:

```text
experiments/hdbscan/benchmark_p01_hdbscan.py
results/hdbscan/benchmark_p01_runtime.json
```

Đánh giá test độc lập:

```text
experiments/hdbscan/evaluate_p01_hdbscan_test.py
results/hdbscan/test/minilm_title_content_eom_15_5_test_assignments.csv
results/hdbscan/test/minilm_title_content_eom_15_5_test_metrics.json
```

Common bundle được xuất tại:

```text
artifacts/common/hdbscan/minilm_title_content_eom_15_5/
├── manifest.json
├── model_summary.json
├── config_used.yaml
├── run_metadata.json
├── runtime_summary.json
├── document_topics.csv
├── topic_summary.csv
├── topic_keywords.csv
├── representative_docs.csv
└── projection_2d.csv
```

`manifest.json` khai báo các file bắt buộc theo common artifact schema. `run_metadata.json`, `runtime_summary.json`, `representative_docs.csv` và `projection_2d.csv` là dữ liệu bổ sung cho run và giao diện. `runtime_summary.json` liên kết benchmark train/validation với kết quả test, đồng thời giữ phạm vi của hai lần đo riêng. File benchmark gốc vẫn được lưu trong `results/hdbscan/`.

`run_metadata.json` cũng liên kết `stability_summary.csv` của E04. Bảng này mô tả 10 seed UMAP từ 42 đến 51 bằng số topic, outlier rate, coverage, ARI giữa các run và keyword Jaccard. UMAP và HDBSCAN được fit trên train; stability không dùng để chọn lại seed đại diện.

## 4. Nội dung các artifact

### `document_topics.csv`

Các cột:

```text
article_id,split,topic_id,topic_probability,membership_strength,soft_primary_topic_id,soft_primary_membership_strength,second_topic_id,second_membership_strength,top_two_share_gap,is_multi_topic,is_outlier,title,category_gold
```

File chứa 9.274 bài thuộc ba split: 6.493 train, 1.391 validation và 1.390 test. Mỗi `article_id` chỉ xuất hiện một lần.

Các trường assignment chính:

- `topic_id`: nhãn cứng do HDBSCAN gán; giá trị `-1` biểu thị outlier.
- `topic_probability`: trường tương thích với common artifact schema.
- `membership_strength`: độ mạnh thành viên HDBSCAN trong khoảng `[0, 1]`, có cùng giá trị với `topic_probability`.
- `is_outlier`: bằng `True` khi `topic_id=-1`.
- `category_gold`: nhãn danh mục tham chiếu, chỉ dùng để đánh giá ARI/NMI sau khi dự đoán; không dùng để fit UMAP hoặc HDBSCAN.

`membership_strength` không phải xác suất đã hiệu chuẩn.

#### Full membership vector và bài chồng lấn

HDBSCAN xuất full soft membership vector trên 101 cụm cho toàn bộ 9.274 bài. Các vector đầy đủ được lưu tại:

```text
results/hdbscan/membership/minilm_title_content_eom_15_5/membership_vectors.npz
```

File NPZ chứa:

- `article_ids`: thứ tự mã bài tương ứng với các hàng của ma trận.
- `splits`: split của từng bài.
- `topic_ids`: topic tương ứng với từng cột của ma trận.
- `membership_vectors`: ma trận kích thước `9274 × 101`.

Bảng tóm tắt được lưu tại:

```text
results/hdbscan/membership/minilm_title_content_eom_15_5/membership_summary.csv
```

Từ full membership vector, exporter bổ sung vào `document_topics.csv`:

- `soft_primary_topic_id`: topic có soft membership strength lớn nhất.
- `soft_primary_membership_strength`: soft membership strength lớn nhất.
- `second_topic_id`: topic có soft membership strength lớn thứ hai.
- `second_membership_strength`: soft membership strength lớn thứ hai.
- `top_two_share_gap`: chênh lệch giữa hai soft membership strength đứng đầu.
- `is_multi_topic`: cờ heuristic nhận diện bài gần ranh giới hoặc có khả năng chồng lấn chủ đề.

Quy tắc hiện tại:

```text
is_multi_topic = top_two_share_gap <= 0.10
```

Theo quy tắc này, artifact hiện có 3.732 bài được đánh dấu `is_multi_topic=True`.

`topic_id` và `soft_primary_topic_id` có ý nghĩa khác nhau. `topic_id` là nhãn cứng của HDBSCAN, còn `soft_primary_topic_id` được lấy từ thứ tự các giá trị trong full membership vector. Hai trường có thể khác nhau ở một số bài và exporter không ép chúng giống nhau.

`top_two_share_gap` và `is_multi_topic` phục vụ phân tích bài biên, bài có confidence thấp và topic chồng lấn. Ngưỡng `0.10` là quy tắc heuristic đã công bố, không phải ngưỡng xác suất được HDBSCAN hiệu chuẩn.

### `topic_summary.csv`

Các cột:

```text
topic_id,topic_name,document_count,percentage,top_keywords
```

File có 102 dòng: 101 cụm thường và một dòng outlier `-1`. Tổng `document_count` là 9.274. `percentage` là tỷ phần trong khoảng `[0, 1]` trên toàn bộ 9.274 bài. Tên cụm thường được tạo tự động từ ba từ khóa đứng đầu và chưa được duyệt thủ công.

### `topic_keywords.csv`

Các cột:

```text
topic_id,rank,keyword,weight
```

Mỗi cụm thường có 10 từ khóa TF-IDF; cụm `-1` không có danh sách từ khóa. File có 1.010 dòng cho 101 cụm. Từ khóa được trích từ văn bản đã xử lý của train, validation và test sau khi cấu hình mô hình đã được chốt; bước này chỉ phục vụ mô tả cụm, không dùng để chọn lại mô hình hoặc tính lại metric test.

### `representative_docs.csv`

Các cột:

```text
topic_id,rank,article_id,title,topic_probability,split
```

File có 505 dòng, tương ứng năm bài đại diện cho mỗi cụm thường: 397 train, 55 validation và 53 test. Trong mỗi cụm, exporter ưu tiên `membership_strength` giảm dần; nếu hòa điểm thì sắp theo `article_id`.

Cả 505 bài được chọn trong artifact hiện tại có `topic_probability=1.0` vì mỗi cụm có ít nhất năm bài đạt mức membership strength đó. Exporter giữ nguyên giá trị HDBSCAN, không tự tạo điểm mới. Khi nhiều bài cùng đạt 1.0, thứ tự `article_id` chỉ giúp kết quả ổn định; nó không bảo đảm năm bài đó đại diện tốt nhất về mặt nội dung.

### `projection_2d.csv`

File có 9.274 điểm, gồm 6.493 train, 1.391 validation và 1.390 test. UMAP 2D được fit trên train, sau đó transform validation và test lên cùng bản đồ. Tọa độ 2D dùng để trực quan hóa; HDBSCAN của run này được fit trên biểu diễn UMAP 5D, không phải tọa độ 2D này.

### Metadata

- `config_used.yaml`: cấu hình thực sự dùng cho run.
- `model_summary.json`: thông tin mô hình, số liệu bundle, runtime và các metrics.
- `run_metadata.json`: nguồn dữ liệu, split, cấu hình, phạm vi runtime và đường dẫn benchmark.
- `manifest.json`: schema và tên các file mà loader chung yêu cầu.

Loader chung đã đọc thành công bundle P01. Việc có `projection_2d.csv` và `representative_docs.csv` trong bundle không tự làm chúng xuất hiện trên Streamlit; phần giao diện cần đọc và hiển thị hai file bổ sung này.

## 5. Topic diversity

`model_summary.json` ghi `metrics.topic_diversity=0.9327`. Giá trị được tính từ `topic_keywords.csv`:

```text
topic diversity = số từ khóa khác nhau / tổng số vị trí từ khóa
                = 942 / 1.010
                ≈ 0,9327
```

Đây là metric của **bundle train + validation + test**, vì từ khóa của bundle được tạo từ cả ba split. Giá trị gần 1 cho thấy danh sách từ khóa giữa các cụm ít trùng lặp hơn; riêng metric này không chứng minh các cụm nhất quán về nội dung. Đây không phải metric riêng của split đang chọn trên giao diện.

## 6. Benchmark runtime

Runtime được đo riêng cho đúng run `minilm_title_content_eom_15_5`, với `time.perf_counter()` và lưu trong:

```text
results/hdbscan/benchmark_p01_runtime.json
```

Phạm vi đo gồm:

1. Tạo embedding MiniLM cho train.
2. Fit UMAP 5D và HDBSCAN trên train.
3. Tạo embedding MiniLM cho validation.
4. UMAP transform và `approximate_predict` cho validation.

Phép đo không gồm tải mô hình, đọc/ghi file, tính metric, trích từ khóa hoặc xuất dữ liệu giao diện và bản đồ 2D.

| Phạm vi | Thời gian |
| --- | ---: |
| Train | 226,20 giây |
| Validation | 53,31 giây |
| Train + validation | 279,50 giây |

Trong `model_summary.json`:

- `runtime_seconds` là runtime **train + validation**, khoảng 279,50 giây; không bao gồm test.
- `metrics.train_runtime_seconds` khoảng 226,20 giây.
- `metrics.val_runtime_seconds` khoảng 53,31 giây.
- Test được đo riêng: encode + transform/dự đoán khoảng 44,86 giây, trong đó `test_predict_seconds` khoảng 6,64 giây. Đây là số từ lần chạy test gần nhất; không cộng hai phép đo độc lập rồi gọi là một lần đo toàn run.

Benchmark ghi cấu hình, môi trường chạy, các giai đoạn đo và `n_repeats=1`. Đây là **một lần đo**. Chỉ so sánh tốc độ trực tiếp với các mô hình khác khi phạm vi đo và điều kiện máy tương đương.

Nếu trang so sánh mô hình đang lọc `train`, `val` hoặc `test`, giao diện cần chủ động đọc metric runtime tương ứng. `runtime_seconds` ở cấp model summary chỉ là phép đo train + validation.

Exporter đọc benchmark train + validation và kết quả test riêng, kiểm tra run, cấu hình và split trước khi ghi metadata. Không dùng `0` cho thời gian chưa đo.

## 7. Đánh giá test độc lập

Run `minilm_title_content_eom_15_5` được chốt trước khi mở test. Script test nạp UMAP và HDBSCAN đã fit trên train, kiểm tra nhãn train của model lưu sẵn khớp assignments M11, rồi encode và dự đoán 1.390 bài test. `category_gold` chỉ được ghép sau khi dự đoán. ARI/NMI tính trên toàn bộ test, bao gồm nhãn `-1`.

- Outlier test: 511/1.390 bài (**36,76%**); coverage **63,24%**.
- ARI test: **0,1244**; NMI test: **0,4214**.
- Assignments và chi tiết thời gian đo nằm ở `results/hdbscan/test/`.

Các kết quả test không dùng để chọn lại embedding, UMAP, HDBSCAN hoặc seed. Việc tạo từ khóa và bài đại diện trên cả ba split là bước mô tả artifact sau đánh giá.

## 8. Kiểm tra

Chạy đánh giá test bằng model đã lưu:

```powershell
python -X utf8 -B -m experiments.hdbscan.evaluate_p01_hdbscan_test
```

Chạy exporter:

```powershell
python -B -m experiments.hdbscan.export_p01_hdbscan
```

Chạy test P01:

```powershell
python -B -m pytest -p no:cacheprovider tests/test_p01_hdbscan_export.py -v
```

Kết quả đã ghi nhận: **12 test P01 passed; 22 test P01 + loader + schema passed**. Các test kiểm tra common bundle, ID ba split, assignments train/validation/test, `membership_strength`, `category_gold`, topic summary và keywords, metadata, representative docs, projection 2D, runtime benchmark và file tổng hợp runtime.

Có thể kiểm tra các giá trị qua loader:

```powershell
python -B -c "from src.artifacts.loader import load_artifact_bundle; b=load_artifact_bundle('artifacts/common/hdbscan/minilm_title_content_eom_15_5'); print('total:', b.model_summary.runtime_seconds); print('train:', b.model_summary.metrics['train_runtime_seconds']); print('val:', b.model_summary.metrics['val_runtime_seconds']); print('topic diversity:', b.model_summary.metrics['topic_diversity'])"
```

Để kiểm tra cùng loader và schema sau thay đổi mới, chạy:

```powershell
python -B -m pytest -p no:cacheprovider tests/test_p01_hdbscan_export.py tests/test_artifact_loader.py tests/test_artifact_schemas.py -v
```

Đã chạy bộ kết hợp sau khi xuất bundle ba split: 21 passed.

Nếu cần **đo lại** runtime trên máy hiện tại, chạy benchmark trước rồi chạy lại exporter. Kết quả đo mới sẽ phụ thuộc vào máy, môi trường và trạng thái cache:

```powershell
python -B -m experiments.hdbscan.benchmark_p01_hdbscan
python -B -m experiments.hdbscan.export_p01_hdbscan
```

## 9. Phạm vi tích hợp giao diện

P01 đã cung cấp dữ liệu HDBSCAN theo common artifact schema, gồm train, validation và test, từ khóa, bài đại diện, bản đồ 2D và metrics runtime. Trang Streamlit cần đọc các file và chọn đúng metric theo split để hiển thị.

`topic_probability` phải được chú thích là membership strength của HDBSCAN. `topic_diversity` là metric toàn run. Khi so sánh runtime giữa các mô hình, giao diện cần nêu rõ phạm vi đo vì các số hiện có có thể được thu thập theo protocol khác nhau.
