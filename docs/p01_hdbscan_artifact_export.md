# P01 — Chuẩn hóa artifact HDBSCAN cho giao diện

## 1. Mục tiêu

Task P01 chuẩn hóa kết quả HDBSCAN theo common artifact schema của nhóm để loader chung, API và Streamlit có thể sử dụng. Phần đóng góp này xuất kết quả của run `minilm_title_content_eom_15_5` cùng dữ liệu bổ sung cho bản đồ 2D và bài báo đại diện.

Mô hình HDBSCAN được fit trên **train**. Nhãn của bài **validation** được gán bằng pipeline đã fit trên train; không fit lại HDBSCAN trên validation. Bundle xuất kết quả của cả hai split để phục vụ hiển thị.

## 2. Run và dữ liệu đầu vào

- Embedding run: `minilm_title_content`.
- Biểu diễn: MiniLM, dùng tiêu đề và 60 từ đầu của nội dung đã xử lý.
- HDBSCAN experiment: `eom_15_5`, với `min_cluster_size=15`, `min_samples=5` và cách chọn cụm `eom`.
- Train: 6.493 bài.
- Validation: 1.391 bài.
- Tổng số bài trong bundle: 7.884.

Các nguồn chính:

- `results/hdbscan/minilm_title_content_assignments.csv`: nhãn và membership strength trên train.
- `results/hdbscan/validation/minilm_title_content_eom_15_5_val_assignments.csv`: nhãn và membership strength trên validation.
- `results/hdbscan/m11_summary.csv`: thông tin run và `config_hash`.
- `data/processed/p0_minimal.csv`: văn bản đã xử lý và tiêu đề.
- `data/splits/split_manifest.csv`: xác định split theo `article_id`.
- `artifacts/embeddings/m10_v1_train_minilm.npz`: embedding train dùng cho phép chiếu 2D.

Exporter đối chiếu `article_id`, split, run và `config_hash` trước khi ghi kết quả.

## 3. Mã nguồn và thư mục đầu ra

Mã xuất artifact:

```text
experiments/hdbscan/export_p01_hdbscan.py
```

Thư mục đầu ra:

```text
artifacts/common/hdbscan/minilm_title_content_eom_15_5/
```

Bundle hiện có chín file:

```text
manifest.json
model_summary.json
config_used.yaml
run_metadata.json
document_topics.csv
topic_summary.csv
topic_keywords.csv
representative_docs.csv
projection_2d.csv
```

`manifest.json` khai báo sáu file thuộc common artifact schema v1.0: chính nó, `model_summary.json`, `config_used.yaml`, `document_topics.csv`, `topic_summary.csv` và `topic_keywords.csv`. Ba file `run_metadata.json`, `representative_docs.csv` và `projection_2d.csv` là dữ liệu bổ sung. Loader chung hiện đọc các bảng chuẩn; ứng dụng cần đọc riêng các CSV bổ sung khi dùng chúng.

## 4. Nội dung từng artifact

### `document_topics.csv`

Mỗi dòng ứng với một `article_id`. File chứa 7.884 bài, gồm 6.493 bài train và 1.391 bài validation, với các cột:

```text
article_id,split,topic_id,topic_probability,is_outlier,title
```

`topic_id` lấy từ `cluster_id` của HDBSCAN. Bài có `topic_id=-1` được đánh dấu `is_outlier=True`.

`topic_probability` chứa `membership_strength` gốc của HDBSCAN trong khoảng `[0, 1]`. Đây là độ mạnh thành viên cụm, **không phải xác suất đã hiệu chuẩn**.

Validation được gộp trực tiếp vào `document_topics.csv`; bundle không sử dụng file `validation_document_topics.csv` riêng.

### `topic_summary.csv`

File có 102 dòng: 101 cụm thường và một dòng outlier `-1`. Các cột:

```text
topic_id,topic_name,document_count,percentage,top_keywords
```

`document_count` và `percentage` được tính trên **toàn bộ 7.884 bài train + validation**. `percentage` được lưu dưới dạng tỷ phần từ `0` đến `1`; giao diện có thể nhân `100` khi hiển thị phần trăm. Tổng `document_count` bằng 7.884.

Tên cụm được tạo tự động bằng cách ghép ba từ khóa TF-IDF đứng đầu. Các tên này chưa được duyệt hoặc đặt lại thủ công.

### `topic_keywords.csv`

File có 1.010 dòng: 10 từ khóa cho mỗi cụm trong 101 cụm thường. Cụm outlier `-1` không được trích từ khóa.

Các cột:

```text
topic_id,rank,keyword,weight
```

Exporter gom văn bản của các bài train và validation đã được gán vào từng cụm, dùng `processed_title` và 60 từ đầu của `processed_content`, sau đó tính TF-IDF để lấy 10 từ khóa đứng đầu. File này tổng hợp theo cụm nên không có cột `split`.

Việc trích từ khóa trên train + validation được thực hiện sau khi đã chốt cấu hình HDBSCAN, nhằm chuẩn bị nội dung hiển thị; không dùng các từ khóa này để chọn lại mô hình trên validation.

### `representative_docs.csv`

File có 505 dòng, tương ứng 5 bài đại diện cho mỗi cụm thường. Các cột:

```text
topic_id,rank,article_id,title,topic_probability,split
```

Bài đại diện được chọn từ cả train và validation bằng cách sắp xếp `membership_strength` giảm dần; các bài hòa điểm được sắp theo `article_id`. Bản artifact hiện có 447 bài train và 58 bài validation trong tập bài đại diện.

Trong run này, cả 505 bài được chọn đều có `topic_probability=1.0`, vì mỗi cụm có ít nhất năm bài đạt giá trị đó. Giá trị này được giữ nguyên từ HDBSCAN, không phải điểm do exporter tự tạo. Cách chọn hiện tại ưu tiên membership_strength; khi nhiều bài cùng đạt 1.0, thứ tự article_id quyết định bài được chọn, nên chưa bảo đảm đó là các bài tiêu biểu nhất về nội dung.

### `projection_2d.csv`

File có 7.884 điểm, gồm 6.493 train và 1.391 validation. Các cột:

```text
article_id,split,topic_id,x,y
```

UMAP 2D được fit trên embedding train, sau đó transform embedding validation lên cùng bản đồ. `topic_id` được ghép theo `article_id` từ assignments HDBSCAN M11. Tọa độ 2D chỉ phục vụ trực quan hóa; HDBSCAN của run này không được fit trên bản đồ 2D.

### Các file metadata

- `config_used.yaml`: ghi cấu hình embedding, UMAP, HDBSCAN, trích từ khóa, chọn bài đại diện và phép chiếu 2D.
- `model_summary.json`: ghi tên mô hình, số cụm, tỷ lệ outlier, số bài theo split và ý nghĩa của các trường hiển thị.
- `run_metadata.json`: ghi run ID, `config_hash`, nguồn dữ liệu, split dùng để fit, các split được xuất và số bài/outlier theo split.
- `manifest.json`: chỉ đến các file bắt buộc của common artifact schema v1.0.

## 5. Kết quả kiểm tra

Bundle hiện có:

- 7.884 `article_id` duy nhất.
- 6.493 bài train và 1.391 bài validation.
- 101 cụm thường.
- 2.092 bài outlier: 1.608 train và 484 validation.
- 5.792 bài được gán vào cụm thường.
- 102 dòng `topic_summary.csv`, tổng `document_count` bằng 7.884.
- 1.010 dòng `topic_keywords.csv`.
- 505 dòng `representative_docs.csv`.
- 7.884 dòng `projection_2d.csv`.

Loader chung đọc thành công bundle chuẩn. Test đối chiếu nhãn và membership strength của cả train lẫn validation với assignments nguồn; đồng thời kiểm tra ID, outlier, số bài theo split, topic summary, keywords, bài đại diện, phép chiếu 2D và metadata.

Kết quả kiểm thử tại thời điểm viết tài liệu: **8 test P01 passed**; khi chạy cùng test loader và schema: **18 passed**.

## 6. Cách chạy lại

Chạy tại thư mục gốc của repository:

```powershell
python -B -m experiments.hdbscan.export_p01_hdbscan
```

Exporter sẽ ghi lại chín file trong thư mục bundle và gọi loader chung để kiểm tra các bảng bắt buộc. Bước tạo `projection_2d.csv` mã hóa validation bằng MiniLM, fit UMAP 2D trên train và transform validation, nên có thể mất thời gian.

Kiểm tra riêng P01:

```powershell
python -B -m pytest -p no:cacheprovider tests/test_p01_hdbscan_export.py -v
```

Kiểm tra cùng loader và schema chung:

```powershell
python -B -m pytest -p no:cacheprovider tests/test_p01_hdbscan_export.py tests/test_artifact_loader.py tests/test_artifact_schemas.py -v
```