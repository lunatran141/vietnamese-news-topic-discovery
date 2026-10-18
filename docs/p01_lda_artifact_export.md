
# P01 — LDA Common Artifact Export

## 1. Mục tiêu

Chuẩn hóa kết quả LDA Final K=15 theo common artifact schema của nhóm để tích hợp với loader, API và ứng dụng Streamlit.

Artifact được xuất từ mô hình LDA Final đã lưu, không thực hiện huấn luyện lại mô hình và không điều chỉnh số lượng topic K dựa trên kết quả Final Test.

## 2. Cấu hình chính thức

| Thành phần | Giá trị |
|---|---|
| Dataset | v1.0 |
| Preprocessing | P2 Segmented |
| Text variant | text |
| Representation | Bag-of-Words |
| Model | Latent Dirichlet Allocation |
| K | 15 |
| Random seed | 42 |
| Fit split | Train + Validation |
| Train documents | 6,493 |
| Validation documents | 1,391 |
| Test documents | 1,390 |
| Total documents | 9,274 |

K=15 đã được leader phê duyệt trong Task 3.7. K=20 chỉ được sử dụng để đối chiếu tại giai đoạn Validation, không sử dụng để đánh giá lại trên Test.

## 3. Nguồn đầu vào

Các dữ liệu và mô hình đầu vào được sử dụng trong quá trình export bao gồm:

- **Mô hình LDA Final:** `artifacts/lda/final/lda_pipeline.joblib`
- **Metadata của mô hình:** `artifacts/lda/final/run_metadata.json`
- **Cấu hình Final:** `artifacts/lda/final/config.json`
- **Dataset:** `data/processed/p2_segmented.csv`
- **Split manifest:** `data/splits/split_manifest.csv`

Các file nguồn của mô hình cần tồn tại trước khi thực hiện export. Quá trình export chỉ tải mô hình đã lưu và suy luận phân phối topic cho các bài báo, không yêu cầu huấn luyện lại LDA Final.

## 4. Cách chạy

Tại thư mục gốc của repository, thực hiện lệnh sau:

```powershell
python -m experiments.lda.export_common_artifacts
```

Script mặc định xuất kết quả vào thư mục:

`artifacts/common/lda/lda_k15_seed42/`

Nếu thư mục đích đã tồn tại, exporter sẽ từ chối ghi đè nhằm bảo vệ các artifact đã được tạo trước đó. Điều này giúp tránh việc vô tình thay đổi hoặc làm mất kết quả thực nghiệm đã được xác minh.

## 5. Các file đầu ra

Bộ common artifact của LDA bao gồm sáu file theo schema thống nhất của nhóm:

| File | Nội dung |
|---|---|
| `document_topics.csv` | Gán topic cho toàn bộ 9,274 bài báo |
| `topic_keywords.csv` | Top 10 từ khóa đại diện cho mỗi topic trong 15 topic |
| `topic_summary.csv` | Thống kê số lượng bài và tỷ lệ phân bố theo từng topic |
| `model_summary.json` | Thông tin tổng quan về mô hình, cấu hình và metadata |
| `config_used.yaml` | Cấu hình được sử dụng trong quá trình xây dựng và export mô hình |
| `manifest.json` | Danh mục các file đầu ra theo common artifact schema |

File `document_topics.csv` bao gồm các trường dữ liệu sau:

- `article_id`: Mã định danh duy nhất của bài báo.
- `split`: Tập dữ liệu mà bài báo thuộc về, bao gồm Train, Validation hoặc Test.
- `topic_id`: Mã topic được gán cho bài báo, có giá trị từ 0 đến 14.
- `topic_probability`: Xác suất của topic được gán, nằm trong khoảng [0,1].
- `is_outlier`: Cờ xác định bài báo có phải outlier hay không.
- `title`: Tiêu đề bài báo.
- `source`: Nguồn xuất bản bài báo.
- `published_at`: Thời điểm xuất bản bài báo.

Mỗi bài báo được gán vào topic có xác suất lớn nhất trong phân phối topic do mô hình LDA dự đoán. Giá trị `topic_probability` được xác định bằng xác suất lớn nhất trong phân phối 15 topic của bài báo tương ứng.

LDA không sử dụng cơ chế phát hiện outlier trong pipeline hiện tại, vì vậy trường `is_outlier` được đặt bằng `False` cho toàn bộ 9,274 bài báo.

## 6. Kết quả xác minh artifact

Sau khi export, bộ artifact được kiểm tra về tính đầy đủ, tính nhất quán và khả năng tương thích với loader chung của nhóm.

| Tiêu chí | Kết quả |
|---|---|
| Tổng số document assignments | 9,274 |
| Số `article_id` duy nhất | 9,274 |
| Số `article_id` trùng lặp | 0 |
| Số `topic_id` không hợp lệ | 0 |
| Số xác suất bị thiếu | 0 |
| Số xác suất không hợp lệ | 0 |
| Số bài được đánh dấu outlier | 0 |
| Số dòng trong `topic_summary.csv` | 15 |
| Số dòng trong `topic_keywords.csv` | 150 |
| Tổng tỷ lệ phân bố topic | Xấp xỉ 1.0 |

Giá trị `topic_probability` nhỏ nhất quan sát được là 0.207650 và lớn nhất là 0.999231. Tất cả giá trị đều nằm trong khoảng [0,1], đáp ứng yêu cầu của common artifact schema.

Kết quả kiểm tra cũng xác nhận có đầy đủ 15 topic và mỗi topic có đúng 10 từ khóa đại diện. Tổng số bài được thống kê trong `topic_summary.csv` bằng 9,274, khớp với tổng số bài trong Dataset v1.0.

Ngoài ra, đã thực hiện kiểm tra tính nhất quán trên 15 bài báo mẫu được lấy từ ba tập Train, Validation và Test. Kết quả cho thấy `topic_id` và `topic_probability` trong common artifact đều khớp với kết quả suy luận trực tiếp từ mô hình LDA Final K=15 đã lưu.

Việc kiểm tra trên 15 bài mẫu chỉ nhằm xác nhận tính nhất quán của quá trình export và không thay thế cho việc xác minh toàn bộ xác suất của 9,274 bài báo.

## 7. Kiểm thử

Các kiểm thử dành riêng cho P01 được triển khai tại:

`tests/test_lda_artifact_export.py`

Để chạy bộ kiểm thử, sử dụng lệnh:

```powershell
python -m pytest tests/test_lda_artifact_export.py -q
```

Bộ kiểm thử được xây dựng nhằm xác minh các yêu cầu sau:

- Common artifact tồn tại và có thể được đọc thành công bằng loader chung của nhóm.
- Dataset có đầy đủ 9,274 document assignments và không có `article_id` trùng lặp.
- Các giá trị `topic_id`, `topic_probability` và `is_outlier` đáp ứng yêu cầu của schema.
- Số lượng topic, số lượng từ khóa và thống kê phân bố topic khớp với cấu hình K=15.
- Số lượng bài theo Train, Validation và Test khớp với split manifest.
- Thông tin trong `manifest.json` nhất quán với `model_summary.json`.
- Exporter từ chối ghi đè thư mục output đã tồn tại.
- Exporter báo lỗi khi thiếu các file nguồn cần thiết và không tạo output dở dang trong trường hợp thiếu nguồn đầu vào.

Các kiểm thử P01 và hệ thống artifact chung đã được thực hiện thành công trong môi trường phát triển trước khi chuẩn bị commit.

## 8. Tích hợp hệ thống

Bộ common artifact LDA được xây dựng theo cùng schema đang sử dụng cho K-Means và BERTopic, cho phép các thành phần khác trong hệ thống truy cập dữ liệu thông qua `load_artifact_bundle()`.

Ví dụ đọc common artifact LDA:

```python
from src.artifacts.loader import load_artifact_bundle

bundle = load_artifact_bundle(
    "artifacts/common/lda/lda_k15_seed42"
)

print(bundle.model_summary.model_name)
print(len(bundle.document_topics))
```

Khi thực thi thành công, loader trả về thông tin mô hình LDA K=15 và bảng document-topic assignments gồm 9,274 bài báo.

Việc sử dụng schema chung giúp thuận tiện cho quá trình tích hợp với API và Streamlit, đồng thời hỗ trợ so sánh kết quả giữa các mô hình khám phá chủ đề trong cùng hệ thống.

## 9. Giới hạn

Common artifact chứa kết quả gán topic cho toàn bộ các bài báo thuộc Train, Validation và Test nhằm phục vụ tích hợp hệ thống và hiển thị kết quả.

Các assignment trên Test không được sử dụng để lựa chọn lại K, tinh chỉnh mô hình hoặc báo cáo thêm kết quả Final Test như một thí nghiệm độc lập. K=15 là cấu hình chính thức đã được phê duyệt trước khi thực hiện đánh giá cuối.

Các file mô hình cục bộ và artifact trung gian thuộc M07–M09 không nằm trong bộ common artifact export. Những file này được lưu riêng để phục vụ việc tái lập và kiểm tra các thực nghiệm trước đó.
