# Hướng dẫn chạy BERTopic baseline - Task M13

## 1. Mục tiêu

Task M13 xây dựng pipeline BERTopic baseline trên Dataset v1 đã được
D04 đóng băng và D05 chia thành các tập train, validation và test.

Pipeline thực hiện các bước:

1. Đọc dữ liệu từ Dataset v1.
2. Chọn đúng tập train theo split chính thức của D05.
3. Tạo embedding cho văn bản tiếng Việt.
4. Giảm chiều embedding bằng UMAP.
5. Phân cụm bằng HDBSCAN.
6. Tạo keyword đại diện cho từng topic.
7. Lưu model và kết quả để API hoặc Streamlit sử dụng lại.

M13 chỉ xây dựng baseline và fit mô hình trên tập train. Việc thử các
bộ tham số khác được thực hiện trong M14.

## 2. Quy định sử dụng dữ liệu

Dataset v1 có tổng cộng 9274 bài viết và được chia như sau:

| Split | Số bài | Mục đích |
|---|---:|---|
| train | 6493 | Fit BERTopic baseline |
| val | 1391 | Tinh chỉnh và chọn tham số trong M14 |
| test | 1390 | Đánh giá cuối cùng |

Các nguyên tắc cần tuân thủ:

- Chỉ fit BERTopic baseline trên tập train.
- Không dùng tập test để chọn mô hình hoặc điều chỉnh tham số.
- Không thay đổi danh sách `article_id` trong các split.
- Không chỉnh sửa trực tiếp Dataset v1 trong task M13.
- Giữ nguyên thứ tự giữa `article_id`, văn bản và topic được gán.

## 3. Dữ liệu đầu vào

Cấu hình mặc định của pipeline nằm tại:

```text
configs/bertopic.yaml
```

Các file dữ liệu chính:

```text
data/processed/p1_normalized.csv
data/splits/split_manifest.csv
configs/dataset_manifest.json
configs/vietnamese_stopwords.txt
```

Ý nghĩa của từng file:

- `p1_normalized.csv`: dữ liệu văn bản đã được chuẩn hóa.
- `split_manifest.csv`: xác định mỗi `article_id` thuộc train, val hay test.
- `dataset_manifest.json`: lưu thông tin và checksum của Dataset v1.
- `vietnamese_stopwords.txt`: danh sách hư từ cần loại khi tạo keyword.

Cột văn bản mặc định được sử dụng là:

```text
text
```

Cột này đã kết hợp nội dung cần thiết của bài báo theo quy ước Dataset v1.

## 4. Cấu hình baseline

Embedding model mặc định:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Model này hỗ trợ nhiều ngôn ngữ, bao gồm tiếng Việt, và có kích thước
phù hợp để chạy baseline trên máy cá nhân.

Các thành phần chính của pipeline:

### SentenceTransformer

Chuyển mỗi bài báo thành một vector embedding. Các bài có nội dung gần
nhau sẽ có vector gần nhau trong không gian embedding.

### UMAP

Giảm số chiều của embedding trước khi phân cụm. Việc giảm chiều giúp
HDBSCAN xử lý dữ liệu nhanh hơn và làm rõ cấu trúc cụm.

Cấu hình baseline:

```text
n_neighbors: 15
n_components: 5
min_dist: 0.0
metric: cosine
random_state: 42
```

### HDBSCAN

Phân cụm bài viết và tự xác định số lượng topic. Những bài không thuộc
rõ một cụm nào có thể được gán vào topic `-1`.

Cấu hình baseline:

```text
min_cluster_size: 15
min_samples: 5
metric: euclidean
cluster_selection_method: eom
prediction_data: true
```

### CountVectorizer

Tạo keyword đại diện cho từng topic. Danh sách stopword tiếng Việt được
dùng để tránh các keyword ít ý nghĩa như `và`, `của`, `các`, `có`.

Vectorizer sử dụng unigram và bigram để keyword có thể gồm một hoặc hai từ.

Toàn bộ tham số chi tiết được lưu trong:

```text
configs/bertopic.yaml
```

## 5. Luồng xử lý

Pipeline M13 hoạt động theo thứ tự:

```text
Dataset v1
    -> đọc split_manifest.csv
    -> chọn các bài thuộc tập train
    -> lấy cột text
    -> tạo embedding
    -> giảm chiều bằng UMAP
    -> phân cụm bằng HDBSCAN
    -> tạo keyword cho từng topic
    -> gán topic cho từng article_id
    -> lưu model và kết quả
```

Mỗi `article_id` phải giữ đúng thứ tự với văn bản và topic tương ứng.
Điều này giúp truy ngược kết quả về bài báo ban đầu.

## 6. Cài đặt môi trường

Từ thư mục gốc của dự án, cài các dependency:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Các thư viện chính được sử dụng:

```text
bertopic
sentence-transformers
umap-learn
hdbscan
pyarrow
pandas
pyyaml
scikit-learn
```

Nếu PowerShell không hiển thị tiếng Việt đúng, chạy:

```powershell
$env:PYTHONUTF8 = "1"
```

Biến này chỉ áp dụng cho cửa sổ terminal hiện tại.

## 7. Chạy kiểm thử

Chạy test cho dataset loader và pipeline BERTopic:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_dataset_loader.py tests\test_bertopic_pipeline.py -v
```

Kết quả hiện tại:

```text
10 passed
```

Các test kiểm tra:

- Dataset hợp lệ được chấp nhận.
- Thiếu hoặc trùng `article_id` sẽ phát sinh lỗi.
- Tập train được đọc đúng theo D05.
- File cấu hình BERTopic được đọc đúng.
- UMAP và HDBSCAN nhận đúng tham số baseline.
- Stopword tiếng Việt được đọc đúng.
- Từ phủ định như `không` không bị loại.
- Embedding nhận đúng batch size và tùy chọn chuẩn hóa.
- Văn bản và embedding được truyền đúng vào BERTopic.

Các unit test sử dụng model giả ở những phần phù hợp nên không tải
embedding model và không huấn luyện BERTopic thật.

## 8. Chạy smoke test

Trước khi chạy toàn bộ dữ liệu, cần chạy thử với một số lượng nhỏ bài viết:

```powershell
.\.venv\Scripts\python.exe -m experiments.run_bertopic --limit 200
```

Smoke test dùng để kiểm tra:

- Dataset có được đọc đúng hay không.
- Embedding model có tải và chạy được hay không.
- UMAP và HDBSCAN có hoạt động hay không.
- Model và các file kết quả có được lưu đúng vị trí hay không.
- Keyword có còn bị chi phối bởi stopword hay không.

Kết quả smoke test được lưu tại:

```text
artifacts/bertopic/bertopic_baseline/smoke_200
```

Smoke test chỉ dùng để kiểm tra kỹ thuật. Số topic và chất lượng topic
từ 200 bài không được dùng làm kết quả baseline chính thức.

## 9. Chạy baseline chính thức

Để chạy trên toàn bộ 6493 bài thuộc tập train:

```powershell
.\.venv\Scripts\python.exe -m experiments.run_bertopic
```

Không truyền `--limit` khi chạy baseline chính thức.

Kết quả được lưu tại:

```text
artifacts/bertopic/bertopic_baseline
```

Quá trình chạy có thể mất thời gian tùy theo cấu hình máy. Lần đầu chạy,
SentenceTransformer có thể cần tải model từ Hugging Face. Các lần sau
sẽ sử dụng model đã được lưu trong bộ nhớ đệm.

## 10. Các file đầu ra

### `document_topics.parquet`

Lưu topic được gán cho từng bài viết.

Các cột chính:

- `article_id`: mã định danh của bài viết.
- `topic`: topic được BERTopic gán.
- `topic_probability`: độ tin cậy của phép gán topic.

Topic `-1` biểu thị bài viết được HDBSCAN xem là nhiễu hoặc chưa thuộc
rõ ràng vào topic nào.

### `topic_info.csv`

Lưu thông tin tổng hợp của từng topic, bao gồm:

- mã topic;
- số lượng bài;
- tên topic;
- các keyword đại diện.

File này được dùng để xem nhanh nội dung và kích thước của các topic.

### `run_metadata.json`

Lưu thông tin của lần chạy:

- `run_id`;
- phiên bản dataset;
- split được sử dụng để fit;
- số lượng bài;
- embedding model;
- đường dẫn file cấu hình.

Metadata giúp xác định model được tạo từ dữ liệu và cấu hình nào.

### `model/`

Chứa BERTopic model đã lưu. API và Streamlit có thể tải model này để
xử lý dữ liệu mà không cần train lại từ đầu.

## 11. Kiểm tra kết quả

Sau khi chạy baseline, xem nhanh các topic bằng PowerShell:

```powershell
Import-Csv artifacts\bertopic\bertopic_baseline\topic_info.csv |
    Select-Object -First 10 Topic, Count, Name |
    Format-Table -AutoSize
```

Cần kiểm tra các điều kiện sau:

- Có nhiều topic khác topic `-1`.
- Keyword phản ánh nội dung của topic.
- Keyword không bị chi phối bởi các hư từ.
- Không phải phần lớn bài viết đều bị gán vào topic `-1`.
- Có đủ 6493 kết quả trong `document_topics.parquet`.
- Các file model và metadata đã được tạo.

Có thể kiểm tra số dòng trong file Parquet bằng:

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; data = pd.read_parquet('artifacts/bertopic/bertopic_baseline/document_topics.parquet'); print(len(data)); print(data['topic'].value_counts().head(10))"
```

Kết quả đầu tiên cần bằng:

```text
6493
```

## 12. Xử lý lỗi thường gặp

### Terminal không hiển thị tiếng Việt

Chạy:

```powershell
$env:PYTHONUTF8 = "1"
```

Sau đó thực hiện lại lệnh trước đó.

### Không tìm thấy thư viện

Cài lại dependency bằng đúng Python trong môi trường ảo:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Không tải được embedding model

Kiểm tra kết nối mạng rồi chạy lại. Model chỉ cần tải ở lần đầu và sẽ
được lưu trong bộ nhớ đệm của máy.

Cảnh báo về việc chưa cung cấp `HF_TOKEN` không phải lỗi bắt buộc.
Model công khai vẫn có thể được tải mà không cần đăng nhập.

### Không tìm thấy dataset hoặc split

Kiểm tra các đường dẫn trong:

```text
configs/bertopic.yaml
```

Đồng thời bảo đảm các thay đổi của D04 và D05 đã được merge vào nhánh
đang làm việc.

### Xuất hiện văn bản rỗng

Kiểm tra cột được khai báo tại:

```yaml
dataset:
  text_variant: "text"
```

Cột này phải tồn tại và không được rỗng đối với các bài được dùng để fit.

### Máy thiếu bộ nhớ

Chạy smoke test trước:

```powershell
.\.venv\Scripts\python.exe -m experiments.run_bertopic --limit 200
```

Nếu bước embedding dùng quá nhiều bộ nhớ, có thể giảm:

```yaml
embedding:
  batch_size: 16
```

Việc thay đổi tham số cho lần chạy chính thức cần được ghi lại trong
metadata hoặc commit cấu hình tương ứng.

## 13. Quy ước lưu kết quả

Các model và artifact sinh ra khi chạy được lưu trong:

```text
artifacts/bertopic/
```

Thư mục này đã được thêm vào `.gitignore` và không được commit lên Git
vì có thể chứa model dung lượng lớn.

Các file cần commit gồm:

- code pipeline;
- file chạy thí nghiệm;
- file cấu hình;
- unit test;
- tài liệu hướng dẫn.

Nếu cần chia sẻ model, nên sử dụng kho lưu trữ artifact hoặc thư mục
dùng chung của nhóm thay vì đưa trực tiếp lên Git.

## 14. Phạm vi giữa M13 và M14

M13 chịu trách nhiệm:

- xây dựng pipeline BERTopic baseline;
- đọc đúng Dataset v1 và split D05;
- tạo embedding;
- fit BERTopic trên train;
- lưu model và kết quả;
- cung cấp lệnh chạy lại thí nghiệm.

M14 chịu trách nhiệm:

- thử các bộ tham số khác;
- đánh giá trên tập validation;
- chọn cấu hình tốt hơn;
- so sánh kết quả giữa các lần chạy.

Tập test vẫn được giữ khóa cho đến bước đánh giá cuối cùng.