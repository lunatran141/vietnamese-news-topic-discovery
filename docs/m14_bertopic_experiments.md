# M14 - Thí nghiệm BERTopic

## 1. Mục tiêu

Thử nghiệm một số cấu hình embedding, UMAP và HDBSCAN để chọn cấu hình BERTopic phù hợp cho tập tin tức tiếng Việt.

Các thí nghiệm chỉ sử dụng:

- Tập train để huấn luyện mô hình.
- Tập validation để so sánh cấu hình.
- Không sử dụng tập test trong quá trình lựa chọn mô hình.

## 2. Dữ liệu sử dụng

- Biến thể dữ liệu: `p1_normalized.csv`
- Số bài train: 6.493
- Số bài validation: 1.391
- Cột văn bản đầu vào: tiêu đề và nội dung đã qua tiền xử lý
- Random seed: 42

Danh sách bài thuộc từng tập được lấy từ `data/splits/split_manifest.csv`.

## 3. Các cấu hình thí nghiệm

### E0 - Multilingual MiniLM

- Embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- UMAP `n_neighbors`: 15
- HDBSCAN `min_cluster_size`: 15

Đây là cấu hình baseline để so sánh với mô hình embedding dành cho tiếng Việt.

### E1 - Vietnamese SBERT

- Embedding: `keepitreal/vietnamese-sbert`
- UMAP `n_neighbors`: 15
- HDBSCAN `min_cluster_size`: 15

Cấu hình này chỉ thay embedding của E0 bằng Vietnamese SBERT.

### E2 - Thay đổi UMAP

- Embedding: `keepitreal/vietnamese-sbert`
- UMAP `n_neighbors`: 30
- HDBSCAN `min_cluster_size`: 15

Cấu hình này giữ nguyên embedding và HDBSCAN của E1, chỉ thay đổi `n_neighbors`.

### E3 - Thay đổi HDBSCAN

- Embedding: `keepitreal/vietnamese-sbert`
- UMAP `n_neighbors`: 15
- HDBSCAN `min_cluster_size`: 30

Cấu hình này giữ nguyên embedding và UMAP của E1, chỉ thay đổi `min_cluster_size`.

## 4. Kết quả

| Cấu hình | Số topic | Outlier train | Outlier validation | Topic diversity | Thời gian |
|---|---:|---:|---:|---:|---:|
| E0 - Multilingual MiniLM | 111 | 31,14% | 42,27% | 0,7000 | 183,30 giây |
| E1 - Vietnamese SBERT | 96 | 24,61% | 35,87% | 0,6813 | 911,34 giây |
| E2 - UMAP neighbors 30 | 93 | 33,61% | 51,91% | 0,6978 | 905,39 giây |
| E3 - HDBSCAN cluster size 30 | 57 | 29,59% | 39,11% | 0,7456 | 997,85 giây |

Trong bảng trên:

- `Số topic` không tính topic `-1`.
- `Outlier rate` là tỷ lệ bài bị HDBSCAN đưa vào topic `-1`.
- `Topic diversity` đo tỷ lệ từ khóa không trùng nhau giữa các topic.
- Outlier validation được dùng để đánh giá khả năng áp dụng mô hình trên dữ liệu chưa dùng để fit.

## 5. Nhận xét

E1 có tỷ lệ outlier thấp nhất trên cả train và validation. Điều này cho thấy Vietnamese SBERT tạo biểu diễn phù hợp hơn với dữ liệu tin tức tiếng Việt trong thí nghiệm này.

E2 không cải thiện kết quả. Khi tăng `n_neighbors` từ 15 lên 30, tỷ lệ outlier validation tăng từ 35,87% lên 51,91%.

E3 tạo ít topic hơn và có topic diversity cao nhất. Tuy nhiên, tỷ lệ outlier validation của E3 vẫn cao hơn E1.

E0 chạy nhanh nhất nhưng có tỷ lệ outlier validation cao hơn E1.

## 6. Cấu hình được chọn

Chọn **E1 - Vietnamese SBERT** làm cấu hình tốt nhất cho bước tiếp theo:

- Embedding: `keepitreal/vietnamese-sbert`
- UMAP `n_neighbors`: 15
- HDBSCAN `min_cluster_size`: 15
- Số topic thu được: 96
- Outlier validation: 35,87%
- Topic diversity: 0,6813

E1 được chọn vì có tỷ lệ outlier thấp nhất trên validation. E3 có diversity cao hơn nhưng số bài bị xem là outlier cũng nhiều hơn.

## 7. Cách chạy lại

Chạy một thí nghiệm cụ thể:

```powershell
.\.venv\Scripts\python.exe -m experiments.run_bertopic_experiments `
  --run-id e1_vietnamese_sbert