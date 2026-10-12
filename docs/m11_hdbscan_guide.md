# Hướng dẫn chạy HDBSCAN semantic clustering — Task M11

## 1. Mục tiêu

M11 thử nghiệm quy trình phân cụm ngữ nghĩa:

**Sentence embedding → UMAP → HDBSCAN**

Embedding được lấy từ bốn artifact đã tạo ở M10. Mỗi bài báo nhận một nhãn cụm; nhãn `-1` biểu thị bài chưa được gán vào cụm.

Thí nghiệm nhằm:

- So sánh kết quả của bốn cấu hình embedding.
- Khảo sát `min_cluster_size` và `min_samples`.
- So sánh hai cách chọn cụm `eom` và `leaf`.
- Lưu nhãn cụm, bài nhiễu và cấu hình tương ứng với từng `article_id`.
- Kiểm tra tính đầy đủ và nhất quán của kết quả.

Theo sheet phân công TV4, M11 tương ứng **mục 4.4 — HDBSCAN experiments**, với đầu ra là **clusters/outliers/configs**.

## 2. Các file chính

- `configs/hdbscan.yaml`: đường dẫn dữ liệu, artifact embedding, tham số UMAP, các cấu hình HDBSCAN và thư mục đầu ra.
- `configs/embeddings.yaml`: cấu hình tạo bốn artifact embedding ở M10.
- `src/features/embeddings.py`: đọc artifact embedding và metadata.
- `src/models/hdbscan.py`: hàm giảm chiều và phân cụm dùng chung.
- `experiments/hdbscan/run_m11_hdbscan.py`: chạy thí nghiệm và lưu CSV.
- `experiments/hdbscan/plot_m11_results.py`: tạo ba biểu đồ từ kết quả.
- `tests/test_embeddings.py`: kiểm tra tạo, lưu và tải lại embedding.
- `tests/test_hdbscan.py`: kiểm tra hàm phân cụm.
- `tests/test_m11_outputs.py`: kiểm tra các CSV đầu ra.
- `docs/m11_hdbscan_guide.md`: hướng dẫn chạy và ghi nhận kết quả.

Runner gọi `reduce_embeddings()` **một lần cho mỗi embedding**, rồi dùng cùng ma trận đã giảm chiều cho tám lần chạy HDBSCAN. Nhờ vậy, khi so sánh tám cấu hình trên cùng một embedding, đầu vào của HDBSCAN được giữ như nhau.

## 3. Dữ liệu đầu vào

M11 sử dụng **Dataset v1.0**:

- File dữ liệu: `data/processed/p0_minimal.csv`.
- File chia tập: `data/splits/split_manifest.csv`.
- Tập sử dụng: `train`.
- Số bài: **6.493**.

Runner dùng `load_split_dataset()` để lấy đúng tập train. Sau đó, runner đối chiếu số bài, `article_id` và thứ tự bài giữa tập train với artifact embedding.

Bốn cấu hình embedding từ M10:

| Cấu hình | Văn bản đưa vào mô hình | Kích thước vector |
| --- | --- | --- |
| `minilm_title_content` | Tiêu đề + 60 từ đầu nội dung | `(6493, 384)` |
| `mpnet_title_content` | Tiêu đề + 60 từ đầu nội dung | `(6493, 768)` |
| `minilm_title_only` | Chỉ tiêu đề | `(6493, 384)` |
| `mpnet_title_only` | Chỉ tiêu đề | `(6493, 768)` |

Hai cấu hình `title_content` sử dụng **60 từ đầu của nội dung**, không sử dụng toàn bộ bài báo. M11 đọc vector từ các file `.npz`; runner không tạo lại embedding.

Trước khi phân cụm, runner so sánh metadata của artifact với `configs/embeddings.yaml`, gồm tên cấu hình, mô hình, `content_words`, cột văn bản, split và các thiết lập mã hóa.

## 4. Cấu hình UMAP

Các thí nghiệm dùng tham số UMAP sau:

```yaml
umap:
  n_neighbors: 15
  n_components: 5
  min_dist: 0.0
  metric: "cosine"
  random_state: 42
```

UMAP chuyển vector embedding từ 384 hoặc 768 chiều xuống **5 chiều** để đưa vào HDBSCAN. File embedding M10 vẫn giữ số chiều gốc.

Mỗi embedding được giảm chiều riêng. Tám cấu hình HDBSCAN của cùng một embedding dùng chung kết quả UMAP.

## 5. Cấu hình HDBSCAN

HDBSCAN dùng `metric: "euclidean"` trên vector 5 chiều sau UMAP.

Ý nghĩa các tham số chính:

- `min_cluster_size`: kích thước tối thiểu để một nhóm được xem là cụm.
- `min_samples`: tham số ảnh hưởng đến mức độ chặt khi xác định vùng có đủ mật độ.
- `cluster_selection_method`: cách chọn cụm từ cấu trúc phân cấp, gồm `eom` và `leaf`.

Sáu cấu hình EOM:

| Tên | `min_cluster_size` | `min_samples` |
| --- | ---: | ---: |
| `eom_15_5` | 15 | 5 |
| `eom_15_10` | 15 | 10 |
| `eom_30_5` | 30 | 5 |
| `eom_30_10` | 30 | 10 |
| `eom_50_5` | 50 | 5 |
| `eom_50_10` | 50 | 10 |

Hai cấu hình Leaf:

| Tên | `min_cluster_size` | `min_samples` |
| --- | ---: | ---: |
| `leaf_15_5` | 15 | 5 |
| `leaf_30_10` | 30 | 10 |

Tổng cộng có **4 embedding × 8 cấu hình HDBSCAN = 32 thí nghiệm**.

Đây là các giá trị được chọn để khảo sát. Kết quả M11 chưa chứng minh một cấu hình là tối ưu.

## 6. Cách chạy

Mở PowerShell tại thư mục gốc `vietnamese-news-topic-discovery`. Chạy từng lệnh và đợi lệnh hiện tại hoàn tất trước khi chạy lệnh tiếp theo:

```powershell
python -B -m experiments.hdbscan.run_m11_hdbscan --embedding minilm_title_content
python -B -m experiments.hdbscan.run_m11_hdbscan --embedding mpnet_title_content
python -B -m experiments.hdbscan.run_m11_hdbscan --embedding minilm_title_only
python -B -m experiments.hdbscan.run_m11_hdbscan --embedding mpnet_title_only
```

Với mỗi embedding, runner:

1. Đọc cấu hình M10 và M11.
2. Kiểm tra bảng tổng hợp cũ, nếu có.
3. Đọc tập train và artifact embedding.
4. Kiểm tra metadata, số bài, kích thước vector, ID và thứ tự ID.
5. Giảm chiều bằng UMAP một lần.
6. Chạy tám cấu hình HDBSCAN.
7. Lưu một file assignments cho embedding đó.
8. Cập nhật các dòng tương ứng trong bảng tổng hợp chung.

Nếu chạy lại một embedding, runner thay kết quả assignments và các dòng summary của embedding đó. Các dòng của embedding khác chỉ được giữ lại khi `config_hash` tương thích.

### `config_hash` dùng để làm gì?

`config_hash` là mã SHA-256 được tính từ cấu hình dataset M11, UMAP, HDBSCAN và metadata cấu hình của bốn embedding M10. Nó giúp phát hiện trường hợp định gộp kết quả từ **hai cấu hình thí nghiệm khác nhau** vào cùng một bảng summary.

Nếu summary cũ thiếu `config_hash` hoặc có hash khác, runner báo lỗi trước khi chạy UMAP. Khi bắt đầu một đợt thí nghiệm với cấu hình mới, cần lưu riêng kết quả cũ rồi dùng thư mục đầu ra phù hợp cho đợt mới.

`config_hash` nhận diện **cấu hình được khai báo**; nó không kiểm tra nội dung từng byte của dataset hay vector. Nếu dữ liệu hoặc artifact thay đổi mà YAML không đổi, cần chạy lại các kết quả chịu ảnh hưởng.

## 7. Kết quả đầu ra

`configs/hdbscan.yaml` đặt:

```yaml
output_dir: "results/hdbscan"
```

Sau khi chạy đủ bốn embedding, có năm CSV chính:

```text
results/hdbscan/
├── minilm_title_content_assignments.csv
├── mpnet_title_content_assignments.csv
├── minilm_title_only_assignments.csv
├── mpnet_title_only_assignments.csv
└── m11_summary.csv
```

### Bốn file assignments

Mỗi file chứa:

**6.493 bài × 8 cấu hình = 51.944 dòng dữ liệu.**

Các cột:

- `article_id`: mã bài báo.
- `cluster_id`: nhãn cụm; `-1` là bài nhiễu.
- `membership_strength`: mức độ thuộc cụm do HDBSCAN trả về.
- `model_name`: tên mô hình embedding.
- `data_version`: phiên bản dữ liệu.
- `embedding_run`: tên cấu hình embedding.
- `experiment`: tên cấu hình HDBSCAN.
- `config_hash`: mã nhận diện cấu hình thí nghiệm dùng để tạo kết quả.

Một `article_id` xuất hiện tám lần trong cùng một file vì bài đó được phân cụm dưới tám cấu hình. Cặp `experiment` + `article_id` phải duy nhất.

`membership_strength` **không phải** độ chính xác của chủ đề hay điểm đánh giá chất lượng toàn bộ cụm.

Nhãn cụm chỉ có ý nghĩa **trong từng lần chạy**. Ví dụ, cụm `1` của EOM không mặc nhiên tương ứng với cụm `1` của Leaf hoặc của embedding khác.

### File `m11_summary.csv`

Bảng tổng hợp có **32 dòng dữ liệu**, mỗi dòng ứng với một cặp embedding và cấu hình HDBSCAN.

Các cột:

- `embedding_run`: tên cấu hình embedding.
- `experiment`: tên cấu hình HDBSCAN.
- `config_hash`: mã nhận diện cấu hình thí nghiệm.
- `min_cluster_size`, `min_samples`, `cluster_selection_method`: tham số HDBSCAN.
- `n_articles`: số bài được phân cụm.
- `n_clusters`: số cụm, không tính nhãn `-1`.
- `n_noise`: số bài mang nhãn `-1`.
- `noise_percent`: tỷ lệ bài nhiễu tính theo phần trăm.
- `smallest_cluster`: số bài trong cụm nhỏ nhất.
- `largest_cluster`: số bài trong cụm lớn nhất.

M11 hiện lưu **nhãn phân cụm và số liệu thí nghiệm** trong CSV. Quy trình hiện tại chưa lưu mô hình UMAP/HDBSCAN đã fit vào `artifacts/hdbscan/`.

## 8. Kiểm tra

Chạy cả test M10 và M11:

```powershell
python -B -m pytest -p no:cacheprovider tests/test_embeddings.py tests/test_hdbscan.py tests/test_m11_outputs.py -v
```

Kết quả đã chạy: **14 passed**, gồm:

- **5 test M10** trong `tests/test_embeddings.py`: kiểm tra số vector, thứ tự ID, NaN, ID trùng, cột P0 và việc lưu/tải lại vector **cùng metadata**.
- **4 test hàm phân cụm** trong `tests/test_hdbscan.py`: kiểm tra việc giữ đúng ID, dùng lại ma trận đã giảm chiều, từ chối số dòng không khớp và từ chối giá trị NaN không hợp lệ.
- **5 test đầu ra M11** trong `tests/test_m11_outputs.py`: kiểm tra 32 cặp cấu hình, số bài và thứ tự ID, nhãn và membership hợp lệ, số liệu summary khớp assignments. Các test này cũng kiểm tra `config_hash` của summary khớp YAML hiện tại và hash của assignments khớp summary.

Các test xác nhận **cấu trúc dữ liệu và cách ghép kết quả**. Chúng chưa chứng minh các bài trong mỗi cụm thực sự cùng chủ đề.

## 9. Kết quả và nhận xét sơ bộ

### MiniLM title + content

- `eom_15_5`: **101 cụm**, **1.608 bài nhiễu**.
- `leaf_15_5`: **126 cụm**, **2.310 bài nhiễu**.

### MPNet title + content — `eom_15_5`

- **110 cụm**.
- **1.660 bài nhiễu**, tương ứng **25,57%**.
- Cụm lớn nhất: **227 bài**.

Trong mẫu tiêu đề đã xem, cụm về thành tích thể thao Việt Nam tại ASIAD khá nhất quán. Cụm sức khỏe rộng hơn, gồm cả dinh dưỡng và trị liệu tâm lý. Một cụm nhỏ về câu chuyện con người có dấu hiệu trộn các bài hiến tạng, tai nạn và hoàn cảnh gia đình.

### MPNet title + content — `leaf_15_5`

- **124 cụm**.
- **2.148 bài nhiễu**, tương ứng **33,08%**.
- Cụm lớn nhất: **104 bài**.

Trong mẫu đã xem, cụm bóng đá câu lạc bộ Việt Nam khá nhất quán. Cụm ngành điện trải rộng từ doanh nghiệp điện đến chính sách giá điện. Một cụm nhỏ vẫn có dấu hiệu trộn nội dung.

### Đánh đổi quan sát được

Ở hai cấu hình MPNet title + content với tham số 15/5, Leaf tạo nhiều cụm hơn, cụm lớn nhất nhỏ hơn, đồng thời có nhiều bài nhiễu hơn EOM.

Tỷ lệ nhiễu thấp **không đủ** để chọn cấu hình tốt nhất. Ví dụ, MPNet title-only với `eom_50_10` chỉ có **0,11%** bài nhiễu, nhưng cụm lớn nhất chứa tới **3.862 bài**. Cụm quá lớn có thể gom nhiều chủ đề khác nhau; cần xem nội dung trước khi đánh giá.

Bài mang nhãn `-1` vẫn có thể là bài báo hợp lệ, thuộc chủ đề hiếm hoặc nằm giữa nhiều chủ đề. Nhãn nhiễu không tự động có nghĩa dữ liệu sai.

Các nhận xét về nội dung cụm trên đây dựa trên **một số mẫu tiêu đề đã kiểm tra**, chưa đại diện cho toàn bộ 32 thí nghiệm và chưa xác định cấu hình tốt nhất.

## 10. Biểu đồ

Tạo hình từ CSV đã lưu:

```powershell
python -B -m experiments.hdbscan.plot_m11_results
```

Lệnh tạo ba hình trong `results/hdbscan/figures/`.

### Tỷ lệ bài nhiễu

![Tỷ lệ bài nhiễu](../results/hdbscan/figures/m11_noise_ratio.png)

Ở cùng tham số 15/5, Leaf có tỷ lệ nhiễu cao hơn EOM trên cả bốn embedding. Con số hiển thị trên hình được làm tròn đến **một chữ số thập phân**; CSV giữ giá trị đến **hai chữ số thập phân**. Vì vậy, tỷ lệ **0,11%** trong CSV hiển thị thành **0,1%** trên hình.

### Số cụm

![Số cụm](../results/hdbscan/figures/m11_cluster_count.png)

Trong các cấu hình EOM đã thử, khi tăng `min_cluster_size` từ 15 lên 50 và giữ cùng `min_samples`, số cụm giảm trên cả bốn embedding. Cần xem nội dung để biết các cụm còn giữ được chủ đề rõ ràng hay đã bị gom quá rộng.

### Phân bố kích thước cụm

![Phân bố kích thước cụm](../results/hdbscan/figures/m11_cluster_size_distribution.png)

Hình so sánh MPNet title + content ở tham số 15/5:

- EOM: **110 cụm**, cụm lớn nhất **227 bài**.
- Leaf: **124 cụm**, cụm lớn nhất **104 bài**.

Leaf tạo các cụm nhỏ hơn trong trường hợp này, nhưng tỷ lệ nhiễu tăng từ **25,57%** lên **33,08%**. Hình không tính nhãn `-1` như một cụm.

## 11. Giới hạn và bước tiếp theo

M11 đã có:

- Code giảm chiều và phân cụm dùng chung.
- Cấu hình cho bốn embedding và tám cách phân cụm.
- Năm CSV chứa kết quả của **32 thí nghiệm**.
- Nhãn cụm gắn với `article_id`.
- Ba biểu đồ mô tả kết quả.
- **9 test liên quan trực tiếp đến M11 đạt**; tính cả 5 test M10 thì **14 test đạt**.
- Nhận xét sơ bộ về một số cụm.

Các chỉ số như số cụm, tỷ lệ nhiễu và kích thước cụm giúp mô tả kết quả, nhưng **chưa đủ để khẳng định chất lượng ngữ nghĩa**. Việc chọn cấu hình cuối cùng cần xem thêm nội dung cụm và theo quy trình đánh giá chung của nhóm.

Phần thí nghiệm HDBSCAN của **M11/mục 4.4** đã sẵn sàng để nhóm review. Công việc tiếp theo của TV4 là **mục 4.5 — Outlier analysis**: xem các bài nhãn `-1`, phân biệt bài lỗi hoặc không liên quan, bài thuộc chủ đề hiếm, bài có nhiều chủ đề và bài hợp lệ bị coi là ngoại lệ; ghi ví dụ kèm nhận xét.