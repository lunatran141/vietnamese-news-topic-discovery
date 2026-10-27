# NMF — Stability qua nhiều random seed

Đã chạy bổ sung stability cho cấu hình NMF được chọn trong báo cáo. Lượt này
kiểm tra độ nhạy theo seed của pipeline TF-IDF/P3, K=10, `nndsvda`; mô hình
Final và số liệu Test tiếp tục thuộc lượt seed 42 đã lưu trước đó.

## Giao thức

- Dataset v1.0, biến thể `p3_no_stopwords`, trường `text`.
- Fit TF-IDF một lần trên 6.493 bài Train, giữ nguyên vocabulary/IDF và thứ
  tự feature giữa các seed. Dùng 50.000 feature theo `configs/nmf.yaml`.
- Fit NMF lại trên Train với seed **42, 52, 62, 72, 82**. Cố định K=10,
  `init=nndsvda`, `solver=cd`, Frobenius loss, `max_iter=500`, `tol=0.0001`.
- Suy diễn trên cùng **1.391 bài Validation**, nối theo `article_id`.
  Loader kiểm tra hash của dữ liệu và ID theo manifest; không biến đổi, tính
  metric hoặc sử dụng bài Test để fit hay chọn seed.
- Đánh giá tất cả **10 cặp seed khác nhau**, không đưa tự so sánh vào thống kê.
- Bộ seed trùng với LDA; HDBSCAN dùng bộ seed riêng. Không dùng bảng này để
  xếp hạng thuật toán, vì representation, số topic và pipeline khác nhau.

Seed truyền vào `NMF.random_state`. Đã kiểm tra mã `_initialize_nmf` của
scikit-learn 1.8.0 đang cài: bước NNDSVD dùng randomized SVD với seed này;
`nndsvda` điền các phần tử 0 bằng trung bình của X. Với coordinate descent,
`shuffle=False` theo mặc định. Vì vậy đây là thay đổi seed trong đúng cấu
hình đang báo cáo, chưa phải thí nghiệm `init=random` hay so sánh kiểu khởi tạo.

## Định nghĩa metric

1. **Keyword Jaccard:** tạo ma trận Jaccard giữa các tập top-10 từ khóa của
   hai mô hình; ghép một-một bằng Hungarian để tối đa tổng Jaccard; lấy trung
   bình của 10 topic đã ghép. Lưu từng phép ghép trong `topic_matches.csv`.
2. **Document cosine:** chuẩn hóa từng hàng W thành topic share; sắp xếp cột
   của mô hình thứ hai theo phép ghép từ khóa trên; tính Cosine cho từng bài
   Validation rồi lấy trung bình. Share NMF không phải xác suất đã hiệu chuẩn.
3. **ARI giữa lần chạy:** so sánh topic có share cao nhất trên cùng article_id
   giữa hai seed. ARI không cần ghép số thứ tự topic và không sử dụng
   `category_gold`; đây là thước đo khác với ARI đối chiếu chuyên mục trên Test.
4. **NPMI:** tính đồng xuất hiện nhị phân ở cấp tài liệu trên Validation,
   dùng top-10 feature và hàm chung `experiments/nmf/metrics.py`.
5. **Topic diversity:** số feature duy nhất chia tổng số vị trí top-10.

Hàng có toàn bộ trọng số topic bằng 0 được đánh dấu `topic_id=-1`; Cosine
của hàng có chuẩn bằng 0 được tính bằng 0 theo hàm chung. Lượt thực tế không
có hàng như vậy. ARI toàn bộ bài và ARI trên phần không có nhãn -1 do đó bằng nhau.

## Kết quả thực tế

| Seed | NPMI Validation | Topic diversity | Vòng lặp | Hội tụ |
|---:|---:|---:|---:|---|
| 42 | 0,449257384 | 1,000000 | 41 | Có |
| 52 | 0,449257384 | 1,000000 | 41 | Có |
| 62 | 0,449257384 | 1,000000 | 41 | Có |
| 72 | 0,449257384 | 1,000000 | 41 | Có |
| 82 | 0,449257384 | 1,000000 | 41 | Có |

| Chỉ số | Số quan sát | Trung bình | Độ lệch chuẩn mẫu |
|---|---:|---:|---:|
| NPMI | 5 seed | 0,449257384 | 0 |
| Topic diversity | 5 seed | 1,000000 | 0 |
| Sai số tái tạo chuẩn hóa | 5 seed | 0,9594766436 | 3,6299 × 10⁻¹⁰ |
| Keyword Jaccard sau ghép | 10 cặp | 1,000000 | 0 |
| ARI giữa lần chạy | 10 cặp | 1,000000 | 0 |
| Cosine topic share đã ghép | 10 cặp | 0,9999999616 | 2,2630 × 10⁻⁸ |

Tất cả bài Validation đều có topic share khác 0, coverage bằng 1. Cả năm
mô hình hội tụ và không sinh cảnh báo hội tụ. Seed 42 tái lập NPMI Validation
K=10 đã lưu ở M05: 0,44925738385583297.

Các tập top-10 từ khóa sau ghép và phân nhóm theo topic chính giống nhau
trong các cặp đã khảo sát. Topic share có sai khác nhỏ: Cosine của từng cặp
nằm trong khoảng **0,9999999162–0,9999999879**. Không diễn giải số làm tròn
1,000000 là các ma trận trọng số bằng nhau hoàn toàn.

Độ lệch chuẩn dùng **ddof=1**, là thống kê mô tả. Các cặp dùng chung mô hình
thành phần nên không độc lập; không dùng SD này như sai số chuẩn hay khoảng
tin cậy. Kết quả chỉ hỗ trợ nhận xét ít nhạy với năm seed trên cấu hình và
split hiện tại. Chưa kiểm tra các kiểu khởi tạo khác, dữ liệu lấy mẫu lại hay
dịch chuyển theo thời gian; stability cũng không thay thế review ngữ nghĩa.
Việc chọn K ở M05 vẫn dựa trên seed 42; không coi lượt stability cố định K=10
là đánh giá lại thứ hạng của mọi giá trị K.

![Kết quả NMF stability trên Validation](../results/stability/submissions/nmf/stability.png)

## Chạy lại trên PowerShell

Không cần kích hoạt venv để chạy các lệnh dưới đây tại thư mục gốc repo:

```powershell
.\.venv\Scripts\python.exe -B -u -m experiments.nmf.run_stability
.\.venv\Scripts\python.exe -B -m experiments.summarize_e04_stability
```

Lượt mặc định đã có kết quả nên lệnh thứ nhất sẽ từ chối ghi đè. Để chạy lại,
dùng thư mục mới:

```powershell
.\.venv\Scripts\python.exe -B -u -m experiments.nmf.run_stability --output results/nmf/stability_rerun_01
```

Nếu muốn đưa lượt mới vào E04, sửa `sources.NMF.directory` trong
`configs/e04_stability.yaml` sang thư mục đó rồi chạy lại lệnh tổng hợp.
Không trộn các seed từ các lượt thuộc môi trường khác nhau vào cùng thống kê.

## Đầu ra và truy vết

`results/stability/submissions/nmf/` chứa:

- `per_seed.csv`, `pairwise.csv`, `summary.csv`: số liệu chưa làm tròn.
- `validation_assignments.csv`: seed, article_id và topic chính cho E04.
- `validation_distributions.npz`: article_id và topic share cùng thứ tự hàng.
- `topic_keywords.csv`, `topic_matches.csv`: từ khóa, trọng số và phép ghép.
- `feature_names.json`: vocabulary fit trên Train, theo thứ tự cột.
- `config_used.json`, `run_metadata.json`: cấu hình, trạng thái hoàn tất,
  phiên bản thư viện, giới hạn thread, SHA-256 input/source/output.
- `run_log.txt`, `stability.png`: log chạy thực tế và hình cho báo cáo.


