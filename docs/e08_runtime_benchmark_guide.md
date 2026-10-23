# E08 — Benchmark năm mô hình trên cùng một máy

Chạy từ thư mục gốc dự án bằng Python trong `.venv`. Bản đo E08 này fit cả năm
phương pháp trên 6.493 bài Train và suy diễn 1.390 bài Test. Mỗi lần lặp chạy
lại toàn bộ pipeline; không tái sử dụng vector của lần lặp trước. BERTopic E1
và HDBSCAN mã hóa văn bản trên CUDA, còn UMAP/HDBSCAN và ba phương pháp từ vựng
chạy trên CPU. Model Transformer cần có sẵn trong Hugging Face cache.

Kiểm tra cấu hình và dữ liệu mà không chạy mô hình:

```powershell
.\.venv\Scripts\python.exe -m experiments.benchmark_all_models_runtime --models all --check
```

Đo chính thức ba lượt liên tiếp:

```powershell
.\.venv\Scripts\python.exe -m experiments.benchmark_all_models_runtime --models all --repeats 3 --device cuda
```

Có thể chọn từng mô hình để chẩn đoán, ví dụ `--models kmeans,nmf`.
Mỗi lần gọi tạo thư mục mới `results/runtime_benchmark/e08_.../` với `runs.json`,
`runs.csv` và `e08_runtime_summary.csv`. Không sửa dataset hay artifact gốc.

`runtime_seconds` cộng thời gian representation, fit, predict và evaluation.
`setup_seconds` của Transformer ghi riêng. `peak_process_rss_mb` là RSS đỉnh
của cả tiến trình Python, được lấy mẫu mỗi 50 ms; đó không phải RAM riêng của
mô hình. `peak_gpu_allocated_mb` là bộ nhớ CUDA do PyTorch cấp phát ở đỉnh,
không phải toàn bộ VRAM của máy. Bảng tổng hợp cũng có thời gian trung bình
theo từng giai đoạn. Chưa đo kích thước model trên đĩa trong bản này nên không
suy ra từ RSS. Độ lệch chuẩn chỉ có ý nghĩa với nhiều lượt chạy; cùng một điều kiện cache,
nguồn điện và các ứng dụng nền càng ổn định thì so sánh càng đáng tin cậy.

Kết quả này có giao thức Train → Test đồng nhất, khác một số artifact final
hiện có dùng Train+Val → Test. Không ghi đè runtime trong common artifact hoặc
thay số báo cáo cũ khi chưa giải thích sự khác nhau về giao thức.
