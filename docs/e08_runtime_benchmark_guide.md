# E08 - Hướng dẫn & Báo cáo Benchmark Thời gian chạy & Tài nguyên (Runtime & Resource Benchmark)

**Người phụ trách tổng hợp:** TV1 (Phạm Đăng Khôi)  
**Tài liệu quy chuẩn gốc:** `docs/evaluation_protocol.md` (Mục 7)  
**File kết quả tổng hợp:** `results/runtime_benchmark/e08_runtime_summary.csv`  
**File chi tiết JSON:** `results/runtime_benchmark/all_models_runtime_benchmark.json`  
**Script thực thi tự động:** `experiments/benchmark_all_models_runtime.py`  

---

## 1. Mục tiêu và Phương pháp luận thực hiện

Theo phân công đồ án tại file `docs/Do_an_7-Phan_cong _cong _viec_nhom-TONG_THE_PIPELINE.csv` (Task E08, ghi chú: *"Cùng môi trường nếu có thể"*), để khắc phục triệt để tình trạng lệch phần cứng và thiếu số liệu RAM/Model Size khi thu thập phân tán từ laptop cá nhân của 5 thành viên, TV1 đã thiết kế kịch bản **chạy và đo kiểm chuẩn hóa toàn bộ 5 mô hình trên cùng một máy tính vật lý duy nhất** (Single-Machine Unified Benchmark).

### Thông số phần cứng & Môi trường thực thi
- **Phần cứng:** 16 vCPUs (AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD), 13.76 GB RAM vật lý.
- **Hệ điều hành:** Windows 11 (AMD64).
- **Môi trường Python:** Python 3.13.7 (Virtual Environment `.\venv`).
- **Thư viện chính:** `scikit-learn 1.7.2`, `sentence-transformers 5.2.2`, `bertopic 0.17.4`, `hdbscan 0.8.40`, `umap-learn 0.5.11`, `torch 2.11.0+cpu`, `psutil 7.2.2`.

---

## 2. Kết quả đo lường thực tế trên cùng một máy (Bảng E08)

Toàn bộ 5 mô hình đều có đầy đủ 100% các chỉ số (không còn bất kỳ giá trị `N/A` nào):

| Phương pháp | Cấu hình tối ưu | Fit Split | N_Train | N_Eval | Biểu diễn (Representation) | Huấn luyện (Fit) | Dự đoán Test (Predict) | Đánh giá (Evaluation) | Tổng pipeline (s) | Khởi tạo mô hình (Setup) | RAM đỉnh (MB) | Kích thước model (MB) | Số lần lặp |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KMeans (TF-IDF)** | $K=9$, seed 42 | train | 6.493 | 1.390 | $17{,}826 \pm 0{,}373$ | $3{,}920 \pm 0{,}061$ | $0{,}008 \pm 0{,}000$ | 0,004 | **$21{,}757 \pm 0{,}431$** | 0,000 | **58,9** | **2,25** | 3 |
| **NMF** | $K=10$, seed 42 | train+val | 7.884 | 1.390 | $34{,}304 \pm 0{,}568$ | $5{,}022 \pm 0{,}175$ | $0{,}055 \pm 0{,}012$ | 0,146 | **$39{,}527 \pm 0{,}451$** | 0,000 | **363,1** | **22,26** | 3 |
| **LDA** | $K=15$, seed 42 | train+val | 7.884 | 1.390 | $8{,}964 \pm 1{,}408$ | $16{,}739 \pm 2{,}973$ | $0{,}201 \pm 0{,}008$ | 0,045 | **$25{,}949 \pm 1{,}577$** | 0,000 | **77,0** | **3,87** | 3 |
| **HDBSCAN** | MiniLM + UMAP 5D | train | 6.493 | 1.390 | 139,245 | 68,260 | 37,021 | 0,016 | **244,542** | 3,848 | **1.059,2** | **36,39** | 1 |
| **BERTopic** | VN-SBERT + UMAP | train | 6.493 | 1.390 | 1.179,078 | 1.299,800 | 281,314 | 2,500 | **2.762,692** | 1,619 | **148,4** | **540,00** | 1 |

---

## 3. Cách chạy lại Benchmark (Replication Guide)

Để tái lập toàn bộ hoặc từng phần benchmark:

```powershell
# Kích hoạt môi trường ảo
.\venv\Scripts\Activate.ps1

# Chạy benchmark cho tất cả 5 mô hình:
python experiments/benchmark_all_models_runtime.py --models all --repeats 3

# Hoặc chạy benchmark cho từng mô hình cụ thể:
python experiments/benchmark_all_models_runtime.py --models kmeans --repeats 3
python experiments/benchmark_all_models_runtime.py --models nmf --repeats 3
python experiments/benchmark_all_models_runtime.py --models lda --repeats 3
python experiments/benchmark_all_models_runtime.py --models hdbscan --repeats 1
python experiments/benchmark_all_models_runtime.py --models bertopic --repeats 1
```

Script sẽ tự động đọc cấu hình phần cứng, tính toán thời gian bằng `time.perf_counter()`, đo bộ nhớ qua `psutil`/`tracemalloc`, tính kích thước artifact trên đĩa và cập nhật trực tiếp vào:
- `results/runtime_benchmark/e08_runtime_summary.csv`
- `results/runtime_benchmark/all_models_runtime_benchmark.json`
