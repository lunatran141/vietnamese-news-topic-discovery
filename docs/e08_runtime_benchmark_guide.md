# E08 - Hướng dẫn Thu thập Benchmark Thời gian chạy & Tài nguyên (Runtime & Resource Benchmark)

**Người phụ trách tổng hợp:** TV1 (Phạm Đăng Khôi)  
**Tài liệu quy chuẩn gốc:** `docs/evaluation_protocol.md` (Mục 7)  
**File biểu mẫu:** `results/runtime_benchmark/e08_runtime_template.csv`

---

## 1. Quy định bắt buộc khi báo cáo số liệu Runtime (Task E08)

Để đảm bảo tính so sánh công bằng và chính xác trong nghiên cứu khoa học, mọi thành viên cần tuân thủ 5 nguyên tắc:
1. **Không so sánh khập khiễng:** Tuyệt đối không so sánh `fit_seconds` của mô hình này với `total_pipeline_seconds` của mô hình khác.
2. **Ghi rõ phạm vi đo (Scope):** Mọi con số phải ghi rõ bao gồm các bước nào (ví dụ: có bao gồm trích xuất vector/nhúng từ hay không, có bao gồm nạp mô hình từ đĩa hay không).
3. **Môi trường phần cứng:** Ghi rõ CPU, RAM vật lý, GPU (nếu dùng), hệ điều hành và phiên bản Python.
4. **Xử lý dữ liệu khuyết:** **Không điền `0`** cho các giá trị chưa đo được; bắt buộc điền **`N/A`**.
5. **Độ lặp lại:** Ghi rõ số lần đo `n_repeats` (khuyến khích $n \ge 3$, báo giá trị trung bình $\pm$ độ lệch chuẩn).

---

## 2. Định nghĩa các trường trong Biểu mẫu (`e08_runtime_template.csv`)

| Tên trường | Đơn vị | Ý nghĩa & Quy chuẩn đo |
| :--- | :---: | :--- |
| `method` | text | Tên phương pháp (KMeans, NMF, LDA, HDBSCAN, BERTopic) |
| `run_id` | text | Mã định danh của run tối ưu báo cáo trong bài báo |
| `fit_split` | text | Tập dữ liệu dùng để fit mô hình (`train` hoặc `train+val`) |
| `n_train` | int | Số lượng bài báo trong tập huấn luyện (6.493 hoặc 7.884) |
| `n_eval` | int | Số lượng bài báo trong tập đánh giá Test (1.390) |
| `representation_seconds` | giây | Thời gian tạo ma trận TF-IDF / BoW hoặc tính Sentence Embeddings |
| `fit_seconds` | giây | Thời gian huấn luyện mô hình (KMeans / NMF / LDA / UMAP+HDBSCAN) |
| `predict_seconds` | giây | Thời gian suy luận / gán nhãn cụm cho tập Test |
| `evaluation_seconds` | giây | Thời gian tính các metric (Purity, NMI, ARI, Coherence...) |
| `total_pipeline_seconds`| giây | Tổng thời gian toàn trình của pipeline test: `representation + fit + predict + evaluation` (loại bỏ I/O đọc ghi file và vẽ đồ thị) |
| `setup_seconds` | giây | Thời gian tải pre-trained model từ đĩa/mạng lần đầu (nếu có); để riêng, không cộng gộp vào `total_pipeline_seconds` |
| `ram_peak_mb` | MB | Bộ nhớ RAM đỉnh tiêu thụ trong quá trình chạy (đo bằng `tracemalloc` hoặc `psutil`) |
| `model_size_mb` | MB | Tổng dung lượng các tệp model/vectorizer lưu trên đĩa |
| `hardware` | text | Tên CPU (ví dụ: Intel Core i7-12700H 14 cores, 16GB RAM) |
| `python_version` | text | Phiên bản Python thực thi (ví dụ: Python 3.13.7) |
| `n_repeats` | int | Số lần chạy lặp lại để lấy trung bình (ví dụ: 1 hoặc 3) |
| `scope_notes` | text | Ghi chú ngắn gọn về các bước nằm trong phép đo |

---

## 3. Mẫu tin nhắn TV1 gửi 4 thành viên trong nhóm

Bạn có thể copy đoạn tin nhắn dưới đây để gửi vào kênh chat của nhóm:

```markdown
Chào các bạn (TV2, TV3, TV4, TV5),

Để hoàn thiện Task E08 (Runtime & Resource Benchmark) cho bài báo cáo NCKH chung của nhóm theo đúng quy chuẩn Mục 7 trong docs/evaluation_protocol.md, mình (TV1) đã tạo biểu mẫu thu thập tại file:
👉 results/runtime_benchmark/e08_runtime_template.csv

Nhờ các bạn cung cấp số liệu thực tế của mô hình mình phụ trách:
- TV1: KMeans (đã đo xong)
- TV2: NMF
- TV3: LDA
- TV4: HDBSCAN
- TV5: BERTopic

Quy định khi điền số liệu:
1. Ghi rõ thời gian từng bước (representation, fit, predict test, evaluation). Không so fit_seconds của mô hình này với total_pipeline của mô hình khác.
2. Thời gian tải model lần đầu (setup_seconds) ghi riêng, không gộp vào total_pipeline.
3. Nếu chưa đo được RAM hoặc Model size thì ghi N/A (tuyệt đối không điền số 0).
4. Ghi kèm thông số cấu hình máy (CPU, RAM, GPU nếu có) và số lần chạy (n_repeats).

Các bạn có thể mở file CSV trên hoặc gửi trực tiếp số liệu vào đây để mình tổng hợp vào bảng chung cho Paper nhé. Cảm ơn các bạn!
```

