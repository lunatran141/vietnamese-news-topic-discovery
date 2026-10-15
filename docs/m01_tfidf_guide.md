# Hướng dẫn Kỹ thuật & Báo cáo Nghiệm thu Task M01: TF-IDF Feature Extraction Pipeline
## 1. Tổng quan & Mục tiêu

Task M01 xây dựng pipeline trích xuất đặc trưng dạng túi từ (bag-of-words) có trọng số TF-IDF (**Term Frequency - Inverse Document Frequency**) chuẩn hóa cho toàn bộ dự án.

Pipeline đáp ứng các yêu cầu cốt lõi:
1. **Zero Data Leakage:** `TfidfVectorizer` được huấn luyện (`fit`) **duy nhất trên tập huấn luyện (Train split: 6,493 bài)**. Tập kiểm định (`Validation: 1,391 bài`) và tập kiểm thử (`Test: 1,390 bài`) được biến đổi (`transform`) thông qua cùng vectorizer đã fit.
2. **Tối ưu hóa siêu tham số có căn cứ thực nghiệm:** Chạy grid search trên 54 tổ hợp tham số và đánh giá tự động trên tập Validation dựa trên 4 tiêu chí định lượng không giám sát.
3. **Chuẩn hóa Artifacts:** Xuất đủ 10 tệp tin đồng bộ trong `artifacts/tfidf/` bao gồm vectorizer, từ vựng, cấu hình tối ưu, metadata thực thi, và các ma trận thưa song song với danh sách `article_id`.

---

## 2. Dữ liệu đầu vào & Tiền xử lý

- **Tập dữ liệu chuẩn:** `data/processed/p2_segmented.csv` (Dataset v1.0, 9,274 bài viết).
- **Biến thể văn bản:** Cột `text` (kết hợp `processed_title` và `processed_content`).
- **Đặc thù ngôn ngữ tiếng Việt:** Văn bản đã được tách từ bằng thư viện `PyVi`
- **Danh sách từ dừng:** `configs/vietnamese_stopwords.txt` (loại bỏ các hư từ, liên từ, trợ từ nhưng bảo lưu từ phủ định như `không`, `chưa`, `chẳng` và từ chuyên mục).

---

## 3. Tiêu chí Đánh giá & Kết quả Tinh chỉnh

### 3.1. Không gian tìm kiếm
- `ngram_range`: `[1, 1]` (unigram từ ghép), `[1, 2]` (unigram + bigram)
- `min_df`: `[2, 3, 5]` (ngưỡng tần số tài liệu tối thiểu)
- `max_df`: `[0.80, 0.85, 0.90]` (ngưỡng lọc từ phổ biến)
- `max_features`: `[10000, 15000, 20000]` (trần kích thước từ vựng)
- `sublinear_tf`: `true` ($1 + \log(TF)$ giúp giảm sự áp đảo của các từ lặp lại nhiều lần trong bài báo dài)
- `norm`: `l2` (chuẩn hóa Euclidean chiều dài vector về 1)

### 3.2. Tiêu chí chọn Best Configuration trên tập Validation
Vì TF-IDF là phương pháp trích xuất đặc trưng không giám sát (unsupervised), thuật toán sử dụng hàm tính điểm phạt heuristic (`heuristic_score`) trên tập Validation:
1. **Không có bài viết rỗng (`val_zero_vector_docs == 0`):** Phạt 10,000 điểm nếu có bài báo bị biến thành vector toàn 0.
2. **Tỷ lệ Out-Of-Vocabulary (OOV) thấp nhất:** Tỷ lệ số token trên tập Validation không có trong từ điển Train phải đạt mức tối thiểu.
3. **Số lượng từ kích hoạt trung bình (`val_mean_active_terms`) trong khoảng tối ưu (80 - 250):** Điểm phạt khoảng cách tính theo $|mean - 150| \times 0.05$.
4. **Kích thước vocabulary đúng trần ngân sách (`vocab_size <= max_features`).**

### 3.3. Kết quả chọn lựa
Toàn bộ nhật ký 54 cấu hình được lưu tại `results/tfidf/tfidf_tuning_validation_results.csv`.

| Chỉ số | Giá trị | Nhận xét |
| :--- | :--- | :--- |
| **Cấu hình tối ưu** | **Candidate 6** | `ngram_range=[1, 1]`, `min_df=2`, `max_df=0.85`, `max_features=20000` |
| **Kích thước từ vựng (Train Vocab)** | **20,000** | Đầy đủ và bao quát miền chủ đề tin tức |
| **Tỷ lệ OOV trên tập Validation** | **2.5578%** | Độ phủ từ vựng cực kỳ xuất sắc (chỉ ~2.5% từ mới) |
| **Số từ kích hoạt trung bình (Val)** | **190.75 từ/bài** | Nằm trong dải lý tưởng [80 - 250], dồi dào tín hiệu cho phân cụm |
| **Số bài báo vector rỗng (Val)** | **0 bài (0.0%)** | Đảm bảo 100% bài viết đều có biểu diễn hợp lệ |
| **Độ thưa ma trận (Sparsity)** | **99.05%** | Tiết kiệm bộ nhớ, phù hợp cho K-Means ma trận thưa |

---

## 4. Đặc tả Artifacts Bàn giao (`artifacts/tfidf/`)

| Tên tệp tin | Định dạng | Kích thước | Mô tả & Hợp đồng giao tiếp |
| :--- | :--- | :--- | :--- |
| `tfidf_vectorizer.joblib` | Joblib / Scikit-Learn | ~888 KB | Model vectorizer đã fit trên Train, sẵn sàng transform văn bản mới |
| `feature_names.json` | JSON Array (20,000 items) | ~302 KB | Danh sách từ vựng theo đúng thứ tự chỉ số cột ma trận |
| `config.json` | JSON Object | ~1 KB | Cấu hình tham số trích xuất đã chốt |
| `run_metadata.json` | JSON Object | ~1.2 KB | Metadata thực thi, shapes, sparsity, OOV, active terms |
| `tfidf_train.npz` | Scipy CSR Matrix | ~11.2 MB | Ma trận đặc trưng Train thưa: shape `(6493, 20000)` |
| `train_article_ids.json` | JSON Array | ~149 KB | Danh sách 6,493 `article_id` khớp từng dòng của `tfidf_train.npz` |
| `tfidf_val.npz` | Scipy CSR Matrix | ~2.3 MB | Ma trận đặc trưng Val thưa: shape `(1391, 20000)` |
| `val_article_ids.json` | JSON Array | ~32 KB | Danh sách 1,391 `article_id` khớp từng dòng của `tfidf_val.npz` |
| `tfidf_test.npz` | Scipy CSR Matrix | ~2.3 MB | Ma trận đặc trưng Test thưa: shape `(1390, 20000)` (**Khóa đến M05**) |
| `test_article_ids.json` | JSON Array | ~32 KB | Danh sách 1,390 `article_id` khớp từng dòng của `tfidf_test.npz` |

---

## 5. Hướng dẫn Tái sử dụng cho TV1 & Các Thành viên khác

### 5.1. Nạp ma trận và Article IDs trong Python

```python
from pathlib import Path
from src.features.tfidf import load_tfidf_artifacts

# Nạp dữ liệu train cho K-Means (TV1) hoặc NMF/LDA (TV2)
artifacts_dir = Path("artifacts/tfidf")
train_bundle = load_tfidf_artifacts(artifacts_dir, split="train")

matrix = train_bundle["matrix"]            # scipy.sparse.csr_matrix (6493, 20000)
article_ids = train_bundle["article_ids"]  # list of str (length 6493)
feature_names = train_bundle["feature_names"] # list of 20000 terms
vectorizer = train_bundle["vectorizer"]    # TfidfVectorizer

print(f"Loaded matrix: {matrix.shape}, first article_id: {article_ids[0]}")
```

### 5.2. Biến đổi văn bản mới

```python
new_articles = ["Chính_phủ ban_hành nghị_định mới về phát_triển trí_tuệ_nhân_tạo."]
new_matrix = vectorizer.transform(new_articles)
print(f"New matrix shape: {new_matrix.shape}") # (1, 20000)
```

---

## 6. Kết quả Kiểm thử

Toàn bộ 59 unit và integration tests (50 bài baseline + 9 bài TF-IDF) đều đã vượt qua 100%:
- `test_stopwords_loading`: Đọc đúng file stopwords tiếng Việt, bỏ qua comment và khoảng trắng.
- `test_create_tfidf_vectorizer_config`: Khởi tạo đúng các tham số.
- `test_fit_transform_no_data_leakage`: Đảm bảo từ vựng tập validation/test không rò rỉ vào train vocab.
- `test_output_sparse_matrix_properties`: Kiểm tra `csr_matrix`, chuẩn hóa L2, không có NaN/Inf.
- `test_article_ids_alignment`: Khớp 1-1 giữa hàng ma trận và danh sách article ID.
- `test_evaluate_validation_vocabulary`: Kiểm tra tính toán OOV, active terms và zero-vector docs.
- `test_save_and_load_artifacts_roundtrip`: Nạp/lưu chính xác nguyên vẹn 10 tệp tin artifacts.
- `test_error_handling`: Bắt lỗi khi dữ liệu rỗng hoặc sai lệch kích thước.
- `test_official_artifacts_exist_and_conform_to_handbook`: Kiểm tra toàn vẹn bộ 10 artifacts chính thức tại `artifacts/tfidf/`.

