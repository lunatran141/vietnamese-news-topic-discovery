# Hướng dẫn thực thi Task D03: Text Preprocessing Variants Pipeline (P0 đến P3)

## 1. Tổng quan & Cấu trúc thư mục

Task D03 nhận dữ liệu sạch sau khử trùng lặp từ Task D02 (`news_articles_deduplicated.csv` gồm 9,274 bài báo) và tạo ra đúng 4 biến thể tiền xử lý độc lập, bảo toàn 100% số dòng, thứ tự `article_id` và các trường metadata.

### Vị trí các tệp liên quan:

| Phân loại | Đường dẫn tệp | Mô tả |
| :--- | :--- | :--- |
| **Dữ liệu đầu vào** | `data/interim/news_articles_deduplicated.csv` | Dữ liệu sạch sau khử trùng lặp ở D02 (9,274 dòng). |
| **Dữ liệu đầu ra sạch** | `data/processed/p0_minimal.csv` | Biến thể P0: Làm sạch tối thiểu (cho BERT / LLM). |
| | `data/processed/p1_normalized.csv` | Biến thể P1: Chuẩn hóa có kiểm soát. |
| | `data/processed/p2_segmented.csv` | Biến thể P2: Tách từ ghép tiếng Việt (cho BoW / TF-IDF). |
| | `data/processed/p3_no_stopwords.csv` | Biến thể P3: Lọc từ dừng, bảo vệ từ phủ định. |
| **Cấu hình tham số** | `configs/preprocessing.yaml` | Cấu hình regex boilerplate, thư viện PyVi, từ phủ định. |
| | `configs/vietnamese_stopwords.txt` | Danh mục từ dừng tiếng Việt chuẩn đã tinh lọc. |
| **Mã nguồn tiền xử lý** | `src/preprocessing/clean_text.py` | Mô-đun làm sạch tối thiểu P0. |
| | `src/preprocessing/normalize_text.py` | Mô-đun chuẩn hóa có kiểm soát P1. |
| | `src/preprocessing/segment_words.py` | Mô-đun tách từ tiếng Việt P2 (`pyvi`). |
| | `src/preprocessing/remove_stopwords.py` | Mô-đun lọc từ dừng P3. |
| | `src/preprocessing/build_variants.py` | Script tổng điều phối tạo 4 file CSV và tính thống kê. |
| | `src/preprocessing/build_splits.py` | Khung sườn (Skeleton) chia tập Train/Val/Test cho D05. |
| **Kiểm thử tự động** | `tests/test_preprocessing.py` | Bộ Pytest kiểm thử 10 tiêu chí nghiệm thu tự động. |
| **Báo cáo & Thống kê** | `results/preprocessing_data/preprocessing_before_after.csv` | Thống kê số lượng token, từ vựng qua 4 biến thể. |
| | `results/preprocessing_data/preprocessing_samples.csv` | 25 mẫu đối chiếu 6 cột song song giữa các biến thể. |
| | `results/preprocessing_data/p1_length_drop_warnings.csv` | Danh sách 29 bài viết bị giảm >20% độ dài ở P1. |
| **Thư mục biểu đồ** | `results/preprocessing_data/figures/` | Bộ 5 đồ thị PNG chất lượng cao (300 DPI). |
| **Chính sách tiền xử lý** | `docs/preprocessing_policy.md` | Tài liệu báo cáo chính sách 4 biến thể và bảo vệ từ phủ định. |
| **Notebook tương tác** | `notebooks/03_preprocessing_variants.ipynb` | Jupyter Notebook phân tích và trực quan hóa chi tiết. |

---

## 2. Chuẩn bị môi trường

Đảm bảo môi trường ảo Python 3.13 đã kích hoạt trên Windows PowerShell và các thư viện cần thiết đã được cài đặt:

```powershell
# Kích hoạt virtualenv
.\venv\Scripts\Activate.ps1

# Cài đặt hoặc cập nhật thư viện cần thiết
pip install -r requirements.txt
```


## 3. Các bước thực thi

Quy trình thực thi Task D03 bao gồm 4 bước tuần tự:

### Bước 1: Chạy Pipeline tạo 4 biến thể dữ liệu (P0 -> P3)

Thực thi module điều phối chính để sinh 4 file CSV tại `data/processed/` và bảng thống kê trước/sau:

```powershell
python -m src.preprocessing.build_variants
```

### Bước 2: Phân tích bổ sung & Xuất file mẫu kiểm toán

Thực thi script phân tích để tính Top 20 từ dừng và xuất 25 mẫu đối chiếu trực quan:

```powershell
python scripts/analyze_d03.py
```

### Bước 3: Chạy bộ kiểm thử tự động (Pytest)

Chạy bộ test suite của Task D03 để nghiệm thu toàn bộ 10 tiêu chí kỹ thuật chốt chặn:

```powershell
pytest tests/test_preprocessing.py -v
```

Hoặc chạy toàn bộ kiểm thử của toàn dự án (D01, D02, D03 - tổng cộng 33 bài test):

```powershell
pytest -v
```

### Bước 4 (Tùy chọn): Mở và tương tác trên Jupyter Notebook

Nếu cần rà soát trực quan số liệu hoặc trình bày báo cáo:

```powershell
jupyter notebook notebooks/03_preprocessing_variants.ipynb
```

---

## 4. Bảng đặc tả 4 biến thể tiền xử lý (Variants Matrix)

| Biến thể | Mục đích kỹ thuật | Thuật toán / Thư viện | Biến đổi chính | Mô hình đích sử dụng | Ràng buộc chốt chặn |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P0 - Minimal Cleaning** | Giữ văn bản sạch tự nhiên, an toàn cú pháp | Unicode NFC, regex, BeautifulSoup | Gỡ HTML, chuẩn khoảng trắng, gỡ ký tự điều khiển | PhoBERT, Multilingual BERT, LLM, BERTopic | Không lowercase, không xóa dấu câu, không tách từ |
| **P1 - Controlled Normalization** | Giảm nhiễu văn bản có kiểm soát | Regex matching, string normalize | Lowercase, thay token `URL`/`EMAIL`, gỡ lặp tiêu đề & boilerplate | Word Embeddings, TF-IDF có kiểm soát | Cảnh báo bài giảm >20% độ dài, fallback chống rỗng |
| **P2 - Word Segmentation** | Nhóm các từ tố tiếng Việt thành từ ghép có nghĩa | `pyvi.ViTokenizer` (v0.1.1) | Nối từ ghép bằng dấu gạch dưới `_` (ví dụ: `kinh_tế`, `công_nghệ`) | BoW, TF-IDF, K-Means, NMF, LDA | Giữ nguyên thứ tự dòng và metadata, ghi rõ config |
| **P3 - Stopword Removal** | Bóc tách tác động của hư từ ngữ pháp | Danh sách lọc chuẩn cố định | Lọc bỏ từ dừng có tần suất cao nhưng ít giá trị phân biệt | Nghiên cứu bóc tách cho mô hình túi từ | không xóa từ phủ định (`không`, `chưa`...) |

---

## 5. Bảng thống kê kết quả & Deliverables nghiệm thu

### 5.1. Bảng số liệu biến chuyển qua 4 biến thể

| Biến thể | Số bài viết | Tổng Token | Số từ TB/bài | Median | P5 | P95 | Kích thước từ vựng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original (Sau Dedup)** | 9,274 | 6,756,487 | 728.54 | 609.0 | 185.0 | 1,514.0 | 115,033 |
| **P0 (Minimal)** | 9,274 | 6,756,482 | 728.54 | 609.0 | 185.0 | 1,514.0 | 113,175 |
| **P1 (Normalized)** | 9,274 | 6,742,822 | 727.07 | 607.0 | 184.0 | 1,512.0 | 100,997 |
| **P2 (Segmented)** | 9,274 | 5,989,915 | 645.88 | 539.0 | 165.0 | 1,340.0 | 61,108 |
| **P3 (No Stopwords)** | 9,274 | 4,327,312 | 466.61 | 386.0 | 118.0 | 973.0 | 60,861 |


