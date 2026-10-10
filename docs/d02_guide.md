# Hướng dẫn thực thi Task D02: Duplicate Audit & Deduplication Pipeline

Tài liệu này hướng dẫn chi tiết quy trình chạy, kiểm thử và phân tích kết quả của **Task D02 — Khử trùng lặp dữ liệu bài báo tiếng Việt** (Deduplication Pipeline).

---

## 1. Tổng quan & Cấu trúc thư mục

Task D02 triển khai đường ống khử trùng lặp 2 tầng (Exact Duplicate và Near Duplicate) trên tập dữ liệu đã qua xác thực từ Task D01 (`news_articles_validated.csv` gồm 9,304 bài báo).

### Sơ đồ luồng xử lý:

```
data/interim/news_articles_validated.csv (9,304 bài)
  │
  ├──► Tầng 1: Exact Deduplication (URL -> Content Hash SHA256 -> Normalized Title)
  │      └── Loại 13 bài trùng tiêu đề
  │
  ├──► Tầng 2: Near Deduplication (MinHash LSH + Dual-Scoring + Union-Find)
  │      └── Loại 33 bài sao chép nội dung / thông cáo báo chí
  │
  ├──► data/interim/news_articles_deduplicated.csv (9,258 bài sạch)
  │
  ├──► results/dedup_data/ (Báo cáo, danh sách loại bỏ, cặp ứng viên)
  │      ├── dedup_removed.csv
  │      ├── dedup_candidates_review.csv
  │      ├── dedup_before_after.csv
  │      └── figures/ (4 biểu đồ trực quan hóa)
  │
  └──► tests/test_deduplication.py (Bộ kiểm thử tự động 14 tests đạt 100%)
```

### Vị trí các tệp liên quan:

| Phân loại | Đường dẫn tệp | Mô tả |
| :--- | :--- | :--- |
| **Dữ liệu đầu vào** | `data/interim/news_articles_validated.csv` | Dữ liệu thẩm định từ D01 (9,304 dòng). |
| **Dữ liệu đầu ra sạch** | `data/interim/news_articles_deduplicated.csv` | Tập bài viết sạch sau dedup (9,258 dòng). |
| **Mã nguồn chính** | `src/preprocessing/deduplicate.py` | Pipeline khử trùng lặp 2 tầng. |
| **Mã trực quan hóa** | `scripts/run_d02_viz.py` | Script sinh biểu đồ, policy md và notebook. |
| **Kiểm thử tự động** | `tests/test_deduplication.py` | Bộ Pytest xác thực 14 tiêu chí kỹ thuật. |
| **Nhật ký loại bỏ** | `results/dedup_data/dedup_removed.csv` | 46 bài viết bị loại kèm bằng chứng truy vết. |
| **Bảng ứng viên** | `results/dedup_data/dedup_candidates_review.csv` | 2,098 cặp ứng viên được phân loại quyết định. |
| **Thống kê trước/sau** | `results/dedup_data/dedup_before_after.csv` | Bảng phân bổ bài viết theo source và category. |
| **Thư mục biểu đồ** | `results/dedup_data/figures/` | 4 đồ thị PNG phân tích tương đồng và phân phối. |
| **Chính sách dedup** | `docs/dedup_policy.md` | Tài liệu báo cáo chính sách và lý giải ngưỡng. |
| **Notebook tương tác** | `notebooks/02_deduplication.ipynb` | Jupyter Notebook phân tích và trực quan hóa. |

---

## 2. Chuẩn bị môi trường

Trước khi chạy, kích hoạt môi trường ảo Python của dự án và đảm bảo các thư viện phụ thuộc đã sẵn sàng:

### Trên Windows (PowerShell):
```powershell
# Kích hoạt virtualenv
.\venv\Scripts\Activate.ps1

# Cài đặt hoặc cập nhật thư viện cần thiết (nếu chưa có)
pip install -r requirements.txt
```

> [!NOTE]
> Task D02 sử dụng thư viện `datasketch` để tạo chỉ mục MinHash LSH hiệu năng cao và `matplotlib`, `seaborn` cho trực quan hóa.

---

## 3. Các bước chạy đường ống (Execution Steps)

Quy trình thực thi D02 bao gồm 3 bước tuần tự:

### Bước 1: Chạy Pipeline khử trùng lặp cốt lõi

Thực thi module khử trùng lặp bằng cách chạy từ thư mục gốc của dự án:

```powershell
python -m src.preprocessing.deduplicate
```

Nếu muốn chạy trên file dữ liệu tùy chỉnh khác:
```powershell
python -m src.preprocessing.deduplicate --input data/interim/news_articles_validated.csv
```

**Quá trình thực thi sẽ in log chi tiết:**
1. **Tầng 1 (Exact Deduplication):**
   - Quét URL chuẩn hóa (loại query param, www, trailing slash).
   - Quét SHA-256 hash của nội dung chuẩn hóa.
   - Quét tiêu đề chuẩn hóa (loại bỏ dấu câu biên và khoảng trắng thừa).
   - Loại 13 bài trùng tiêu đề chính xác (9,304 -> 9,291 bài).
2. **Tầng 2 (Near Deduplication):**
   - Sinh chữ ký MinHash 128 hàm băm với `seed=42`.
   - Lập chỉ mục LSH với ngưỡng lọc 0.70.
   - Tìm ra 2,098 cặp ứng viên tiềm năng.
   - Tính Dual-scoring (Jaccard tiêu đề và Jaccard nội dung).
   - Gom cụm các cặp `remove` bằng cấu trúc Union-Find và chọn bài giữ lại theo Tie-breaking.
   - Loại 33 bài sao chép (9,291 -> 9,258 bài).
3. **Lưu trữ kết quả:**
   - Xuất tập sạch `news_articles_deduplicated.csv` vào `data/interim/`.
   - Xuất 3 bảng báo cáo vào `results/dedup_data/`.

---

### Bước 2: Sinh biểu đồ phân tích và cập nhật tài liệu báo cáo

Chạy script trực quan hóa độc lập:

```powershell
python scripts/run_d02_viz.py
```

**Script này sẽ tự động tạo:**
1. **4 biểu đồ chất lượng cao** trong `results/dedup_data/figures/`:
   - `similarity_distribution.png`: Phân bố điểm Jaccard nội dung và tiêu đề kèm vạch ngưỡng quyết định.
   - `dedup_before_after.png`: Biểu đồ cột so sánh số lượng bài trước và sau dedup theo từng cặp (Nguồn báo $\times$ Chuyên mục).
   - `duplicates_by_category.png`: Số lượng và tỷ lệ % trùng lặp theo 7 chuyên mục.
   - `cross_source_heatmap.png`: Ma trận nhiệt trùng lặp chéo giữa các tòa soạn báo (Tuổi Trẻ, VietNamNet, VnExpress).
2. **Tài liệu chính sách:** `docs/dedup_policy.md` ghi nhận toàn bộ số liệu thống kê mới nhất và nhúng hình ảnh trực quan.
3. **Jupyter Notebook:** `notebooks/02_deduplication.ipynb` được sinh hoàn chỉnh và sẵn sàng để chạy lại từng bước.

---

### Bước 3: Chạy bộ kiểm thử tự động (Pytest)

Chạy bộ test suite của Task D02 để kiểm chứng 14 tiêu chí kỹ thuật:

```powershell
pytest tests/test_deduplication.py -v
```

Hoặc chạy toàn bộ kiểm thử của cả dự án (gồm Task D01 và Task D02):

```powershell
pytest -v
```

**Danh sách các ca kiểm thử được xác thực:**
- `test_row_count_integrity`: Bảo toàn số dòng ($9,304 - 46 = 9,258$).
- `test_no_exact_duplicates_remaining`: 0 bài trùng URL, 0 trùng hash nội dung, 0 trùng tiêu đề chuẩn hóa trong tập sạch.
- `test_balanced_removal_rate`: Không chuyên mục hay nguồn báo nào bị xóa quá 50% số bài.
- `test_review_sample_size`: Tập candidate có $\ge 30$ cặp `remove` (thực tế: 33) và $\ge 30$ cặp `keep_both` (thực tế: 2,046).
- `test_determinism_and_sorting`: Chạy lại 2 lần cho ra file byte-identical 100% (mã băm SHA256 trùng khớp) và `article_id` được sắp xếp tăng dần.
- `test_deliverables_exist`: Kiểm tra đầy đủ sự hiện diện của các tệp đầu ra.
- `test_dedup_schema`, `test_removed_schema`, `test_candidates_schema`, `test_before_after_schema`: Đúng số cột, đúng tên cột và đúng định dạng.
- `test_transitive_closure`: Kiểm thử thuật toán Union-Find gom cụm bắc cầu $A \sim B \sim C$.
- `test_no_false_positive_across_categories`: Đảm bảo không xóa nhầm bài viết giữa các chuyên mục khác nhau.
- `test_article_id_uniqueness`: `article_id` trong tập sạch là duy nhất 100%.
- `test_removed_not_in_dedup`: Tập giao giữa bài bị loại và tập sạch là tập rỗng.

---

### Bước 4 (Tùy chọn): Mở và tương tác trên Jupyter Notebook

Nếu cần rà soát lại trực quan hoặc trình bày báo cáo:

```powershell
jupyter notebook notebooks/02_deduplication.ipynb
```

Notebook đã được thiết lập sẵn mã nguồn để đọc dữ liệu từ `results/dedup_data/` và hiển thị đầy đủ các bảng dữ liệu, phân bố tương đồng và đồ thị trực quan.

---

## 4. Bảng tra cứu ngưỡng quyết định (Decision Matrix)

| Quyết định | Điều kiện áp dụng | Số lượng | Hành động trong Pipeline |
| :--- | :--- | :--- | :--- |
| **`remove`** | `content_sim >= 0.85` HOẶC (`title_sim >= 0.60` VÀ `content_sim >= 0.70`) | 33 cặp | Gom cụm Union-Find, loại bỏ bản sao và giữ lại 1 bản ghi tốt nhất. |
| **`manual_review`** | `0.80 <= content_sim < 0.85` | 19 cặp | Vùng biên nhạy cảm. Hệ thống tự động **giữ lại cả hai** để bảo toàn ngữ liệu. |
| **`keep_both`** | `content_sim < 0.80` | 2,046 cặp | Bài viết cùng sự kiện nhưng hành văn riêng biệt. **Giữ cả hai bài**. |

### Quy tắc chọn bản ghi giữ lại (Tie-breaking Rule):
Khi hai hoặc nhiều bài viết thuộc cùng một cụm trùng lặp, bài được giữ lại (`keeper_id`) được chọn theo thứ tự ưu tiên nghiêm ngặt:
1. **Nội dung dài hơn:** `len(content)` giảm dần (ưu tiên bài viết đầy đủ, không bị cắt ngắn).
2. **Có ngày xuất bản hợp lệ:** `published_at` không rỗng (ưu tiên bài có metadata thời gian chuẩn).
3. **ID xuất hiện trước:** `article_id` nhỏ hơn theo thứ tự từ điển (đảm bảo tính tất định).
4. **Tuyệt đối không dùng `category_gold`:** Để tránh gây thiên vị phân phối chuyên mục.

---

## 5. Xử lý sự cố thường gặp (Troubleshooting)

### 1. Lỗi mã hóa Unicode trên Windows (`charmap codec can't encode...`)
- **Nguyên nhân:** Console Windows sử dụng bảng mã mặc định `cp1252` thay vì `UTF-8`.
- **Khắc phục:** Script `scripts/run_d02_viz.py` đã tích hợp cấu hình `sys.stdout.reconfigure(encoding="utf-8")`. Nếu chạy lệnh PowerShell trực tiếp in tiếng Việt, hãy chạy trước lệnh:
  ```powershell
  [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
  ```

### 2. Thiếu thư viện `datasketch`
- **Thông báo lỗi:** `ModuleNotFoundError: No module named 'datasketch'`
- **Khắc phục:** Chạy lệnh cài đặt:
  ```powershell
  pip install datasketch
  ```

### 3. Kiểm thử Pytest báo file không khớp (Determinism failure)
- **Nguyên nhân:** File CSV bị mở và chỉnh sửa thủ công bằng Excel làm thay đổi định dạng xuống dòng (CRLF/LF) hoặc mất UTF-8 BOM.
- **Khắc phục:** Chạy lại `python -m src.preprocessing.deduplicate` để tạo lại các file chuẩn `utf-8-sig`.

