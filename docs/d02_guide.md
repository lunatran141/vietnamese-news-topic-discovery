# Hướng dẫn thực thi Task D02: Duplicate Audit & Deduplication Pipeline


## 1. Tổng quan & Cấu trúc thư mục

Task D02 triển khai đường ống khử trùng lặp 2 tầng (Exact Duplicate và Near Duplicate) trên tập dữ liệu đã qua xác thực từ Task D01 (`news_articles_validated.csv` gồm 9,304 bài báo).

### Sơ đồ luồng xử lý:


### Vị trí các tệp liên quan:

| Phân loại | Đường dẫn tệp | Mô tả |
| :--- | :--- | :--- |
| **Dữ liệu đầu vào** | `data/interim/news_articles_validated.csv` | Dữ liệu thẩm định từ D01 (9,304 dòng). |
| **Dữ liệu đầu ra sạch** | `data/interim/news_articles_deduplicated.csv` | Tập bài viết sạch sau dedup (9,274 dòng). |
| **Mã nguồn chính** | `src/preprocessing/deduplicate.py` | Pipeline khử trùng lặp 2 tầng an toàn. |
| **Kiểm thử tự động** | `tests/test_deduplication.py` | Bộ Pytest xác thực 16 tiêu chí kỹ thuật. |
| **Nhật ký loại bỏ** | `results/dedup_data/dedup_removed.csv` | 30 bài viết bị loại kèm bằng chứng truy vết. |
| **Bảng ứng viên** | `results/dedup_data/dedup_candidates_review.csv` | 2,122 cặp ứng viên được phân loại quyết định và cờ xung đột category. |
| **Thống kê trước/sau** | `results/dedup_data/dedup_before_after.csv` | Bảng phân bổ bài viết theo source và category. |
| **Thư mục biểu đồ** | `results/dedup_data/figures/` | 4 đồ thị PNG phân tích tương đồng và phân phối. |
| **Chính sách dedup** | `results/dedup_data/dedup_policy.md` | Tài liệu báo cáo chính sách và lý giải chọn ngưỡng. |
| **Notebook tương tác** | `notebooks/02_deduplication.ipynb` | Jupyter Notebook phân tích và trực quan hóa. |

---

## 2. Chuẩn bị môi trường

Trước khi chạy, kích hoạt môi trường ảo Python của dự án và đảm bảo các thư viện phụ thuộc đã sẵn sàng:

### Trên Windows (PowerShell):
```powershell
# Kích hoạt virtualenv
.\venv\Scripts\Activate.ps1

# Cài đặt hoặc cập nhật thư viện cần thiết
pip install -r requirements.txt
```


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

### Bước 2: Sinh biểu đồ phân tích và cập nhật tài liệu báo cáo

Chạy script trực quan hóa độc lập:

```powershell
python scripts/run_d02_viz.py
```

### Bước 3: Chạy bộ kiểm thử tự động (Pytest)

Chạy bộ test suite của Task D02 để kiểm chứng các tiêu chí kỹ thuật:

```powershell
pytest tests/test_deduplication.py -v
```

Hoặc chạy toàn bộ kiểm thử của cả dự án (gồm Task D01 và Task D02 - tổng cộng 23 tests):

```powershell
pytest -v
```

### Bước 4 (Tùy chọn): Mở và tương tác trên Jupyter Notebook

Nếu cần rà soát lại trực quan hoặc trình bày báo cáo:

```powershell
jupyter notebook notebooks/02_deduplication.ipynb
```

---

## 4. Bảng tra cứu ngưỡng quyết định (Decision Matrix)

| Quyết định | Điều kiện áp dụng | Số lượng | Hành động trong Pipeline |
| :--- | :--- | :--- | :--- |
| **`remove`** | `content_sim >= 0.85` HOẶC (`title_sim >= 0.60` VÀ `content_sim >= 0.70` VÀ `days_diff <= 2d`) không có xung đột chuyên mục | 30 cặp | Gom cụm Union-Find, loại bỏ bản sao và giữ lại 1 bản ghi tốt nhất. |
| **`manual_review`** | `0.80 <= content_sim < 0.85` HOẶC có xung đột `category_conflict == True` | 18 cặp | Vùng biên nhạy cảm và xung đột nhãn. Hệ thống tự động **giữ lại cả hai** để bảo toàn ground truth. |
| **`keep_both`** | `content_sim < 0.80` HOẶC trùng tiêu đề nhưng khác ngày đăng (`days_diff > 2d`, `content_sim < 0.70`) | 2,074 cặp | Bài viết cùng sự kiện hoặc tin tức định kỳ (giá xăng, thể thao). **Giữ cả hai bài**. |

### Quy tắc chọn bản ghi giữ lại:
Khi hai hoặc nhiều bài viết thuộc cùng một cụm trùng lặp, bài được giữ lại (`keeper_id`) được chọn theo thứ tự ưu tiên nghiêm ngặt:
1. **Nội dung dài hơn:** `len(content)` giảm dần (ưu tiên bài viết đầy đủ, không bị cắt ngắn).
2. **Có ngày xuất bản hợp lệ:** `published_at` không rỗng (ưu tiên bài có metadata thời gian chuẩn).
3. **ID xuất hiện trước:** `article_id` nhỏ hơn theo thứ tự từ điển (đảm bảo tính tất định).
4. **Tuyệt đối không dùng `category_gold`:** Để tránh gây thiên vị phân phối chuyên mục.

---

## 5. Xử lý sự cố thường gặp

### 1. Thiếu thư viện `datasketch`
- **Thông báo lỗi:** `ModuleNotFoundError: No module named 'datasketch'`
- **Khắc phục:** Chạy lệnh cài đặt:
  ```powershell
  pip install datasketch
  ```

### 2. Kiểm thử Pytest báo file không khớp (Determinism failure)
- **Nguyên nhân:** File CSV bị mở và chỉnh sửa thủ công bằng Excel làm thay đổi định dạng xuống dòng (CRLF/LF) hoặc mất UTF-8 BOM.
- **Khắc phục:** Chạy lại `python -m src.preprocessing.deduplicate` để tạo lại các file chuẩn `utf-8-sig`.
