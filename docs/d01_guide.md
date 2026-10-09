# Hướng dẫn chạy & Nghiệm thu Task D01: EDA & Data Quality

Tài liệu hướng dẫn quy trình kiểm tra chất lượng dữ liệu, chạy phân tích EDA và kiểm thử tự động cho Task D01.

---

## 1. Dữ liệu đầu vào
- File dữ liệu thô: `data/raw/news_articles.csv`
- Quy mô: 9,304 bài báo từ 3 nguồn (VnExpress, Tuổi Trẻ, VietnamNet) phát hành tháng 09/2026.
- Kiểm tra tính toàn vẹn (SHA-256):
  `548b35953368857373e8f26639531b96ab83b742ed6432bff0b6462661b5077a`

---

## 2. Cách chạy kiểm tra và thẩm định dữ liệu 
Thực thi script thẩm định schema và miền giá trị trên dòng lệnh:

```bash
python -m src.preprocessing.validate_data --expected-rows 9304 --year 2026 --month 9
```

Script sẽ kiểm tra: schema 7 cột bắt buộc, tính duy nhất của `article_id`, không khuyết thiếu dữ liệu, ngày tháng thuộc tháng 09/2026 và miền giá trị nhãn hợp lệ.

---

## 3. Cách chạy Notebook EDA
Mở và thực thi toàn bộ luồng phân tích trên Jupyter Notebook:

```bash
jupyter notebook notebooks/01_data_eda.ipynb
```
Hoặc chọn **Run All** trong VS Code / Jupyter Lab để tái lập toàn bộ thống kê 4 tầng (Schema, Phân bố, Nhiễu văn bản, Trùng lặp).

---

## 4. Danh mục Output & Kết quả bàn giao
- Dữ liệu trung gian sau thẩm định:
  - `data/interim/news_articles_validated.csv` (9,304 bản ghi)
- Báo cáo chất lượng và mẫu kiểm toán:
  - `results/data_quality/data_quality_summary.csv` (Bảng 31 chỉ số kiểm tra)
  - `results/data_quality/issue_samples.csv` (46 mẫu lỗi/nhiễu: HTML, URL, exact duplicate, near duplicate)
  - `results/data_quality/eda_report.md` (Báo cáo tổng hợp kết quả EDA và khuyến nghị cho D02/D03)

---

## 5. Cách chạy kiểm thử tự động
Chạy bộ kiểm thử tự động 7 tiêu chí nghiệm thu bằng Pytest. Bộ test đảm bảo: mã hash SHA-256 dữ liệu raw được bảo toàn, schema hợp lệ và tập dữ liệu `interim` được sinh ra đầy đủ.

```bash
pytest tests/test_data_quality.py -v
```
