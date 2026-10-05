# Pipeline Thu Thập Dữ Liệu Báo Chí Tiếng Việt (News Crawler Pipeline)

Module thu thập, làm sạch và hợp nhất dữ liệu văn bản tin tức tiếng Việt từ 3 tòa báo lớn (VnExpress, Tuổi Trẻ, VietnamNet) phục vụ đề tài nghiên cứu Khám phá và Phân cụm Chủ đề (Topic Discovery & Clustering).

---

## 1. Môi trường & Cài đặt Dependencies

* **Phiên bản Python khuyến nghị:** `Python 3.10` trở lên (đã kiểm thử tương thích tốt trên `Python 3.11` và `Python 3.13`).
* **Cài đặt môi trường ảo và gói phụ thuộc:**

```powershell
# 1. Tạo môi trường ảo
python -m venv .venv

# 2. Kích hoạt môi trường trên Windows PowerShell
.\.venv\Scripts\Activate.ps1

# 3. Cài đặt các thư viện từ requirements.txt
pip install -r requirements.txt
```

* **Các thư viện cốt lõi:**
  * `requests>=2.31.0`: Gửi HTTP requests với Session và Retry adapter.
  * `beautifulsoup4>=4.12.0`: Phân tích cú pháp DOM HTML.
  * `feedparser>=6.0.10`: Hỗ trợ phân tích dữ liệu RSS/XML.
  * `pandas>=2.0.0`: Quản lý DataFrame, xử lý và làm sạch dữ liệu bảng.
  * `tqdm>=4.66.0`: Thanh tiến trình trực quan hóa khi crawl.
  * `urllib3>=2.0.0`: Quản lý Connection Pool.

---

## 2. Phạm vi Dữ liệu & Nguồn Báo Thu Thập

* **Khoảng thời gian mục tiêu:** Từ ngày `01/09/2026` đến hết ngày `30/09/2026` (chuẩn múi giờ `Asia/Ho_Chi_Minh` / `UTC+7`).
* **3 Nguồn báo chính thức:**
  * **VnExpress:** `vnexpress.net`
  * **Tuổi Trẻ:** `tuoitre.vn`
  * **VietnamNet:** `vietnamnet.vn`
* **7 Chuyên mục chuẩn hóa (`category_gold`):**
  1. `Kinh doanh`
  2. `Khoa học công nghệ`
  3. `Thể thao`
  4. `Giáo dục`
  5. `Sức khỏe`
  6. `Pháp luật`
  7. `Giải trí`

---

## 3. Câu Lệnh Chạy Chuẩn & Cấu Hình Thử Nghiệm

Mọi câu lệnh đều được thực thi tại thư mục gốc của repository (`vietnamese-news-topic-discovery`):

```powershell
python -m src.data.crawlers.run_all [CÁC_THAM_SỐ]
```

### Bảng cấu hình các kịch bản chạy thực tế

| Kịch bản | Mục đích kiểm tra | Câu lệnh chi tiết |
| :--- | :--- | :--- |
| **Smoke Test (~20 bài)** | Kiểm tra kết nối mạng và parser từng trang | `python -m src.data.crawlers.run_all --from-date 2026-09-01 --to-date 2026-09-02 --max-pages-vne 1 --max-pages-ttr 2 --max-pages-vnn 1 --out data/raw/_test_20.csv` |
| **Quick Test (~100 bài)** | Kiểm tra độ tương thích schema và logic khử trùng lặp | `python -m src.data.crawlers.run_all --from-date 2026-09-01 --to-date 2026-09-03 --max-pages-vne 2 --max-pages-ttr 3 --max-pages-vnn 2 --out data/raw/_test_100.csv` |
| **Mini Corpus (~500 bài)** | Đánh giá sơ bộ phân bố chuyên mục và kiểm thử EDA | `python -m src.data.crawlers.run_all --from-date 2026-09-01 --to-date 2026-09-10 --max-pages-vne 5 --max-pages-ttr 8 --max-pages-vnn 5 --out data/raw/_test_500.csv` |
| **Full Production (~3.000+ bài)** | Cào trọn vẹn tập dữ liệu tháng 9/2026 | `python -m src.data.crawlers.run_all --from-date 2026-09-01 --to-date 2026-09-30 --max-pages-vne 40 --max-pages-ttr 50 --max-pages-vnn 30 --out data/raw/news_articles.csv` |

### Chế độ gộp nhanh từ các file có sẵn
Nếu đã có sẵn 3 file `vnexpress_articles.csv`, `tuoitre_articles.csv`, `vietnamnet_articles.csv` trong thư mục `data/raw/`:

```powershell
python -m src.data.crawlers.run_all --merge-only
```

---

## 4. Đặc Tả Dữ Liệu Đầu Ra

* **Đường dẫn lưu trữ:** `data/raw/news_articles.csv`
* **Mã hóa ký tự (Encoding):** `utf-8-sig` (hiển thị tiếng Việt có dấu chuẩn xác trên Excel và Pandas).
* **Quy chuẩn Schema 7 trường dữ liệu:**

| Cột | Kiểu | Mô tả chi tiết | Ví dụ |
| :--- | :--- | :--- | :--- |
| `article_id` | `string` | Khóa chính duy nhất, SHA-1 từ URL gốc kèm tiền tố báo | `VNE_c3d7f9a12b4e` |
| `title` | `string` | Tiêu đề bài viết (đã chuẩn hóa khoảng trắng) | `Giá vàng hôm nay 15/9: Bật tăng mạnh...` |
| `content` | `string` | Toàn bộ văn bản (sapo + thân bài ghép nối) | `Sáng ngày 15/9, thị trường ghi nhận biến động...` |
| `published_at` | `string` | Thời gian xuất bản định dạng ISO 8601 kèm múi giờ | `2026-09-15T08:30:00+07:00` |
| `source` | `string` | Nguồn báo (`vnexpress`, `tuoitre`, `vietnamnet`) | `vnexpress` |
| `url` | `string` | URL gốc (đã bóc tách bỏ query params và hash) | `https://vnexpress.net/...` |
| `category_gold` | `string` | Nhãn chuyên mục được quy về 1 trong 7 nhóm chuẩn | `Kinh doanh` |

---

## 5. Thời Gian Thực Thi, Giới Hạn Đã Biết & Xử Lý Sự Cố

### Thời gian chạy đo lường thực tế
* **VnExpress:** ~10 – 15 phút (~3.000 bài).
* **Tuổi Trẻ:** ~8 – 10 phút (~4000 bài, tối ưu nhờ cơ chế dừng sớm theo URL).
* **VietnamNet:** ~10 - 15 phút (~3000 bài).
* **Toàn bộ pipeline:** **Khoảng 30 - 45 phút** cho tập dữ liệu đầy đủ ~10.000 bài.

### Giới hạn kỹ thuật đã biết
* **Bài viết Multimedia:** Các liên kết dạng Video, Podcast, Ảnh, Infographic bị loại bỏ tự động do mật độ từ ngữ thấp (`MIN_WORD_COUNT < 50 từ`).
* **Trùng lặp sự kiện:** Pipeline xử lý trùng lặp tuyệt đối theo URL và ID (giữ lại bản ghi có độ dài văn bản lớn nhất). Bài toán trùng lặp nội dung giữa các báo khác nhau (Near-deduplication) được bàn giao cho tầng Preprocessing xử lý.

### Cách tiếp tục khi gặp sự cố gián đoạn
Nếu tiến trình bị đứt quãng giữa chừng khi đang cào nguồn thứ 3, không cần chạy lại từ đầu. Sử dụng các cờ bỏ qua (`--skip-*`) để cào bù nguồn còn thiếu:

```powershell
# Ví dụ: VnExpress và Tuổi Trẻ đã hoàn thành, chỉ chạy cào VietnamNet rồi tự động gộp
python -m src.data.crawlers.run_all --skip-vne --skip-ttr
```

---

## 6. Đạo Đức Thu Thập Dữ Liệu & Nguyên Tắc Lịch Sự

* **Mục đích phi thương mại:** Dữ liệu thu thập phục vụ độc quyền cho mục đích học thuật, nghiên cứu thuật toán phân cụm trong khuôn khổ môn học Xử lý Ngôn ngữ Tự nhiên.
* **Kiểm soát tải máy chủ (Rate Limiting):**
  * Giới hạn tối đa 6 luồng worker song song (`MAX_WORKERS = 6`).
  * Duy trì độ trễ giữa các yêu cầu: `DELAY_ARTICLE = 0.1s` và `DELAY_PAGE = 0.3s` để tránh gây quá tải hạ tầng đối tác.
* **Tối ưu băng thông:** Tận dụng nén `gzip` và cơ chế lọc ngày sớm (Early Stopping) ngay trên URL để không tải các trang HTML dư thừa nằm ngoài phạm vi nghiên cứu.